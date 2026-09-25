# OpenManus Platform

An autonomous software engineer built on the OpenManus agent. You describe what to
build; the agent plans, writes code, runs it, tests it, fixes it, and ships it to
GitHub, while you watch and steer from a web UI that shares a live browser with
the agent.

Deployment: see [`../DEPLOY.md`](../DEPLOY.md). Roadmap: [`BUILD_PLAN.md`](BUILD_PLAN.md).

## Unified Assistant

Each project now has one persistent assistant composer with an action selector:

- **Discuss** answers questions without changing project files.
- Discuss mode receives a bounded, read-only snapshot of the project tree and
  common architecture files such as README, package manifests, Dockerfiles,
  and dependency files, so it can explain a cloned repository without a mode
  switch. Secrets and generated directories are excluded.
- **Inspect project** starts a read-only task that analyzes the repository and
  reports findings without intentional edits, commits, or pushes.
- **Make changes** starts the autonomous coding workflow for implementation,
  testing, and project changes.

The paperclip button is in the lower-left of the composer. Selected uploads
appear as attachment chips and are sent with the next Discuss message. For
Inspect and Make changes, uploads remain in the project workspace and the task
is told to inspect them when relevant.

## How a build task runs (v0.7)

```
prompt ─▶ PlatformManus (plan → act → observe, up to AGENT_MAX_STEPS)
             tools: workspace bash / python / file editor, platform_git,
                    platform_browser (shared), ask_human, terminate
        ─▶ independent validation (CodingLoop: install deps, run the project's tests/build)
             ├─ pass  ─▶ task succeeded
             └─ fail  ─▶ repair pass with the exact failing output (≤ AGENT_MAX_REPAIR_CYCLES)
```

- **Workspace-scoped tools.** Every project has its own directory
  (`workspace/projects/<id>`). Shell, Python and file edits are confined to it; each
  command gets a timeout (`AGENT_COMMAND_TIMEOUT`) and an environment with API keys
  and tokens removed.
- **Independent validation.** The platform, not the agent, decides success. It
  detects Node (npm/yarn/pnpm), Python (pytest, project venv), Rust and Go projects,
  installs dependencies once per lockfile change, and runs tests/build.
- **Human in the loop.** The agent can ask a question (`ask_human`); the UI shows a
  reply box and the task waits up to `HUMAN_REPLY_TIMEOUT`. Messages you send while
  it works are injected at its next step. Tasks can be stopped and retried.
- **Shared browser.** One Playwright Chromium per project, used by both the agent
  (`platform_browser`: navigate, click, type, scroll, screenshot, …) and you
  (click on the live screenshot, type, press keys). Screenshots are passed to the
  model when it supports images.
- **GitHub.** `platform_git` gives the agent status/diff/log/init/branch/commit/
  push/pull-request/publish. The token stays server-side (a short-lived
  `GIT_ASKPASS` helper), never in remotes or the agent's environment.
- **Durable.** Projects, tasks, events and checkpoints are in SQLite. After a
  restart, interrupted tasks are re-queued; the event stream resumes from
  `Last-Event-ID`.
- **Project uploads.** The Files panel accepts common PDF, Word, spreadsheet,
  presentation, text, ZIP/archive, audio, video and image files. Uploads are
  stored under the project workspace, are visible to the Build agent, and are
  retained by the persistent Docker volume. Archives are stored but never
  extracted automatically.
- **Fail fast on setup problems.** A missing/invalid model configuration fails the
  task immediately with a clear hint; permanent provider errors (401/403/404) are
  not retried.

## API overview

| Area | Endpoints |
| --- | --- |
| Status | `GET /api/health` (no auth), `GET /api/status` (model, GitHub, auth) |
| Projects | `POST/GET /api/projects`, `GET /api/projects/{id}`, `GET /api/projects/{id}/files` |
| Uploads | `POST/GET /api/projects/{id}/uploads`, `GET /api/projects/{id}/uploads/{file_id}` |
| Chat | `GET /api/projects/{id}/chat`, `POST /api/projects/{id}/chat` (persistent project conversation) |
| Tasks | `POST /api/tasks` (`{project_id, prompt}`; 409 if one is running in that project), `GET /api/projects/{id}/tasks`, `GET /api/tasks/{id}`, `POST /api/tasks/{id}/cancel`, `POST /api/tasks/{id}/resume`, `POST /api/tasks/{id}/messages` |
| Events | `GET /api/tasks/{id}/events` (SSE, resumable), `GET /api/tasks/{id}/events/history` |
| Git | `GET /api/projects/{id}/git/status\|diff\|log`, `POST …/git/branch\|commit\|push` |
| GitHub | `GET /api/github/user\|repos`, `POST /api/projects/{id}/github/connect\|pull-request\|publish` |
| Browser | `POST/GET /api/browser/sessions`, `…/{sid}/navigate\|click\|type\|press\|click_at\|keyboard`, `GET …/{sid}/screenshot`, `GET /api/browser/projects/{id}/session` |

Event types streamed to the UI include `agent.thought`, `agent.tool_call`,
`agent.tool_result`, `agent.question`, `human.reply`, `coding.validation`,
`coding.repair`, `github.pull_request` and the `task.*` lifecycle events.

## Running locally

```bash
pip install -r requirements-platform.txt
playwright install chromium
cp config/config.example.toml config/config.toml   # model + API key (vision-capable recommended)
python run_platform.py                              # http://127.0.0.1:8000
```

Optional: `GITHUB_TOKEN` (repo scope) plus an optional `GITHUB_CLASSIC_TOKEN` fallback, `PLATFORM_PASSWORD` (Basic auth),
`PLATFORM_DATA_DIR` (database + workspaces location, default `workspace/`).

## Tests

```bash
python -m pytest -q test_platform.py test_coding_loop.py test_v05.py \
  test_shared_browser_bridge.py test_browser_platform.py test_v07_agent.py
```

The suite needs no API keys, network or Chromium: the LLM, GitHub API and browser
page are faked; git operations run against local bare repositories.

## Known limitations

- The agent runs inside the platform container (no per-task sandbox yet). Keys
  are scrubbed from its environment, but files in the container, including the
  generated `config/config.toml`, are readable to it.
- Browser sessions live in the server process and are not restored after a restart.
- Tasks in different projects run concurrently in one process; size the instance
  accordingly.
