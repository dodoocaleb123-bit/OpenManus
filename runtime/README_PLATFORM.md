# OpenManus Platform

An autonomous software engineer built on the OpenManus agent. You describe what to
build; the agent plans, writes code, runs it, tests it, fixes it, and ships it to
GitHub, while you watch and steer from a web UI that shares a live browser with
the agent.

Deployment: see [`../DEPLOY.md`](../DEPLOY.md). Roadmap: [`BUILD_PLAN.md`](BUILD_PLAN.md).

## Unified Assistant

Assistant replies support rendered TeX/LaTeX mathematics through MathJax,
including display equations, fractions, aligned systems, superscripts, and
common AMS constructs. MathJax is loaded by the web client; if the CDN is
unavailable, the original formula text remains visible as a fallback.

Each project has one persistent conversation composer with explicit request modes:

- **Auto** chooses a response, read-only inspection, or implementation based on
  the request. Project context is loaded only when the question is project-related.
- **Answer only** does not load repository excerpts or start project tools.
- **Inspect project** reads bounded, relevant local files and answers without
  starting the build agent or intentionally changing project state.
- **Plan only** returns a plan and risks without running tools or changing files.
- **Implement & verify** starts the autonomous coding workflow for the requested
  change and independent validation.

The selector is saved in the browser. Bounded lexical retrieval prefers relevant
source files and common project documentation; secret-like filenames, generated
directories, and oversized content are excluded. A per-project Memory pane lets
you save up to 6,000 characters of stable preferences and constraints in local
SQLite. Credentials are rejected, and memory/file excerpts are treated as
untrusted context rather than instructions.

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

- **Project tools and guardrails.** Every project has its own workspace. Agent
  processes start there, get command timeouts (`AGENT_COMMAND_TIMEOUT`), and use
  an environment with platform credentials removed. Secret-like configuration
  files are blocked in the file editor, and common credential assignments/tokens
  are redacted from shell/Python results. This is defense in depth, not a
  per-task operating-system sandbox.
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
  push/pull-request/publish. Push, pull-request, and repository-publish actions
  require the user to ask for that action explicitly or approve it in the UI.
  The token stays server-side (a short-lived `GIT_ASKPASS` helper), never in
  remotes or the agent's environment.
- **Durable.** Projects, tasks, events and checkpoints are in SQLite. After a
  restart, interrupted tasks are re-queued; the event stream resumes from
  `Last-Event-ID`. Task-launch retries can include `Idempotency-Key`; replaying a
  matching key returns the original task, while mismatched requests are rejected.
- **Measured quality and speed.** Assistant replies persist total response time and
  time-to-first-token when available. The Metrics pane and `GET /api/metrics` show
  local response/TTFT aggregates, deterministic verification outcomes, and
  task helpfulness feedback. Metrics are derived from local SQLite and are not
  uploaded as telemetry.
- **Task history.** Tasks can be filtered locally by text and status. A completed
  task's saved reply retains its task association and feedback controls when
  reopened.
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
| Status | `GET /api/health` (no auth), `GET /api/status` (model, GitHub, auth), `GET /api/metrics` (task, verification, recovery, latency, feedback, and audit summaries) |
| Projects | `POST/GET /api/projects`, `GET /api/projects/{id}`, `GET /api/projects/{id}/files` |
| Repository intelligence | `GET /api/projects/{id}/repository-map` (bounded file/language/Python-symbol map) |
| Project memory | `GET/PUT /api/projects/{id}/memory` (local-only; 6,000-character limit; credential-like values rejected) |
| Uploads | `POST/GET /api/projects/{id}/uploads`, `GET /api/projects/{id}/uploads/{file_id}` |
| Chat | `GET /api/projects/{id}/chat`, `POST /api/projects/{id}/chat` (persistent conversation; `mode` may be `auto`, `answer`, `inspect`, `plan`, or `implement`) |
| Tasks | `POST /api/tasks` (`{project_id, prompt}`; optional `Idempotency-Key` header), `GET /api/projects/{id}/tasks`, `GET /api/tasks/{id}`, `POST /api/tasks/{id}/cancel`, `POST /api/tasks/{id}/resume`, `POST /api/tasks/{id}/messages`, `POST /api/tasks/{id}/feedback` (`{rating: 1|-1}`) |
| Events | `GET /api/tasks/{id}/events` (SSE, resumable), `GET /api/tasks/{id}/events/history` |
| Git | `GET /api/projects/{id}/git/status\|diff\|log`, `POST …/git/branch\|commit\|push` |
| GitHub | `GET /api/github/user\|repos`, `POST /api/projects/{id}/github/connect\|pull-request\|publish` |
| Browser | `POST/GET /api/browser/sessions`, `…/{sid}/navigate\|click\|type\|press\|click_at\|keyboard`, `GET …/{sid}/screenshot`, `GET /api/browser/projects/{id}/session` |

Event types streamed to the UI include `agent.thought`, `agent.tool_call`,
`agent.tool_result`, `agent.question`, `human.reply`, `coding.validation`,
`coding.repair`, `github.pull_request` and the `task.*` lifecycle events.

Project deletion, build-history deletion, repository publishing, Git push, and
pull-request creation require the `X-OpenManus-Confirm: true` header on direct API
calls. The web UI supplies it only from the corresponding user-triggered control;
project/build deletion and repository publishing also show confirmation dialogs.
The autonomous Git tool separately checks for an explicit matching request or
asks the user through the task UI before push/PR/publish actions.

## Running locally

```bash
pip install -r requirements-platform.txt
playwright install chromium
cp config/config.example.toml config/config.toml   # model + API key (vision-capable recommended)
python run_platform.py                              # http://127.0.0.1:8000
```

Optional: `GITHUB_TOKEN` (repo scope) plus an optional `GITHUB_CLASSIC_TOKEN` fallback, `PLATFORM_PASSWORD` (Basic auth),
`PLATFORM_REQUIRE_AUTH=true` (fail closed if a password is missing),
`PLATFORM_MAX_CONCURRENT_TASKS`, `PLATFORM_MAX_UPLOAD_MB`, `PLATFORM_IMAGE_MAX_TOKENS` (default `6144`),
`PLATFORM_CHAT_MAX_TOKENS` (default `2400`), `REASONING_LLM_MODEL`,
`REASONING_LLM_BASE_URL`, `REASONING_LLM_API_KEY_01` … `_10`,
`REASONING_LLM_ENABLED=true`, `REASONING_LLM_MODE=complex|always|off`,
`REASONING_LLM_MAX_TOKENS` (default `1200`), and `PLATFORM_DATA_DIR`
(database + workspaces location, default `workspace/`).

## Tests

```bash
python -m pytest -q test_platform.py test_coding_loop.py test_v05.py \
  test_shared_browser_bridge.py test_browser_platform.py test_v07_agent.py
```

The suite needs no API keys, network or Chromium: the LLM, GitHub API and browser
page are faked; git operations run against local bare repositories.

## Known limitations

- The agent runs inside the platform container (no per-task sandbox yet). Keys
  are scrubbed from its process environment, secret-like files are blocked from
  the file-editor tool, and common values are redacted from tool output. The
  container filesystem is not a security boundary against malicious project code.
- Browser sessions live in the server process and are not restored after a restart.
- Tasks in different projects may run concurrently unless
  `PLATFORM_MAX_CONCURRENT_TASKS` is set. The Render template sets this to `1`.
- The right context panel includes GitHub, Files, and Evidence tabs. Evidence shows
  plans, validation, artifacts, recovery state, checkpoints, and source URLs from
  recorded task evidence.
- Assistant messages containing fenced `mermaid` blocks are rendered as diagrams
  in the browser with Mermaid; if the CDN is unavailable, the source remains
  visible as a code block. Image math questions receive a larger output budget
  and bounded completion retries so the assistant must finish all visible parts.
- Repository maps are intentionally bounded and metadata-only; they do not read
  arbitrary source contents. Full task-level Docker isolation, durable browser
  profiles, multi-agent roles, app hosting, and RL trajectory integration remain
  follow-on subsystems rather than being represented as complete here.
- The optional reasoning profile is disabled by default. When enabled, it plans and
  reviews complex or risky tasks, while Qwen remains the only model that executes
  tools; deterministic validation remains authoritative.
