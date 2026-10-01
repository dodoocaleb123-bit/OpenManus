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

Each project has one persistent conversation composer and no task/conversation
mode selector. Every new message is sent to DeepSeek first, unchanged; DeepSeek
chooses a direct response or an ordered, registry-validated capability workflow.
Simple conversation goes through a compact DeepSeek control decision; the long
executable-capability directory is sent only when DeepSeek decides specialist
work is needed. This avoids overwhelming local DeepSeek with a full catalog
for greetings or ordinary questions. On local Ollama, the control decision
requests JSON mode and will retry once with a larger output budget if the
model returns only reasoning or malformed JSON. Both attempts use DeepSeek,
never a keyword classifier or substitute model.
For a direct reply, DeepSeek's `respond` decision is sufficient: a numeric
capability ID is not required. The platform records the generic conversation
capability as bookkeeping and grants no tools on this route.
Only when DeepSeek asks for workspace context does the server add bounded local
file excerpts. Secret-like filenames, generated directories, and oversized
content are excluded. A per-project Memory pane stores up to 6,000 characters
in local SQLite; credentials are rejected and both memory and files are treated
as untrusted reference data, not instructions.

The paperclip button is in the lower-left of the composer. Selected uploads
appear as attachment chips and are included in the same DeepSeek-first message.
DeepSeek may assign image analysis to Gemma; project code, terminal, and preview
work goes to the configured Qwen Coder role. During active tasks, follow-up
messages also go through DeepSeek, which decides whether to answer conversationally
or continue the active workflow. Declared user-action capabilities pause the task
and resume only after the user replies.

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
| Chat | `GET /api/projects/{id}/chat`, `POST /api/projects/{id}/chat` (persistent conversation; no client-selected route mode) |
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
`REASONING_LLM_ENABLED=true`,
`REASONING_LLM_MAX_TOKENS` (default `1200`), `RESEARCH_LLM_MODEL`,
`RESEARCH_LLM_BASE_URL`, `RESEARCH_LLM_API_KEY` or `RESEARCH_LLM_API_KEY_01` … `_10`,
`CREATIVITY_LLM_MODEL`, `CREATIVITY_LLM_BASE_URL`, `CREATIVITY_LLM_API_KEY` or
`CREATIVITY_LLM_API_KEY_01` … `_10`, and `PLATFORM_DATA_DIR`
(database + workspaces location, default `workspace/`).

`OPENMANUS_CONTROLLER_MAX_TOKENS` (default `3072`, minimum `2048`) sets the
first DeepSeek control-decision output budget independently of the older
`REASONING_LLM_MAX_TOKENS` review setting. If a request fails at the decision
stage, the chat error now includes a safe validation reason; on a local Docker
deployment inspect `docker logs --tail 100 openmanus` for the underlying Ollama
error. A working model configuration on this server cannot prove that the
model is reachable inside your laptop's container.

For the zero-cost Ollama setup in [`.env.example`](.env.example), install only
the models you want on the host that runs Ollama. Example roles are
`qwen2.5-coder:7b` (coding), `gemma3:4b` (vision), `deepseek-r1:7b`
(required control unit), `qwen2.5:3b` (research), and `llama3.2:3b` (design).
Ollama's Q4_K_M catalog entries for the last two are approximately 1.9 GB and
2.0 GB. Downloads are free but consume disk/RAM and carry different license
terms. A role is configured only when its own model settings are loaded; named
roles never silently fall back to another model. The Docker container reaches
host Ollama at `host.docker.internal:11434`.

**DeepSeek is the mandatory control unit for every new user message.** It
receives the message before any answer/task branch, chooses either a direct
response or a validated capability workflow, and selects the capability IDs
and handlers. **Qwen Coder** executes selected software, terminal, preview, and
GitHub work; **Gemma 3** analyzes selected image/screenshot work; **Qwen2.5 3B**
conducts selected source-grounded web research; and **Llama 3.2 3B** supplies
selected creative direction. The platform validates DeepSeek's plan against
the local capability/model registry and executes the selected handlers. The
registry also inventories UI/platform features and known limits, but those are
not offered as executable task steps; the planner sees only direct-answer,
model-backed, and supported user-handoff capabilities. DeepSeek's selected order
is retained, so a user step can block coding or pause after a verified build.
Scheduled prompts are routed through the same DeepSeek controller. It
does not use a keyword classifier or an automatic fallback route; if DeepSeek
is unavailable or returns an invalid plan, the request fails visibly without
being reclassified by another layer. Named roles are pinned to their configured
endpoint so a failed DeepSeek, research, or design model cannot silently become
the default or a cloud model.

Chat has no answer/inspect/plan/build selector. Messages sent while a task is
running also go through DeepSeek first: it decides whether the message is a
conversational reply or compatible guidance for the active workflow. Compatible
guidance resumes the task; a request needing handlers outside the active plan is
rejected with a retry-after-completion message rather than being silently
misrouted. Explicit UI controls such as Stop and confirmation dialogs remain
platform actions. The platform—not the model—retains authorization and
confirmation gates for login, payments, destructive actions, browser takeover,
publishing, and pushes.

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
- DeepSeek is required for new chat and task-routing requests. Set
  `REASONING_LLM_MODEL` and `REASONING_LLM_ENABLED=true` to enable it. There is
  deliberately no regex/keyword fallback when that local model is unavailable.
- DeepSeek chooses the workflow; capability IDs, owning handlers, dependencies,
  local model availability, and permission gates are validated deterministically
  before execution. This deterministic validation is not an intent classifier
  and cannot change the selected route into a different one.
- `capabilities.json` is a routing/validation catalog, not 400 separately
  implemented plugins. Only capabilities connected to an existing handler and
  tool have executable behavior; unsupported or uncompleted capabilities must
  fail visibly rather than being reported as complete.
- Workspace shell commands are classified as `safe`, `confirmation`, or `blocked`.
  Dependency installs, recursive deletion, privileged operations, and external Git
  changes require human approval. Remote scripts piped into a shell and direct
  disk-formatting operations are blocked. Approval prompts show the exact command
  and the policy reason.
- Resume requests preserve the previous checkpoint and recovery metadata, record a
  `resume_requested` checkpoint, and re-queue through the normal durable lifecycle.
