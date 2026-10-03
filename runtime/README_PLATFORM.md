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

Each project has one persistent composer; there is no conversation/task or
answer/build mode selector. Every valid, non-empty user message that passes API
authentication, project access, and attachment-ownership checks is passed
verbatim to DeepSeek before any model/tool workflow decision. There is no
keyword classifier, intent gate, or numbered task catalogue. Each message
creates a persisted task, including greetings and ordinary questions. DeepSeek
can answer directly with an empty specialist workflow, delegate selected steps
to local model roles/tools, pause for a user action, or combine implementation
with a final synthesized answer.

DeepSeek receives human-readable descriptions of the configured model roles
and available OpenManus tools, then returns ordered workflow steps with
objectives, dependencies, and selected tools. The platform validates only the
workflow structure, configured adapters, project permissions, protected-action
approvals, and evidence after DeepSeek's decision; it does not decide what kind
of request the user made. On local Ollama, JSON mode is requested and DeepSeek
retries once with a larger output budget if the response is only reasoning or
malformed JSON. If DeepSeek asks for workspace context, the server adds bounded
local file excerpts.
Secret-like filenames, generated directories, and oversized content are
excluded. A per-project Memory pane stores up to 6,000 characters in local
SQLite; credentials are rejected and both memory and files are treated as
untrusted reference data, not instructions.

The paperclip button is in the lower-left of the composer. Selected uploads
appear as attachment chips and are included in the same DeepSeek-first message.
DeepSeek may assign image analysis to Gemma; project code, terminal, and preview
work can go to the configured Qwen Coder role. During active tasks, follow-up
messages also go through DeepSeek, which decides whether the message needs an
answer, a continuation of the active workflow, or both. User-action workflow
steps pause and resume after the user replies.

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

After rebuilding and starting the container, test the *actual* host Ollama from
inside it, without creating projects or sending data to an external API:

```powershell
docker exec openmanus python -m deploy.check_multimodel
```

If the image under review is not deployed yet, build it as
`openmanus-platform-review` from this branch and run a **disposable** test
container instead, leaving your existing `openmanus` container and data volume
untouched (run when no other Ollama task is active):

```powershell
docker build -t openmanus-platform-review .
docker run --rm --add-host=host.docker.internal:host-gateway --env-file ..\.env openmanus-platform-review python -m deploy.check_multimodel
```

This sequentially checks the **exact configured tags**, asks all five local
models for one answer (including a generated red image for Gemma), tests the
real DeepSeek controller on `Hellooo`, and requests Ollama to unload each model
before the next. It prints pass/fail and elapsed time, not credentials or raw
model text. A missing research/creativity role is a failure, not a substitute
call to the default coder. Passing this probe is **not** the complete ZIP
acceptance test: the multi-step build, screenshot, preview, GitHub and restart
scenario must also pass on the laptop before calling the whole update complete.

**DeepSeek is the mandatory control unit for every new user message**, whether
it arrives through the project composer, the task API, or a scheduled prompt.
It receives the message before any task workflow is created and selects model
handlers and ordered capabilities. **Qwen Coder** executes selected software,
terminal, preview, and GitHub work; **Gemma 3** analyzes selected image/screenshot
work; **Qwen2.5 3B** conducts selected source-grounded web research; and
**Llama 3.2 3B** supplies selected creative direction. DeepSeek's selected
user-facing output capability depends on earlier specialist results, so build
and research requests can finish with a synthesized explanation. The platform
validates DeepSeek's plan against the local capability/model registry and
executes the selected handlers. The registry also inventories UI/platform
features and known limits; only capabilities with supported adapters are
offered for execution. A selected user action pauses the workflow and resumes
after a reply. If DeepSeek is unavailable or returns an invalid plan, the request
fails visibly; it is not reclassified by another layer. Named roles are pinned
to their configured local endpoint so a failed specialist cannot silently
become the default or a cloud model.

Messages sent to an active task are also planned by DeepSeek. In the current
implementation, such a follow-up can continue only through capability steps
already selected for that active task; a genuinely new capability workflow is
rejected until the current task ends. Explicit UI controls such as Stop and
confirmation dialogs remain platform actions. The platform—not the model—retains
authorization and confirmation gates for login, payments, destructive actions,
browser takeover, publishing, and pushes.

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
- DeepSeek chooses model handlers, tools, objectives, and dependencies from
  descriptions in its prompt. The platform has no numbered task catalogue or
  registry-based planner; it validates only descriptive step structure and
  actual runtime adapters/permissions after the model decides.
- Unsupported tools, unavailable model roles, uncompleted steps, and missing
evidence must fail visibly rather than being reported as complete. Model-status
and plugin metadata endpoints are informational/runtime support surfaces and
are not consulted as a task-selection catalogue by DeepSeek.
- Workspace shell commands are classified as `safe`, `confirmation`, or `blocked`.
  Dependency installs, recursive deletion, privileged operations, and external Git
  changes require human approval. Remote scripts piped into a shell and direct
  disk-formatting operations are blocked. Approval prompts show the exact command
  and the policy reason.
- Resume requests preserve the previous checkpoint and recovery metadata, record a
  `resume_requested` checkpoint, and re-queue through the normal durable lifecycle.
