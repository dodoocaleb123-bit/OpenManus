# OpenManus Platform Build Plan

## Implemented

### v0.1
- Platform API
- project workspaces
- task orchestration
- SSE task events
- minimal web UI

### v0.2
- GitHub repository discovery
- repository cloning
- branch creation
- git status/diff/commit/push

### v0.3
- SQLite durable projects/tasks/events
- task attempts/checkpoints
- restart recovery
- task history/resume

### v0.4
- project-scoped Playwright browser sessions
- navigation/click/type/press
- screenshots
- human browser controls in UI

### v0.5
- shared browser session attachment to tasks
- OpenManus agent receives `platform_browser`
- agent and human control the same live browser session
- browser screenshots can be returned directly to the agent as image tool output
- cross-project browser attachment is rejected

## Next implementation order

1. Isolated execution workers per task (Docker first, stronger VM/microVM later)
2. Durable browser workers + reconnect/resume
3. Repository map/index and codebase-aware planning
4. Deployment/hosting service for built apps (logs, health checks, domains, TLS)
5. Multi-agent roles (planner / coder / reviewer)
6. Project/conversation/repository memory
7. OpenManus-RL trajectory capture, evaluation and training integration

## v0.6 — Autonomous coding loop

Implemented:
- independent workspace validation after every agent implementation pass
- automatic detection for common Python/Node/Rust/Go projects
- persisted validation results and coding iteration/checkpoint
- up to three repair cycles driven by concrete failing command output
- task only reaches `SUCCEEDED` after validation passes; otherwise it ends as validation-exhausted failure
- validation lifecycle events are visible through the existing task event stream

## v0.7 — Autonomous engineer loop, human-in-the-loop, GitHub automation

Implemented:
- `PlatformManus` agent with workspace-confined bash/python/file tools, per-command
  timeouts and credential-scrubbed environments; project venv/node_modules on PATH
- the agent drives git itself through `platform_git` (status/diff/log/init/branch/
  commit/push/pull request/publish new repo); token only via short-lived askpass
- human channel: `ask_human` questions with in-UI replies, live messages injected
  into the running agent, stop/cancel and retry
- validation after every pass with repair cycles; dependency setup cached per lockfile
- shared browser: one session per project for agent and human, click-on-screenshot
  and keyboard takeover, screenshots given to vision-capable models
- model adapter: vision/reasoning capability detection with env overrides, fail-fast
  on permanent provider errors, pre-flight configuration check with setup hint
- resumable SSE (`Last-Event-ID`), `/api/status`, cancelled status, restart recovery
- new three-column UI: projects/history, live activity feed, browser + GitHub/files
- container ships Node.js 22, npm, build tools, git, Chromium; Render plan 1c-2g
