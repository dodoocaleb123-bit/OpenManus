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

1. Durable browser workers + reconnect/resume
2. Autonomous build/test/debug/repair loop
3. Repository map/index and codebase-aware planning
4. GitHub branch/commit/PR automation with review checkpoints
5. Isolated execution workers (Docker first, stronger VM/microVM later)
6. Deployment/hosting service with logs, health checks, domains and TLS
7. Project/conversation/repository memory
8. OpenManus-RL trajectory capture, evaluation and training integration
9. Multi-user accounts, per-user GitHub connections and public signups (needs #5)

## v0.6 — Autonomous coding loop

Implemented:
- independent workspace validation after every agent implementation pass
- automatic detection for common Python/Node/Rust/Go projects
- persisted validation results and coding iteration/checkpoint
- up to three repair cycles driven by concrete failing command output
- task only reaches `SUCCEEDED` after validation passes; otherwise it ends as validation-exhausted failure
- validation lifecycle events are visible through the existing task event stream

## v0.7 — Deployment readiness

Implemented:
- HTTP Basic login (PLATFORM_ADMIN_USER / PLATFORM_ADMIN_PASSWORD) on everything except the health check
- HOST/PORT configuration and binding on 0.0.0.0 for deployment
- durable data directory via PLATFORM_DATA_DIR (SQLite + project workspaces on a mounted volume)
- model configuration from OPENMANUS_LLM_* / OPENMANUS_VISION_* environment variables (secrets stay off disk)
- config loads without a Daytona section (daytona_api_key now optional)
- GitHub token scrubbed from agent-executed code (python/bash tools, workspace validation commands)
- startup task recovery runs in the background: the API no longer blocks on replaying interrupted tasks
- SSE task streams send heartbeats and resume via Last-Event-ID after reconnects (survives proxy idle/streaming caps)
- task Resume no longer replays the previous attempt's terminal event in the UI
- Docker image starts the platform (was: interactive bash), with Node for validation and Chromium for the shared browser
- DEPLOY.md: step-by-step Railway deployment guide

## v0.8 — Correctness for real projects

Implemented:
- project git operations refuse to touch any repository the workspace is merely inside of (previously they acted on the platform's own checkout)
- GitHub connection and branch state are persisted (previously lost on page reload)
- create-branch returns the branch name to the UI (was empty; the UI showed "branch: unknown")
- Node validation passes `--runInBand` only to Jest (other runners reject it and failed every cycle); npm's default "no test specified" placeholder is not treated as a validator
- every implementation/repair agent run starts with a fresh step budget (repair cycles previously inherited leftovers)
- a rejected browser attachment no longer leaves an orphaned queued task
- project workspaces are keyed by project id (duplicate names can no longer share a folder)
