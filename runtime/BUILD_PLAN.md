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

## v0.6 — Autonomous coding loop

Implemented:
- independent workspace validation after every agent implementation pass
- automatic detection for common Python/Node/Rust/Go projects
- persisted validation results and coding iteration/checkpoint
- up to three repair cycles driven by concrete failing command output
- task only reaches `SUCCEEDED` after validation passes; otherwise it ends as validation-exhausted failure
- validation lifecycle events are visible through the existing task event stream
