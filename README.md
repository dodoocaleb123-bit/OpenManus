# OpenManus Platform v0.8

This build extends the OpenManus runtime with a product/platform layer for autonomous software development.

## Implemented

- OpenManus agent runtime retained as the execution engine.
- GitHub repository discovery and connection.
- Isolated per-project workspaces.
- Git branches, status, diff, commit and push.
- SQLite persistence for projects, tasks and events.
- Restart-safe task state.
- Automatic recovery of tasks interrupted while running.
- Task attempts/checkpoints.
- Persistent event history and SSE streaming.
- Task resume endpoint and task-history UI.
- Deployment readiness: login, configurable host/port/data directory, secrets from environment variables, non-blocking startup recovery, reconnecting live task streams. See [DEPLOY.md](DEPLOY.md).
- Correctness for real projects: safe project git operations, persisted GitHub/branch state, fair repair-cycle step budgets, runner-agnostic Node validation.

## Run

```bash
cd runtime
python run_platform.py
```

Open `http://127.0.0.1:8000`. The server binds `0.0.0.0` and honours `HOST` and `PORT`.

The GitHub integration reads `GITHUB_TOKEN` from the server environment (never visible to agent-executed code). Real agent execution still requires the normal OpenManus model configuration and dependencies — or just the `OPENMANUS_LLM_*` environment variables described in [DEPLOY.md](DEPLOY.md).

## Deploy

See [DEPLOY.md](DEPLOY.md) for deploying to Railway (or any Docker host) behind a login, with persistent storage.

## Architecture

```text
Web UI
  ↓
FastAPI Platform API
  ↓
Durable Project/Task/Event Store (SQLite)
  ↓
Agent Orchestrator
  ↓
OpenManus Manus Agent
  ├── planning
  ├── browser/MCP
  ├── Python
  ├── file editing
  ├── shell
  └── sandbox integrations
```

OpenManus-RL remains in `training/` as the separate training/evaluation layer.

## Next platform layers

1. Persistent browser/computer sessions, screenshots and human takeover.
2. Autonomous build/test/debug/repair loop with structured checkpoints.
3. GitHub pull-request creation and review workflow.
4. Isolated execution service.
5. Deployment/hosting with logs, health checks, domains and TLS.
