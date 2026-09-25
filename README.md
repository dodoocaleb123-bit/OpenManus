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
- Persistent project chat mode alongside autonomous Build mode.
- Gemini/OpenAI-compatible conversational API with SQLite chat history.

## Run

```bash
cd runtime
python run_platform.py
```

Open `http://127.0.0.1:8000`.

The GitHub integration reads `GITHUB_TOKEN` from the server environment. Real agent execution still requires the normal OpenManus model configuration and dependencies.

## Deploy

The platform ships as a single Docker image (`runtime/Dockerfile`) with a Render Blueprint (`render.yaml`). See [DEPLOY.md](DEPLOY.md) for the Render walkthrough, environment variables, sizing, and how to run the same image on any Docker host.

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
