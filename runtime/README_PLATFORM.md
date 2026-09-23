# OpenManus Platform

## v0.5 — Shared Browser Agent Bridge

This build adds a project-scoped browser that can be controlled by both the human UI and the OpenManus agent.

### Shared browser flow

1. Start a browser from the Browser panel.
2. Enter a task and click **Run agent**.
3. The task automatically attaches the currently selected project's browser session.
4. OpenManus receives a `platform_browser` tool for that exact session.
5. The agent can navigate, click, type, press keys, extract page text, wait, and capture screenshots.
6. The human can continue observing and controlling the same browser from the UI.

The existing Browser Use CLI/MCP integration remains available for agents that need its richer browser-use toolset. The new bridge is deliberately additive: it provides a deterministic shared session owned by the platform so human and agent actions operate on the same browser instance.

### Important runtime requirement

Install dependencies from `requirements.txt` and install the Chromium runtime:

```bash
pip install -r requirements.txt
playwright install chromium
```

### Current limitation

The browser process itself is currently tied to the running platform server. Project/task metadata is durable, but an active browser session is not yet restored after a server restart. The next browser iteration can move sessions into isolated long-lived workers/VMs and add reconnection/resume semantics.
