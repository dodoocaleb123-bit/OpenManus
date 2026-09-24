from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import build_router
from app.platform.auth import BasicAuthMiddleware
from app.platform.orchestrator import AgentOrchestrator
from app.platform.browser import BrowserManager
from app.platform.browser_api import build_browser_router
from app.platform.store import PlatformStore

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


logger = logging.getLogger("openmanus.platform")


def create_app(
    root: Path | None = None,
    password: str | None = None,
    username: str | None = None,
    agent_factory=None,
    check_llm: bool = True,
) -> FastAPI:
    """Build the platform app.

    ``password``/``username`` enable HTTP Basic auth; they default to the
    ``PLATFORM_PASSWORD`` / ``PLATFORM_USERNAME`` environment variables. Leaving
    the password unset (the local-development default) disables auth entirely.
    """
    workspace_root = Path(root) if root is not None else Path(os.environ.get("PLATFORM_DATA_DIR") or ROOT / "workspace")
    store = PlatformStore(workspace_root)
    browsers = BrowserManager(workspace_root)
    orchestrator = AgentOrchestrator(store, browsers, agent_factory=agent_factory, check_llm=check_llm)

    async def _recover_safely() -> None:
        try:
            await orchestrator.recover()
        except Exception:
            logger.exception("Task recovery failed")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Recover interrupted tasks in the background: startup (and Render's
        # health check) must never wait for old agent runs to be replayed.
        recovery = asyncio.create_task(_recover_safely())
        yield
        recovery.cancel()
        await orchestrator.shutdown()
        await browsers.close_all()

    application = FastAPI(title="OpenManus Platform", version="0.7.0", lifespan=lifespan)
    application.state.store = store
    application.state.orchestrator = orchestrator
    application.state.browsers = browsers
    application.include_router(build_router(store, orchestrator))
    application.include_router(build_browser_router(store, browsers))
    application.mount("/static", StaticFiles(directory=WEB), name="static")

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB / "index.html")

    password = password if password is not None else os.environ.get("PLATFORM_PASSWORD", "")
    if password:
        username = username or os.environ.get("PLATFORM_USERNAME") or "admin"
        application.add_middleware(BasicAuthMiddleware, username=username, password=password)

    return application


app = create_app()
