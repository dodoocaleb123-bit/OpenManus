from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import build_router
from app.platform.orchestrator import AgentOrchestrator
from app.platform.browser import BrowserManager
from app.platform.browser_api import build_browser_router
from app.platform.store import PlatformStore

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def create_app(root: Path | None = None) -> FastAPI:
    workspace_root = Path(root) if root is not None else ROOT / "workspace"
    store = PlatformStore(workspace_root)
    browsers = BrowserManager(workspace_root)
    orchestrator = AgentOrchestrator(store, browsers)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await orchestrator.recover()
        yield
        await browsers.close_all()

    application = FastAPI(title="OpenManus Platform", version="0.5.0", lifespan=lifespan)
    application.include_router(build_router(store, orchestrator))
    application.include_router(build_browser_router(store, browsers))
    application.mount("/static", StaticFiles(directory=WEB), name="static")

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB / "index.html")

    return application


app = create_app()
