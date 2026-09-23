from __future__ import annotations

import asyncio
import base64
import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import build_router
from app.platform.orchestrator import AgentOrchestrator
from app.platform.browser import BrowserManager
from app.platform.browser_api import build_browser_router
from app.platform.store import PlatformStore

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

logger = logging.getLogger("openmanus.platform")


def _data_root() -> Path:
    """Durable data location: PLATFORM_DATA_DIR (e.g. a mounted volume) or ./workspace."""
    return Path(os.environ.get("PLATFORM_DATA_DIR") or (ROOT / "workspace"))


def _require_basic_auth(application: FastAPI, username: str, password: str) -> None:
    """Single-user HTTP Basic auth for the whole app except the health check."""

    realm = 'Basic realm="OpenManus Platform", charset="UTF-8"'

    @application.middleware("http")
    async def basic_auth(request: Request, call_next):
        if request.url.path == "/api/health":
            return await call_next(request)
        header = request.headers.get("Authorization", "")
        authorized = False
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:]).decode("utf-8")
                user, _, secret = decoded.partition(":")
                authorized = hmac.compare_digest(
                    user, username
                ) and hmac.compare_digest(secret, password)
            except Exception:
                authorized = False
        if not authorized:
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": realm},
            )
        return await call_next(request)


def create_app(root: Path | None = None) -> FastAPI:
    workspace_root = Path(root) if root is not None else _data_root()
    store = PlatformStore(workspace_root)
    browsers = BrowserManager(workspace_root)
    orchestrator = AgentOrchestrator(store, browsers)

    async def _recover_safely() -> None:
        try:
            await orchestrator.recover()
        except Exception:
            logger.exception("Task recovery failed")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Recover interrupted tasks in the background: startup must never block
        # on replaying old agent runs.
        asyncio.create_task(_recover_safely())
        yield
        await browsers.close_all()

    application = FastAPI(title="OpenManus Platform", version="0.8.0", lifespan=lifespan)
    application.include_router(build_router(store, orchestrator))
    application.include_router(build_browser_router(store, browsers))
    application.mount("/static", StaticFiles(directory=WEB), name="static")

    password = os.environ.get("PLATFORM_ADMIN_PASSWORD", "")
    if password:
        username = os.environ.get("PLATFORM_ADMIN_USER", "admin")
        _require_basic_auth(application, username, password)
    else:
        logger.warning(
            "PLATFORM_ADMIN_PASSWORD is not set: the platform is running WITHOUT "
            "authentication. Set it before exposing the server to a network."
        )

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB / "index.html")

    return application


app = create_app()
