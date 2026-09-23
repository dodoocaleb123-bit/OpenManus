from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.platform.browser import BrowserError, BrowserManager
from app.platform.store import PlatformStore


class BrowserCreate(BaseModel):
    project_id: str
    url: str | None = None
    headless: bool = True

class Navigate(BaseModel):
    url: str = Field(min_length=1, max_length=2000)

class SelectorAction(BaseModel):
    selector: str = Field(min_length=1, max_length=1000)

class TypeAction(SelectorAction):
    text: str = Field(max_length=10000)
    submit: bool = False

class PressAction(SelectorAction):
    key: str = Field(min_length=1, max_length=100)


def build_browser_router(store: PlatformStore, browsers: BrowserManager) -> APIRouter:
    router = APIRouter(prefix="/api/browser")

    def get_session(session_id: str):
        session = browsers.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Browser session not found")
        return session

    @router.post("/sessions")
    async def create_session(body: BrowserCreate):
        if not store.get_project(body.project_id):
            raise HTTPException(status_code=404, detail="Project not found")
        try:
            session = await browsers.create(body.project_id, body.url, body.headless)
            return await session.status()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/sessions")
    async def list_sessions():
        return [await s.status() for s in browsers.sessions.values()]

    @router.get("/sessions/{session_id}")
    async def session_status(session_id: str):
        return await get_session(session_id).status()

    @router.post("/sessions/{session_id}/navigate")
    async def navigate(session_id: str, body: Navigate):
        try:
            return await get_session(session_id).navigate(body.url)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/click")
    async def click(session_id: str, body: SelectorAction):
        try:
            return await get_session(session_id).click(body.selector)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/type")
    async def type_text(session_id: str, body: TypeAction):
        try:
            return await get_session(session_id).type_text(body.selector, body.text, body.submit)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/press")
    async def press(session_id: str, body: PressAction):
        try:
            return await get_session(session_id).press(body.selector, body.key)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/sessions/{session_id}/screenshot")
    async def screenshot(session_id: str):
        try:
            path = await get_session(session_id).screenshot()
            return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/sessions/{session_id}")
    async def close_session(session_id: str):
        if not browsers.get(session_id):
            raise HTTPException(status_code=404, detail="Browser session not found")
        await browsers.close(session_id)
        return {"closed": True, "id": session_id}

    return router
