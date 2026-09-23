from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class BrowserError(RuntimeError):
    pass


class BrowserSession:
    def __init__(self, session_id: str, project_id: str, root: Path):
        self.id = session_id
        self.project_id = project_id
        self.root = root
        self.session_dir = root / "browser" / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._lock = asyncio.Lock()

    async def start(self, url: str | None = None, headless: bool = True) -> dict[str, Any]:
        async with self._lock:
            if self.page:
                if url:
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                return await self.status()
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise BrowserError("Playwright is not installed. Install requirements.txt first.") from exc
            self.playwright = await async_playwright().start()
            try:
                self.browser = await self.playwright.chromium.launch(headless=headless)
                self.context = await self.browser.new_context(viewport={"width": 1440, "height": 900})
                self.page = await self.context.new_page()
                if url:
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception:
                await self.close()
                raise
            return await self.status()

    async def navigate(self, url: str) -> dict[str, Any]:
        if not self.page:
            await self.start(url)
        else:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return await self.status()

    async def screenshot(self) -> Path:
        if not self.page:
            raise BrowserError("Browser session is not running")
        path = self.session_dir / "latest.png"
        await self.page.screenshot(path=str(path), full_page=False)
        return path

    async def click(self, selector: str) -> dict[str, Any]:
        if not self.page:
            raise BrowserError("Browser session is not running")
        await self.page.locator(selector).first.click(timeout=15000)
        return await self.status()

    async def type_text(self, selector: str, text: str, submit: bool = False) -> dict[str, Any]:
        if not self.page:
            raise BrowserError("Browser session is not running")
        await self.page.locator(selector).first.fill(text, timeout=15000)
        if submit:
            await self.page.locator(selector).first.press("Enter")
        return await self.status()

    async def press(self, selector: str, key: str) -> dict[str, Any]:
        if not self.page:
            raise BrowserError("Browser session is not running")
        await self.page.locator(selector).first.press(key, timeout=15000)
        return await self.status()

    async def evaluate(self, expression: str) -> Any:
        if not self.page:
            raise BrowserError("Browser session is not running")
        return await self.page.evaluate(expression)

    async def status(self) -> dict[str, Any]:
        if not self.page:
            return {"id": self.id, "project_id": self.project_id, "running": False, "url": None, "title": None}
        return {
            "id": self.id,
            "project_id": self.project_id,
            "running": not self.page.is_closed(),
            "url": self.page.url,
            "title": await self.page.title(),
        }

    async def close(self) -> None:
        for obj in (self.context, self.browser, self.playwright):
            if obj is None:
                continue
            try:
                await obj.close() if hasattr(obj, "close") else await obj.stop()
            except Exception:
                pass
        self.page = self.context = self.browser = self.playwright = None


class BrowserManager:
    """Owns browser processes and exposes controlled human/agent actions."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.sessions: dict[str, BrowserSession] = {}
        self._lock = asyncio.Lock()

    async def create(self, project_id: str, url: str | None = None, headless: bool = True) -> BrowserSession:
        session = BrowserSession(uuid4().hex[:12], project_id, self.root)
        async with self._lock:
            self.sessions[session.id] = session
        try:
            await session.start(url, headless=headless)
        except Exception:
            self.sessions.pop(session.id, None)
            raise
        return session

    def get(self, session_id: str) -> BrowserSession | None:
        return self.sessions.get(session_id)

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        for session_id in list(self.sessions):
            await self.close(session_id)
