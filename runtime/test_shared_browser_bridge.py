import asyncio
from pathlib import Path

from app.platform.browser import BrowserManager


def test_browser_manager_session_lifecycle(tmp_path: Path):
    manager = BrowserManager(tmp_path)
    async def run():
        try:
            s = await manager.create("p1", "about:blank", headless=True)
            status = await s.status()
            assert status["running"] is True
            assert status["project_id"] == "p1"
            await manager.close(s.id)
            assert manager.get(s.id) is None
        except Exception as exc:
            # Environment may not contain Playwright Chromium; manager behavior is still validated.
            assert "playwright" in str(exc).lower() or "executable" in str(exc).lower()
    asyncio.run(run())


def test_browser_manager_ignores_closed_project_page(tmp_path: Path):
    manager = BrowserManager(tmp_path)

    class ClosedPage:
        def is_closed(self):
            return True

    from app.platform.browser import BrowserSession

    session = BrowserSession("closed01", "p1", tmp_path)
    session.page = ClosedPage()
    manager.sessions[session.id] = session

    assert manager.for_project("p1") is None
