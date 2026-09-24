from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Awaitable, Callable, Optional

from app.platform.browser import BrowserSession
from app.tool.base import BaseTool, ToolResult

# Called once when the tool has to start a browser on its own, so the platform
# can tell the UI which session to display.
SessionFactory = Callable[[Optional[str]], Awaitable[BrowserSession]]


class PlatformBrowserTool(BaseTool):
    """Direct bridge from the OpenManus agent to the browser session visible in the UI."""

    name: str = "platform_browser"
    description: str = (
        "Control the project's shared browser. This is the SAME browser the user sees (and can take over) "
        "in the platform UI. Use it to research the web and to test the apps you build: start the app's "
        "server in the background with the bash tool, then navigate to http://localhost:<port>. "
        "Take a `screenshot` after important actions to visually verify the page. "
        "Selectors are Playwright selectors: CSS (`#login`, `input[name=email]`), text (`text=Sign in`) "
        "or role (`role=button[name=\"Save\"]`)."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "navigate", "back", "click", "type", "press", "select",
                    "scroll", "extract", "screenshot", "wait",
                ],
                "description": "Browser action to perform",
            },
            "url": {"type": "string", "description": "URL for navigate"},
            "selector": {"type": "string", "description": "Element selector for click/type/press/select"},
            "text": {"type": "string", "description": "Text to type"},
            "submit": {"type": "boolean", "description": "Press Enter after typing"},
            "key": {"type": "string", "description": "Keyboard key to press, e.g. Enter, Tab, ArrowDown"},
            "value": {"type": "string", "description": "Option value/label for select"},
            "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction"},
            "expression": {
                "type": "string",
                "description": "JavaScript expression for extract (default: visible page text)",
            },
            "seconds": {"type": "number", "description": "Seconds to wait (max 30)"},
        },
        "required": ["action"],
    }

    session: Any = None
    session_factory: Any = None
    max_extract_chars: int = 15000

    class Config:
        arbitrary_types_allowed = True

    async def _session(self, url: str | None = None) -> BrowserSession:
        if self.session is not None and self.session.page is not None:
            return self.session
        if self.session is not None:
            await self.session.start(url)
            return self.session
        if self.session_factory is None:
            raise RuntimeError("No browser session is available for this task")
        self.session = await self.session_factory(url)
        return self.session

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action")
        try:
            if action == "navigate":
                url = kwargs.get("url")
                if not url:
                    return self.fail_response("navigate requires `url`")
                session = await self._session(url)
                return self.success_response(await session.navigate(url))

            session = await self._session()
            if action == "status":
                return self.success_response(await session.status())
            if action == "back":
                return self.success_response(await session.back())
            if action == "click":
                return self.success_response(await session.click(kwargs["selector"]))
            if action == "type":
                return self.success_response(
                    await session.type_text(kwargs["selector"], kwargs.get("text", ""), bool(kwargs.get("submit")))
                )
            if action == "press":
                return self.success_response(await session.press(kwargs.get("selector") or "body", kwargs["key"]))
            if action == "select":
                return self.success_response(await session.select_option(kwargs["selector"], kwargs.get("value", "")))
            if action == "scroll":
                return self.success_response(await session.scroll(kwargs.get("direction", "down")))
            if action == "extract":
                expression = kwargs.get("expression") or "document.body.innerText"
                value = await session.evaluate(expression)
                text = value if isinstance(value, str) else json.dumps(value, default=str)
                if len(text) > self.max_extract_chars:
                    text = text[: self.max_extract_chars] + f"\n…[truncated, {len(text)} chars total]"
                return self.success_response(text)
            if action == "wait":
                await asyncio.sleep(max(0.0, min(float(kwargs.get("seconds", 1)), 30.0)))
                return self.success_response(await session.status())
            if action == "screenshot":
                data = base64.b64encode(await session.screenshot_jpeg()).decode("ascii")
                # Refresh the PNG the UI polls so the human sees the same frame.
                try:
                    await session.screenshot()
                except Exception:
                    pass
                return ToolResult(
                    output=f"Screenshot captured from {await session.status()}", base64_image=data
                )
            return self.fail_response(f"Unknown browser action: {action}")
        except KeyError as exc:
            return self.fail_response(f"Missing required argument {exc} for action '{action}'")
        except Exception as exc:
            return self.fail_response(str(exc))
