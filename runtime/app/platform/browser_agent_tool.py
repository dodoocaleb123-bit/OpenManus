from __future__ import annotations

import base64
from typing import Any

from app.tool.base import BaseTool, ToolResult
from app.platform.browser import BrowserSession


class PlatformBrowserTool(BaseTool):
    """Direct bridge from the OpenManus agent to the browser session visible in the UI."""

    name: str = "platform_browser"
    description: str = (
        "Control the project's shared browser session. This is the SAME browser the user can see "
        "and control in the platform UI. Use it for web navigation, clicking, typing, pressing keys, "
        "reading page text, and screenshots."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "navigate", "click", "type", "press", "extract", "screenshot", "wait"],
                "description": "Browser action to perform",
            },
            "url": {"type": "string", "description": "URL for navigate"},
            "selector": {"type": "string", "description": "CSS selector for click/type/press"},
            "text": {"type": "string", "description": "Text to type"},
            "key": {"type": "string", "description": "Keyboard key to press"},
            "expression": {"type": "string", "description": "JavaScript expression for extracting page data"},
            "seconds": {"type": "number", "description": "Seconds to wait"},
        },
        "required": ["action"],
    }

    session: Any = None

    class Config:
        arbitrary_types_allowed = True

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action")
        try:
            if action == "status":
                return self.success_response(await self.session.status())
            if action == "navigate":
                return self.success_response(await self.session.navigate(kwargs["url"]))
            if action == "click":
                return self.success_response(await self.session.click(kwargs["selector"]))
            if action == "type":
                return self.success_response(await self.session.type_text(kwargs["selector"], kwargs.get("text", "")))
            if action == "press":
                return self.success_response(await self.session.press(kwargs["selector"], kwargs["key"]))
            if action == "extract":
                expression = kwargs.get("expression") or "document.body.innerText"
                value = await self.session.evaluate(expression)
                return self.success_response(value)
            if action == "wait":
                import asyncio
                await asyncio.sleep(max(0, min(float(kwargs.get("seconds", 1)), 30)))
                return self.success_response(await self.session.status())
            if action == "screenshot":
                path = await self.session.screenshot()
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                return ToolResult(output=f"Screenshot captured from {await self.session.status()}", base64_image=data)
            return self.fail_response(f"Unknown browser action: {action}")
        except Exception as exc:
            return self.fail_response(str(exc))
