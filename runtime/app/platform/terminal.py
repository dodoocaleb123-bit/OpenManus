"""Persistent project terminal sessions used by the UI and local agent runtime."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.platform.agent import WorkspaceBash


class ProjectTerminalManager:
    """Keep one workspace-rooted bash session per project.

    This is deliberately container-local. It does not provide access to the
    Windows host that runs Docker Desktop; host PowerShell requires a separate
    authenticated bridge and is not silently emulated here.
    """

    def __init__(self):
        self.sessions: dict[str, WorkspaceBash] = {}
        self._lock = asyncio.Lock()

    async def execute(self, project_id: str, workspace: Path, command: str) -> dict[str, Any]:
        if not command.strip():
            raise ValueError("A terminal command is required")
        async with self._lock:
            terminal = self.sessions.get(project_id)
            if terminal is None:
                async def approve(_question: str) -> str:
                    # The command was submitted directly by the authenticated
                    # user through the terminal panel.
                    return "yes"
                terminal = WorkspaceBash(workspace=Path(workspace), request_approval=approve)
                self.sessions[project_id] = terminal
            result = await terminal.execute(command=command)
        return {
            "command": command,
            "output": getattr(result, "output", None) or "",
            "error": getattr(result, "error", None) or "",
            "system": getattr(result, "system", None) or "",
            "evidence": getattr(result, "evidence", {}) or {},
        }

    async def close_all(self) -> None:
        async with self._lock:
            for terminal in self.sessions.values():
                terminal.close()
            self.sessions.clear()
