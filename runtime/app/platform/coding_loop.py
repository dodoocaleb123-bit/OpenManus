from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from app.utils.env import scrubbed_env


class ValidationResult:
    def __init__(self, ok: bool, command: str, output: str):
        self.ok = ok
        self.command = command
        self.output = output

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "command": self.command, "output": self.output}


class CodingLoop:
    """Build/test/diagnose/repair loop around an OpenManus coding run.

    The agent performs edits. This service independently verifies the resulting
    workspace and, when verification fails, feeds the concrete failure back to
    the same agent for another repair pass.
    """

    def __init__(self, workspace: str | Path, max_repair_cycles: int = 3):
        self.workspace = Path(workspace).resolve()
        self.max_repair_cycles = max(0, max_repair_cycles)

    def detect_commands(self) -> list[list[str]]:
        p = self.workspace
        commands: list[list[str]] = []
        if (p / "package.json").exists():
            try:
                data = json.loads((p / "package.json").read_text(encoding="utf-8"))
                scripts = data.get("scripts", {})
                if "test" in scripts:
                    commands.append(["npm", "test", "--", "--runInBand"])
                elif "test" in scripts:
                    commands.append(["npm", "test"])
                if "build" in scripts:
                    commands.append(["npm", "run", "build"])
            except Exception:
                commands.append(["npm", "test"])
        if (p / "pyproject.toml").exists() or (p / "pytest.ini").exists() or (p / "tests").exists():
            commands.append([os.environ.get("PYTEST", "pytest"), "-q"])
        elif (p / "requirements.txt").exists() and any(p.glob("test*.py")):
            commands.append([os.environ.get("PYTEST", "pytest"), "-q"])
        if (p / "Cargo.toml").exists():
            commands.append(["cargo", "test"])
        if (p / "go.mod").exists():
            commands.append(["go", "test", "./..."])
        return commands

    async def run_command(self, command: list[str], timeout: int = 180) -> ValidationResult:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=scrubbed_env(),
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ValidationResult(False, " ".join(command), f"Command timed out after {timeout}s")
        text = out.decode(errors="replace")
        return ValidationResult(proc.returncode == 0, " ".join(command), text[-20000:])

    async def validate(self) -> list[ValidationResult]:
        commands = self.detect_commands()
        if not commands:
            return [ValidationResult(True, "(no automatic validator detected)", "No supported test/build command was detected.")]
        results: list[ValidationResult] = []
        for command in commands:
            result = await self.run_command(command)
            results.append(result)
            if not result.ok:
                break
        return results
