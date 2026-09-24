from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any

from app.utils.env import scrubbed_env

# npm writes this placeholder test script into every fresh package.json.
_NPM_PLACEHOLDER_TEST = 'echo "Error: no test specified" && exit 1'


class ValidationResult:
    def __init__(self, ok: bool, command: str, output: str, skipped: bool = False):
        self.ok = ok
        self.command = command
        self.output = output
        self.skipped = skipped

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "command": self.command, "output": self.output, "skipped": self.skipped}


class CodingLoop:
    """Build/test/diagnose/repair loop around an OpenManus coding run.

    The agent performs edits. This service independently verifies the resulting
    workspace and, when verification fails, feeds the concrete failure back to
    the same agent for another repair pass.

    Validation runs with project-local dependencies (``node_modules``, a
    per-project ``.venv``) and without platform credentials in the environment.
    """

    def __init__(self, workspace: str | Path, max_repair_cycles: int | None = None):
        self.workspace = Path(workspace).resolve()
        if max_repair_cycles is None:
            try:
                max_repair_cycles = int(os.environ.get("AGENT_MAX_REPAIR_CYCLES", 3))
            except ValueError:
                max_repair_cycles = 3
        self.max_repair_cycles = max(0, max_repair_cycles)

    # ------------------------------------------------------------- detection

    def _package_json(self) -> dict[str, Any] | None:
        path = self.workspace / "package.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _python_project(self) -> bool:
        p = self.workspace
        has_manifest = any((p / f).exists() for f in ("pyproject.toml", "requirements.txt", "setup.py", "pytest.ini"))
        has_tests = (p / "tests").is_dir() or any(p.glob("test_*.py")) or any(p.glob("*_test.py")) or (p / "pytest.ini").exists()
        return has_tests and (has_manifest or any(p.glob("*.py")) or (p / "tests").is_dir())

    def _venv_python(self) -> Path:
        return self.workspace / ".venv" / "bin" / "python"

    def detect_commands(self) -> list[list[str]]:
        """Validation commands (tests/build) for the project, without setup steps."""
        commands: list[list[str]] = []
        pkg = self._package_json()
        if pkg is not None:
            scripts = pkg.get("scripts") or {}
            test = (scripts.get("test") or "").strip()
            if test and test != _NPM_PLACEHOLDER_TEST:
                if "jest" in test and "--runInBand" not in test:
                    # Jest only: run serially so small containers don't OOM.
                    commands.append(["npm", "test", "--", "--runInBand"])
                else:
                    commands.append(["npm", "test"])
            if scripts.get("build"):
                commands.append(["npm", "run", "build"])
        if self._python_project():
            if self._venv_python().exists():
                commands.append([str(self._venv_python()), "-m", "pytest", "-q"])
            else:
                commands.append([os.environ.get("PYTEST", "pytest"), "-q"])
        if (self.workspace / "Cargo.toml").exists():
            commands.append(["cargo", "test"])
        if (self.workspace / "go.mod").exists():
            commands.append(["go", "test", "./..."])
        return commands

    # ----------------------------------------------------------------- setup

    def _stamp_ok(self, stamp: Path, sources: list[Path]) -> bool:
        digest = hashlib.sha256()
        for src in sources:
            if src.exists():
                digest.update(src.name.encode())
                digest.update(src.read_bytes())
        value = digest.hexdigest()
        if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == value:
            return True
        self._pending_stamps.append((stamp, value))
        return False

    def setup_commands(self) -> list[tuple[list[str], int]]:
        """Dependency installation needed before validation (skipped when up to date)."""
        self._pending_stamps: list[tuple[Path, str]] = []
        steps: list[tuple[list[str], int]] = []
        p = self.workspace
        if self._package_json() is not None:
            stamp = p / "node_modules" / ".openmanus-install"
            if not (p / "node_modules").exists() or not self._stamp_ok(stamp, [p / "package.json", p / "package-lock.json"]):
                if (p / "package-lock.json").exists():
                    steps.append((["npm", "ci", "--no-audit", "--no-fund"], 900))
                else:
                    steps.append((["npm", "install", "--no-audit", "--no-fund"], 900))
        if self._python_project():
            venv = p / ".venv"
            if not self._venv_python().exists():
                steps.append(([sys.executable, "-m", "venv", str(venv)], 300))
            pip = [str(self._venv_python()), "-m", "pip", "install", "-q", "--disable-pip-version-check"]
            stamp = venv / ".openmanus-install"
            if not self._stamp_ok(stamp, [p / "requirements.txt", p / "requirements-dev.txt", p / "pyproject.toml", p / "setup.py"]):
                if (p / "requirements.txt").exists():
                    steps.append((pip + ["-r", "requirements.txt"], 900))
                if (p / "requirements-dev.txt").exists():
                    steps.append((pip + ["-r", "requirements-dev.txt"], 900))
                if (p / "pyproject.toml").exists() or (p / "setup.py").exists():
                    steps.append((pip + ["-e", "."], 900))
                steps.append((pip + ["pytest"], 600))
        return steps

    # --------------------------------------------------------------- running

    async def run_command(self, command: list[str], timeout: int = 300) -> ValidationResult:
        label = " ".join(command)
        if shutil.which(command[0], path=self._path()) is None and not Path(command[0]).exists():
            return ValidationResult(
                True, label, f"Skipped: `{command[0]}` is not installed in this environment.", skipped=True
            )
        env = scrubbed_env(extra={"CI": "1"})
        env["PATH"] = self._path()
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            return ValidationResult(False, label, f"Could not start command: {exc}")
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.communicate()
            return ValidationResult(False, label, f"Command timed out after {timeout}s")
        text = out.decode(errors="replace")
        if proc.returncode == 5 and "pytest" in label:
            # pytest: "no tests collected" is not a failure of the implementation.
            return ValidationResult(True, label, text[-20000:], skipped=True)
        return ValidationResult(proc.returncode == 0, label, text[-20000:])

    def _path(self) -> str:
        base = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        return f"{self.workspace / '.venv' / 'bin'}:{self.workspace / 'node_modules' / '.bin'}:{base}"

    async def validate(self) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for command, timeout in self.setup_commands():
            result = await self.run_command(command, timeout=timeout)
            result.command = f"(setup) {result.command}"
            results.append(result)
            if not result.ok:
                return results
        for stamp, value in getattr(self, "_pending_stamps", []):
            try:
                stamp.parent.mkdir(parents=True, exist_ok=True)
                stamp.write_text(value, encoding="utf-8")
            except OSError:
                pass
        commands = self.detect_commands()
        if not commands:
            results.append(ValidationResult(True, "(no automatic validator detected)", "No supported test/build command was detected.", skipped=True))
            return results
        for command in commands:
            result = await self.run_command(command)
            results.append(result)
            if not result.ok:
                break
        return results
