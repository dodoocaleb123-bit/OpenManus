"""The platform's autonomous software-engineering agent.

``PlatformManus`` is the OpenManus ``Manus`` agent (BaseAgent -> ToolCallAgent
-> Manus, preserved as-is) wired into the platform:

* a real terminal, Python runner and file editor, all rooted in the project
  workspace, with platform credentials scrubbed from their environment;
* the project's shared browser (the one the user sees), opened on demand;
* git/GitHub through the platform (branch, commit, push, pull request,
  publish) so the GitHub token never enters the agent's shell;
* human-in-the-loop: ``ask_human`` asks the user in the UI and waits for the
  reply, and messages the user sends mid-task are injected at the next step;
* every thought, tool call and tool result is emitted as a task event, so the
  UI shows what the agent is doing live ("feedback after every action").
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from pydantic import Field

from app.agent.manus import Manus
from app.logger import logger
from app.schema import Message, ToolCall
from app.tool import Terminate, ToolCollection
from app.tool.base import BaseTool, CLIResult, ToolResult
from app.tool.bash import Bash, _BashSession
from app.tool.str_replace_editor import StrReplaceEditor
from app.utils.env import scrubbed_env

EmitFn = Callable[[str, str, dict], Awaitable[None]]


def _int_env(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.environ.get(name, default))))
    except ValueError:
        return default


def _clip(text: Any, limit: int) -> str:
    s = text if isinstance(text, str) else str(text)
    return s if len(s) <= limit else s[:limit] + f"\n…[{len(s) - limit} more chars]"


# --------------------------------------------------------------------- tools


def workspace_env(workspace: Path) -> dict[str, str]:
    """Environment for agent processes: no credentials, project venv first on PATH."""
    venv_bin = workspace / ".venv" / "bin"
    path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    env = scrubbed_env(extra={"HOME": os.environ.get("HOME", str(workspace)), "CI": "1"})
    env["PATH"] = f"{venv_bin}:{workspace / 'node_modules' / '.bin'}:{path}"
    env["PWD"] = str(workspace)
    env.setdefault("npm_config_update_notifier", "false")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return env


class _WorkspaceBashSession(_BashSession):
    def __init__(self, workspace: Path, timeout: float):
        super().__init__()
        self.workspace = workspace
        self._timeout = timeout

    async def start(self):
        if self._started:
            return
        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            env=workspace_env(self.workspace),
        )
        self._started = True

    def kill(self) -> None:
        """Kill the shell and everything it started (including background servers)."""
        if not self._started or self._process.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class WorkspaceBash(Bash):
    """Persistent bash session rooted in the project workspace."""

    description: str = (
        "Run a bash command in a persistent shell. The shell starts in the project workspace and keeps "
        "state (cwd, variables) between calls. Node.js/npm, Python 3 and git are installed.\n"
        "* Long-running processes (dev servers, watchers) MUST run in the background with output "
        "redirected, e.g. `npm run dev > dev.log 2>&1 &`, then check `tail dev.log` or open the app "
        "with the platform_browser tool.\n"
        "* Python dependencies: use the project virtualenv (`python3 -m venv .venv` once; `.venv/bin` "
        "is already first on PATH, so `pip install ...` installs into it).\n"
        "* Do not run git push or GitHub commands here: the shell has no credentials. Use the "
        "platform_git tool for commits, pushes, pull requests and publishing.\n"
        "* If a command times out the shell is restarted; re-run slow commands in the background."
    )

    workspace: Path = Field(default_factory=Path.cwd)
    timeout_seconds: float = 300.0

    async def execute(self, command: str | None = None, restart: bool = False, **kwargs) -> CLIResult:
        if restart or (self._session is not None and getattr(self._session, "_timed_out", False)):
            self.close()
            self._session = None
            if restart:
                self._session = _WorkspaceBashSession(self.workspace, self.timeout_seconds)
                await self._session.start()
                return CLIResult(system="tool has been restarted.")
        if self._session is None:
            self._session = _WorkspaceBashSession(self.workspace, self.timeout_seconds)
            await self._session.start()
        if command is None:
            return CLIResult(error="no command provided.")
        return await self._session.run(command)

    def close(self) -> None:
        if isinstance(self._session, _WorkspaceBashSession):
            self._session.kill()
        self._session = None


class WorkspacePython(BaseTool):
    """Run a Python snippet as a subprocess inside the project workspace."""

    name: str = "python_execute"
    description: str = (
        "Execute a Python 3 script in the project workspace (a fresh process each call; use print() to "
        "see results). Uses the project's .venv if it exists. Prefer the bash tool for running project "
        "commands and tests."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 60, max 600)."},
        },
        "required": ["code"],
    }
    workspace: Path = Field(default_factory=Path.cwd)

    async def execute(self, code: str, timeout: int = 60, **kwargs) -> ToolResult:
        timeout = max(1, min(int(timeout or 60), 600))
        venv_python = self.workspace / ".venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else sys.executable
        proc = await asyncio.create_subprocess_exec(
            python, "-",
            cwd=str(self.workspace),
            env=workspace_env(self.workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(code.encode()), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            return ToolResult(error=f"Execution timed out after {timeout} seconds")
        text = out.decode(errors="replace")
        if proc.returncode != 0:
            return ToolResult(error=f"exit code {proc.returncode}\n{text}")
        return ToolResult(output=text or "(no output)")


class WorkspaceEditor(StrReplaceEditor):
    """StrReplaceEditor that also accepts paths relative to the project workspace."""

    workspace: Path = Field(default_factory=Path.cwd)

    async def execute(self, *, command: str, path: str, **kwargs: Any) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = (self.workspace / p).resolve()
        if command == "create":
            # Make retries safe: local models often repeat a create call after a
            # transient/tool-text error. If the requested file already contains
            # the requested content, report it as verified instead of failing.
            requested = kwargs.get("file_text")
            if p.is_file() and requested is not None:
                try:
                    existing = p.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    existing = None
                if existing == requested:
                    return f"File already exists at: {p}; existing content matches the requested content and was verified."
            # The base editor can't create files in folders that don't exist yet
            # (e.g. src/app.py in a brand-new project).
            p.parent.mkdir(parents=True, exist_ok=True)
        return await super().execute(command=command, path=str(p), **kwargs)


class PlatformAskHuman(BaseTool):
    """Ask the user a question in the platform UI and wait for the answer."""

    name: str = "ask_human"
    description: str = (
        "Ask the user a question and wait for their reply in the platform UI. Use it only when you are "
        "genuinely blocked: missing credentials or accounts, an ambiguous requirement with materially "
        "different outcomes, or an action that needs explicit approval. Do not use it for progress updates."
    )
    parameters: dict = {
        "type": "object",
        "properties": {"inquire": {"type": "string", "description": "The question for the user."}},
        "required": ["inquire"],
    }
    ask: Any = None  # async (question) -> answer | None

    async def execute(self, inquire: str, **kwargs) -> str:
        if self.ask is None:
            return "No user is available to answer. Proceed with your best judgement and state your assumptions."
        answer = await self.ask(inquire)
        if answer is None:
            return (
                "The user did not answer in time. Proceed with your best judgement, choose the safest "
                "reasonable option and state your assumption in the final summary."
            )
        return f"The user replied: {answer}"


# ------------------------------------------------------------ system prompt

PLATFORM_SYSTEM_PROMPT = """\
You are OpenManus, an autonomous software engineer working inside a persistent cloud workspace.
You don't just generate code: you build it, run it, test it, look at it, diagnose failures, fix them
and verify the result before reporting completion.

PROJECT
- Name: {project_name}
- Workspace (your working directory): {workspace}
- GitHub repository: {repository}
- Current branch: {branch}

TOOLS
- bash: persistent shell in the workspace (Node.js/npm, Python 3, git; project .venv first on PATH).
- python_execute: run a Python script in the workspace.
- str_replace_editor: view/create/edit files (relative paths resolve against the workspace).
- platform_browser: the project's shared browser. The user watches it live and may take over.
  Use it for research and to test web apps you run locally (http://localhost:<port>). Before navigating,
  verify the URL is actually listening from bash. Use `screenshot` to see the page, and judge the visual
  result critically. A successful shell command or a model-generated statement is not proof that the
  server is running; the browser result is the proof.
- web_search: search the internet.
- platform_git: connect an existing GitHub repository, status/diff/log, init, branches, commit, push,
  create_pull_request and publish_repository (creates a new GitHub repo). Credentials are handled for you.
- ask_human: ask the user only when genuinely blocked.
- terminate: finish.

WORKING METHOD (Plan -> Act -> Observe -> Verify -> Repeat)
1. Understand the request. Inspect the existing project first (list files, read README/package
   manifests) before changing anything.
2. Write a short plan for yourself, then execute it step by step.
3. After every action, read the result. On errors, diagnose the real cause from the output and fix it. If a file-create action reports that the file already exists, do not repeat `create`: use `view` or a safe edit, verify its contents, and continue.
4. Verify with evidence: install dependencies, run the build and the tests, start the app and check it
   in the browser for UI work. Add or update tests for new behaviour when the project has a test setup.
5. Keep all files inside the workspace. Never print, log or commit secrets. Keep a .gitignore with
   node_modules/, .venv/, build outputs and .env files.
6. Git: when the user asks for commits/PRs (or a repository is connected and the change is
   complete), work on a feature branch rather than the default branch, commit with clear messages,
   then push/open a pull request via platform_git. Do not try to push from bash.
7. Finish with `terminate` only when the task is done and verified (or truly impossible). Your final
   message before terminating must summarise what you changed, how you verified it, and anything the
   user must do next.
8. If a model or tool request times out, do not claim the project is complete. Preserve any useful partial
   work, explain the timeout clearly, and allow the task to be retried.

After you finish, the platform independently runs the project's build/tests. If they fail, you will
get the failing output and must repair it.
"""


# -------------------------------------------------------------------- agent


class PlatformManus(Manus):
    """Manus wired into the platform (workspace tools, live events, human loop)."""

    name: str = "OpenManus"
    max_steps: int = 40
    max_observe: int = 12000
    emit: Any = Field(default=None, exclude=True)
    inbox: Any = Field(default=None, exclude=True)  # asyncio.Queue[str] of user messages

    @classmethod
    async def create_for_project(
        cls,
        *,
        project_name: str,
        workspace: str | Path,
        repository: str | None,
        branch: str | None,
        emit: EmitFn | None = None,
        inbox: asyncio.Queue | None = None,
        ask: Callable[[str], Awaitable[Optional[str]]] | None = None,
        extra_tools: list[BaseTool] | None = None,
        llm: Any = None,
        max_steps: int | None = None,
    ) -> "PlatformManus":
        ws = Path(workspace).resolve()
        tools: list[BaseTool] = [
            WorkspaceBash(workspace=ws, timeout_seconds=float(_int_env("AGENT_COMMAND_TIMEOUT", 300, 30, 3600))),
            WorkspacePython(workspace=ws),
            WorkspaceEditor(workspace=ws),
        ]
        try:
            from app.tool.web_search import WebSearch

            tools.append(WebSearch())
        except Exception as exc:  # optional search backends missing
            logger.warning(f"web_search unavailable: {exc}")
        tools += list(extra_tools or [])
        tools += [PlatformAskHuman(ask=ask), Terminate()]
        system_prompt = PLATFORM_SYSTEM_PROMPT.format(
            project_name=project_name,
            workspace=ws,
            repository=repository or "none connected",
            branch=branch or "none",
        )
        return await cls.create(
            available_tools=ToolCollection(*tools),
            system_prompt=system_prompt,
            llm=llm,
            max_steps=max_steps or _int_env("AGENT_MAX_STEPS", 40, 5, 500),
            emit=emit,
            inbox=inbox,
        )

    async def _emit(self, type_: str, message: str, data: dict | None = None) -> None:
        if self.emit is None:
            return
        try:
            await self.emit(type_, message, data or {})
        except Exception as exc:  # events must never break the agent
            logger.warning(f"event emit failed: {exc}")

    def _drain_inbox(self) -> list[str]:
        messages: list[str] = []
        if self.inbox is None:
            return messages
        while True:
            try:
                messages.append(self.inbox.get_nowait())
            except asyncio.QueueEmpty:
                return messages

    async def think(self) -> bool:
        for text in self._drain_inbox():
            self.memory.add_message(
                Message.user_message(f"Message from the user (sent while you were working):\n{text}")
            )
            await self._emit("agent.user_message", "Delivered your message to the agent", {"message": text})
        await self._emit("agent.step", f"Step {self.current_step}/{self.max_steps}", {"step": self.current_step})
        should_act = await super().think()
        last = self.memory.messages[-1] if self.memory.messages else None
        content = last.content if last is not None and last.role == "assistant" else ""
        calls = [
            {"name": c.function.name, "arguments": _clip(c.function.arguments or "", 1500)}
            for c in (self.tool_calls or [])
        ]
        if content or calls:
            summary = _clip(content, 400) if content else ", ".join(c["name"] for c in calls)
            await self._emit("agent.thought", summary or "thinking", {"content": _clip(content, 6000), "tool_calls": calls})
        return should_act

    async def execute_tool(self, command: ToolCall) -> str:
        name = command.function.name if command and command.function else "?"
        args = command.function.arguments if command and command.function else ""
        await self._emit("agent.tool_call", f"{name}", {"tool": name, "arguments": _clip(args or "", 3000)})
        result = await super().execute_tool(command)
        head = result[:400]
        ok = not (head.startswith("Error") or "\nError: " in head)
        data: dict[str, Any] = {"tool": name, "ok": ok, "output": _clip(result, 4000)}
        if self._current_base64_image:
            data["image"] = True
        await self._emit("agent.tool_result", f"{name} {'done' if ok else 'failed'}", data)
        return result

    def final_summary(self) -> str:
        """The agent's last substantive message, used as the task result."""
        for msg in reversed(self.memory.messages):
            if msg.role == "assistant" and msg.content and msg.content.strip():
                return msg.content.strip()
        return "Task completed."

    def close_processes(self) -> None:
        tool = self.available_tools.tool_map.get("bash")
        if isinstance(tool, WorkspaceBash):
            tool.close()


__all__ = [
    "PlatformManus",
    "WorkspaceBash",
    "WorkspacePython",
    "WorkspaceEditor",
    "PlatformAskHuman",
    "workspace_env",
]
