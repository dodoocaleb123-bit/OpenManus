from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from app.logger import logger
from app.platform.browser import BrowserManager
from app.platform.coding_loop import CodingLoop
from app.platform.llm_check import llm_problem
from app.platform.models import TERMINAL_STATUSES, Event, Project, Task, TaskStatus
from app.platform.store import PlatformStore

AgentFactory = Callable[..., Awaitable[Any]]


def _human_timeout() -> float:
    try:
        return max(30.0, float(os.environ.get("HUMAN_REPLY_TIMEOUT", 900)))
    except ValueError:
        return 900.0


class TaskNotRunning(RuntimeError):
    pass


class AgentOrchestrator:
    """Durable execution boundary around the OpenManus agent.

    Runs each task as a background asyncio task and owns its live channels:
    event emission, cancellation, the agent's questions to the human and
    messages the human sends while the agent works.
    """

    def __init__(
        self,
        store: PlatformStore,
        browsers: BrowserManager | None = None,
        agent_factory: AgentFactory | None = None,
        check_llm: bool = True,
    ):
        self.store = store
        self.browsers = browsers
        self.agent_factory = agent_factory or self._default_agent_factory
        self.check_llm = check_llm
        self._running: dict[str, asyncio.Task | None] = {}
        self._inboxes: dict[str, asyncio.Queue] = {}
        self._questions: dict[str, tuple[str, asyncio.Future]] = {}

    # ------------------------------------------------------------ lifecycle

    def start(self, task: Task) -> None:
        """Run ``task`` in the background (no-op if it is already running)."""
        if task.id in self._running:
            return
        self._running[task.id] = None  # reserve before the coroutine starts
        self._running[task.id] = asyncio.create_task(self.run_task(task), name=f"task-{task.id}")

    def is_running(self, task_id: str) -> bool:
        return task_id in self._running

    async def recover(self) -> None:
        for task in await self.store.recover_interrupted():
            await self.store.emit(Event(task_id=task.id, type="task.recovered", message="Task re-queued after process interruption"))
            self.start(task)

    async def resume_task(self, task: Task) -> None:
        if task.id in self._running and self._running[task.id] is not None:
            return
        task.status = TaskStatus.QUEUED
        task.finished_at = None
        task.error = None
        task.result = None
        task.checkpoint = "resumed"
        await self.store.save_task(task)
        await self.store.emit(Event(task_id=task.id, type="task.resumed", message="Task resumed"))
        self._running.pop(task.id, None)
        self.start(task)

    async def cancel(self, task_id: str) -> bool:
        handle = self._running.get(task_id)
        if handle is None or handle.done():
            return False
        handle.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(handle), timeout=15)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
        return True

    async def shutdown(self) -> None:
        for task_id in list(self._running):
            await self.cancel(task_id)

    # ------------------------------------------------------- human channel

    def pending_question(self, task_id: str) -> Optional[str]:
        entry = self._questions.get(task_id)
        return entry[0] if entry and not entry[1].done() else None

    async def send_message(self, task_id: str, text: str) -> str:
        """Answer the agent's pending question, or queue guidance for its next step."""
        text = text.strip()
        if not text:
            raise ValueError("Message is empty")
        entry = self._questions.get(task_id)
        if entry and not entry[1].done():
            entry[1].set_result(text)
            await self.store.emit(Event(task_id=task_id, type="human.reply", message=text))
            return "answered"
        if task_id not in self._running or task_id not in self._inboxes:
            raise TaskNotRunning("Task is not running")
        self._inboxes[task_id].put_nowait(text)
        await self.store.emit(Event(task_id=task_id, type="human.message", message=text))
        return "queued"

    async def _ask(self, task: Task, question: str) -> Optional[str]:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._questions[task.id] = (question, future)
        timeout = _human_timeout()
        task.checkpoint = "awaiting_human"
        await self.store.save_task(task)
        await self.store.emit(Event(task_id=task.id, type="agent.question", message=question, data={"question": question, "timeout_seconds": timeout}))
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            await self.store.emit(Event(task_id=task.id, type="agent.question_timeout", message="No reply received; the agent will continue with its best judgement"))
            return None
        finally:
            self._questions.pop(task.id, None)
            task.checkpoint = "running"
            await self.store.save_task(task)

    # ---------------------------------------------------------------- agent

    async def _default_agent_factory(self, *, project: Project, task: Task, emit, inbox, ask, extra_tools):
        from app.platform.agent import PlatformManus

        return await PlatformManus.create_for_project(
            project_name=project.name,
            workspace=project.workspace,
            repository=project.repository,
            branch=project.branch,
            emit=emit,
            inbox=inbox,
            ask=ask,
            extra_tools=extra_tools,
        )

    def _browser_tool(self, task: Task, project: Project):
        if self.browsers is None:
            return None
        from app.platform.browser_agent_tool import PlatformBrowserTool

        existing = self.browsers.get(task.browser_session_id) if task.browser_session_id else None
        if existing is None:
            existing = self.browsers.for_project(project.id)

        async def factory(url: Optional[str]):
            session = await self.browsers.get_or_create(project.id, url)
            task.browser_session_id = session.id
            await self.store.save_task(task)
            await self.store.emit(Event(task_id=task.id, type="browser.started", message=f"Agent opened the shared browser {session.id}", data={"session_id": session.id}))
            return session

        return PlatformBrowserTool(session=existing, session_factory=factory)

    async def run_task(self, task: Task) -> None:
        self._running.setdefault(task.id, None)
        task.status = TaskStatus.RUNNING
        task.attempt += 1
        task.started_at = datetime.now(timezone.utc)
        task.finished_at = None
        task.error = None
        task.checkpoint = "running"
        await self.store.save_task(task)
        await self.store.emit(Event(task_id=task.id, type="task.started", message="Agent started", data={"attempt": task.attempt}))

        agent = None
        inbox: asyncio.Queue = asyncio.Queue()
        self._inboxes[task.id] = inbox
        try:
            project = self.store.get_project(task.project_id)
            if not project:
                raise RuntimeError("Project no longer exists")
            if self.check_llm and (problem := llm_problem()):
                raise RuntimeError(problem)

            await self.store.emit(Event(task_id=task.id, type="workspace.ready", message=f"Workspace ready: {project.workspace}", data={"workspace": project.workspace}))

            async def emit(type_: str, message: str, data: dict) -> None:
                await self.store.emit(Event(task_id=task.id, type=type_, message=message, data=data))

            extra_tools: list[Any] = []
            browser_tool = self._browser_tool(task, project)
            if browser_tool is not None:
                extra_tools.append(browser_tool)
                if browser_tool.session is not None:
                    await emit("browser.attached", f"Attached shared browser session {browser_tool.session.id}", {"session_id": browser_tool.session.id})
            from app.platform.git_tool import PlatformGitTool

            extra_tools.append(PlatformGitTool(store=self.store, project_id=project.id, on_event=emit))

            agent = await self.agent_factory(
                project=project, task=task, emit=emit, inbox=inbox,
                ask=lambda q: self._ask(task, q), extra_tools=extra_tools,
            )

            await emit("agent.running", "Executing autonomous coding workflow", {})
            loop = CodingLoop(project.workspace)
            max_cycles = loop.max_repair_cycles
            agent.current_step = 0
            history = await asyncio.to_thread(self.store.list_chat_messages, task.project_id, 20)
            conversation = "\n".join(f"{item.role.upper()}: {item.content}" for item in history)
            conversation = conversation[-30000:]
            await agent.run(
                f"RECENT PROJECT CONVERSATION:\n{conversation}\n\n"
                f"CURRENT USER MESSAGE:\n{task.prompt}\n\n"
                "UNIFIED CONVERSATION REQUIREMENT: Treat this as one continuous project conversation. "
                "First understand the user's intent. If the user is asking a question, requesting an explanation, "
                "or asking for inspection only, read the relevant project files and respond without editing files "
                "or changing project state. If the user explicitly asks to build, modify, fix, refactor, test, "
                "commit, or otherwise change the project, implement the request and verify it. For any code change, "
                "inspect the project first, run appropriate tests/builds, and visually check UI work when relevant. "
                "Do not claim to have changed or verified anything without evidence."
            )
            task.coding_iteration = 1
            task.checkpoint = "implementation_complete"
            await self.store.save_task(task)
            await emit("coding.implementation", "Initial implementation pass completed", {"iteration": 1})

            await emit("coding.validating", "Running independent build/test validation", {})
            validation_results = await loop.validate()
            for cycle in range(max_cycles + 1):
                task.coding_iteration = cycle + 1
                payload = [r.as_dict() for r in validation_results]
                task.validation = {"results": payload, "passed": all(r.ok for r in validation_results)}
                task.checkpoint = "validation_passed" if task.validation["passed"] else "validation_failed"
                await self.store.save_task(task)
                await emit("coding.validation", "Validation passed" if task.validation["passed"] else "Validation failed", {"iteration": task.coding_iteration, "results": payload})
                if task.validation["passed"] or cycle >= max_cycles:
                    break
                failures = "\n\n".join(f"COMMAND: {r.command}\nOUTPUT:\n{r.output}" for r in validation_results if not r.ok)
                repair_prompt = (
                    f"The implementation pass is complete, but independent validation failed in project '{project.name}'.\n"
                    f"Workspace: {project.workspace}\n\n{failures}\n\n"
                    "Diagnose the actual failure, inspect the relevant code, make the smallest correct repair, and run the appropriate tests again. "
                    "Do not just explain the failure; modify the workspace to fix it. Preserve existing working behavior."
                )
                await emit("coding.repair", f"Starting repair cycle {cycle + 1}", {"cycle": cycle + 1, "failures": failures[-12000:]})
                agent.current_step = 0
                await agent.run(repair_prompt)
                validation_results = await loop.validate()

            summary = agent.final_summary() if hasattr(agent, "final_summary") else "Task completed."
            task.result = summary
            assistant_reply = summary or task.error or "Task completed."
            await asyncio.to_thread(store.add_chat_message, task.project_id, "assistant", assistant_reply)
            if task.validation and task.validation.get("passed"):
                task.status = TaskStatus.SUCCEEDED
                task.checkpoint = "validated"
                await emit("task.succeeded", "Assistant response completed and validation passed", {"result": summary, "validation": task.validation})
            else:
                task.status = TaskStatus.FAILED
                task.error = "Automatic validation still fails."
                task.checkpoint = "validation_exhausted"
                await emit("task.failed", task.error, {"result": summary, "validation": task.validation})
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.error = "Cancelled by user"
            task.checkpoint = "cancelled"
            if agent is not None and hasattr(agent, "final_summary"):
                task.result = agent.final_summary()
            await self.store.emit(Event(task_id=task.id, type="task.cancelled", message="Task cancelled"))
        except Exception as exc:
            logger.exception(f"Task {task.id} failed")
            task.error = str(exc) or exc.__class__.__name__
            task.status = TaskStatus.FAILED
            task.checkpoint = "failed"
            await self.store.emit(Event(task_id=task.id, type="task.failed", message=task.error, data={"error": task.error}))
        finally:
            task.finished_at = datetime.now(timezone.utc)
            if task.status not in TERMINAL_STATUSES:
                task.status = TaskStatus.FAILED
            await self.store.save_task(task)
            if agent is not None:
                for closer in ("close_processes", "cleanup"):
                    fn = getattr(agent, closer, None)
                    if fn is None:
                        continue
                    try:
                        result = fn()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass
            question = self._questions.pop(task.id, None)
            if question and not question[1].done():
                question[1].cancel()
            self._inboxes.pop(task.id, None)
            self._running.pop(task.id, None)
