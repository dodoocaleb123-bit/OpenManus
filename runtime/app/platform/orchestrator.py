from __future__ import annotations

from datetime import datetime, timezone

from app.platform.models import Event, Task, TaskStatus
from app.platform.store import PlatformStore
from app.platform.browser import BrowserManager
from app.platform.coding_loop import CodingLoop


class AgentOrchestrator:
    """Durable execution boundary around the existing OpenManus agent."""

    def __init__(self, store: PlatformStore, browsers: BrowserManager | None = None):
        self.store = store
        self.browsers = browsers
        self._running: set[str] = set()

    async def recover(self) -> None:
        for task in await self.store.recover_interrupted():
            await self.store.emit(Event(task_id=task.id, type="task.recovered", message="Task re-queued after process interruption"))
            await self.run_task(task)

    async def run_task(self, task: Task) -> None:
        if task.id in self._running:
            return
        self._running.add(task.id)
        task.status = TaskStatus.RUNNING
        task.attempt += 1
        task.started_at = datetime.now(timezone.utc)
        task.finished_at = None
        task.error = None
        await self.store.save_task(task)
        await self.store.emit(Event(task_id=task.id, type="task.started", message="Agent started", data={"attempt": task.attempt}))

        project = self.store.get_project(task.project_id)
        if not project:
            task.status = TaskStatus.FAILED
            task.error = "Project no longer exists"
            task.finished_at = datetime.now(timezone.utc)
            await self.store.save_task(task)
            await self.store.emit(Event(task_id=task.id, type="task.failed", message=task.error))
            self._running.discard(task.id)
            return

        await self.store.emit(Event(task_id=task.id, type="workspace.ready", message=f"Workspace ready: {project.workspace}", data={"workspace": project.workspace}))
        agent = None
        try:
            from app.agent.manus import Manus
            agent = await Manus.create()
            if self.browsers and task.browser_session_id:
                browser = self.browsers.get(task.browser_session_id)
                if browser is not None:
                    from app.platform.browser_agent_tool import PlatformBrowserTool
                    agent.available_tools.add_tools(PlatformBrowserTool(session=browser))
                    await self.store.emit(Event(task_id=task.id, type="browser.attached", message=f"Attached shared browser session {browser.id}"))
            prompt = (
                f"You are working on project '{project.name}'.\n"
                f"Project workspace: {project.workspace}\n"
                f"Git repository: {project.repository or 'none'}\n"
                f"Current branch: {project.branch or 'none'}\n"
                + (f"Shared browser session: {task.browser_session_id}\n" if task.browser_session_id else "")
                + "Use this workspace for project files. Do not modify files outside it unless explicitly asked.\n\n"
                + f"USER TASK:\n{task.prompt}"
            )
            await self.store.emit(Event(task_id=task.id, type="agent.running", message="Executing autonomous coding workflow"))
            loop = CodingLoop(project.workspace)
            max_cycles = loop.max_repair_cycles
            # Each agent run gets its full step budget: Manus keeps current_step
            # between runs, so without this the first repair pass would inherit
            # the leftovers of the implementation pass.
            agent.current_step = 0
            result = await agent.run(prompt + "\n\nAUTONOMOUS CODING REQUIREMENT: Inspect the project, implement the requested change, and verify your work. Do not stop merely because files were edited; use available tools to inspect and test the implementation.")
            task.coding_iteration = 1
            task.checkpoint = "implementation_complete"
            await self.store.save_task(task)
            await self.store.emit(Event(task_id=task.id, type="coding.implementation", message="Initial implementation pass completed", data={"iteration": 1}))

            validation_results = await loop.validate()
            for cycle in range(max_cycles + 1):
                task.coding_iteration = cycle + 1
                payload = [r.as_dict() for r in validation_results]
                task.validation = {"results": payload, "passed": all(r.ok for r in validation_results)}
                task.checkpoint = "validation_passed" if task.validation["passed"] else "validation_failed"
                await self.store.save_task(task)
                await self.store.emit(Event(task_id=task.id, type="coding.validation", message=("Validation passed" if task.validation["passed"] else "Validation failed"), data={"iteration": task.coding_iteration, "results": payload}))
                if task.validation["passed"]:
                    break
                if cycle >= max_cycles:
                    break
                failures = "\n\n".join(f"COMMAND: {r.command}\nOUTPUT:\n{r.output}" for r in validation_results if not r.ok)
                repair_prompt = (
                    f"The implementation pass is complete, but independent validation failed in project '{project.name}'.\n"
                    f"Workspace: {project.workspace}\n\n{failures}\n\n"
                    "Diagnose the actual failure, inspect the relevant code, make the smallest correct repair, and run the appropriate tests again. "
                    "Do not just explain the failure; modify the workspace to fix it. Preserve existing working behavior."
                )
                await self.store.emit(Event(task_id=task.id, type="coding.repair", message=f"Starting repair cycle {cycle + 1}", data={"cycle": cycle + 1, "failures": failures[-12000:]}))
                agent.current_step = 0
                await agent.run(repair_prompt)
                validation_results = await loop.validate()

            task.result = str(result) if result is not None else "Task completed."
            if task.validation and task.validation.get("passed"):
                task.status = TaskStatus.SUCCEEDED
                task.checkpoint = "validated"
                await self.store.emit(Event(task_id=task.id, type="task.succeeded", message="Implementation completed and validation passed", data={"result": task.result, "validation": task.validation}))
            else:
                task.status = TaskStatus.FAILED
                task.error = "Implementation completed but automatic validation still fails."
                task.checkpoint = "validation_exhausted"
                await self.store.emit(Event(task_id=task.id, type="task.failed", message=task.error, data={"validation": task.validation}))
        except Exception as exc:
            task.error = str(exc)
            task.status = TaskStatus.FAILED
            task.checkpoint = "failed"
            await self.store.emit(Event(task_id=task.id, type="task.failed", message=str(exc), data={"error": str(exc)}))
        finally:
            task.finished_at = datetime.now(timezone.utc)
            await self.store.save_task(task)
            if agent is not None:
                try:
                    await agent.cleanup()
                except Exception:
                    pass
            self._running.discard(task.id)

    async def resume_task(self, task: Task) -> None:
        if task.id in self._running:
            return
        task.status = TaskStatus.QUEUED
        task.finished_at = None
        task.error = None
        task.result = None
        task.checkpoint = "resumed"
        await self.store.save_task(task)
        await self.store.emit(Event(task_id=task.id, type="task.resumed", message="Task resumed"))
        await self.run_task(task)
