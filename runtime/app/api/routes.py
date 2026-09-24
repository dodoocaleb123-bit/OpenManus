from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.platform.github import GitHubClient
from app.platform.git import GitError, GitWorkspace, github_remote
from app.platform.models import TaskStatus
from app.platform.orchestrator import AgentOrchestrator
from app.platform.store import PlatformStore

TERMINAL_EVENT_TYPES = {"task.succeeded", "task.failed"}

# Keep well under Railway's 15-minute streaming cap; clients reconnect with
# Last-Event-ID and resume where they left off.
SSE_HEARTBEAT_INTERVAL = 25.0
SSE_MAX_LIFETIME = 840.0
SSE_POLL_INTERVAL = 1.0


async def sse_event_stream(
    store: PlatformStore,
    task_id: str,
    after_id: str | None = None,
    heartbeat_interval: float = SSE_HEARTBEAT_INTERVAL,
    max_lifetime: float = SSE_MAX_LIFETIME,
    poll_interval: float = SSE_POLL_INTERVAL,
):
    """Yield task events as Server-Sent Event frames.

    ``after_id`` implements Last-Event-ID resume: events up to and including
    that id are skipped, so an auto-reconnecting EventSource sees no duplicates.
    Idle connections get ``: ping`` comments so proxies do not close them.
    """
    started = time.monotonic()
    last_beat = started
    index = 0
    if after_id:
        past = await store.list_events(task_id)
        index = next((i + 1 for i, e in enumerate(past) if e.id == after_id), 0)
    while True:
        events = await store.list_events(task_id)
        while index < len(events):
            event = events[index]
            index += 1
            yield f"id: {event.id}\ndata: {json.dumps(event.model_dump(mode='json'))}\n\n"
            last_beat = time.monotonic()
            if event.type in TERMINAL_EVENT_TYPES:
                return
        task = store.get_task(task_id)
        if task is None or task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
            return
        now = time.monotonic()
        if now - started >= max_lifetime:
            return
        if now - last_beat >= heartbeat_interval:
            yield ": ping\n\n"
            last_beat = now
        await asyncio.sleep(poll_interval)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    repository: str | None = None
    branch: str | None = None


class TaskCreate(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1)
    browser_session_id: str | None = None


class GitConnect(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    branch: str | None = Field(default=None, max_length=100)


class GitBranch(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class GitCommit(BaseModel):
    message: str = Field(min_length=1, max_length=200)


def build_router(store: PlatformStore, orchestrator: AgentOrchestrator) -> APIRouter:
    router = APIRouter(prefix="/api")

    def project_or_404(project_id: str):
        project = store.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    @router.get("/health")
    async def health():
        return {"status": "ok", "service": "openmanus-platform"}

    @router.post("/projects")
    async def create_project(body: ProjectCreate):
        return store.create_project(body.name.strip(), body.repository, body.branch)

    @router.get("/projects")
    async def list_projects():
        return list(store.projects.values())

    @router.get("/github/user")
    async def github_user():
        try:
            return await GitHubClient().get_user()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}") from exc

    @router.get("/github/repos")
    async def github_repositories():
        try:
            return await GitHubClient().list_repositories()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}") from exc

    @router.get("/github/repos/{owner}/{repo}")
    async def github_repository(owner: str, repo: str):
        try:
            return await GitHubClient().get_repository(owner, repo)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}") from exc

    @router.post("/projects/{project_id}/github/connect")
    async def connect_github(project_id: str, body: GitConnect):
        project = project_or_404(project_id)
        try:
            remote = github_remote(body.owner, body.repo)
            await GitWorkspace(project.workspace).clone(remote, body.branch)
            git = GitWorkspace(project.workspace)
            project.repository = f"{body.owner}/{body.repo}"
            project.branch = await git.current_branch()
            project.git_ready = True
            store.update_project(project)  # persist the connection across reloads
            return {"project": project, "git": await git.status()}
        except (GitError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/projects/{project_id}/git/status")
    async def git_status(project_id: str):
        project = project_or_404(project_id)
        try:
            return await GitWorkspace(project.workspace).status()
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/projects/{project_id}/git/diff")
    async def git_diff(project_id: str):
        project = project_or_404(project_id)
        try:
            return {"diff": await GitWorkspace(project.workspace).diff()}
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/git/branch")
    async def git_branch(project_id: str, body: GitBranch):
        project = project_or_404(project_id)
        try:
            git = GitWorkspace(project.workspace)
            branch = await git.create_branch(body.name)
            project.branch = branch
            store.update_project(project)  # persist the branch across reloads
            return {"branch": branch, "project": project}
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/git/commit")
    async def git_commit(project_id: str, body: GitCommit):
        project = project_or_404(project_id)
        try:
            git = GitWorkspace(project.workspace)
            result = await git.commit(body.message)
            return {"result": result, "status": await git.status()}
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/git/push")
    async def git_push(project_id: str):
        project = project_or_404(project_id)
        try:
            result = await GitWorkspace(project.workspace).push(project.branch)
            return {"result": result}
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/tasks/{task_id}/resume")
    async def resume_task(task_id: str):
        task = store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status.value == "running":
            raise HTTPException(status_code=409, detail="Task is already running")
        asyncio.create_task(orchestrator.resume_task(task))
        return task

    @router.get("/tasks/{task_id}/events/history")
    async def task_event_history(task_id: str):
        if not store.get_task(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        return await store.list_events(task_id)

    @router.post("/tasks")
    async def create_task(body: TaskCreate):
        try:
            # Validate the browser attachment first so a rejected request does
            # not leave an orphaned task behind.
            if body.browser_session_id:
                # The browser must belong to the same project; cross-project attachment is forbidden.
                browsers = getattr(orchestrator, "browsers", None)
                session = browsers.get(body.browser_session_id) if browsers else None
                if session is None or session.project_id != body.project_id:
                    raise ValueError("Browser session does not belong to this project")
            task = store.create_task(body.project_id, body.prompt.strip())
            if body.browser_session_id:
                task.browser_session_id = body.browser_session_id
            await store.save_task(task)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        asyncio.create_task(orchestrator.run_task(task))
        return task

    @router.get("/projects/{project_id}/tasks")
    async def project_tasks(project_id: str):
        if not store.get_project(project_id):
            raise HTTPException(status_code=404, detail="Project not found")
        return [task for task in store.tasks.values() if task.project_id == project_id]

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        task = store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @router.get("/tasks/{task_id}/events")
    async def task_events(task_id: str, request: Request, last_event_id: str | None = None):
        if not store.get_task(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        # Browsers send Last-Event-ID automatically when EventSource reconnects.
        after_id = request.headers.get("last-event-id") or last_event_id
        return StreamingResponse(
            sse_event_stream(store, task_id, after_id=after_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
