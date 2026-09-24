from __future__ import annotations

import json
import os
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.platform.git import GitError, GitWorkspace
from app.platform.git_service import ProjectGit
from app.platform.github import GitHubClient, GitHubError
from app.platform.llm_check import llm_status
from app.platform.models import TERMINAL_EVENT_TYPES, TERMINAL_STATUSES, Task, TaskStatus
from app.platform.orchestrator import AgentOrchestrator, TaskNotRunning
from app.platform.store import PlatformStore

# Proxies (Render included) close idle HTTP connections; a comment frame every
# 20s keeps the stream alive. Streams also end after ~14 minutes and the
# browser's EventSource reconnects with Last-Event-ID, resuming seamlessly.
SSE_HEARTBEAT_INTERVAL = 20.0
SSE_MAX_LIFETIME = 840.0


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    repository: str | None = None
    branch: str | None = None


class TaskCreate(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1, max_length=50000)
    browser_session_id: str | None = None


class TaskMessage(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


class GitConnect(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    branch: str | None = Field(default=None, max_length=100)


class GitBranch(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class GitCommit(BaseModel):
    message: str = Field(min_length=1, max_length=200)


class PullRequestCreate(BaseModel):
    title: str = Field(min_length=1, max_length=250)
    body: str = Field(default="", max_length=60000)
    base: str | None = Field(default=None, max_length=100)
    draft: bool = False


class RepositoryPublish(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    private: bool = True
    description: str = Field(default="", max_length=350)


async def sse_event_stream(
    store: PlatformStore,
    task_id: str,
    after_id: str | None = None,
    heartbeat_interval: float = SSE_HEARTBEAT_INTERVAL,
    max_lifetime: float = SSE_MAX_LIFETIME,
):
    """Yield task events as Server-Sent Event frames with Last-Event-ID resume."""
    started = time.monotonic()
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
            if event.type in TERMINAL_EVENT_TYPES:
                return
        task = store.get_task(task_id)
        if task is None or task.status in TERMINAL_STATUSES:
            return
        if time.monotonic() - started >= max_lifetime:
            return
        before = len(events)
        await store.wait_for_event(task_id, heartbeat_interval)
        if len(await store.list_events(task_id)) == before:
            yield ": ping\n\n"


def build_router(store: PlatformStore, orchestrator: AgentOrchestrator) -> APIRouter:
    router = APIRouter(prefix="/api")

    def project_or_404(project_id: str):
        project = store.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    def task_or_404(task_id: str) -> Task:
        task = store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.pending_question = orchestrator.pending_question(task_id)
        return task

    def git_error(exc: Exception) -> HTTPException:
        if isinstance(exc, RuntimeError) and "GITHUB_TOKEN" in str(exc):
            return HTTPException(status_code=503, detail=str(exc))
        return HTTPException(status_code=400, detail=str(exc))

    @router.get("/health")
    async def health():
        return {"status": "ok", "service": "openmanus-platform"}

    @router.get("/status")
    async def platform_status():
        """Setup status for the UI: is the model configured, is GitHub connected."""
        llm = llm_status()
        return {
            "llm": {k: llm.get(k) for k in ("configured", "problem", "model", "api_type", "supports_images")},
            "github": {"token_configured": bool(os.getenv("GITHUB_TOKEN"))},
            "auth": {"enabled": bool(os.getenv("PLATFORM_PASSWORD"))},
        }

    # ---------------------------------------------------------- projects

    @router.post("/projects")
    async def create_project(body: ProjectCreate):
        try:
            return store.create_project(body.name.strip(), body.repository, body.branch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/projects")
    async def list_projects():
        return list(store.projects.values())

    @router.get("/projects/{project_id}")
    async def get_project(project_id: str):
        return project_or_404(project_id)

    @router.get("/projects/{project_id}/files")
    async def project_files(project_id: str):
        """Top-level listing of the workspace (for a quick look at what the agent built)."""
        project = project_or_404(project_id)
        root = GitWorkspace(project.workspace).path
        skip = {".git", "node_modules", ".venv", "__pycache__", ".next", ".pytest_cache"}
        entries = []
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if any(part in skip for part in rel.parts):
                continue
            if len(rel.parts) > 3:
                continue
            entries.append({"path": str(rel), "dir": path.is_dir()})
            if len(entries) >= 400:
                break
        return {"workspace": project.workspace, "entries": entries}

    # ------------------------------------------------------------ github

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
            return await ProjectGit(store, project).connect(body.owner, body.repo, body.branch)
        except (GitError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/github/pull-request")
    async def create_pull_request(project_id: str, body: PullRequestCreate):
        project = project_or_404(project_id)
        try:
            return await ProjectGit(store, project).pull_request(body.title, body.body, body.base, body.draft)
        except (GitError, GitHubError, ValueError, RuntimeError) as exc:
            raise git_error(exc) from exc

    @router.post("/projects/{project_id}/github/publish")
    async def publish_repository(project_id: str, body: RepositoryPublish):
        project = project_or_404(project_id)
        try:
            result = await ProjectGit(store, project).publish(body.name, body.private, body.description)
            return {**result, "project": store.get_project(project_id)}
        except (GitError, GitHubError, ValueError, RuntimeError) as exc:
            raise git_error(exc) from exc

    # --------------------------------------------------------------- git

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

    @router.get("/projects/{project_id}/git/log")
    async def git_log(project_id: str):
        project = project_or_404(project_id)
        try:
            return {"log": await GitWorkspace(project.workspace).log(20)}
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/git/branch")
    async def git_branch(project_id: str, body: GitBranch):
        project = project_or_404(project_id)
        svc = ProjectGit(store, project)
        try:
            branch = await svc.create_branch(body.name)
            return {"branch": branch, "project": svc.project}
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/git/commit")
    async def git_commit(project_id: str, body: GitCommit):
        project = project_or_404(project_id)
        svc = ProjectGit(store, project)
        try:
            result = await svc.commit(body.message)
            return {"result": result, "status": await svc.git.status()}
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/git/push")
    async def git_push(project_id: str):
        project = project_or_404(project_id)
        try:
            return {"result": await ProjectGit(store, project).push()}
        except GitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------- tasks

    @router.post("/tasks")
    async def create_task(body: TaskCreate):
        if not store.get_project(body.project_id):
            raise HTTPException(status_code=404, detail="Project not found")
        # Validate the browser attachment first so a rejected request does not
        # leave an orphaned task behind.
        if body.browser_session_id:
            browsers = getattr(orchestrator, "browsers", None)
            session = browsers.get(body.browser_session_id) if browsers else None
            if session is None or session.project_id != body.project_id:
                raise HTTPException(status_code=400, detail="Browser session does not belong to this project")
        running = [
            t for t in store.tasks.values()
            if t.project_id == body.project_id and orchestrator.is_running(t.id)
        ]
        if running:
            raise HTTPException(
                status_code=409,
                detail="An agent task is already running in this project. Send it a message, or cancel it first.",
            )
        task = store.create_task(body.project_id, body.prompt.strip())
        if body.browser_session_id:
            task.browser_session_id = body.browser_session_id
            await store.save_task(task)
        orchestrator.start(task)
        return task

    @router.get("/projects/{project_id}/tasks")
    async def project_tasks(project_id: str):
        if not store.get_project(project_id):
            raise HTTPException(status_code=404, detail="Project not found")
        tasks = [task for task in store.tasks.values() if task.project_id == project_id]
        for task in tasks:
            task.pending_question = orchestrator.pending_question(task.id)
        return tasks

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        return task_or_404(task_id)

    @router.post("/tasks/{task_id}/resume")
    async def resume_task(task_id: str):
        task = task_or_404(task_id)
        if orchestrator.is_running(task_id) or task.status == TaskStatus.RUNNING:
            raise HTTPException(status_code=409, detail="Task is already running")
        await orchestrator.resume_task(task)
        return task

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str):
        task_or_404(task_id)
        if not await orchestrator.cancel(task_id):
            raise HTTPException(status_code=409, detail="Task is not running")
        return task_or_404(task_id)

    @router.post("/tasks/{task_id}/messages")
    async def message_task(task_id: str, body: TaskMessage):
        """Answer the agent's question, or send guidance it will read at its next step."""
        task_or_404(task_id)
        try:
            delivery = await orchestrator.send_message(task_id, body.message)
        except TaskNotRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"delivery": delivery}

    @router.get("/tasks/{task_id}/events/history")
    async def task_event_history(task_id: str):
        task_or_404(task_id)
        return await store.list_events(task_id)

    @router.get("/tasks/{task_id}/events")
    async def task_events(task_id: str, request: Request, last_event_id: str | None = None):
        task_or_404(task_id)
        # Browsers send Last-Event-ID automatically when EventSource reconnects.
        after_id = request.headers.get("last-event-id") or last_event_id
        return StreamingResponse(
            sse_event_stream(store, task_id, after_id=after_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
