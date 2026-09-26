from __future__ import annotations

import asyncio
import base64
import copy
from io import BytesIO
import json
import mimetypes
import os
import re
import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from openai import RateLimitError
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from tenacity import RetryError

from app.config import config
from app.llm import LLM
from app.platform.git import GitError, GitWorkspace
from app.platform.git_service import ProjectGit
from app.platform.github import GitHubClient, GitHubError
from app.platform.llm_check import llm_problem, llm_status
from app.platform.models import TERMINAL_EVENT_TYPES, TERMINAL_STATUSES, Task, TaskStatus, UploadedFile
from app.platform.orchestrator import AgentOrchestrator, TaskNotRunning
from app.platform.store import PlatformStore

# Proxies (Render included) close idle HTTP connections; a comment frame every
# 20s keeps the stream alive. Streams also end after ~14 minutes and the
# browser's EventSource reconnects with Last-Event-ID, resuming seamlessly.
SSE_HEARTBEAT_INTERVAL = 20.0
SSE_MAX_LIFETIME = 840.0


def _rate_limit_exception(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "rate limit" in message or "quota" in message or "too many requests" in message:
        return True
    if isinstance(exc, RateLimitError) or exc.__class__.__name__ == "RateLimitError":
        return True
    if isinstance(exc, RetryError):
        last = exc.last_attempt.exception()
        return bool(last and (isinstance(last, RateLimitError) or last.__class__.__name__ == "RateLimitError"))
    return False


def _image_payload(path: Path, content_type: str | None) -> tuple[bytes, str]:
    """Return a bounded image payload suitable for a vision API request."""
    raw = path.read_bytes()
    mime = content_type or "image/jpeg"
    if len(raw) <= 5 * 1024 * 1024 and mime != "image/svg+xml":
        try:
            with Image.open(BytesIO(raw)) as image:
                if max(image.size) <= 2048:
                    return raw, mime
        except Exception:
            return raw, mime
    if mime == "image/svg+xml":
        return raw, mime
    try:
        with Image.open(BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=True)
            return output.getvalue(), "image/jpeg"
    except Exception:
        return raw, mime


def _workspace_context(root: Path) -> str:
    """Build a bounded, secret-aware snapshot for read-only conversational inspection."""
    ignored = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    secret_names = {".env", ".env.local", "config.toml", "config.json"}
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in ignored for part in path.relative_to(root).parts):
            continue
        if path.name in secret_names or path.name.endswith(".sqlite"):
            continue
        files.append(path)
    tree = "\n".join(str(path.relative_to(root)) for path in files[:300]) or "(workspace has no readable files)"
    sections = [f"Repository file tree (bounded):\n{tree}"]
    preferred = ("README.md", "README", "package.json", "pyproject.toml", "requirements.txt", "Cargo.toml", "go.mod", "Dockerfile")
    total = 0
    for path in files:
        if path.name not in preferred or total >= 120_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        excerpt = text[: 120_000 - total]
        sections.append(f"\n--- {path.relative_to(root)} ---\n{excerpt}")
        total += len(excerpt)
    return "\n".join(sections)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    repository: str | None = None
    branch: str | None = None


class ProjectRename(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class TaskCreate(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1, max_length=50000)
    browser_session_id: str | None = None


class TaskMessage(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


class ChatMessageCreate(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    attachment_ids: list[str] = Field(default_factory=list, max_length=4)


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


UPLOAD_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".odt", ".rtf", ".xls", ".xlsx", ".ods", ".ppt", ".pptx",
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".html", ".htm",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".svg",
}
MAX_UPLOAD_BYTES = max(1, int(os.environ.get("PLATFORM_MAX_UPLOAD_MB", "250"))) * 1024 * 1024
IMAGE_ORDINALS = {
    "first": 0, "1": 0, "second": 1, "2": 1, "third": 2, "3": 2,
    "fourth": 3, "4": 3, "fifth": 4, "5": 4,
}


def referenced_images(text: str, uploads: list[UploadedFile], attachment_ids: list[str]) -> list[UploadedFile]:
    """Return only images explicitly attached or referenced by this message."""
    images = [item for item in uploads if (item.content_type or "").startswith("image/") and Path(item.stored_path).is_file()]
    selected: list[UploadedFile] = []
    by_id = set(attachment_ids)
    selected.extend(item for item in images if item.id in by_id)
    lowered = text.casefold()
    if re.search(r"\b(do not|don't|dont|without|not)\s+(use|inspect|look at|consider)\b.{0,40}\b(image|picture|photo)\b", lowered):
        return []

    for item in images:
        filename = item.filename.casefold()
        stem = Path(item.filename).stem.casefold()
        if filename in lowered or (stem and len(stem) >= 3 and re.search(rf"\b{re.escape(stem)}\b", lowered)):
            selected.append(item)

    ordinal_matches = re.findall(
        r"\b(first|second|third|fourth|fifth|1|2|3|4|5)\s+(?:uploaded\s+)?(?:image|picture|photo)\b|\b(?:image|picture|photo)\s*(?:number\s*)?(1|2|3|4|5)\b",
        lowered,
    )
    for match in ordinal_matches:
        ordinal = next((part for part in match if part), None)
        if ordinal is not None:
            index = IMAGE_ORDINALS[ordinal]
            if index < len(images):
                selected.append(images[index])

    if re.search(r"\b(this|that|the|last|latest|previous|current)\s+(image|picture|photo)\b|\b(image|picture|photo)\s+(above|attached|shown)\b", lowered):
        selected.append(images[-1] if images else None)
    selected_ids = list(dict.fromkeys(item.id for item in selected if item))[-4:]
    return [item for item in images if item.id in selected_ids]


def response_needs_continuation(answer: str, finish_reason: str | None) -> bool:
    """Detect the common provider responses that stop mid-answer."""
    if (finish_reason or "").lower() in {"length", "max_tokens", "token_limit"}:
        return True
    ending = answer.strip().splitlines()[-1].strip() if answer.strip() else ""
    if not ending:
        return True
    return bool(re.search(
        r"(?:[:;,]|\b(?:and|or|but|because|therefore|so|to|the|a|an|of|is|are|was|were|will|I|I'll|Express|Step\s+\d+\.?)$)",
        ending,
        re.IGNORECASE,
    ))


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
    chat_locks: dict[str, asyncio.Lock] = {}

    def project_or_404(project_id: str):
        project = store.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    def chat_lock(project_id: str) -> asyncio.Lock:
        return chat_locks.setdefault(project_id, asyncio.Lock())

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
            "llm": {k: llm.get(k) for k in ("configured", "problem", "model", "api_type", "supports_images", "api_key_count", "failover_configured", "fallback_model", "fallback_configured")},
            "github": {
                "token_configured": bool(os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_CLASSIC_TOKEN")),
                "classic_fallback_configured": bool(os.getenv("GITHUB_CLASSIC_TOKEN")),
            },
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

    @router.patch("/projects/{project_id}")
    async def rename_project(project_id: str, body: ProjectRename):
        project_or_404(project_id)
        try:
            return store.rename_project(project_id, body.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/projects/{project_id}")
    async def delete_project(project_id: str):
        project_or_404(project_id)
        if any(t.project_id == project_id and orchestrator.is_running(t.id) for t in store.tasks.values()):
            raise HTTPException(status_code=409, detail="Stop the running build before deleting this project.")
        workspace = store.delete_project(project_id)
        if workspace:
            root = Path(workspace).resolve()
            projects_root = (store.root / "projects").resolve()
            if root.parent == projects_root and root != projects_root:
                shutil.rmtree(root, ignore_errors=True)
        return {"deleted": project_id}

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

    @router.get("/projects/{project_id}/uploads")
    async def project_uploads(project_id: str):
        project_or_404(project_id)
        return await asyncio.to_thread(store.list_uploaded_files, project_id)

    @router.get("/projects/{project_id}/files/download")
    async def download_workspace_file(project_id: str, path: str):
        """Download a regular file from the selected project's workspace."""
        project = project_or_404(project_id)
        root = Path(project.workspace).resolve()
        target = (root / path).resolve()
        if target == root or root not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="Workspace file not found")
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type, filename=target.name)

    @router.post("/projects/{project_id}/uploads")
    async def upload_project_file(project_id: str, file: UploadFile = File(...)):
        project = project_or_404(project_id)
        raw_name = file.filename or ""
        if "\x00" in raw_name:
            raise HTTPException(status_code=400, detail="Filename contains an invalid character")
        original_name = Path(raw_name.replace("\\", "/")).name
        suffix = Path(original_name).suffix.lower()
        if not original_name or original_name in {".", ".."}:
            raise HTTPException(status_code=400, detail="A filename is required")
        if suffix not in UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix or 'no extension'}")
        file_id = uuid4().hex[:12]
        metadata = UploadedFile(
            id=file_id,
            project_id=project_id,
            filename=original_name,
            stored_path=str(Path(project.workspace) / "uploads" / f"{file_id}{suffix}"),
            content_type=file.content_type,
        )
        # Keep uploads inside the project workspace; do not extract archives automatically.
        target = Path(metadata.stored_path).resolve()
        upload_root = (Path(project.workspace) / "uploads").resolve()
        if upload_root not in target.parents:
            raise HTTPException(status_code=400, detail="Invalid upload path")
        target.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            with target.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit")
                    destination.write(chunk)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc
        finally:
            await file.close()
        metadata.size = total
        await asyncio.to_thread(store.add_uploaded_file, metadata)
        return metadata

    @router.get("/projects/{project_id}/uploads/{file_id}")
    async def download_project_file(project_id: str, file_id: str):
        project = project_or_404(project_id)
        uploaded = await asyncio.to_thread(store.get_uploaded_file, file_id)
        if uploaded is None or uploaded.project_id != project_id:
            raise HTTPException(status_code=404, detail="Uploaded file not found")
        path = Path(uploaded.stored_path).resolve()
        root = Path(project.workspace).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Uploaded file is missing")
        return FileResponse(path, media_type=uploaded.content_type or "application/octet-stream", filename=uploaded.filename)

    # -------------------------------------------------------------- chat

    @router.get("/projects/{project_id}/chat")
    async def project_chat(project_id: str):
        project_or_404(project_id)
        return await asyncio.to_thread(store.list_chat_messages, project_id, 200)

    @router.post("/projects/{project_id}/chat")
    async def send_chat_message(project_id: str, body: ChatMessageCreate):
        project = project_or_404(project_id)
        text = body.message.strip()
        async with chat_lock(project_id):
            user_message = await asyncio.to_thread(store.add_chat_message, project_id, "user", text)
            if re.fullmatch(r"(?:hi|hello|hey|good morning|good afternoon|good evening)[!. ]*", text, re.IGNORECASE):
                assistant_message = await asyncio.to_thread(
                    store.add_chat_message,
                    project_id,
                    "assistant",
                    "Hello! How can I help you today?",
                )
                return {"user": user_message, "assistant": assistant_message}
            if re.search(r"\b(are you|is it|what is|what's|how is)\b.*\b(building|working|progress|status|process)\b|\b(in progress|still building|still working)\b", text, re.IGNORECASE):
                project_tasks = [task for task in store.tasks.values() if task.project_id == project_id]
                latest = max(project_tasks, key=lambda task: task.created_at, default=None)
                if latest is not None and latest.status in {TaskStatus.RUNNING, TaskStatus.QUEUED}:
                    status_reply = f"Yes. The build is currently {latest.status.value} in this conversation. You can watch its Processing steps or tell me to stop it."
                elif latest is not None and latest.status == TaskStatus.SUCCEEDED:
                    status_reply = "The latest build completed successfully and was validated."
                elif latest is not None and latest.status == TaskStatus.FAILED:
                    status_reply = f"No. The latest build is not running because it failed: {latest.error or 'the task reported an unknown error.'}"
                elif latest is not None and latest.status == TaskStatus.CANCELLED:
                    status_reply = "No. The latest build was cancelled and is not running."
                else:
                    status_reply = "No build is currently running for this project."
                assistant_message = await asyncio.to_thread(store.add_chat_message, project_id, "assistant", status_reply)
                return {"user": user_message, "assistant": assistant_message}
            if re.search(r"\b(what can you do|what are you capable|your capabilities|can you access github|can you interact with github)\b", text, re.IGNORECASE):
                github_ready = bool(os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_CLASSIC_TOKEN"))
                capabilities = (
                    "I can work as your project assistant. I can:\n\n"
                    "- inspect and explain your project files and repository structure;\n"
                    "- create, edit, rename, and verify files;\n"
                    "- run commands, tests, builds, and local validation;\n"
                    "- use the shared browser to test web applications and inspect screenshots;\n"
                    "- connect to an existing GitHub repository through the GitHub panel;\n"
                    "- inspect Git status and diffs, create branches, commit changes, push changes, open pull requests, and publish projects;\n"
                    "- keep build progress in this conversation; and\n"
                    "- let you download generated project files from the Files panel.\n\n"
                    f"GitHub access is currently {'configured' if github_ready else 'not configured'} in this deployment. "
                    f"This project is {'connected to ' + project.repository if project.repository else 'not yet connected to a repository'}. "
                    "When you ask me to make a change, I can carry it out rather than merely describe the steps."
                )
                assistant_message = await asyncio.to_thread(store.add_chat_message, project_id, "assistant", capabilities)
                return {"user": user_message, "assistant": assistant_message}
            if problem := llm_problem():
                raise HTTPException(status_code=503, detail=problem)
            history = await asyncio.to_thread(store.list_chat_messages, project_id, 20)
            messages = [{"role": item.role, "content": item.content} for item in history]
            # Keep casual conversation fast on CPU-only local models. Load
            # repository context only for questions that clearly need it.
            code_intent = bool(re.search(
                r"\b(code|file|project|repository|repo|bug|error|function|class|api|docker|github|html|css|javascript|python|architecture|workspace|test)\b",
                text,
                re.IGNORECASE,
            ))
            uploads = await asyncio.to_thread(store.list_uploaded_files, project_id)
            image_uploads = referenced_images(text, uploads, body.attachment_ids)
            if image_uploads and messages:
                payloads = []
                for image in image_uploads:
                    raw, image_mime = await asyncio.to_thread(_image_payload, Path(image.stored_path), image.content_type)
                    payloads.append({"filename": image.filename, "data": base64.b64encode(raw).decode("ascii"), "mime": image_mime})
                messages[-1]["content"] += "\nReferenced image(s): " + ", ".join(item["filename"] for item in payloads)
                messages[-1]["base64_images"] = payloads
            workspace_context = await asyncio.to_thread(_workspace_context, Path(project.workspace)) if code_intent else "No repository context was loaded for this general conversational reply."
            system = {
                "role": "system",
                "content": (
                    "You are OpenManus Chat, a helpful conversational software-engineering assistant. "
                    "You are chatting with the owner of project {name}. The project workspace is {workspace}. "
                    "Answer naturally and concisely. You can discuss ideas, explain code, plan features, and "
                    "answer questions. This is the conversational side of one unified project assistant. "
                    "For mathematics, use clean readable plain text: write fractions as a/b, equations on "
                    "separate lines, and multiplication with × or *. Do not use LaTeX delimiters such as \\[ \\], "
                    "$$, or inline $...$, and do not emit commands such as \\frac, \\begin, or \\boxed. "
                    "The assistant also has project tools for implementation tasks: it can inspect and edit files, "
                    "run commands and tests, use the shared browser, and use the connected GitHub integration for "
                    "repository status, branches, commits, pushes, pull requests, and publishing. Do not claim an "
                    "action has already happened unless a build task or tool result provides evidence. If the user "
                    "asks for implementation, the interface will deliver it to the build workflow. You have "
                    "read-only repository context below; use it to answer architecture and code questions. "
                    "Uploaded files in this project: {uploads}. "
                    "Only images explicitly attached or referenced by the latest user message should be inspected. "
                    "Finish every solution completely; never stop after a heading, colon, or unfinished sentence.\n\n{context}"
                ).format(
                    name=project.name,
                    workspace=project.workspace,
                    uploads=", ".join(item.filename for item in uploads) or "none",
                    context=workspace_context,
                ),
            }
            try:
                chat_llm = LLM(config_name="vision") if image_uploads and "vision" in config.llm else LLM()
                # format_messages removes internal base64 fields while preparing
                # a request. Keep an untouched copy so a bounded continuation
                # can include the image again if the provider cuts off early.
                original_messages = copy.deepcopy(messages)
                answer = await chat_llm.ask(
                    copy.deepcopy(original_messages),
                    system_msgs=[system],
                    stream=False,
                    temperature=0.3,
                    max_tokens=768 if image_uploads else 512,
                )
                for _ in range(2) if image_uploads else range(0):
                    if not response_needs_continuation(answer, getattr(chat_llm, "last_finish_reason", None)):
                        break
                    continuation_messages = copy.deepcopy(original_messages)
                    continuation_messages.append({"role": "assistant", "content": answer})
                    continuation_messages.append({
                        "role": "user",
                        "content": "Continue the previous answer from exactly where it stopped. "
                        "Do not repeat the completed text; finish the current step and provide the final answer.",
                    })
                    answer += "\n" + await chat_llm.ask(
                        continuation_messages,
                        system_msgs=[system],
                        stream=False,
                        temperature=0.3,
                        max_tokens=768 if image_uploads else 512,
                    )
            except RateLimitError as exc:
                raise HTTPException(
                    status_code=429,
                    detail="The Gemini API rate limit or quota was reached. Please wait, check your Gemini API quota, or try again later.",
                ) from exc
            except Exception as exc:
                if _rate_limit_exception(exc):
                    raise HTTPException(
                        status_code=429,
                        detail="The Gemini API rate limit or quota was reached. Please wait, check your Gemini API quota, or try again later.",
                    ) from exc
                raise HTTPException(status_code=502, detail=f"Chat model request failed: {exc}") from exc
            assistant_message = await asyncio.to_thread(store.add_chat_message, project_id, "assistant", answer.strip())
            return {"user": user_message, "assistant": assistant_message}

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
        prompt = body.prompt.strip()
        task = store.create_task(body.project_id, prompt)
        await asyncio.to_thread(store.add_chat_message, body.project_id, "user", prompt)
        await asyncio.to_thread(
            store.add_chat_message,
            body.project_id,
            "assistant",
            "I’ll work on that in this conversation. I’ll inspect the project, make the requested changes, and verify the result before reporting back.",
        )
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

    @router.delete("/tasks/{task_id}")
    async def delete_task(task_id: str):
        task = task_or_404(task_id)
        if orchestrator.is_running(task_id) or task.status in {TaskStatus.RUNNING, TaskStatus.QUEUED}:
            raise HTTPException(status_code=409, detail="Stop the running build before deleting it.")
        if not store.delete_task(task_id):
            raise HTTPException(status_code=404, detail="Build activity not found")
        return {"deleted": task_id}

    @router.post("/tasks/{task_id}/resume")
    async def resume_task(task_id: str):
        task = task_or_404(task_id)
        if orchestrator.is_running(task_id) or task.status == TaskStatus.RUNNING:
            raise HTTPException(status_code=409, detail="Task is already running")
        await orchestrator.resume_task(task)
        return task

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, body: TaskMessage | None = None):
        task = task_or_404(task_id)
        if body and body.message.strip():
            await asyncio.to_thread(store.add_chat_message, task.project_id, "user", body.message.strip())
        if not await orchestrator.cancel(task_id):
            raise HTTPException(status_code=409, detail="Task is not running")
        if body and body.message.strip():
            await asyncio.to_thread(store.add_chat_message, task.project_id, "assistant", "I stopped the running build. Its conversation and activity remain available.")
        return task_or_404(task_id)

    @router.post("/tasks/{task_id}/messages")
    async def message_task(task_id: str, body: TaskMessage):
        """Answer the agent's question, or send guidance it will read at its next step."""
        task_or_404(task_id)
        try:
            delivery = await orchestrator.send_message(task_id, body.message)
            task = store.get_task(task_id)
            if task:
                await asyncio.to_thread(store.add_chat_message, task.project_id, "user", body.message.strip())
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
