from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from app.platform.models import ChatMessage, TERMINAL_EVENT_TYPES, TERMINAL_STATUSES, Event, Project, Task, TaskStatus, UploadedFile


class PlatformStore:
    """Durable SQLite-backed project/task/event store.

    SQLite is intentionally used for the first durable implementation: it keeps the
    platform single-node and easy to run locally while giving us restart-safe state.
    The API boundary can later be moved to PostgreSQL without changing callers.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "platform.db"
        self._lock = asyncio.Lock()
        self._conditions: dict[str, asyncio.Condition] = defaultdict(asyncio.Condition)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    repository TEXT,
                    branch TEXT,
                    git_ready INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    result TEXT,
                    error TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    checkpoint TEXT,
                    browser_session_id TEXT,
                    coding_iteration INTEGER NOT NULL DEFAULT 0,
                    validation TEXT,
                    plan TEXT,
                    evidence TEXT,
                    artifacts TEXT,
                    recovery TEXT,
                    idempotency_key TEXT,
                    last_heartbeat TEXT,
                    execution_mode TEXT NOT NULL DEFAULT 'implement',
                    first_token_ms INTEGER,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    response_time_ms INTEGER,
                    time_to_first_token_ms INTEGER,
                    task_id TEXT,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    content_type TEXT,
                    size INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_chat_project ON chat_messages(project_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_uploads_project ON uploaded_files(project_id, created_at);
                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, name TEXT NOT NULL,
                    data TEXT NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON task_checkpoints(task_id, created_at);
                CREATE TABLE IF NOT EXISTS project_memory (
                    project_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE TABLE IF NOT EXISTS task_feedback (
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id),
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(tasks)").fetchall()}
            if "browser_session_id" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN browser_session_id TEXT")
            if "coding_iteration" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN coding_iteration INTEGER NOT NULL DEFAULT 0")
            if "validation" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN validation TEXT")
            if "plan" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN plan TEXT")
            for name, definition in (("evidence", "TEXT"), ("artifacts", "TEXT"), ("recovery", "TEXT"), ("idempotency_key", "TEXT"), ("last_heartbeat", "TEXT")):
                if name not in columns:
                    db.execute(f"ALTER TABLE tasks ADD COLUMN {name} {definition}")
            if "execution_mode" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN execution_mode TEXT NOT NULL DEFAULT 'implement'")
            if "first_token_ms" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN first_token_ms INTEGER")
            chat_columns = {row[1] for row in db.execute("PRAGMA table_info(chat_messages)").fetchall()}
            if "response_time_ms" not in chat_columns:
                db.execute("ALTER TABLE chat_messages ADD COLUMN response_time_ms INTEGER")
            if "time_to_first_token_ms" not in chat_columns:
                db.execute("ALTER TABLE chat_messages ADD COLUMN time_to_first_token_ms INTEGER")
            if "task_id" not in chat_columns:
                db.execute("ALTER TABLE chat_messages ADD COLUMN task_id TEXT")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_idempotency_key ON tasks(project_id,idempotency_key) WHERE idempotency_key IS NOT NULL")
            db.execute("CREATE INDEX IF NOT EXISTS idx_chat_task ON chat_messages(task_id,created_at) WHERE task_id IS NOT NULL")

    @staticmethod
    def _dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    @staticmethod
    def _project(row: sqlite3.Row) -> Project:
        return Project(
            id=row["id"], name=row["name"], workspace=row["workspace"],
            repository=row["repository"], branch=row["branch"],
            git_ready=bool(row["git_ready"]), created_at=PlatformStore._dt(row["created_at"]),
        )

    @staticmethod
    def _task(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"], project_id=row["project_id"], prompt=row["prompt"],
            status=TaskStatus(row["status"]), created_at=PlatformStore._dt(row["created_at"]),
            started_at=PlatformStore._dt(row["started_at"]), finished_at=PlatformStore._dt(row["finished_at"]),
            result=row["result"], error=row["error"], attempt=row["attempt"], checkpoint=row["checkpoint"],
            browser_session_id=row["browser_session_id"] if "browser_session_id" in row.keys() else None,
            coding_iteration=row["coding_iteration"] if "coding_iteration" in row.keys() else 0,
            validation=json.loads(row["validation"]) if row["validation"] else None,
            plan=json.loads(row["plan"]) if "plan" in row.keys() and row["plan"] else None,
            evidence=json.loads(row["evidence"]) if "evidence" in row.keys() and row["evidence"] else {},
            artifacts=json.loads(row["artifacts"]) if "artifacts" in row.keys() and row["artifacts"] else [],
            recovery=json.loads(row["recovery"]) if "recovery" in row.keys() and row["recovery"] else {},
            idempotency_key=row["idempotency_key"] if "idempotency_key" in row.keys() else None,
            last_heartbeat=PlatformStore._dt(row["last_heartbeat"]) if "last_heartbeat" in row.keys() else None,
            execution_mode=row["execution_mode"] if "execution_mode" in row.keys() else "implement",
            first_token_ms=row["first_token_ms"] if "first_token_ms" in row.keys() else None,
        )

    @staticmethod
    def _event(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"], task_id=row["task_id"], type=row["type"], message=row["message"],
            data=json.loads(row["data"]), created_at=PlatformStore._dt(row["created_at"]),
        )

    @property
    def projects(self) -> dict[str, Project]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM projects ORDER BY created_at").fetchall()
        return {r["id"]: self._project(r) for r in rows}

    @property
    def tasks(self) -> dict[str, Task]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
        return {r["id"]: self._task(r) for r in rows}

    def get_project(self, project_id: str) -> Project | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._project(row) if row else None

    def get_task(self, task_id: str) -> Task | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def get_task_by_idempotency_key(self, project_id: str, key: str) -> Task | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE project_id=? AND idempotency_key=?", (project_id, key)).fetchone()
        return self._task(row) if row else None

    def create_project(self, name: str, repository: str | None = None, branch: str | None = None) -> Project:
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}", name):
            raise ValueError("Project name may contain letters, numbers, spaces, underscores, dots and hyphens")
        # Workspaces are keyed by the unique project id, so duplicate or
        # case-variant project names can never share (or clobber) a folder.
        project = Project(name=name, repository=repository, branch=branch, workspace="")
        project.workspace = str(self.root / "projects" / project.id)
        Path(project.workspace).mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "INSERT INTO projects(id,name,workspace,repository,branch,git_ready,created_at) VALUES(?,?,?,?,?,?,?)",
                (project.id, project.name, project.workspace, project.repository, project.branch, int(project.git_ready), project.created_at.isoformat()),
            )
        return project

    def update_project(self, project: Project) -> Project:
        with self._connect() as db:
            db.execute("UPDATE projects SET name=?, workspace=?, repository=?, branch=?, git_ready=? WHERE id=?",
                       (project.name, project.workspace, project.repository, project.branch, int(project.git_ready), project.id))
        return project

    def rename_project(self, project_id: str, name: str) -> Project:
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}", name):
            raise ValueError("Project name may contain letters, numbers, spaces, underscores, dots and hyphens")
        project = self.get_project(project_id)
        if not project:
            raise KeyError(project_id)
        project.name = name
        return self.update_project(project)

    def delete_task(self, task_id: str) -> bool:
        with self._connect() as db:
            db.execute("DELETE FROM events WHERE task_id=?", (task_id,))
            db.execute("DELETE FROM task_checkpoints WHERE task_id=?", (task_id,))
            db.execute("DELETE FROM task_feedback WHERE task_id=?", (task_id,))
            result = db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        return result.rowcount > 0

    def delete_project(self, project_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT workspace FROM projects WHERE id=?", (project_id,)).fetchone()
            if not row:
                return None
            db.execute("DELETE FROM events WHERE task_id IN (SELECT id FROM tasks WHERE project_id=?)", (project_id,))
            db.execute("DELETE FROM task_checkpoints WHERE task_id IN (SELECT id FROM tasks WHERE project_id=?)", (project_id,))
            db.execute("DELETE FROM task_feedback WHERE project_id=?", (project_id,))
            db.execute("DELETE FROM tasks WHERE project_id=?", (project_id,))
            db.execute("DELETE FROM uploaded_files WHERE project_id=?", (project_id,))
            db.execute("DELETE FROM chat_messages WHERE project_id=?", (project_id,))
            db.execute("DELETE FROM project_memory WHERE project_id=?", (project_id,))
            db.execute("DELETE FROM projects WHERE id=?", (project_id,))
        return row["workspace"]

    def create_task(self, project_id: str, prompt: str, execution_mode: str = "implement", idempotency_key: str | None = None) -> Task:
        if not self.get_project(project_id):
            raise KeyError(f"Unknown project: {project_id}")
        if execution_mode not in {"implement", "inspect"}:
            raise ValueError("Task execution mode must be implement or inspect")
        task = Task(project_id=project_id, prompt=prompt, execution_mode=execution_mode, idempotency_key=idempotency_key)
        with self._connect() as db:
            db.execute("INSERT INTO tasks(id,project_id,prompt,status,created_at,attempt,browser_session_id,coding_iteration,validation,plan,evidence,artifacts,recovery,idempotency_key,last_heartbeat,execution_mode,first_token_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (task.id, task.project_id, task.prompt, task.status.value, task.created_at.isoformat(), task.attempt, task.browser_session_id, task.coding_iteration, json.dumps(task.validation) if task.validation else None, json.dumps(task.plan) if task.plan else None, json.dumps(task.evidence), json.dumps(task.artifacts), json.dumps(task.recovery), task.idempotency_key, task.last_heartbeat.isoformat() if task.last_heartbeat else None, task.execution_mode, task.first_token_ms))
        return task

    @staticmethod
    def _chat_message(row: sqlite3.Row) -> ChatMessage:
        return ChatMessage(
            id=row["id"], project_id=row["project_id"], role=row["role"],
            content=row["content"], created_at=PlatformStore._dt(row["created_at"]),
            response_time_ms=row["response_time_ms"] if "response_time_ms" in row.keys() else None,
            time_to_first_token_ms=row["time_to_first_token_ms"] if "time_to_first_token_ms" in row.keys() else None,
            task_id=row["task_id"] if "task_id" in row.keys() else None,
        )

    def add_chat_message(self, project_id: str, role: str, content: str, response_time_ms: int | None = None, time_to_first_token_ms: int | None = None, task_id: str | None = None) -> ChatMessage:
        if not self.get_project(project_id):
            raise KeyError(f"Unknown project: {project_id}")
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Chat message role must be user, assistant, or system")
        response_time_ms = max(0, int(response_time_ms)) if response_time_ms is not None else None
        time_to_first_token_ms = max(0, int(time_to_first_token_ms)) if time_to_first_token_ms is not None else None
        if task_id is not None:
            task = self.get_task(task_id)
            if not task or task.project_id != project_id:
                raise KeyError(f"Unknown task for project: {task_id}")
        message = ChatMessage(project_id=project_id, role=role, content=content, task_id=task_id, response_time_ms=response_time_ms, time_to_first_token_ms=time_to_first_token_ms)
        with self._connect() as db:
            db.execute(
                "INSERT INTO chat_messages(id,project_id,role,content,created_at,response_time_ms,time_to_first_token_ms,task_id) VALUES(?,?,?,?,?,?,?,?)",
                (message.id, message.project_id, message.role, message.content, message.created_at.isoformat(), message.response_time_ms, message.time_to_first_token_ms, message.task_id),
            )
        return message

    def get_project_memory(self, project_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT content,updated_at FROM project_memory WHERE project_id=?", (project_id,)).fetchone()
        return {"project_id": project_id, "content": row["content"], "updated_at": row["updated_at"]} if row else {"project_id": project_id, "content": "", "updated_at": None}

    def set_project_memory(self, project_id: str, content: str) -> dict:
        if not self.get_project(project_id):
            raise KeyError(f"Unknown project: {project_id}")
        content = str(content).strip()
        if len(content) > 6000:
            raise ValueError("Project memory must be 6,000 characters or fewer")
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            if content:
                db.execute(
                    "INSERT INTO project_memory(project_id,content,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(project_id) DO UPDATE SET content=excluded.content,updated_at=excluded.updated_at",
                    (project_id, content, updated_at),
                )
            else:
                db.execute("DELETE FROM project_memory WHERE project_id=?", (project_id,))
        return {"project_id": project_id, "content": content, "updated_at": updated_at if content else None}

    def set_task_feedback(self, task_id: str, rating: int) -> dict:
        if rating not in {-1, 1}:
            raise ValueError("Feedback rating must be -1 or 1")
        task = self.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                "INSERT INTO task_feedback(task_id,project_id,rating,created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET rating=excluded.rating,created_at=excluded.created_at",
                (task_id, task.project_id, rating, created_at),
            )
        return {"task_id": task_id, "rating": rating, "created_at": created_at}

    def feedback_metrics(self) -> dict:
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) AS helpful, "
                "SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) AS not_helpful FROM task_feedback"
            ).fetchone()
        total = int(row["total"] or 0)
        helpful = int(row["helpful"] or 0)
        return {
            "rated_tasks": total,
            "helpful": helpful,
            "not_helpful": int(row["not_helpful"] or 0),
            "helpful_rate": round(helpful / total, 4) if total else None,
        }

    def latency_metrics(self) -> dict:
        with self._connect() as db:
            rows = db.execute(
                "SELECT response_time_ms,time_to_first_token_ms FROM chat_messages "
                "WHERE role='assistant' AND response_time_ms IS NOT NULL "
                "ORDER BY created_at DESC,rowid DESC LIMIT 500"
            ).fetchall()
        totals = [int(row["response_time_ms"]) for row in rows if row["response_time_ms"] is not None]
        first = [int(row["time_to_first_token_ms"]) for row in rows if row["time_to_first_token_ms"] is not None]

        def summarize(values: list[int]) -> dict:
            if not values:
                return {"samples": 0, "average_ms": None, "p95_ms": None}
            ordered = sorted(values)
            p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
            return {
                "samples": len(values),
                "average_ms": round(sum(values) / len(values)),
                "p95_ms": ordered[p95_index],
            }

        return {"response_time": summarize(totals), "time_to_first_token": summarize(first)}

    def list_chat_messages(self, project_id: str, limit: int = 100) -> list[ChatMessage]:
        limit = max(1, min(int(limit), 500))
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM chat_messages WHERE project_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        return [self._chat_message(row) for row in reversed(rows)]

    @staticmethod
    def _uploaded_file(row: sqlite3.Row) -> UploadedFile:
        return UploadedFile(
            id=row["id"], project_id=row["project_id"], filename=row["filename"],
            stored_path=row["stored_path"], content_type=row["content_type"],
            size=row["size"], created_at=PlatformStore._dt(row["created_at"]),
        )

    def add_uploaded_file(self, file: UploadedFile) -> UploadedFile:
        with self._connect() as db:
            db.execute(
                "INSERT INTO uploaded_files(id,project_id,filename,stored_path,content_type,size,created_at) VALUES(?,?,?,?,?,?,?)",
                (file.id, file.project_id, file.filename, file.stored_path, file.content_type, file.size, file.created_at.isoformat()),
            )
        return file

    def list_uploaded_files(self, project_id: str) -> list[UploadedFile]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM uploaded_files WHERE project_id=? ORDER BY created_at, rowid", (project_id,)).fetchall()
        return [self._uploaded_file(row) for row in rows]

    def get_uploaded_file(self, file_id: str) -> UploadedFile | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM uploaded_files WHERE id=?", (file_id,)).fetchone()
        return self._uploaded_file(row) if row else None

    async def save_task(self, task: Task) -> None:
        async with self._lock:
            with self._connect() as db:
                db.execute("""UPDATE tasks SET status=?, started_at=?, finished_at=?, result=?, error=?, attempt=?, checkpoint=?, browser_session_id=?, coding_iteration=?, validation=?, plan=?, evidence=?, artifacts=?, recovery=?, idempotency_key=?, last_heartbeat=?, execution_mode=?, first_token_ms=? WHERE id=?""",
                           (task.status.value, task.started_at.isoformat() if task.started_at else None,
                            task.finished_at.isoformat() if task.finished_at else None, task.result, task.error,
                            task.attempt, task.checkpoint, task.browser_session_id, task.coding_iteration, json.dumps(task.validation) if task.validation else None, json.dumps(task.plan) if task.plan else None, json.dumps(task.evidence), json.dumps(task.artifacts), json.dumps(task.recovery), task.idempotency_key, task.last_heartbeat.isoformat() if task.last_heartbeat else None, task.execution_mode, task.first_token_ms, task.id))

    async def save_checkpoint(self, task_id: str, name: str, data: dict) -> None:
        from uuid import uuid4
        async with self._lock:
            with self._connect() as db:
                db.execute("INSERT INTO task_checkpoints(id,task_id,name,data,created_at) VALUES(?,?,?,?,?)", (uuid4().hex[:12], task_id, name, json.dumps(data), datetime.now(timezone.utc).isoformat()))

    def list_checkpoints(self, task_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT name,data,created_at FROM task_checkpoints WHERE task_id=? ORDER BY created_at", (task_id,)).fetchall()
        return [{"name": row["name"], "data": json.loads(row["data"]), "created_at": row["created_at"]} for row in rows]

    async def emit(self, event: Event) -> None:
        async with self._lock:
            with self._connect() as db:
                db.execute("INSERT INTO events(id,task_id,type,message,data,created_at) VALUES(?,?,?,?,?,?)",
                           (event.id, event.task_id, event.type, event.message, json.dumps(event.data), event.created_at.isoformat()))
        async with self._conditions[event.task_id]:
            self._conditions[event.task_id].notify_all()

    async def stream_events(self, task_id: str) -> AsyncIterator[Event]:
        index = 0
        while True:
            with self._connect() as db:
                rows = db.execute("SELECT * FROM events WHERE task_id=? ORDER BY created_at, rowid", (task_id,)).fetchall()
            while index < len(rows):
                event = self._event(rows[index]); index += 1
                yield event
                if event.type in TERMINAL_EVENT_TYPES:
                    return
            task = self.get_task(task_id)
            if task and task.status in TERMINAL_STATUSES:
                return
            async with self._conditions[task_id]:
                try:
                    await asyncio.wait_for(self._conditions[task_id].wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass

    async def wait_for_event(self, task_id: str, timeout: float) -> None:
        """Block until a new event is emitted for ``task_id`` or ``timeout`` elapses."""
        condition = self._conditions[task_id]
        async with condition:
            try:
                await asyncio.wait_for(condition.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    async def recover_interrupted(self) -> list[Task]:
        """Convert tasks left RUNNING by a process crash back to QUEUED."""
        recovered: list[Task] = []
        async with self._lock:
            with self._connect() as db:
                rows = db.execute("SELECT * FROM tasks WHERE status=?", (TaskStatus.RUNNING.value,)).fetchall()
                for row in rows:
                    db.execute("UPDATE tasks SET status=?, error=? WHERE id=?",
                               (TaskStatus.QUEUED.value, "Process interrupted; task re-queued for recovery.", row["id"]))
        for row in rows:
            task = self.get_task(row["id"])
            if task:
                recovered.append(task)
        return recovered

    async def list_events(self, task_id: str) -> list[Event]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM events WHERE task_id=? ORDER BY created_at, rowid", (task_id,)).fetchall()
        return [self._event(r) for r in rows]
