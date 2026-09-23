from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from app.platform.models import Event, Project, Task, TaskStatus


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
                CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, created_at);
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(tasks)").fetchall()}
            if "browser_session_id" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN browser_session_id TEXT")
            if "coding_iteration" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN coding_iteration INTEGER NOT NULL DEFAULT 0")
            if "validation" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN validation TEXT")

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

    def create_project(self, name: str, repository: str | None = None, branch: str | None = None) -> Project:
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}", name):
            raise ValueError("Project name may contain letters, numbers, spaces, underscores, dots and hyphens")
        project = Project(name=name, repository=repository, branch=branch,
                          workspace=str(self.root / "projects" / name))
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

    def create_task(self, project_id: str, prompt: str) -> Task:
        if not self.get_project(project_id):
            raise KeyError(f"Unknown project: {project_id}")
        task = Task(project_id=project_id, prompt=prompt)
        with self._connect() as db:
            db.execute("INSERT INTO tasks(id,project_id,prompt,status,created_at,attempt,browser_session_id,coding_iteration,validation) VALUES(?,?,?,?,?,?,?,?,?)",
                       (task.id, task.project_id, task.prompt, task.status.value, task.created_at.isoformat(), task.attempt, task.browser_session_id, task.coding_iteration, json.dumps(task.validation) if task.validation else None))
        return task

    async def save_task(self, task: Task) -> None:
        async with self._lock:
            with self._connect() as db:
                db.execute("""UPDATE tasks SET status=?, started_at=?, finished_at=?, result=?, error=?, attempt=?, checkpoint=?, browser_session_id=?, coding_iteration=?, validation=? WHERE id=?""",
                           (task.status.value, task.started_at.isoformat() if task.started_at else None,
                            task.finished_at.isoformat() if task.finished_at else None, task.result, task.error,
                            task.attempt, task.checkpoint, task.browser_session_id, task.coding_iteration, json.dumps(task.validation) if task.validation else None, task.id))

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
                if event.type in {"task.succeeded", "task.failed"}:
                    return
            task = self.get_task(task_id)
            if task and task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
                return
            async with self._conditions[task_id]:
                try:
                    await asyncio.wait_for(self._conditions[task_id].wait(), timeout=5)
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
