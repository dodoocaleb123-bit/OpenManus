"""Local-first Phase C automation primitives."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import shlex
import signal
import sqlite3
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4


def detect_preview_command(root: Path, port: int = 3000) -> list[str] | None:
    """Choose a conservative local web-server command for a project workspace."""
    root = Path(root)
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if isinstance(scripts, dict):
            script = next((name for name in ("dev", "start", "serve") if name in scripts), None)
            if script:
                return ["npm", "run", script]
    if (root / "streamlit_app.py").is_file():
        return ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", str(port)]
    if (root / "index.html").is_file():
        return ["python", "-m", "http.server", str(port), "--bind", "0.0.0.0"]
    for filename in ("app.py", "main.py", "server.py"):
        if (root / filename).is_file():
            return ["python", filename]
    return None


def parse_preview_command(command: str | None, root: Path, port: int = 3000) -> list[str]:
    if command and command.strip():
        parts = shlex.split(command, posix=False)
        if not parts or any("\x00" in part for part in parts):
            raise ValueError("Preview command must not be empty or contain NUL bytes")
        return parts
    detected = detect_preview_command(root, port)
    if detected is None:
        raise ValueError("Could not detect a web app entry point; provide a preview command")
    return detected


async def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 30.0) -> None:
    """Wait until a preview server accepts HTTP connections on the given port."""
    deadline = asyncio.get_running_loop().time() + timeout

    def probe() -> bool:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/", timeout=1) as response:
                return response.status < 500
        except Exception:
            return False

    while asyncio.get_running_loop().time() < deadline:
        if await asyncio.to_thread(probe):
            return
        await asyncio.sleep(0.25)
    raise TimeoutError(f"Preview server did not respond on http://{host}:{port} within {timeout:g}s")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

class AutomationStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init()
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn
    def _init(self):
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS scheduled_automations (
              id TEXT PRIMARY KEY, project_id TEXT NOT NULL, prompt TEXT NOT NULL,
              interval_seconds INTEGER NOT NULL, next_run TEXT NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1, last_run TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS managed_processes (
              id TEXT PRIMARY KEY, project_id TEXT NOT NULL, command TEXT NOT NULL,
              cwd TEXT NOT NULL, pid INTEGER, status TEXT NOT NULL, created_at TEXT NOT NULL,
              stopped_at TEXT, last_output TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS connectors (
              id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
              endpoint TEXT NOT NULL, secret TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL
            );
            """)
    def create_schedule(self, project_id: str, prompt: str, interval_seconds: int) -> dict[str, Any]:
        interval = max(60, min(int(interval_seconds), 31_536_000))
        now = datetime.now(timezone.utc)
        item = {"id": str(uuid4()), "project_id": project_id, "prompt": prompt, "interval_seconds": interval, "next_run": now.timestamp() + interval, "enabled": True, "last_run": None, "created_at": utcnow()}
        with self._connect() as db:
            db.execute("INSERT INTO scheduled_automations VALUES (?,?,?,?,?,?,?,?)", (item["id"], item["project_id"], item["prompt"], interval, datetime.fromtimestamp(item["next_run"], timezone.utc).isoformat(), 1, None, item["created_at"]))
        return item
    def list_schedules(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM scheduled_automations" + (" WHERE project_id=?" if project_id else "") + " ORDER BY created_at DESC", (project_id,) if project_id else ()).fetchall()
        return [dict(row) for row in rows]
    def delete_schedule(self, schedule_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM scheduled_automations WHERE id=?", (schedule_id,)).rowcount > 0
    def due_schedules(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM scheduled_automations WHERE enabled=1 AND next_run<=?", (utcnow(),)).fetchall()
            for row in rows:
                db.execute("UPDATE scheduled_automations SET next_run=?, last_run=? WHERE id=?", (datetime.fromtimestamp(datetime.now(timezone.utc).timestamp()+int(row["interval_seconds"]), timezone.utc).isoformat(), utcnow(), row["id"]))
        return [dict(row) for row in rows]
    def add_connector(self, project_id: str, name: str, endpoint: str, secret: str) -> dict[str, Any]:
        item = {"id": str(uuid4()), "project_id": project_id, "name": name, "endpoint": endpoint, "secret": secret, "enabled": True, "created_at": utcnow()}
        with self._connect() as db:
            db.execute("INSERT INTO connectors VALUES (?,?,?,?,?,?,?)", tuple(item.values()))
        item["secret"] = "configured"
        return item
    def list_connectors(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT id,project_id,name,endpoint,enabled,created_at FROM connectors WHERE project_id=? ORDER BY created_at DESC", (project_id,)).fetchall()
        return [dict(row) for row in rows]
    def get_connector(self, connector_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM connectors WHERE id=? AND enabled=1", (connector_id,)).fetchone()
        return dict(row) if row else None
    def record_process(self, process_id: str, project_id: str, command: list[str], cwd: str, pid: int, status: str = "running") -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO managed_processes (id,project_id,command,cwd,pid,status,created_at,last_output) VALUES (?,?,?,?,?,?,?,?)", (process_id, project_id, json.dumps(command), cwd, pid, status, utcnow(), ""))
    def list_processes(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT id,project_id,command,cwd,pid,status,created_at,stopped_at,last_output FROM managed_processes WHERE project_id=? ORDER BY created_at DESC", (project_id,)).fetchall()
        return [dict(row) for row in rows]

    def update_process(self, process_id: str, *, status: str | None = None, output: str | None = None) -> None:
        fields, values = [], []
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if output is not None:
            fields.append("last_output=?")
            values.append(output[-10000:])
        if not fields:
            return
        values.append(process_id)
        with self._connect() as db:
            db.execute(f"UPDATE managed_processes SET {', '.join(fields)} WHERE id=?", values)

class ProcessManager:
    """Supervise commands without a shell, confined to a project workspace."""
    def __init__(self, records: AutomationStore | None = None):
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self._drainers: dict[str, asyncio.Task] = {}
        self.records = records

    async def _drain_output(self, process_id: str, proc: asyncio.subprocess.Process) -> None:
        output = ""
        if proc.stdout is not None:
            async for raw in proc.stdout:
                output = (output + raw.decode(errors="replace"))[-10000:]
                if self.records:
                    self.records.update_process(process_id, output=output)
        code = await proc.wait()
        if self.records:
            self.records.update_process(process_id, status="exited" if code == 0 else "failed")
    async def start(self, process_id: str, command: list[str], cwd: Path, project_id: str | None = None) -> dict[str, Any]:
        if not command or any("\x00" in part for part in command):
            raise ValueError("A non-empty safe command list is required")
        proc = await asyncio.create_subprocess_exec(*command, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        self.processes[process_id] = proc
        self._drainers[process_id] = asyncio.create_task(self._drain_output(process_id, proc))
        if self.records and project_id:
            self.records.record_process(process_id, project_id, command, str(cwd), proc.pid)
        return {"id": process_id, "pid": proc.pid, "status": "running", "command": command, "cwd": str(cwd)}
    async def stop(self, process_id: str) -> bool:
        proc = self.processes.get(process_id)
        if not proc or proc.returncode is not None:
            return False
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        if self.records:
            self.records.update_process(process_id, status="stopping")
        return True
    async def shutdown(self) -> None:
        for process_id in list(self.processes):
            await self.stop(process_id)
    async def status(self, process_id: str) -> dict[str, Any]:
        proc = self.processes.get(process_id)
        if not proc:
            return {"id": process_id, "status": "unknown"}
        return {"id": process_id, "pid": proc.pid, "status": "running" if proc.returncode is None else "exited", "returncode": proc.returncode}

async def deliver_connector(endpoint: str, secret: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, default=str).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    def send():
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json", "X-OpenManus-Signature": f"sha256={signature}"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as response:
            return {"status": response.status}
    return await asyncio.to_thread(send)

def verify_webhook(raw_body: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


class AutomationRunner:
    def __init__(self, records: AutomationStore, launch: Callable[[str, str], Awaitable[Any]], interval: float = 15.0):
        self.records = records
        self.launch = launch
        self.interval = max(5.0, interval)
        self._task: asyncio.Task | None = None
        self._stopped = False

    async def _run(self):
        while not self._stopped:
            for schedule in self.records.due_schedules():
                try:
                    await self.launch(schedule["project_id"], schedule["prompt"])
                except Exception:
                    # A failed scheduled launch must not stop future schedules.
                    pass
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self._task is None:
            self._stopped = False
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
