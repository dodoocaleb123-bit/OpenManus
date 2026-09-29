"""First-class, bounded artifact discovery for platform tasks."""
from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_IGNORED = {".git", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv"}
_MAX_FILES = 200
_MAX_BYTES = 25 * 1024 * 1024


def _kind(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix in {".md", ".txt", ".rst"}:
        return "document"
    if suffix in {".csv", ".tsv", ".xlsx", ".xls", ".ods"}:
        return "spreadsheet"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return "image"
    if suffix in {".pdf", ".docx", ".pptx"}:
        return "document"
    if suffix in {".html", ".css", ".js", ".ts", ".py", ".json", ".yaml", ".yml"}:
        return "source"
    return "file"


def artifact_record(path: Path, root: Path, *, task_id: str | None = None, verified: bool = True) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()[:_MAX_BYTES]).hexdigest() if stat.st_size <= _MAX_BYTES else None
    return {
        "path": relative,
        "filename": path.name,
        "kind": _kind(path),
        "mime": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        "size": stat.st_size,
        "sha256": digest,
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verified": verified,
    }


def discover_artifacts(root: str | Path, *, baseline: dict[str, tuple[int, int]] | None = None, task_id: str | None = None) -> list[dict[str, Any]]:
    """Return bounded metadata for files created or changed since a baseline."""
    workspace = Path(root).resolve()
    baseline = baseline or {}
    result: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or any(part in _IGNORED for part in path.parts):
            continue
        try:
            stat = path.stat()
            key = path.relative_to(workspace).as_posix()
            current = (stat.st_size, stat.st_mtime_ns)
            if key in baseline and baseline[key] == current:
                continue
            result.append(artifact_record(path, workspace, task_id=task_id))
            if len(result) >= _MAX_FILES:
                break
        except (OSError, ValueError):
            continue
    return result


def workspace_baseline(root: str | Path) -> dict[str, tuple[int, int]]:
    workspace = Path(root).resolve()
    baseline: dict[str, tuple[int, int]] = {}
    for path in workspace.rglob("*"):
        if path.is_file() and not any(part in _IGNORED for part in path.parts):
            try:
                stat = path.stat()
                baseline[path.relative_to(workspace).as_posix()] = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                pass
    return baseline


def artifact_summary(artifacts: Iterable[dict[str, Any] | str]) -> dict[str, Any]:
    items = [item if isinstance(item, dict) else {"path": str(item)} for item in artifacts]
    return {"count": len(items), "kinds": sorted({str(item.get("kind", "file")) for item in items}), "items": items[:_MAX_FILES]}
