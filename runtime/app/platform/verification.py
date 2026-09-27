"""Centralized, evidence-producing verification and recovery helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class VerificationEngine:
    """Produce durable, machine-readable checks before a task claims success."""

    @staticmethod
    def workspace_check(workspace: str | Path) -> dict[str, Any]:
        root = Path(workspace).resolve()
        exists = root.is_dir()
        files = []
        if exists:
            for path in root.rglob("*"):
                if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                    files.append(str(path.relative_to(root)))
        return {
            "check": "workspace",
            "passed": exists,
            "workspace": str(root),
            "file_count": len(files),
            "files_sample": files[:100],
        }

    @staticmethod
    def validation_check(validation: dict[str, Any] | None) -> dict[str, Any]:
        if not validation:
            return {"check": "validation", "passed": False, "reason": "No validation result was recorded."}
        results = validation.get("results") or []
        passed = bool(validation.get("passed")) and all(item.get("ok", False) for item in results if isinstance(item, dict))
        return {"check": "validation", "passed": passed, "result_count": len(results), "results": results[-20:]}

    @staticmethod
    def artifact_check(workspace: str | Path, artifacts: list[dict[str, Any] | str] | None) -> dict[str, Any]:
        root = Path(workspace).resolve()
        checked = []
        for item in artifacts or []:
            raw = item if isinstance(item, str) else item.get("path") or item.get("url")
            if not raw:
                checked.append({"path": raw, "exists": False})
                continue
            if str(raw).startswith(("http://", "https://")):
                checked.append({"path": str(raw), "exists": True, "external": True})
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            safe = path == root or root in path.parents
            checked.append({"path": str(path.relative_to(root)) if safe and path != root else str(path), "exists": safe and path.is_file(), "safe": safe})
        return {"check": "artifacts", "passed": all(item["exists"] for item in checked), "items": checked}

    @classmethod
    def task_completion(cls, workspace: str | Path, validation: dict[str, Any] | None, artifacts: list[dict[str, Any] | str] | None = None, *, skipped: bool = False) -> dict[str, Any]:
        checks = [cls.workspace_check(workspace)]
        if skipped:
            checks.append({"check": "validation", "passed": True, "skipped": True})
        else:
            checks.append(cls.validation_check(validation))
        if artifacts:
            checks.append(cls.artifact_check(workspace, artifacts))
        return {"passed": all(check.get("passed", False) for check in checks), "checks": checks}


class FailureClassifier:
    """Map raw failures to safe recovery categories and bounded policies."""

    @staticmethod
    def classify(error: BaseException | str) -> str:
        text = str(error).casefold()
        name = error.__class__.__name__.casefold() if isinstance(error, BaseException) else ""
        if any(token in text or token in name for token in ("rate limit", "quota", "429", "authentication", "apierror", "apiconnection")):
            return "provider_failure"
        if any(token in text or token in name for token in ("browser", "playwright", "targetclosed", "session")):
            return "browser_failure"
        if any(token in text or token in name for token in ("timeout", "timed out", "apitimeout")):
            return "temporary_failure"
        if any(token in text for token in ("validation", "test failed", "exit code", "build failed")):
            return "validation_failure"
        if any(token in text for token in ("tool", "command", "permission denied")):
            return "tool_failure"
        if any(token in text for token in ("confirm", "confirmation", "destructive")):
            return "confirmation_required"
        return "unknown_failure"

    @staticmethod
    def policy(category: str, attempt: int, max_attempts: int = 3) -> dict[str, Any]:
        retryable = category in {"provider_failure", "browser_failure", "temporary_failure", "tool_failure"}
        return {"category": category, "retryable": retryable and attempt < max_attempts, "attempt": attempt, "max_attempts": max_attempts}


def stable_operation_id(task_id: str, action: str, payload: Any = None) -> str:
    """Create an idempotency token for externally visible operations."""
    encoded = json.dumps(payload, sort_keys=True, default=str) if payload is not None else ""
    return hashlib.sha256(f"{task_id}:{action}:{encoded}".encode()).hexdigest()[:24]
