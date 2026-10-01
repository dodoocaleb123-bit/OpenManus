"""Phase D platform primitives.

This module deliberately keeps Phase D declarative and local-first:
- capability profiles describe what a model can do; they do not load models;
- plugins are signed/declared manifests, never arbitrary code execution;
- project policies and memberships are enforced at the API boundary;
- trajectory evaluation is deterministic and auditable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any


BUILTIN_CAPABILITIES: dict[str, dict[str, Any]] = {
    "qwen2.5-coder:7b": {
        "provider": "ollama", "vision": False, "tools": True, "coding": True,
        "reasoning": "moderate", "planning": False, "research": "limited",
    },
    "gemma3:4b": {
        "provider": "ollama", "vision": True, "tools": False, "coding": "limited",
        "reasoning": "moderate", "planning": False, "research": False,
    },
    "deepseek-r1:7b": {
        "provider": "ollama", "vision": False, "tools": False, "coding": False,
        "reasoning": "strong", "planning": True, "research": "limited",
    },
    "qwen2.5:3b": {
        "provider": "ollama", "vision": False, "tools": False, "coding": False,
        "reasoning": "moderate", "planning": False, "research": "strong",
    },
    "llama3.2:3b": {
        "provider": "ollama", "vision": False, "tools": False, "coding": False,
        "reasoning": "moderate", "planning": False, "research": False,
        "creativity": "strong", "design": "strong",
    },
}

CAPABILITY_OWNERS: dict[str, str] = {
    "reasoning": "deepseek",
    "software_engineering": "qwen_coder",
    "github": "qwen_coder",
    "image_analysis": "gemma3",
    "research": "qwen2.5_3b",
    "creativity": "llama3.2_3b",
    "beautiful_design": "llama3.2_3b",
    "user_action": "user",
    "protected_operation": "platform",
    "unassigned": "deepseek",
}

DEFAULT_POLICY: dict[str, Any] = {
    "enabled": True,
    "max_steps": 40,
    "max_runtime_seconds": 2700,
    "max_tool_calls": 120,
    "max_repair_cycles": 2,
    "max_files_changed": 100,
    "allowed_tools": [],
    "blocked_tools": [],
    "allowed_domains": [],
    "require_confirmation_for": ["destructive", "publish", "credential_change"],
    "allow_external_network": True,
}

ROLE_ORDER = {"viewer": 0, "editor": 1, "owner": 2}
PLUGIN_NAME = re.compile(r"^[a-z][a-z0-9._-]{1,63}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def capability_registry() -> list[dict[str, Any]]:
    """Return built-ins plus configured model profiles without exposing keys."""
    profiles = {name: {**value, "model": name, "source": "builtin"} for name, value in BUILTIN_CAPABILITIES.items()}
    for env_name, role in (("LOCAL_LLM_MODEL", "local"), ("VISION_LLM_MODEL", "vision"), ("HEAVY_CODING_LLM_MODEL", "heavy_coding"), ("REASONING_LLM_MODEL", "reasoning"), ("RESEARCH_LLM_MODEL", "research"), ("CREATIVITY_LLM_MODEL", "creativity")):
        model = os.environ.get(env_name, "").strip()
        if model and model not in profiles:
            profiles[model] = {
                "model": model, "provider": "configured", "source": "environment", "role": role,
                "vision": role == "vision", "tools": role in {"local", "heavy_coding"},
                "coding": role in {"local", "heavy_coding"}, "reasoning": "strong" if role == "reasoning" else "moderate",
                "planning": role == "reasoning", "research": role in {"local", "vision", "research"},
                "creativity": role == "creativity", "design": role == "creativity",
            }
        elif model:
            profiles[model]["role"] = role
    return list(profiles.values())


def normalize_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(DEFAULT_POLICY)
    if policy:
        value.update(policy)
    for key in ("max_steps", "max_runtime_seconds", "max_tool_calls", "max_repair_cycles", "max_files_changed"):
        try:
            value[key] = max(0, min(int(value[key]), 100000))
        except (TypeError, ValueError):
            value[key] = DEFAULT_POLICY[key]
    for key in ("allowed_tools", "blocked_tools", "allowed_domains", "require_confirmation_for"):
        raw = value.get(key, [])
        value[key] = [str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, list) else []
    value["enabled"] = bool(value.get("enabled", True))
    value["allow_external_network"] = bool(value.get("allow_external_network", True))
    return value


def validate_plugin_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a declarative plugin manifest; executable entrypoints are rejected."""
    name = str(manifest.get("name", "")).strip().lower()
    if not PLUGIN_NAME.fullmatch(name):
        raise ValueError("Plugin name must be 2-64 lowercase letters, numbers, dots, hyphens, or underscores")
    version = str(manifest.get("version", "")).strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Plugin version must use semantic version format, for example 1.0.0")
    capabilities = manifest.get("capabilities", [])
    permissions = manifest.get("permissions", [])
    if not isinstance(capabilities, list) or not isinstance(permissions, list):
        raise ValueError("Plugin capabilities and permissions must be arrays")
    forbidden = {"execute_code", "shell_unrestricted", "read_secrets", "install_package"}
    requested = {str(item).strip() for item in permissions}
    if requested & forbidden:
        raise ValueError("Plugin requests a forbidden permission: " + sorted(requested & forbidden)[0])
    canonical = {
        "name": name,
        "version": version,
        "display_name": str(manifest.get("display_name") or name)[:120],
        "description": str(manifest.get("description") or "")[:1000],
        "capabilities": sorted({str(item).strip() for item in capabilities if str(item).strip()}),
        "permissions": sorted(requested),
        "homepage": str(manifest.get("homepage") or "")[:500],
        "source": str(manifest.get("source") or "local")[:200],
        "enabled": bool(manifest.get("enabled", False)),
    }
    canonical["manifest_hash"] = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    canonical["created_at"] = str(manifest.get("created_at") or utc_now())
    return canonical


def role_allows(role: str | None, required: str) -> bool:
    return ROLE_ORDER.get(str(role or "viewer"), -1) >= ROLE_ORDER[required]


def evaluate_trajectory(task: Any, checkpoints: list[dict[str, Any]], feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce an explainable score; this never overrides task status or verification."""
    status = getattr(getattr(task, "status", None), "value", getattr(task, "status", ""))
    validation = getattr(task, "validation", None) or {}
    passed = bool(validation.get("passed"))
    signals = {
        "terminal_success": status == "succeeded",
        "deterministic_validation_passed": passed,
        "has_checkpoints": bool(checkpoints),
        "has_artifacts": bool(getattr(task, "artifacts", [])),
        "recovered": bool(getattr(task, "recovery", {})),
        "repair_cycles": max(0, int(getattr(task, "coding_iteration", 0) or 0) - 1),
        "user_feedback": feedback.get("rating") if feedback else None,
    }
    score = 0.0
    score += 0.45 if signals["terminal_success"] else 0.0
    score += 0.30 if signals["deterministic_validation_passed"] else 0.0
    score += 0.10 if signals["has_checkpoints"] else 0.0
    score += 0.05 if signals["has_artifacts"] else 0.0
    score += 0.10 if signals["user_feedback"] == 1 else (-0.20 if signals["user_feedback"] == -1 else 0.0)
    score -= min(0.20, signals["repair_cycles"] * 0.04)
    return {
        "task_id": task.id, "project_id": task.project_id, "score": round(max(0.0, min(1.0, score)), 4),
        "success": signals["terminal_success"] and signals["deterministic_validation_passed"],
        "signals": signals, "checkpoint_count": len(checkpoints), "evaluated_at": utc_now(),
        "learning_safe": True,
        "learning_note": "Use this record to compare trajectories; never automatically change tools, permissions, or prompts from one example.",
    }
