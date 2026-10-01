"""Versioned capability registry and deterministic handler resolution.

The registry is the source of truth for planning. Models receive selected
records, never an unbounded prompt containing the entire inventory.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("capabilities.json")


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    records = payload.get("capabilities", [])
    ids = [int(item["id"]) for item in records]
    if payload.get("count") != 400 or ids != list(range(1, 401)) or len({*ids}) != 400:
        raise ValueError("Capability registry must contain every ID from 1 through 400 exactly once")
    return payload


def registry_version() -> str:
    return str(load_registry().get("schema_version", "unknown"))


def capabilities() -> list[dict[str, Any]]:
    return list(load_registry()["capabilities"])


def get_capability(capability_id: int) -> dict[str, Any]:
    for record in capabilities():
        if int(record["id"]) == int(capability_id):
            return record
    raise KeyError(f"Unknown capability ID: {capability_id}")


def search_capabilities(query: str, limit: int = 12) -> list[dict[str, Any]]:
    terms = {part for part in str(query).casefold().split() if len(part) > 2}
    ranked: list[tuple[int, dict[str, Any]]] = []
    for record in capabilities():
        haystack = " ".join([record["name"], record["category_label"], *record.get("aliases", [])]).casefold()
        score = sum(2 if term in record["name"].casefold() else 1 for term in terms if term in haystack)
        if score:
            ranked.append((score, record))
    return [record for _, record in sorted(ranked, key=lambda x: (-x[0], x[1]["id"]))[: max(1, limit)]]


def summary_for_planner() -> list[dict[str, Any]]:
    return [{"id": r["id"], "name": r["name"], "category": r["category"], "handler": r["handler"], "requires_user": r["requires_user"], "platform_only": r["platform_only"], "verification": r["verification"]} for r in capabilities()]


def model_status_from_env() -> list[dict[str, Any]]:
    import os
    roles = {
        "qwen_coder": ("LOCAL_LLM_MODEL", "coding", "qwen2.5-coder:7b"),
        "gemma3": ("VISION_LLM_MODEL", "vision", "gemma3:4b"),
        "deepseek": ("REASONING_LLM_MODEL", "reasoning", "deepseek-r1:7b"),
        "qwen2.5_3b": ("RESEARCH_LLM_MODEL", "research", "qwen2.5:3b"),
        "llama3.2_3b": ("CREATIVITY_LLM_MODEL", "creativity", "llama3.2:3b"),
    }
    result = []
    for role, (variable, capability, default) in roles.items():
        result.append({"role": role, "model": os.getenv(variable, default), "configured": bool(os.getenv(variable)), "capability": capability, "status": "configured" if os.getenv(variable) else "default"})
    return result
