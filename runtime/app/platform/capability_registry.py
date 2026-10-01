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
VALID_HANDLERS = {"qwen_coder", "gemma3", "deepseek", "qwen2.5_3b", "llama3.2_3b", "user", "platform"}


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    records = payload.get("capabilities", [])
    ids = [int(item["id"]) for item in records]
    if payload.get("count") != 400 or ids != list(range(1, 401)) or len({*ids}) != 400:
        raise ValueError("Capability registry must contain every ID from 1 through 400 exactly once")
    for record in records:
        required = {"id", "name", "category", "handler", "required_inputs", "expected_outputs", "input_schema", "output_schema", "dependencies", "verification", "requires_user", "platform_only", "requires_confirmation", "may_run_in_parallel", "failure_policy", "aliases", "supporting_handlers"}
        missing = required - set(record)
        if missing:
            raise ValueError(f"Capability {record.get('id')} is missing metadata: {sorted(missing)}")
        if record["handler"] not in VALID_HANDLERS or any(item not in VALID_HANDLERS for item in record["supporting_handlers"]):
            raise ValueError(f"Capability {record['id']} has an unknown handler")
        if not record["aliases"] or not isinstance(record["input_schema"], dict) or not isinstance(record["output_schema"], dict):
            raise ValueError(f"Capability {record['id']} has invalid aliases or schemas")
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
    return [{"id": r["id"], "name": r["name"], "category": r["category"], "handler": r["handler"], "supporting_handlers": r["supporting_handlers"], "requires_user": r["requires_user"], "platform_only": r["platform_only"], "verification": r["verification"], "input_schema": r["input_schema"], "output_schema": r["output_schema"]} for r in capabilities()]


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a DeepSeek/control-unit plan against the current registry."""
    step_ids = {str(step.get("step_id")) for step in plan.get("steps", [])}
    if len(step_ids) != len(plan.get("steps", [])) or None in step_ids:
        raise ValueError("Control-unit plan step IDs must be unique and non-empty")
    for step in plan.get("steps", []):
        record = get_capability(int(step["capability_id"]))
        if step.get("handler") != record["handler"]:
            raise ValueError(f"Capability {record['id']} must run on {record['handler']}, not {step.get('handler')}")
        if any(str(dep) not in step_ids for dep in step.get("depends_on", [])):
            raise ValueError(f"Capability {record['id']} contains an unknown dependency")
    return plan


def model_status_from_env() -> list[dict[str, Any]]:
    from app.platform.model_profiles import status_from_env
    return status_from_env()
