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


def validate_json_schema(value: Any, schema: dict[str, Any], *, path: str = "result") -> None:
    """Validate the bounded JSON-Schema subset used by capability contracts."""
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: schema must be an object")
    expected = schema.get("type")
    checks = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected in checks and not checks[expected]:
        raise ValueError(f"{path}: expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is not in enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key}: required field is missing")
        for key, child in schema.get("properties", {}).items():
            if key in value:
                validate_json_schema(value[key], child, path=f"{path}.{key}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value[:100]):
            validate_json_schema(item, schema["items"], path=f"{path}[{index}]")


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
        if not isinstance(record["dependencies"], list) or any(int(dep) == int(record["id"]) for dep in record["dependencies"]):
            raise ValueError(f"Capability {record['id']} has invalid dependencies")
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
        present_capabilities = {int(item["capability_id"]) for item in plan.get("steps", [])}
        missing = set(int(dep) for dep in record.get("dependencies", [])) - present_capabilities
        if missing:
            raise ValueError(f"Capability {record['id']} is missing declared dependencies: {sorted(missing)}")
        validate_json_schema(step.get("input", {}), record["input_schema"], path=f"{step['step_id']}.input")
    return plan


def model_status_from_env() -> list[dict[str, Any]]:
    from app.platform.model_profiles import status_from_env
    return status_from_env()
