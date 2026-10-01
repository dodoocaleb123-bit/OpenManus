"""Typed multi-model handoffs and dependency-aware execution plans."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from app.platform.capability_registry import get_capability, registry_version, validate_json_schema


class HandoffStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_USER = "needs_user"


@dataclass(frozen=True)
class HandoffRequest:
    task_id: str
    step_id: str
    capability_id: int
    handler: str
    user_request: str
    approved_context: dict[str, Any] = field(default_factory=dict)
    dependency_outputs: dict[str, Any] = field(default_factory=dict)
    required_output_schema: dict[str, Any] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    allowed_files: tuple[str, ...] = ()
    completion_rules: tuple[str, ...] = ()
    failure_rules: tuple[str, ...] = ()
    handoff_id: str = field(default_factory=lambda: uuid4().hex[:16])
    registry_version: str = field(default_factory=registry_version)

    def as_dict(self) -> dict[str, Any]:
        record = get_capability(self.capability_id)
        return {"handoff_id": self.handoff_id, "task_id": self.task_id, "step_id": self.step_id, "capability_id": self.capability_id, "handler": self.handler, "user_request": self.user_request, "approved_context": self.approved_context, "dependency_outputs": self.dependency_outputs, "required_output_schema": self.required_output_schema or {"type": "object"}, "allowed_tools": list(self.allowed_tools or record.get("tools", [])), "allowed_files": list(self.allowed_files), "completion_rules": list(self.completion_rules or record.get("verification", [])), "failure_rules": list(self.failure_rules or [record.get("failure_policy", "report")]), "registry_version": self.registry_version}


@dataclass
class HandoffResult:
    status: HandoffStatus
    request: HandoffRequest
    structured_result: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    verification: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "step_id": self.request.step_id, "capability_id": self.request.capability_id, "handler": self.request.handler, "structured_result": self.structured_result, "summary": self.summary, "evidence_refs": self.evidence_refs, "files_changed": self.files_changed, "sources": self.sources, "verification": self.verification, "error": self.error, "registry_version": self.request.registry_version}


def validate_result(result: HandoffResult) -> None:
    if result.status == HandoffStatus.COMPLETED:
        if not (result.evidence_refs or result.verification or result.structured_result):
            raise ValueError(f"Completed capability {result.request.capability_id} has no evidence")
        record = get_capability(result.request.capability_id)
        validate_json_schema(result.as_dict(), record["output_schema"], path=f"capability[{result.request.capability_id}].output")
    if result.status == HandoffStatus.FAILED and not result.error:
        raise ValueError("Failed handoff must include an error")


def order_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topologically order plan steps and reject cycles or unknown dependencies."""
    by_id = {str(step["step_id"]): step for step in steps}
    if len(by_id) != len(steps):
        raise ValueError("Plan step IDs must be unique")
    unknown = {str(dep) for step in steps for dep in step.get("depends_on", [])} - set(by_id)
    if unknown:
        raise ValueError("Plan contains unknown dependencies: " + ", ".join(sorted(unknown)))
    output: list[dict[str, Any]] = []
    pending = dict(by_id)
    while pending:
        ready = [step for step in pending.values() if all(str(dep) not in pending for dep in step.get("depends_on", []))]
        if not ready:
            raise ValueError("Capability plan contains a dependency cycle or unknown dependency")
        ready.sort(key=lambda step: str(step["step_id"]))
        for step in ready:
            output.append(step)
            pending.pop(str(step["step_id"]))
    return output
