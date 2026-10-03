"""Typed specialist handoffs and dependency-aware workflow ordering."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class HandoffStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_USER = "needs_user"


@dataclass(frozen=True)
class HandoffRequest:
    task_id: str
    step_id: str
    handler: str
    objective: str
    user_request: str
    approved_context: dict[str, Any] = field(default_factory=dict)
    dependency_outputs: dict[str, Any] = field(default_factory=dict)
    expected_output: str = "A concise, evidence-grounded structured result for the assigned objective."
    allowed_tools: tuple[str, ...] = ()
    allowed_files: tuple[str, ...] = ()
    completion_rules: tuple[str, ...] = ()
    failure_rules: tuple[str, ...] = ()
    handoff_id: str = field(default_factory=lambda: uuid4().hex[:16])

    def as_dict(self) -> dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "handler": self.handler,
            "objective": self.objective,
            "user_request": self.user_request,
            "approved_context": self.approved_context,
            "dependency_outputs": self.dependency_outputs,
            "expected_output": self.expected_output,
            "allowed_tools": list(self.allowed_tools),
            "allowed_files": list(self.allowed_files),
            "completion_rules": list(self.completion_rules),
            "failure_rules": list(self.failure_rules),
        }


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
        return {
            "status": self.status.value,
            "step_id": self.request.step_id,
            "handler": self.request.handler,
            "objective": self.request.objective,
            "structured_result": self.structured_result,
            "summary": self.summary,
            "evidence_refs": self.evidence_refs,
            "files_changed": self.files_changed,
            "sources": self.sources,
            "verification": self.verification,
            "error": self.error,
        }


def validate_result(result: HandoffResult) -> None:
    if result.status == HandoffStatus.COMPLETED:
        if not isinstance(result.structured_result, dict):
            raise ValueError(f"Completed workflow step {result.request.step_id} must return a JSON object")
        if not (result.evidence_refs or result.verification or result.structured_result):
            raise ValueError(f"Completed workflow step {result.request.step_id} has no evidence")
    if result.status == HandoffStatus.FAILED and not result.error:
        raise ValueError("Failed handoff must include an error")


def order_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topologically order model-authored steps; reject cycles and missing links."""
    by_id = {str(step["step_id"]): step for step in steps}
    if len(by_id) != len(steps):
        raise ValueError("Workflow step IDs must be unique")
    unknown = {str(dep) for step in steps for dep in step.get("depends_on", [])} - set(by_id)
    if unknown:
        raise ValueError("Workflow contains unknown dependencies: " + ", ".join(sorted(unknown)))
    output: list[dict[str, Any]] = []
    pending = dict(by_id)
    while pending:
        ready = [step for step in pending.values() if all(str(dep) not in pending for dep in step.get("depends_on", []))]
        if not ready:
            raise ValueError("Workflow contains a dependency cycle")
        for step in ready:
            output.append(step)
            pending.pop(str(step["step_id"]))
    return output
