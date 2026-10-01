"""Runtime state machine for registry-backed multi-model task execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.platform.handoffs import HandoffResult, HandoffStatus, order_steps


@dataclass
class StepState:
    step_id: str
    capability_id: int
    handler: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    attempts: int = 0
    handoff_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"step_id": self.step_id, "capability_id": self.capability_id, "handler": self.handler, "depends_on": self.depends_on, "status": self.status, "attempts": self.attempts, "handoff_id": self.handoff_id, "started_at": self.started_at, "finished_at": self.finished_at, "error": self.error, "evidence": self.evidence}


class ExecutionStateMachine:
    """Deterministic orchestration state independent from model text."""

    def __init__(self, plan: dict[str, Any]):
        raw_steps = list(plan.get("steps") or [])
        self.registry_version = plan.get("registry_version", "unknown")
        self.steps: dict[str, StepState] = {}
        for raw in order_steps(raw_steps):
            step_id = str(raw["step_id"])
            self.steps[step_id] = StepState(step_id, int(raw["capability_id"]), str(raw["handler"]), [str(x) for x in raw.get("depends_on", [])])
        self.events: list[dict[str, Any]] = []

    def ready(self) -> list[StepState]:
        return [step for step in self.steps.values() if step.status == "pending" and all(self.steps[dep].status == "completed" for dep in step.depends_on)]

    def start(self, step_id: str, handoff_id: str | None = None) -> StepState:
        step = self.steps[step_id]
        if step.status != "pending":
            raise ValueError(f"Step {step_id} cannot start from {step.status}")
        if any(self.steps[dep].status != "completed" for dep in step.depends_on):
            raise ValueError(f"Step {step_id} has incomplete dependencies")
        step.status = "running"
        step.attempts += 1
        step.handoff_id = handoff_id
        step.started_at = datetime.now(timezone.utc).isoformat()
        return step

    def finish(self, result: HandoffResult | None = None, *, evidence: dict[str, Any] | None = None) -> StepState:
        if result is not None:
            step = self.steps[result.request.step_id]
            step.handoff_id = result.request.handoff_id
            if result.status == HandoffStatus.NEEDS_USER:
                step.status = "waiting_for_user"
            elif result.status == HandoffStatus.FAILED:
                step.status = "failed"
            elif result.status == HandoffStatus.BLOCKED:
                step.status = "blocked"
            else:
                step.status = "completed"
            step.error = (result.error or {}).get("message") if result.error else None
            step.evidence = result.as_dict()
        else:
            running = [step for step in self.steps.values() if step.status == "running"]
            if len(running) != 1:
                raise ValueError("finish() requires exactly one running step when no handoff is supplied")
            step = running[0]
            step.status = "completed"
            step.evidence = evidence or {}
        step.finished_at = datetime.now(timezone.utc).isoformat()
        return step

    def fail(self, step_id: str, error: str, *, retryable: bool = True) -> StepState:
        step = self.steps[step_id]
        step.status = "retryable_failure" if retryable and step.attempts < 3 else "failed"
        step.error = error[:1000]
        step.finished_at = datetime.now(timezone.utc).isoformat()
        return step

    def pause_for_user(self, step_id: str, action: str) -> StepState:
        step = self.steps[step_id]
        step.status = "waiting_for_user"
        step.evidence = {"required_action": action, "checkpoint": "waiting_for_user"}
        return step

    def as_dict(self) -> dict[str, Any]:
        values = list(self.steps.values())
        return {"registry_version": self.registry_version, "status": "failed" if any(s.status == "failed" for s in values) else "waiting_for_user" if any(s.status == "waiting_for_user" for s in values) else "completed" if values and all(s.status == "completed" for s in values) else "running", "steps": [step.as_dict() for step in values], "ready_steps": [step.step_id for step in self.ready()]}
