"""Runtime state machine for DeepSeek-authored multi-model workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.platform.handoffs import HandoffResult, HandoffStatus, order_steps


@dataclass
class StepState:
    step_id: str
    handler: str
    objective: str
    depends_on: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    git_actions: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    status: str = "pending"
    attempts: int = 0
    handoff_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "handler": self.handler,
            "objective": self.objective,
            "depends_on": self.depends_on,
            "tools": self.tools,
            "git_actions": self.git_actions,
            "requires_confirmation": self.requires_confirmation,
            "status": self.status,
            "attempts": self.attempts,
            "handoff_id": self.handoff_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "evidence": self.evidence,
        }


class ExecutionStateMachine:
    """Dependency and pause state independent from model-generated prose."""

    def __init__(self, plan: dict[str, Any]):
        raw_steps = list(plan.get("steps") or [])
        self.steps: dict[str, StepState] = {}
        for raw in order_steps(raw_steps):
            step_id = str(raw["step_id"])
            # The objective fallback allows tasks persisted by older releases
            # to be inspected/recovered without retaining their registry.
            objective = str(raw.get("objective") or raw.get("name") or "Previously planned workflow step")
            self.steps[step_id] = StepState(
                step_id=step_id,
                handler=str(raw["handler"]),
                objective=objective,
                depends_on=[str(value) for value in raw.get("depends_on", [])],
                tools=[str(value) for value in raw.get("tools", [])],
                git_actions=[str(value) for value in raw.get("git_actions", [])],
                requires_confirmation=bool(raw.get("requires_confirmation", False)),
            )
        self.events: list[dict[str, Any]] = []

    def ready(self) -> list[StepState]:
        return [
            step for step in self.steps.values()
            if step.status == "pending" and all(self.steps[dep].status == "completed" for dep in step.depends_on)
        ]

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
        status = (
            "failed" if any(step.status == "failed" for step in values)
            else "waiting_for_user" if any(step.status == "waiting_for_user" for step in values)
            else "completed" if not values or all(step.status == "completed" for step in values)
            else "running"
        )
        return {"status": status, "steps": [step.as_dict() for step in values], "ready_steps": [step.step_id for step in self.ready()]}
