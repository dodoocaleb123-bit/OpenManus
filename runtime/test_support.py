from __future__ import annotations

from typing import Any, Callable

from app.config import config
from app.platform.control_unit import ControlUnit


def build_test_plan(
    prompt: str,
    *,
    capability_ids: list[int] | None = None,
    attachment_ids: list[str] | None = None,
    browser_session_id: str | None = None,
    needs_workspace_context: bool = True,
) -> dict[str, Any]:
    """Construct a registry-validated fake DeepSeek workflow decision."""
    selected_ids = list(capability_ids if capability_ids is not None else [167])
    if not set(selected_ids).intersection(range(167, 174)):
        selected_ids.append(167)
    plan = ControlUnit().plan_from_capability_ids(
        prompt,
        selected_ids,
        attachment_ids=attachment_ids or [],
        browser_session_id=browser_session_id,
    )
    plan.update(
        needs_workspace_context=needs_workspace_context,
        summary="Test controller selected capabilities",
        requires_confirmation=any(step.get("requires_confirmation") for step in plan["steps"]),
        deepseek={
            "model": "deepseek-r1:7b",
            "intent": "test",
            "summary": "Test controller selected capabilities",
            "rationale": ["explicit test fixture decision"],
            "raw_response": "{}",
        },
    )
    return plan


def install_fake_controller(
    monkeypatch,
    *,
    capability_ids: list[int] | None = None,
    needs_workspace_context: bool = True,
    selector: Callable[[str, dict[str, Any]], Any] | None = None,
):
    """Mock DeepSeek capability selection while recording exact user input."""
    calls: list[dict[str, Any]] = []
    reasoning = config.llm.get("reasoning") or config.llm.get("default")
    if reasoning is not None:
        monkeypatch.setitem(config.llm, "reasoning", reasoning)
    monkeypatch.setattr("app.api.routes.reasoning_enabled", lambda: True)

    async def fake_plan(llm, *, prompt, attachment_ids=None, browser_session_id=None, context=None):
        context = context or {}
        calls.append({
            "prompt": prompt,
            "attachment_ids": list(attachment_ids or []),
            "browser_session_id": browser_session_id,
            "context": context,
        })
        picked = selector(prompt, context) if selector else list(capability_ids if capability_ids is not None else [167])
        return build_test_plan(
            prompt,
            capability_ids=picked,
            attachment_ids=attachment_ids,
            browser_session_id=browser_session_id,
            needs_workspace_context=needs_workspace_context,
        )

    monkeypatch.setattr("app.api.routes.make_authoritative_plan", fake_plan)
    return calls


def stub_orchestrator_start(client):
    """Prevent background execution in route tests; return captured task objects."""
    started = []
    client.app.state.orchestrator.start = lambda task: started.append(task)
    return started
