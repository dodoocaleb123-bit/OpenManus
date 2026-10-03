from __future__ import annotations

from typing import Any, Callable

from app.config import config
from app.platform.control_unit import ControlUnit

_DEFAULT_TOOLS = {
    "qwen_coder": ["bash", "python_execute", "str_replace_editor"],
    "gemma3": [],
    "qwen2.5_3b": ["web_search"],
    "llama3.2_3b": [],
    "user": ["ask_human"],
}
_OBJECTIVES = {
    "qwen_coder": "Implement the requested project changes and validate them",
    "gemma3": "Analyze the supplied image or requested visual reference",
    "qwen2.5_3b": "Research the request and return source-grounded findings",
    "llama3.2_3b": "Review the requested design and provide practical guidance",
    "user": "Ask the user for the required information or action, then resume",
}


def build_test_plan(
    prompt: str,
    *,
    handlers: list[str] | None = None,
    step_specs: list[dict[str, Any]] | None = None,
    attachment_ids: list[str] | None = None,
    browser_session_id: str | None = None,
    needs_workspace_context: bool = True,
) -> dict[str, Any]:
    """Construct a fake DeepSeek plan from descriptive handlers and tool choices."""
    if step_specs is not None:
        workflow = [dict(step) for step in step_specs]
    else:
        workflow = []
        prior: str | None = None
        for index, handler in enumerate(handlers or [], start=1):
            step_id = f"step-{index}"
            workflow.append({
                "step_id": step_id,
                "handler": handler,
                "objective": _OBJECTIVES.get(handler, f"Complete the {handler} work needed for the request"),
                "depends_on": [prior] if prior else [],
                "tools": list(_DEFAULT_TOOLS.get(handler, [])),
                "git_actions": [],
                "requires_confirmation": False,
            })
            prior = step_id
    plan = ControlUnit().build_workflow(
        prompt,
        workflow,
        attachment_ids=attachment_ids or [],
        browser_session_id=browser_session_id,
    )
    selected_handlers = sorted({step["handler"] for step in plan["steps"]})
    selected_tools = sorted({tool for step in plan["steps"] for tool in step["tools"]})
    summary = "Test controller selected a descriptive workflow"
    plan.update(
        needs_workspace_context=needs_workspace_context,
        summary=summary,
        intent="test",
        selected_handlers=selected_handlers,
        selected_tools=selected_tools,
        requires_confirmation=any(step.get("requires_confirmation") for step in plan["steps"]),
        deepseek={
            "model": "deepseek-r1:7b",
            "intent": "test",
            "summary": summary,
            "rationale": ["explicit test fixture decision"],
            "raw_response": "{}",
        },
    )
    return plan


def install_fake_controller(
    monkeypatch,
    *,
    handlers: list[str] | None = None,
    step_specs: list[dict[str, Any]] | None = None,
    needs_workspace_context: bool = True,
    selector: Callable[[str, dict[str, Any]], Any] | None = None,
):
    """Mock DeepSeek planning while recording exact user input and context."""
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
        selected = selector(prompt, context) if selector else None
        selected_specs = selected if isinstance(selected, list) and selected and isinstance(selected[0], dict) else step_specs
        selected_handlers = selected if isinstance(selected, list) and (not selected or isinstance(selected[0], str)) else handlers
        return build_test_plan(
            prompt,
            handlers=selected_handlers,
            step_specs=selected_specs,
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
