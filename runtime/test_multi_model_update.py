from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.platform.capability_registry import capabilities, get_capability, load_registry, validate_plan
from app.platform.control_unit import ControlUnit
from app.platform.execution_state import ExecutionStateMachine
from app.platform.model_profiles import profiles
from app.platform.orchestrator import _is_design_only_task
from app.platform.handoffs import HandoffRequest, HandoffResult, HandoffStatus, order_steps, validate_result
from app.platform.reasoning import make_authoritative_plan
from app.server import create_app


def test_registry_contains_exactly_400_capabilities_and_expected_handlers():
    payload = load_registry()
    assert payload["count"] == 400
    records = capabilities()
    assert [item["id"] for item in records] == list(range(1, 401))
    assert len({item["id"] for item in records}) == 400
    assert get_capability(1)["handler"] == "qwen_coder"
    assert get_capability(131)["handler"] == "gemma3"
    assert get_capability(135)["handler"] == "llama3.2_3b"
    assert get_capability(147)["handler"] == "qwen2.5_3b"
    assert get_capability(166)["handler"] == "deepseek"
    assert get_capability(225)["handler"] == "user"
    assert get_capability(257)["handler"] == "platform"
    assert all(item["input_schema"]["type"] == "object" for item in records)
    assert all(item["output_schema"]["type"] == "object" for item in records)
    assert all("supporting_handlers" in item for item in records)


def test_control_unit_selects_ordered_handlers_and_current_attachments():
    plan = ControlUnit().plan(
        "Analyze this screenshot, research dashboard patterns, design and build the improved app",
        attachment_ids=["upload-current"],
        mode="implement",
    )
    assert plan["controller"] == "deepseek"
    assert plan["attachment_ids"] == ["upload-current"]
    assert {step["handler"] for step in plan["steps"]} >= {"gemma3", "qwen2.5_3b", "llama3.2_3b", "qwen_coder"}
    assert plan["evidence_required"] is True
    assert all("input_schema" in step and "output_schema" in step for step in plan["steps"])
    validate_plan(plan)


def test_five_typed_model_profiles_are_present():
    assert {profile.role for profile in profiles()} == {"qwen_coder", "gemma3", "deepseek", "qwen2.5_3b", "llama3.2_3b"}


def test_order_steps_rejects_cycles_and_preserves_dependencies():
    ordered = order_steps([
        {"step_id": "b", "depends_on": ["a"]},
        {"step_id": "a", "depends_on": []},
    ])
    assert [step["step_id"] for step in ordered] == ["a", "b"]
    with pytest.raises(ValueError):
        order_steps([{ "step_id": "a", "depends_on": ["b"] }, {"step_id": "b", "depends_on": ["a"]}])


def test_state_machine_enforces_dependencies_and_user_pause():
    state = ExecutionStateMachine({"registry_version": "test", "steps": [
        {"step_id": "design", "capability_id": 135, "handler": "llama3.2_3b", "depends_on": []},
        {"step_id": "user", "capability_id": 225, "handler": "user", "depends_on": ["design"]},
    ]})
    state.start("design")
    state.finish(evidence={"brief": True})
    state.pause_for_user("user", "Take over the browser")
    assert state.as_dict()["status"] == "waiting_for_user"
    assert state.steps["user"].evidence["required_action"] == "Take over the browser"


def test_design_only_requests_do_not_count_as_implementation():
    assert _is_design_only_task("Design a beautiful mobile dashboard with a modern palette")
    assert not _is_design_only_task("Design and build a beautiful mobile dashboard")


def test_completed_handoff_requires_evidence():
    request = HandoffRequest(task_id="t", step_id="s", capability_id=167, handler="deepseek", user_request="answer")
    with pytest.raises(ValueError):
        validate_result(HandoffResult(HandoffStatus.COMPLETED, request))
    result = HandoffResult(HandoffStatus.COMPLETED, request, structured_result={"answer": "ok"})
    validate_result(result)


@pytest.mark.asyncio
async def test_deepseek_authoritative_plan_selects_and_validates_registry_ids():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"

        async def ask(self, messages, **kwargs):
            return '{"summary":"research and build","intent":"engineering","capability_ids":[147,135,1],"excluded_capabilities":[],"requires_confirmation":false,"rationale":["source evidence before implementation"]}'

    plan = await make_authoritative_plan(FakeDeepSeek(), prompt="Research this URL, design and build the result")
    assert plan["authoritative"] is True
    assert plan["controller"] == "deepseek"
    assert [step["handler"] for step in plan["steps"]][:2] == ["deepseek", "qwen2.5_3b"]
    assert any(step["handler"] == "llama3.2_3b" for step in plan["steps"])


def test_registry_and_model_status_endpoints(tmp_path: Path):
    app = create_app(tmp_path, check_llm=False)
    with TestClient(app) as client:
        registry = client.get("/api/capabilities/registry")
        assert registry.status_code == 200
        assert registry.json()["count"] == 400
        status = client.get("/api/capabilities/models/status")
        assert status.status_code == 200
        roles = {item["role"] for item in status.json()["models"]}
        assert roles == {"qwen_coder", "gemma3", "deepseek", "qwen2.5_3b", "llama3.2_3b"}
        assert "api_key" not in status.text.lower()
        platform_status = client.get("/api/status")
        assert platform_status.status_code == 200
        assert "deepseek_bridge" in platform_status.json()
        assert '"api_key":' not in platform_status.text.lower()
