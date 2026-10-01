from __future__ import annotations

from pathlib import Path

import pytest
import httpx
from fastapi.testclient import TestClient
from openai import BadRequestError

from app.platform.capability_registry import capabilities, get_capability, load_registry, validate_plan
from app.platform.control_unit import ControlUnit
from app.platform.execution_state import ExecutionStateMachine
from app.platform.model_profiles import profiles
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
    plan = ControlUnit().plan_from_capability_ids(
        "Analyze this screenshot, research dashboard patterns, design and build the improved app",
        [131, 147, 135, 1],
        attachment_ids=["upload-current"],
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
        last_messages = None
        last_system = None

        async def ask(self, messages, system_msgs=None, **kwargs):
            FakeDeepSeek.last_messages = messages
            FakeDeepSeek.last_system = system_msgs
            if "Executable capability directory" not in system_msgs[0]["content"]:
                return '{"route":"execute","needs_workspace_context":true,"summary":"research and build","intent":"engineering","capability_ids":[],"handlers":["qwen2.5_3b","llama3.2_3b","qwen_coder"],"rationale":[]}'
            return '{"route":"execute","needs_workspace_context":true,"summary":"research and build","intent":"engineering","capability_ids":[147,135,1],"excluded_capabilities":[],"requires_confirmation":false,"rationale":["source evidence before implementation"]}'

    raw_prompt = "  Research this URL, design and build the result  "
    plan = await make_authoritative_plan(FakeDeepSeek(), prompt=raw_prompt)
    assert plan["authoritative"] is True
    assert plan["controller"] == "deepseek"
    assert [step["handler"] for step in plan["steps"]][:2] == ["deepseek", "qwen2.5_3b"]
    assert any(step["handler"] == "llama3.2_3b" for step in plan["steps"])
    assert FakeDeepSeek.last_messages == [{"role": "user", "content": raw_prompt}]
    offered = FakeDeepSeek.last_system[0]["content"]
    assert "257\tplatform" not in offered
    assert "Delete projects" not in offered
    assert "Executable capability directory" in offered


@pytest.mark.asyncio
async def test_greeting_routes_through_deepseek_without_full_catalog_and_recovers_think_only_reply():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"
        base_url = "http://host.docker.internal:11434/v1"

        def __init__(self):
            self.calls = []

        async def ask(self, messages, system_msgs=None, **kwargs):
            self.calls.append((messages, system_msgs, kwargs))
            if len(self.calls) == 1:
                return "<think>working through a greeting</think>"
            return '{"route":"respond","needs_workspace_context":false,"summary":"Greet the user","intent":"conversation","capability_ids":[166],"handlers":[],"rationale":[]}'

    llm = FakeDeepSeek()
    plan = await make_authoritative_plan(llm, prompt="Hellooo")
    assert plan["route"] == "respond"
    assert plan["selected_capabilities"] == [166]
    assert len(llm.calls) == 2
    assert all(messages == [{"role": "user", "content": "Hellooo"}] for messages, _, _ in llm.calls)
    assert all("Executable capability directory" not in system[0]["content"] for _, system, _ in llm.calls)
    assert all(kwargs["response_format"] == {"type": "json_object"} for _, _, kwargs in llm.calls)
    assert llm.calls[1][2]["max_tokens"] >= 4096


@pytest.mark.asyncio
async def test_old_ollama_json_mode_rejection_retries_same_deepseek_model():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"
        base_url = "http://host.docker.internal:11434/v1"

        def __init__(self):
            self.calls = []

        async def ask(self, messages, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("response_format"):
                raise BadRequestError(
                    "unknown response_format",
                    response=httpx.Response(400, request=httpx.Request("POST", self.base_url + "/chat/completions")),
                    body={},
                )
            return '{"route":"respond","needs_workspace_context":false,"summary":"Greet the user","intent":"conversation","capability_ids":[166],"handlers":[]}'

    llm = FakeDeepSeek()
    plan = await make_authoritative_plan(llm, prompt="Hello")
    assert plan["route"] == "respond"
    assert [call["response_format"] for call in llm.calls] == [{"type": "json_object"}, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [
    '{"route":"respond"}',
    '{"route":"respond","needs_workspace_context":false,"summary":"Hello","capability_ids":[]}',
    '{"route":"respond","needs_workspace_context":false,"summary":"Hello","capability_ids":[1]}',
    '{"route":"respond","needs_workspace_context":false,"summary":"Hello","capability_ids":["167"]}',
])
async def test_deepseek_direct_reply_does_not_require_a_numeric_capability_id(decision):
    class FakeDeepSeek:
        model = "deepseek-r1:7b"

        async def ask(self, messages, **kwargs):
            return decision

    plan = await make_authoritative_plan(FakeDeepSeek(), prompt="Hellooo")
    assert plan["route"] == "respond"
    assert plan["selected_capabilities"] == [166]
    assert {step["handler"] for step in plan["steps"]} == {"deepseek"}


@pytest.mark.asyncio
async def test_deepseek_cannot_select_an_unimplemented_platform_capability_as_work():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"

        async def ask(self, messages, system_msgs=None, **kwargs):
            if "Executable capability directory" not in system_msgs[0]["content"]:
                return '{"route":"execute","needs_workspace_context":false,"summary":"delete project","intent":"project operation","capability_ids":[],"handlers":["qwen_coder"],"rationale":[]}'
            return '{"route":"execute","needs_workspace_context":false,"summary":"delete project","intent":"project operation","capability_ids":[179],"excluded_capabilities":[],"requires_confirmation":true,"rationale":[]}'

    with pytest.raises(ValueError, match="capability outside its chosen handler directory"):
        await make_authoritative_plan(FakeDeepSeek(), prompt="Delete the project")


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
        assert platform_status.json()["deepseek_control"]["required"] is True
        assert '"api_key":' not in platform_status.text.lower()
