from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import BadRequestError

from app.platform.control_unit import ControlUnit
from app.platform.execution_state import ExecutionStateMachine
from app.platform.handoffs import HandoffRequest, HandoffResult, HandoffStatus, order_steps, validate_result
from app.platform.model_profiles import profiles
from app.platform.reasoning import make_authoritative_plan
from app.api.routes import available_planner_tools
from app.server import create_app

TOOLS = ["bash", "python_execute", "str_replace_editor", "web_search", "platform_browser", "platform_git", "ask_human", "terminate"]
ROLES = ["reasoning", "heavy_coding", "vision", "research", "creativity"]


def workflow_steps():
    return [
        {"step_id": "research", "handler": "qwen2.5_3b", "objective": "Find reliable implementation references", "depends_on": [], "tools": ["web_search"], "git_actions": []},
        {"step_id": "design", "handler": "llama3.2_3b", "objective": "Recommend a design informed by the research", "depends_on": ["research"], "tools": [], "git_actions": []},
        {"step_id": "build", "handler": "qwen_coder", "objective": "Implement and validate the requested project", "depends_on": ["research", "design"], "tools": ["bash", "python_execute", "str_replace_editor"], "git_actions": []},
    ]


def test_deepseek_authored_workflow_uses_descriptive_handlers_tools_and_dependencies():
    plan = ControlUnit().build_workflow("Research, design, build, and explain", workflow_steps())
    assert plan["authoritative"] is True
    assert [step["step_id"] for step in plan["steps"]] == ["research", "design", "build"]
    assert plan["steps"][0]["handler"] == "qwen2.5_3b"
    assert plan["steps"][2]["tools"] == ["bash", "python_execute", "str_replace_editor"]
    assert plan["steps"][2]["depends_on"] == ["research", "design"]
    assert "capability_id" not in str(plan)


def test_direct_deepseek_answer_needs_no_specialist_workflow_step():
    plan = ControlUnit().build_workflow("Say hello", [])
    assert plan["authoritative"] is True
    assert plan["steps"] == []


def test_planner_tool_context_reflects_runtime_browser_availability():
    class Runtime:
        def __init__(self, browsers):
            self.browsers = browsers

    without_browser = available_planner_tools(Runtime(None))
    with_browser = available_planner_tools(Runtime(object()))
    assert {"bash", "python_execute", "str_replace_editor", "ask_human", "terminate"} <= set(without_browser)
    assert "platform_browser" not in without_browser
    assert "platform_browser" in with_browser
    assert "platform_git" in without_browser


def test_control_unit_rejects_unknown_handler_tool_and_coding_without_tools():
    with pytest.raises(ValueError, match="unavailable workflow handler"):
        ControlUnit().build_workflow("x", [{"step_id": "x", "handler": "unknown", "objective": "x"}])
    with pytest.raises(ValueError, match="unavailable tool"):
        ControlUnit().build_workflow("x", [{"step_id": "x", "handler": "qwen_coder", "objective": "x", "tools": ["shell_magic"]}])
    with pytest.raises(ValueError, match="must explicitly select"):
        ControlUnit().build_workflow("x", [{"step_id": "x", "handler": "qwen_coder", "objective": "x", "tools": []}])
    with pytest.raises(ValueError, match="web_search or platform_browser"):
        ControlUnit().build_workflow("x", [{"step_id": "x", "handler": "qwen2.5_3b", "objective": "research", "tools": []}])


def test_order_steps_rejects_cycles_and_preserves_dependencies():
    ordered = order_steps([
        {"step_id": "build", "depends_on": ["research"]},
        {"step_id": "research", "depends_on": []},
    ])
    assert [step["step_id"] for step in ordered] == ["research", "build"]
    with pytest.raises(ValueError, match="cycle"):
        order_steps([{"step_id": "a", "depends_on": ["b"]}, {"step_id": "b", "depends_on": ["a"]}])


def test_state_machine_enforces_dependencies_and_user_pause():
    state = ExecutionStateMachine({"steps": [
        {"step_id": "design", "handler": "llama3.2_3b", "objective": "Create design guidance", "depends_on": [], "tools": []},
        {"step_id": "user", "handler": "user", "objective": "Take over the browser", "depends_on": ["design"], "tools": ["ask_human"]},
    ]})
    state.start("design")
    state.finish(evidence={"brief": True})
    state.pause_for_user("user", "Take over the browser")
    assert state.as_dict()["status"] == "waiting_for_user"
    assert state.steps["user"].evidence["required_action"] == "Take over the browser"


def test_completed_handoff_requires_result_evidence():
    request = HandoffRequest(task_id="t", step_id="answer", handler="qwen_coder", objective="Edit README", user_request="Update README")
    with pytest.raises(ValueError):
        validate_result(HandoffResult(HandoffStatus.COMPLETED, request))
    result = HandoffResult(HandoffStatus.COMPLETED, request, structured_result={"summary": "updated"})
    validate_result(result)


@pytest.mark.asyncio
async def test_deepseek_receives_verbatim_message_and_model_tool_descriptions_in_one_plan_call():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"
        base_url = "http://host.docker.internal:11434/v1"

        def __init__(self):
            self.calls = []

        async def ask(self, messages, system_msgs=None, **kwargs):
            self.calls.append((messages, system_msgs, kwargs))
            return '{"summary":"Research then implement","intent":"engineering","needs_workspace_context":true,"workflow":' + __import__("json").dumps(workflow_steps()) + ',"requires_confirmation":false,"rationale":["Research grounds the build"]}'

    llm = FakeDeepSeek()
    prompt = "  Research this topic, design and build a small app  "
    plan = await make_authoritative_plan(llm, prompt=prompt, context={"available_model_roles": ROLES, "available_tools": TOOLS})
    assert plan["authoritative"] is True
    assert [step["handler"] for step in plan["steps"]] == ["qwen2.5_3b", "llama3.2_3b", "qwen_coder"]
    assert plan["steps"][-1]["depends_on"] == ["research", "design"]
    assert llm.calls[0][0] == [{"role": "user", "content": prompt}]
    system = llm.calls[0][1][0]["content"]
    assert "MODEL AND TOOL DESCRIPTIONS" in system
    assert "qwen_coder" in system and "platform_git" in system
    assert "400" not in system and "capability_id" not in system
    assert "When asked your name, identify yourself as OpenManus" in system
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_openmanus_name_question_is_direct_and_identity_is_explicit_in_planner_prompt():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"
        base_url = "http://host.docker.internal:11434/v1"

        def __init__(self):
            self.calls = []

        async def ask(self, messages, system_msgs=None, **kwargs):
            self.calls.append((messages, system_msgs, kwargs))
            return '{"summary":"Identify as OpenManus","intent":"identity_question","needs_workspace_context":false,"workflow":[]}'

    llm = FakeDeepSeek()
    plan = await make_authoritative_plan(llm, prompt="My name is Caleb, your creator. What is yours?")
    assert plan["steps"] == []
    system = llm.calls[0][1][0]["content"]
    assert "When asked your name, identify yourself as OpenManus" in system
    assert "select no tools or specialist workflow steps" in system


@pytest.mark.asyncio
async def test_deepseek_must_not_select_unconfigured_handler_or_unavailable_tool():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"

        def __init__(self, workflow):
            self.workflow = workflow

        async def ask(self, *args, **kwargs):
            import json
            return json.dumps({"summary": "run", "intent": "work", "needs_workspace_context": False, "workflow": self.workflow})

    coder = [{"step_id": "build", "handler": "qwen_coder", "objective": "Build", "depends_on": [], "tools": ["bash"]}]
    with pytest.raises(ValueError, match="not configured"):
        await make_authoritative_plan(FakeDeepSeek(coder), prompt="Build", context={"available_model_roles": ["reasoning"], "available_tools": TOOLS})
    with pytest.raises(ValueError, match="not available in this runtime"):
        await make_authoritative_plan(FakeDeepSeek(coder), prompt="Build", context={"available_model_roles": ROLES, "available_tools": ["web_search"]})


@pytest.mark.asyncio
async def test_same_model_retries_when_local_ollama_rejects_json_mode():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"
        base_url = "http://host.docker.internal:11434/v1"

        def __init__(self):
            self.calls = []

        async def ask(self, messages, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("response_format") and len(self.calls) == 1:
                raise BadRequestError(
                    "unknown response_format",
                    response=httpx.Response(400, request=httpx.Request("POST", self.base_url + "/chat/completions")),
                    body={},
                )
            return '{"summary":"Greet","intent":"greeting","needs_workspace_context":false,"workflow":[]}'

    llm = FakeDeepSeek()
    result = await make_authoritative_plan(llm, prompt="Hello")
    assert result["steps"] == []
    assert [call["response_format"] for call in llm.calls] == [{"type": "json_object"}, None]


@pytest.mark.asyncio
async def test_deepseek_repairs_valid_json_with_empty_workflow_summary():
    class FakeDeepSeek:
        model = "deepseek-r1:7b"
        base_url = "http://host.docker.internal:11434/v1"

        def __init__(self):
            self.calls = []

        async def ask(self, messages, system_msgs=None, **kwargs):
            self.calls.append((messages, system_msgs, kwargs))
            if len(self.calls) == 1:
                return '{"summary":"","intent":"greeting","needs_workspace_context":false,"workflow":[]}'
            return '{"summary":"Greet the user","intent":"greeting","needs_workspace_context":false,"workflow":[]}'

    llm = FakeDeepSeek()
    plan = await make_authoritative_plan(llm, prompt="Hello")
    assert plan["summary"] == "Greet the user"
    assert len(llm.calls) == 2
    assert "summary" in llm.calls[1][0][0]["content"]


def test_model_status_remains_but_numbered_task_registry_endpoint_is_removed(tmp_path: Path):
    app = create_app(tmp_path, check_llm=False)
    with TestClient(app) as client:
        assert client.get("/api/capabilities/registry").status_code == 404
        status = client.get("/api/capabilities/models/status")
        assert status.status_code == 200
        roles = {item["role"] for item in status.json()["models"]}
        assert roles == {"qwen_coder", "gemma3", "deepseek", "qwen2.5_3b", "llama3.2_3b"}
        assert "api_key" not in status.text.lower()
        platform_status = client.get("/api/status")
        assert platform_status.status_code == 200
        assert platform_status.json()["deepseek_control"]["required"] is True
        assert '"api_key":' not in platform_status.text.lower()


def test_five_model_profiles_remain_available_for_descriptive_planning():
    assert {profile.role for profile in profiles()} == {"qwen_coder", "gemma3", "deepseek", "qwen2.5_3b", "llama3.2_3b"}
