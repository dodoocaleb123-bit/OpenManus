from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import config
from app.platform.automation import AutomationStore, detect_preview_command, parse_preview_command, verify_webhook
from app.server import create_app
from test_support import install_fake_controller
from app.platform.specialists import ROLES, bounded_gather
from app.platform.control_unit import ControlUnit


def test_specialist_roles_map_to_local_model_profiles():
    assert {role.model_config_name for role in ROLES.values()} >= {
        "reasoning", "research", "heavy_coding", "vision", "creativity", "default"
    }


def test_bounded_gather_never_exceeds_limit():
    active = 0
    peak = 0
    async def worker(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return value * 2
    assert asyncio.run(bounded_gather(range(8), worker, limit=2)) == [i * 2 for i in range(8)]
    assert peak <= 2


def test_registry_assigns_only_the_explicitly_selected_capability_handlers():
    plan = ControlUnit().plan_from_capability_ids("opaque user request", [147, 135, 1])
    assert {step["handler"] for step in plan["steps"]} >= {"qwen2.5_3b", "llama3.2_3b", "qwen_coder"}
    same_selection = ControlUnit().plan_from_capability_ids("unrelated words", [147, 135, 1])
    assert [step["handler"] for step in plan["steps"]] == [step["handler"] for step in same_selection["steps"]]


def test_specialist_capability_owners_are_explicit_model_roles():
    assert ROLES["planner"].model_config_name == "reasoning"
    assert ROLES["researcher"].model_config_name == "research"
    assert ROLES["designer"].model_config_name == "creativity"
    assert ROLES["coder"].model_config_name == "heavy_coding"


def test_gemma_browser_review_is_ordered_after_qwen_build_and_declares_browser_tool():
    plan = ControlUnit().plan_from_capability_ids("Build and visually inspect a web app", [132])
    by_id = {step["capability_id"]: step for step in plan["steps"]}
    assert by_id[132]["handler"] == "gemma3"
    assert "browser" in by_id[132]["tools"]
    assert by_id[1]["step_id"] in by_id[132]["depends_on"]
    ordered = [step["capability_id"] for step in plan["steps"]]
    assert ordered.index(1) < ordered.index(132)


def test_user_handoff_respects_deepseek_selected_order_around_coding():
    before = ControlUnit().plan_from_capability_ids("Take over, then build", [231, 1])
    before_by_id = {step["capability_id"]: step for step in before["steps"]}
    assert before_by_id[231]["step_id"] in before_by_id[1]["depends_on"]

    after = ControlUnit().plan_from_capability_ids("Build, then let the user take over", [1, 231])
    after_by_id = {step["capability_id"]: step for step in after["steps"]}
    assert after_by_id[1]["step_id"] in after_by_id[231]["depends_on"]
    assert after_by_id[231]["step_id"] not in after_by_id[1]["depends_on"]


def test_scheduled_prompt_is_deepseek_routed_before_answer_or_execution(tmp_path, monkeypatch):
    from app.platform.control_unit import ControlUnit

    monkeypatch.setitem(config.llm, "reasoning", config.llm["default"])
    monkeypatch.setattr("app.server.reasoning_enabled", lambda: True)
    seen = []

    async def fake_plan(llm, *, prompt, attachment_ids=None, browser_session_id=None, context=None):
        seen.append((prompt, context))
        route = "respond" if prompt == "just explain this" else "execute"
        ids = [167] if route == "respond" else [1]
        plan = ControlUnit().plan_from_capability_ids(prompt, ids)
        plan.update(route=route, needs_workspace_context=False, summary="Selected by fake DeepSeek", deepseek={"model": "deepseek-test", "intent": "test"})
        return plan

    class FakeLLM:
        model = "deepseek-test"

        def __init__(self, *args, **kwargs):
            pass

        async def ask(self, messages, **kwargs):
            return "DeepSeek scheduled answer."

    monkeypatch.setattr("app.server.make_authoritative_plan", fake_plan)
    monkeypatch.setattr("app.server.LLM", FakeLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "scheduled"}).json()
    started = []
    client.app.state.orchestrator.start = started.append

    import asyncio

    direct = asyncio.run(client.app.state.scheduler.launch(project["id"], "just explain this"))
    assert direct["kind"] == "chat"
    assert direct["assistant"].content == "DeepSeek scheduled answer."
    assert direct["assistant"].response_time_ms is not None
    created = asyncio.run(client.app.state.scheduler.launch(project["id"], "run the selected work"))
    assert created.plan["control_unit_plan"]["authoritative"] is True
    assert created.plan["route"] == "execute"
    assert started == [created]
    assert [entry[0] for entry in seen] == ["just explain this", "run the selected work"]


def test_automation_store_persists_schedule_and_connector(tmp_path: Path):
    records = AutomationStore(tmp_path / "platform.db")
    schedule = records.create_schedule("project-1", "Run the tests", 60)
    assert records.list_schedules("project-1")[0]["id"] == schedule["id"]
    connector = records.add_connector("project-1", "CI", "https://example.com/hook", "s" * 32)
    assert records.list_connectors("project-1")[0]["name"] == "CI"
    assert records.get_connector(connector["id"])["secret"] == "s" * 32


def test_signed_webhook_verification():
    body = b'{"event":"push"}'
    signature = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()
    assert verify_webhook(body, signature, "secret")
    assert not verify_webhook(body, signature, "wrong")


def test_preview_command_detection_and_override(tmp_path: Path):
    (tmp_path / "index.html").write_text("<h1>Preview</h1>")
    assert detect_preview_command(tmp_path, 3210) == ["python", "-m", "http.server", "3210", "--bind", "0.0.0.0"]
    assert parse_preview_command("npm run dev", tmp_path) == ["npm", "run", "dev"]
    (tmp_path / "package.json").write_text('{"scripts":{"dev":"vite"}}')
    assert parse_preview_command(None, tmp_path) == ["npm", "run", "dev"]


def test_phase_c_routes_and_process_manager(tmp_path, monkeypatch):
    install_fake_controller(monkeypatch, capability_ids=[1], route="execute")
    client = TestClient(create_app(tmp_path, check_llm=False))
    client.app.state.orchestrator.start = lambda task: None
    project = client.post("/api/projects", json={"name": "phase-c"}).json()
    project_id = project["id"]
    roles = client.get(f"/api/projects/{project_id}/specialists")
    assert roles.status_code == 200
    schedule = client.post(f"/api/projects/{project_id}/schedules", json={"prompt": "Run tests", "interval_seconds": 60})
    assert schedule.status_code == 200
    assert client.get(f"/api/projects/{project_id}/schedules").json()
    process = client.post(f"/api/projects/{project_id}/processes", json={"command": ["python", "-c", "print('ok')"]})
    assert process.status_code == 200
    process_id = process.json()["id"]
    assert client.get(f"/api/processes/{process_id}").status_code == 200
    terminal = client.post(f"/api/projects/{project_id}/terminal", json={"command": "printf terminal-ok"})
    assert terminal.status_code == 200
    assert "terminal-ok" in terminal.json()["output"]
    connector = client.post(f"/api/projects/{project_id}/connectors", json={"name": "test", "endpoint": "https://example.com/hook", "secret": "x" * 32})
    assert connector.status_code == 200
    connector_id = connector.json()["id"]
    body = json.dumps({"event": "push"}).encode()
    signature = "sha256=" + hmac.new(b"x" * 32, body, hashlib.sha256).hexdigest()
    response = client.post(f"/api/connectors/{connector_id}/webhook", content=body, headers={"X-OpenManus-Signature": signature})
    assert response.status_code == 200
    assert response.json()["accepted"] is True
