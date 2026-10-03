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


def test_model_authored_workflow_lists_only_selected_handlers_and_tools():
    steps = [
        {"step_id": "research", "handler": "qwen2.5_3b", "objective": "Research the question", "tools": ["web_search"]},
        {"step_id": "design", "handler": "llama3.2_3b", "objective": "Create a design brief", "depends_on": ["research"], "tools": []},
        {"step_id": "build", "handler": "qwen_coder", "objective": "Build the result", "depends_on": ["research", "design"], "tools": ["bash", "str_replace_editor"]},
    ]
    plan = ControlUnit().build_workflow("opaque user request", steps)
    assert {step["handler"] for step in plan["steps"]} >= {"qwen2.5_3b", "llama3.2_3b", "qwen_coder"}
    assert plan["steps"][-1]["tools"] == ["bash", "str_replace_editor"]
    assert "capability_id" not in str(plan)


def test_specialist_capability_owners_are_explicit_model_roles():
    assert ROLES["planner"].model_config_name == "reasoning"
    assert ROLES["researcher"].model_config_name == "research"
    assert ROLES["designer"].model_config_name == "creativity"
    assert ROLES["coder"].model_config_name == "heavy_coding"


def test_gemma_browser_review_can_depend_on_the_qwen_build():
    plan = ControlUnit().build_workflow("Build and visually inspect a web app", [
        {"step_id": "build", "handler": "qwen_coder", "objective": "Build the web app", "depends_on": [], "tools": ["bash", "str_replace_editor", "platform_browser"]},
        {"step_id": "visual-review", "handler": "gemma3", "objective": "Review the rendered app", "depends_on": ["build"], "tools": ["platform_browser"]},
    ])
    by_id = {step["step_id"]: step for step in plan["steps"]}
    assert by_id["visual-review"]["handler"] == "gemma3"
    assert "platform_browser" in by_id["visual-review"]["tools"]
    assert "build" in by_id["visual-review"]["depends_on"]


def test_user_handoff_respects_deepseek_selected_order_around_coding():
    before = ControlUnit().build_workflow("Take over, then build", [
        {"step_id": "user", "handler": "user", "objective": "Take over the browser", "depends_on": [], "tools": ["ask_human"]},
        {"step_id": "build", "handler": "qwen_coder", "objective": "Build after the user action", "depends_on": ["user"], "tools": ["bash"]},
    ])
    assert before["steps"][1]["depends_on"] == ["user"]

    after = ControlUnit().build_workflow("Build, then let the user take over", [
        {"step_id": "build", "handler": "qwen_coder", "objective": "Build the first stage", "depends_on": [], "tools": ["bash"]},
        {"step_id": "user", "handler": "user", "objective": "Verify the first stage", "depends_on": ["build"], "tools": ["ask_human"]},
    ])
    assert after["steps"][1]["depends_on"] == ["build"]


def test_every_scheduled_prompt_becomes_a_deepseek_planned_task(tmp_path, monkeypatch):
    from app.platform.control_unit import ControlUnit

    monkeypatch.setitem(config.llm, "reasoning", config.llm["default"])
    monkeypatch.setattr("app.server.reasoning_enabled", lambda: True)
    seen = []

    async def fake_plan(llm, *, prompt, attachment_ids=None, browser_session_id=None, context=None):
        seen.append((prompt, context))
        steps = [{"step_id": "build", "handler": "qwen_coder", "objective": "Run the selected project work", "tools": ["bash", "python_execute", "str_replace_editor"]}] if prompt == "run the selected work" else []
        plan = ControlUnit().build_workflow(prompt, steps)
        plan.update(needs_workspace_context=False, summary="Selected by fake DeepSeek", intent="test", selected_handlers=sorted({step["handler"] for step in steps}), selected_tools=sorted({tool for step in steps for tool in step["tools"]}), deepseek={"model": "deepseek-test", "intent": "test"})
        return plan

    monkeypatch.setattr("app.server.make_authoritative_plan", fake_plan)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "scheduled"}).json()
    started = []
    client.app.state.orchestrator.start = started.append

    import asyncio

    direct = asyncio.run(client.app.state.scheduler.launch(project["id"], "just explain this"))
    assert direct.plan["planner"] == "deepseek"
    assert "route" not in direct.plan
    assert direct.prompt == "just explain this"
    created = asyncio.run(client.app.state.scheduler.launch(project["id"], "run the selected work"))
    assert created.plan["control_unit_plan"]["authoritative"] is True
    assert "route" not in created.plan
    assert started == [direct, created]
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
    install_fake_controller(monkeypatch, handlers=["qwen_coder"])
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
