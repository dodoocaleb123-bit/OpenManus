from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.platform.phase_d import evaluate_trajectory, normalize_policy, validate_plugin_manifest
from app.platform.store import PlatformStore
from app.server import create_app
from test_support import install_fake_controller


def test_phase_d_policy_and_manifest_validation():
    policy = normalize_policy({"max_steps": "8", "blocked_tools": ["bash"]})
    assert policy["max_steps"] == 8
    assert policy["blocked_tools"] == ["bash"]
    manifest = validate_plugin_manifest({"name": "safe-tool", "version": "1.0.0", "capabilities": ["search"], "permissions": ["network_read"]})
    assert manifest["manifest_hash"] and manifest["enabled"] is False
    try:
        validate_plugin_manifest({"name": "unsafe-tool", "version": "1.0.0", "permissions": ["execute_code"]})
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("unsafe plugin permission was accepted")


def test_phase_d_store_initializes_owner_policy_and_registry(tmp_path: Path):
    store = PlatformStore(tmp_path / "data")
    project = store.create_project("demo")
    assert store.get_project_policy(project.id)["policy"]["enabled"] is True
    assert store.get_project_role(project.id, "admin") == "owner"
    assert store.list_model_capabilities()
    plugin = store.install_plugin_manifest({"name": "safe-tool", "version": "1.0.0", "capabilities": ["search"], "permissions": []})
    assert store.list_plugins()[0]["manifest_hash"] == plugin["manifest_hash"]


def test_phase_d_api_and_evaluation(tmp_path: Path, monkeypatch):
    install_fake_controller(monkeypatch, capability_ids=[1])
    app = create_app(tmp_path, check_llm=False)
    with TestClient(app) as client:
        client.app.state.orchestrator.start = lambda task: None
        project = client.post("/api/projects", json={"name": "phase-d"}).json()
        assert client.get("/api/capabilities/models").status_code == 200
        assert client.post("/api/plugins", json={"manifest": {"name": "safe-tool", "version": "1.0.0", "capabilities": [], "permissions": []}}).status_code == 200
        assert client.get(f"/api/projects/{project['id']}/policy").status_code == 200
        assert client.put(f"/api/projects/{project['id']}/policy", json={"policy": {"max_steps": 8}}).status_code == 200
        assert client.put(f"/api/projects/{project['id']}/members", json={"user_id": "reviewer", "role": "viewer"}).status_code == 200
        task = client.post("/api/tasks", json={"project_id": project["id"], "prompt": "create a file"}).json()
        assert task["project_id"] == project["id"]
        evaluation = client.post(f"/api/tasks/{task['id']}/evaluation")
        assert evaluation.status_code == 200
        assert "score" in evaluation.json()


def test_phase_d_team_auth_and_viewer_role(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLATFORM_REQUIRE_AUTH", "true")
    monkeypatch.setenv("PLATFORM_USERNAME", "admin")
    monkeypatch.setenv("PLATFORM_PASSWORD", "admin-secret")
    monkeypatch.setenv("PLATFORM_USERS_JSON", json.dumps({"bob": "bob-secret"}))
    app = create_app(tmp_path, check_llm=False)
    admin = {"Authorization": "Basic " + base64.b64encode(b"admin:admin-secret").decode()}
    bob = {"Authorization": "Basic " + base64.b64encode(b"bob:bob-secret").decode()}
    with TestClient(app) as client:
        project = client.post("/api/projects", json={"name": "team"}, headers=admin).json()
        assert client.put(f"/api/projects/{project['id']}/members", json={"user_id": "bob", "role": "viewer"}, headers=admin).status_code == 200
        assert client.get(f"/api/projects/{project['id']}/policy", headers=bob).status_code == 200
        assert client.post("/api/tasks", json={"project_id": project["id"], "prompt": "edit a file"}, headers=bob).status_code == 403


def test_trajectory_score_is_advisory(tmp_path: Path):
    store = PlatformStore(tmp_path / "data")
    project = store.create_project("score")
    task = store.create_task(project.id, "test")
    task.status = "succeeded"
    task.validation = {"passed": True}
    result = evaluate_trajectory(task, [{"name": "validated"}], {"rating": 1})
    assert result["success"] is True
    assert result["learning_safe"] is True
