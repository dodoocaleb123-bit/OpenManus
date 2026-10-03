from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.platform.models import TaskStatus
from app.platform.store import PlatformStore
from app.platform.verification import FailureClassifier, VerificationEngine, stable_operation_id
from app.server import create_app
from test_support import install_fake_controller


def test_verification_engine_requires_workspace_and_validation(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "README.md").write_text("ok", encoding="utf-8")
    result = VerificationEngine.task_completion(
        workspace,
        {"passed": True, "results": [{"command": "pytest", "ok": True}]},
    )
    assert result["passed"] is True
    assert any(check["check"] == "workspace" for check in result["checks"])

    failed = VerificationEngine.task_completion(workspace, {"passed": False, "results": []})
    assert failed["passed"] is False


def test_failure_classifier_and_operation_ids_are_stable():
    assert FailureClassifier.classify("provider quota exceeded") == "provider_failure"
    assert FailureClassifier.classify("browser target closed") == "browser_failure"
    assert FailureClassifier.policy("provider_failure", 1)["retryable"] is True
    assert FailureClassifier.policy("provider_failure", 3)["retryable"] is False
    assert stable_operation_id("task-1", "push", {"branch": "main"}) == stable_operation_id("task-1", "push", {"branch": "main"})


def test_store_persists_task_evidence_and_checkpoints(tmp_path):
    async def scenario():
        store = PlatformStore(tmp_path)
        project = store.create_project("evidence")
        task = store.create_task(project.id, "verify the workspace")
        task.evidence = {"verification": {"passed": True}}
        task.artifacts = [{"path": "README.md", "kind": "document"}]
        task.recovery = {"category": "provider_failure", "retryable": True}
        task.checkpoint = "validated"
        await store.save_task(task)
        await store.save_checkpoint(task.id, "validated", {"passed": True})
        loaded = store.get_task(task.id)
        assert loaded is not None
        assert loaded.evidence["verification"]["passed"] is True
        assert loaded.artifacts[0]["path"] == "README.md"
        assert loaded.recovery["retryable"] is True
        assert store.list_checkpoints(task.id)[0]["name"] == "validated"

    asyncio.run(scenario())


def test_task_evidence_endpoint_exposes_persisted_contract(tmp_path, monkeypatch):
    install_fake_controller(monkeypatch, handlers=["qwen_coder"])
    client = TestClient(create_app(tmp_path, check_llm=False))
    client.app.state.orchestrator.start = lambda task: None
    project = client.post("/api/projects", json={"name": "evidence-api"}).json()
    task = client.post("/api/tasks", json={"project_id": project["id"], "prompt": "create a file named note.txt containing hello"}).json()
    response = client.get(f"/api/tasks/{task['id']}/evidence")
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task["id"]
    assert "evidence" in payload and "recovery" in payload
