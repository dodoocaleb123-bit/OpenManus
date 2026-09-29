from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.platform.automation import AutomationStore, verify_webhook
from app.server import create_app
from app.platform.specialists import bounded_gather, select_specialists


def test_specialist_selection_matches_multimodal_task():
    roles = {item["name"] for item in select_specialists(intent="coding", complexity="heavy", browser=True, has_images=True, requires_artifacts=True)}
    assert {"planner", "researcher", "coder", "vision", "verifier", "artifact"} <= roles


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


def test_phase_c_routes_and_process_manager(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
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
    connector = client.post(f"/api/projects/{project_id}/connectors", json={"name": "test", "endpoint": "https://example.com/hook", "secret": "x" * 32})
    assert connector.status_code == 200
    connector_id = connector.json()["id"]
    body = json.dumps({"event": "push"}).encode()
    signature = "sha256=" + hmac.new(b"x" * 32, body, hashlib.sha256).hexdigest()
    response = client.post(f"/api/connectors/{connector_id}/webhook", content=body, headers={"X-OpenManus-Signature": signature})
    assert response.status_code == 200
    assert response.json()["accepted"] is True
