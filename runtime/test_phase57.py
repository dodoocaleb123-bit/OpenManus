from pathlib import Path

from fastapi.testclient import TestClient

from app.platform.repository_map import build_repository_map
from app.platform.sources import extract_sources
from app.server import create_app


def test_irreversible_api_actions_require_confirmation(tmp_path: Path):
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "demo"}).json()
    response = client.delete(f"/api/projects/{project['id']}")
    assert response.status_code == 428
    response = client.delete(
        f"/api/projects/{project['id']}",
        headers={"X-OpenManus-Confirm": "true"},
    )
    assert response.status_code == 200


def test_repository_map_is_bounded_and_reports_python_symbols(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "app.py").write_text("class Demo:\n    pass\n\ndef run():\n    return 1\n", encoding="utf-8")
    (project_root / ".env").write_text("SECRET=do-not-read", encoding="utf-8")
    result = build_repository_map(project_root)
    assert result["file_count"] == 1
    assert result["languages"] == {"python": 1}
    assert result["files"][0]["symbols"] == {"classes": ["Demo"], "functions": ["run"]}


def test_sources_are_structured_and_deduplicated():
    evidence = {
        "tool_results": [
            {"evidence": {"url": "https://example.com/pricing", "title": "Pricing"}},
            {"evidence": {"text": "See https://example.com/pricing and https://docs.example.com/guide."}},
        ]
    }
    sources = extract_sources(evidence)
    assert [item["url"] for item in sources] == [
        "https://example.com/pricing",
        "https://docs.example.com/guide",
    ]
    assert sources[0]["title"] == "Pricing"


def test_metrics_and_evidence_include_sources(tmp_path: Path):
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "metrics"}).json()
    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["projects"] == 1
    assert "task_statuses" in metrics.json()
    task = client.post(
        "/api/tasks",
        json={"project_id": project["id"], "prompt": "research https://example.com"},
    ).json()
    evidence = client.get(f"/api/tasks/{task['id']}/evidence")
    assert evidence.status_code == 200
    assert "sources" in evidence.json()
