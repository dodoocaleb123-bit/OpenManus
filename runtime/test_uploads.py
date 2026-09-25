from __future__ import annotations

from fastapi.testclient import TestClient

from app.server import create_app


def test_upload_list_download_and_persist(tmp_path):
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "uploads"}).json()

    response = client.post(
        f"/api/projects/{project['id']}/uploads",
        files={"file": ("meeting notes.pdf", b"%PDF-test", "application/pdf")},
    )
    assert response.status_code == 200
    uploaded = response.json()
    assert uploaded["filename"] == "meeting notes.pdf"
    assert uploaded["size"] == 9
    assert (tmp_path / "projects" / project["id"] / "uploads").exists()

    files = client.get(f"/api/projects/{project['id']}/uploads")
    assert files.status_code == 200
    assert files.json()[0]["id"] == uploaded["id"]

    download = client.get(f"/api/projects/{project['id']}/uploads/{uploaded['id']}")
    assert download.status_code == 200
    assert download.content == b"%PDF-test"
    assert "meeting%20notes.pdf" in download.headers["content-disposition"]

    fresh = TestClient(create_app(tmp_path, check_llm=False))
    assert len(fresh.get(f"/api/projects/{project['id']}/uploads").json()) == 1


def test_upload_rejects_unsupported_extension_and_path_traversal(tmp_path):
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "uploads"}).json()

    unsupported = client.post(
        f"/api/projects/{project['id']}/uploads",
        files={"file": ("malware.exe", b"no", "application/octet-stream")},
    )
    assert unsupported.status_code == 415

    safe_name = client.post(
        f"/api/projects/{project['id']}/uploads",
        files={"file": ("../../notes.txt", b"safe", "text/plain")},
    )
    assert safe_name.status_code == 200
    assert safe_name.json()["filename"] == "notes.txt"
    assert not (tmp_path / "notes.txt").exists()
