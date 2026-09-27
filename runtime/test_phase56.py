import base64

import pytest
from fastapi.testclient import TestClient

from app.server import create_app


def test_required_auth_fails_closed_without_password(monkeypatch, tmp_path):
    monkeypatch.setenv("PLATFORM_REQUIRE_AUTH", "true")
    monkeypatch.delenv("PLATFORM_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="PLATFORM_PASSWORD"):
        create_app(tmp_path, check_llm=False)


def test_auth_and_security_headers(monkeypatch, tmp_path):
    monkeypatch.setenv("PLATFORM_PASSWORD", "test-secret")
    monkeypatch.setenv("PLATFORM_USERNAME", "admin")
    monkeypatch.setenv("PLATFORM_REQUIRE_AUTH", "true")
    app = create_app(tmp_path, check_llm=False)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.headers["x-content-type-options"] == "nosniff"
        assert health.headers["x-frame-options"] == "DENY"

        protected = client.get("/api/projects")
        assert protected.status_code == 401
        token = base64.b64encode(b"admin:test-secret").decode()
        authorized = client.get("/api/projects", headers={"Authorization": f"Basic {token}"})
        assert authorized.status_code == 200
        assert authorized.headers["cache-control"] == "no-store"
