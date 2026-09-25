from __future__ import annotations

from fastapi.testclient import TestClient
import httpx
from openai import RateLimitError

from app.server import create_app


class CapturingLLM:
    captured = None

    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, stream=False):
        CapturingLLM.captured = messages
        return "I can see the attached image."


class RateLimitedLLM:
    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, stream=False):
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
        raise RateLimitError("quota exceeded", response=response, body={})


def make_project(client):
    return client.post("/api/projects", json={"name": "vision"}).json()


def test_chat_sends_uploaded_image_as_vision_content(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", CapturingLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    upload = client.post(
        f"/api/projects/{project['id']}/uploads",
        files={"file": ("diagram.png", b"fake-image-bytes", "image/png")},
    ).json()

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "What is in this image?", "attachment_ids": [upload["id"]]},
    )
    assert response.status_code == 200
    latest = CapturingLLM.captured[-1]
    assert latest["base64_image_mime"] == "image/png"
    assert latest["base64_image"]


def test_chat_returns_actionable_rate_limit_error(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", RateLimitedLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Hello"})
    assert response.status_code == 429
    assert "rate limit or quota" in response.json()["detail"]
