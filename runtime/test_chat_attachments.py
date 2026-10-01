from __future__ import annotations

from fastapi.testclient import TestClient
import httpx
from openai import RateLimitError

from app.server import create_app
from test_support import install_fake_controller


class CapturingLLM:
    captured = None
    model = "deepseek-r1:7b"

    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
        CapturingLLM.captured = messages
        return "DeepSeek direct response."


class RateLimitedLLM:
    model = "deepseek-r1:7b"

    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
        raise RateLimitError("quota exceeded", response=response, body={})


def make_project(client):
    return client.post("/api/projects", json={"name": "vision"}).json()


def test_deepseek_delegates_attached_image_to_vision_capability(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch, capability_ids=[131], route="execute")
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    upload = client.post(
        f"/api/projects/{project['id']}/uploads",
        files={"file": ("diagram.png", b"fake-image-bytes", "image/png")},
    ).json()
    client.app.state.orchestrator.start = lambda task: None

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "What is in this image?", "attachment_ids": [upload["id"]]},
    )
    assert response.status_code == 200
    assert response.json()["kind"] == "task"
    assert calls[0]["attachment_ids"] == [upload["id"]]
    assert calls[0]["context"]["current_attachments"][0]["filename"] == "diagram.png"
    assert upload["id"] in response.json()["task"]["plan"]["attachment_ids"]
    assert "gemma3" in {step["handler"] for step in response.json()["task"]["plan"]["control_unit_plan"]["steps"]}


def test_old_image_is_not_reattached_by_prompt_keywords_or_filename(tmp_path, monkeypatch):
    def choose_route(prompt, context):
        if any(item["filename"] == "diagram.png" for item in context.get("current_attachments", [])):
            return "execute", [131]
        return "respond", [167]

    calls = install_fake_controller(monkeypatch, selector=choose_route)
    monkeypatch.setattr("app.api.routes.LLM", CapturingLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    upload = client.post(
        f"/api/projects/{project['id']}/uploads",
        files={"file": ("diagram.png", b"fake-image-bytes", "image/png")},
    ).json()

    unrelated = client.post(f"/api/projects/{project['id']}/chat", json={"message": "What can you help me with?"})
    assert unrelated.status_code == 200
    assert calls[0]["context"]["current_attachments"] == []
    assert "base64_images" not in CapturingLLM.captured[-1]

    client.app.state.orchestrator.start = lambda task: None
    related = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Please describe the last uploaded image diagram.png."},
    )
    assert related.status_code == 200
    assert calls[1]["context"]["current_attachments"] == []
    assert calls[1]["attachment_ids"] == []
    assert related.json()["kind"] == "chat"
    assert "base64_images" not in CapturingLLM.captured[-1]

    # Only a deliberate current-message attachment makes the existing upload
    # visible to Gemma; text alone cannot reselect an older project's image.
    attached = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Please describe diagram.png.", "attachment_ids": [upload["id"]]},
    )
    assert attached.status_code == 200
    assert calls[2]["attachment_ids"] == [upload["id"]]
    assert attached.json()["kind"] == "task"


def test_multiple_current_images_and_question_are_sent_to_deepseek_only_when_attached(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch, capability_ids=[131], route="execute")
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    first = client.post(
        f"/api/projects/{project['id']}/uploads",
        files={"file": ("equation.png", b"first", "image/png")},
    ).json()
    second = client.post(
        f"/api/projects/{project['id']}/uploads",
        files={"file": ("diagram.png", b"second", "image/png")},
    ).json()
    client.app.state.orchestrator.start = lambda task: None
    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={
            "message": "Solve the math in the attached images and explain your reasoning.",
            "attachment_ids": [second["id"], first["id"]],
        },
    )
    assert response.status_code == 200
    assert calls[0]["prompt"] == "Solve the math in the attached images and explain your reasoning."
    assert calls[0]["attachment_ids"] == [second["id"], first["id"]]
    assert {item["filename"] for item in calls[0]["context"]["current_attachments"]} == {"equation.png", "diagram.png"}
    assert response.json()["task"]["plan"]["attachment_ids"] == [second["id"], first["id"]]


def test_chat_returns_actionable_rate_limit_error(tmp_path, monkeypatch):
    install_fake_controller(monkeypatch)
    monkeypatch.setattr("app.api.routes.LLM", RateLimitedLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Please answer this question"})
    assert response.status_code == 429
    assert "rate limit or quota" in response.json()["detail"]
