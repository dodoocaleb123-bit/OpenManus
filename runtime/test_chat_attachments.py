from __future__ import annotations

import httpx
from fastapi.testclient import TestClient
from openai import RateLimitError

from app.server import create_app
from test_support import install_fake_controller, stub_orchestrator_start


def make_project(client):
    project = client.post("/api/projects", json={"name": "vision"}).json()
    stub_orchestrator_start(client)
    return project


def upload(client, project_id, filename, data=b"fake-image-bytes"):
    return client.post(
        f"/api/projects/{project_id}/uploads",
        files={"file": (filename, data, "image/png")},
    ).json()


def test_deepseek_delegates_attached_image_to_vision_model(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch, handlers=["gemma3"])
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    image = upload(client, project["id"], "diagram.png")
    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "What is in this image?", "attachment_ids": [image["id"]]},
    )
    assert response.status_code == 200
    assert response.json()["kind"] == "task"
    assert calls[0]["attachment_ids"] == [image["id"]]
    assert calls[0]["context"]["current_attachments"][0]["filename"] == "diagram.png"
    assert image["id"] in response.json()["task"]["plan"]["attachment_ids"]
    steps = response.json()["task"]["plan"]["control_unit_plan"]["steps"]
    assert {step["handler"] for step in steps} == {"gemma3"}


def test_old_image_is_not_reattached_by_prompt_keywords_or_filename(tmp_path, monkeypatch):
    def choose(prompt, context):
        if any(item["filename"] == "diagram.png" for item in context.get("current_attachments", [])):
            return ["gemma3"]
        return []

    calls = install_fake_controller(monkeypatch, selector=choose)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    image = upload(client, project["id"], "diagram.png")

    unrelated = client.post(f"/api/projects/{project['id']}/chat", json={"message": "What can you help me with?"})
    assert unrelated.status_code == 200
    assert unrelated.json()["kind"] == "task"
    assert calls[0]["context"]["current_attachments"] == []

    related = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Please describe the last uploaded image diagram.png."},
    )
    assert related.status_code == 200
    assert related.json()["kind"] == "task"
    assert calls[1]["context"]["current_attachments"] == []
    assert calls[1]["attachment_ids"] == []

    attached = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Please describe diagram.png.", "attachment_ids": [image["id"]]},
    )
    assert attached.status_code == 200
    assert calls[2]["attachment_ids"] == [image["id"]]
    assert attached.json()["kind"] == "task"


def test_multiple_current_images_and_question_reach_deepseek_as_one_workflow(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch, handlers=["gemma3"])
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)
    first = upload(client, project["id"], "equation.png", b"first")
    second = upload(client, project["id"], "diagram.png", b"second")
    text = "Solve the math in the attached images and explain your reasoning."
    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": text, "attachment_ids": [second["id"], first["id"]]},
    )
    assert response.status_code == 200
    assert calls[0]["prompt"] == text
    assert calls[0]["attachment_ids"] == [second["id"], first["id"]]
    assert {item["filename"] for item in calls[0]["context"]["current_attachments"]} == {"equation.png", "diagram.png"}
    assert response.json()["task"]["plan"]["attachment_ids"] == [second["id"], first["id"]]
    handlers = {step["handler"] for step in response.json()["task"]["plan"]["control_unit_plan"]["steps"]}
    assert handlers == {"gemma3"}


def test_chat_planner_returns_actionable_rate_limit_error(tmp_path, monkeypatch):
    install_fake_controller(monkeypatch)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = make_project(client)

    async def rate_limited(*args, **kwargs):
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
        raise RateLimitError("quota exceeded", response=response, body={})

    monkeypatch.setattr("app.api.routes.make_authoritative_plan", rate_limited)
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Please answer this question"})
    assert response.status_code == 429
    assert "rate limit or quota" in response.json()["detail"]
