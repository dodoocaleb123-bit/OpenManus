from __future__ import annotations

from fastapi.testclient import TestClient

from app.server import create_app


class FakeChatLLM:
    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, stream=False):
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        return f"Gemini reply to: {user_messages[-1]} (turns={len(user_messages)})"


def test_project_chat_persists_messages_and_context(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()

    first = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Hello"})
    assert first.status_code == 200
    assert first.json()["assistant"]["content"] == "Gemini reply to: Hello (turns=1)"

    second = client.post(f"/api/projects/{project['id']}/chat", json={"message": "What did I say?"})
    assert second.status_code == 200
    assert second.json()["assistant"]["content"] == "Gemini reply to: What did I say? (turns=2)"

    history = client.get(f"/api/projects/{project['id']}/chat")
    assert history.status_code == 200
    assert [(m["role"], m["content"]) for m in history.json()] == [
        ("user", "Hello"),
        ("assistant", "Gemini reply to: Hello (turns=1)"),
        ("user", "What did I say?"),
        ("assistant", "Gemini reply to: What did I say? (turns=2)"),
    ]

    # A fresh application instance reads the same SQLite conversation.
    fresh = TestClient(create_app(tmp_path, check_llm=False))
    assert len(fresh.get(f"/api/projects/{project['id']}/chat").json()) == 4


def test_project_chat_rejects_unknown_project(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    assert client.get("/api/projects/missing/chat").status_code == 404
    assert client.post("/api/projects/missing/chat", json={"message": "Hi"}).status_code == 404


def test_project_chat_reports_model_configuration_problem(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: "LLM is not configured")
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Hello"})
    assert response.status_code == 503
    assert response.json()["detail"] == "LLM is not configured"
