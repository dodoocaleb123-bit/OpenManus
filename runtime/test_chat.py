from __future__ import annotations

from fastapi.testclient import TestClient

from app.server import create_app


class FakeChatLLM:
    last_system = None

    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
        FakeChatLLM.last_system = system_msgs
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        return f"Gemini reply to: {user_messages[-1]} (turns={len(user_messages)})"


def test_project_chat_persists_messages_and_context(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()

    first = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Tell me something"})
    assert first.status_code == 200
    assert first.json()["assistant"]["content"] == "Gemini reply to: Tell me something (turns=1)"

    second = client.post(f"/api/projects/{project['id']}/chat", json={"message": "What did I say?"})
    assert second.status_code == 200
    assert second.json()["assistant"]["content"] == "Gemini reply to: What did I say? (turns=2)"

    history = client.get(f"/api/projects/{project['id']}/chat")
    assert history.status_code == 200
    assert [(m["role"], m["content"]) for m in history.json()] == [
        ("user", "Tell me something"),
        ("assistant", "Gemini reply to: Tell me something (turns=1)"),
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
    assert client.post(f"/api/projects/missing/chat", json={"message": "Hi"}).status_code == 404


def test_project_chat_includes_read_only_workspace_context(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "repo"}).json()
    workspace = tmp_path / "projects" / project["id"]
    (workspace / "README.md").write_text("Repository architecture overview", encoding="utf-8")
    (workspace / "app.py").write_text("print('hello')", encoding="utf-8")
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Explain the python architecture"})
    assert response.status_code == 200
    system = FakeChatLLM.last_system[0]["content"]
    assert "README.md" in system
    assert "Repository architecture overview" in system
    assert "app.py" in system


def test_project_chat_routes_change_requests_to_the_build_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "unified"}).json()
    app = client.app
    started = []
    app.state.orchestrator.start = lambda task: started.append(task)

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Please update the project README with setup instructions."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "task"
    assert payload["task"]["prompt"] == "Please update the project README with setup instructions."
    assert started and started[0].id == payload["task"]["id"]


def test_project_chat_requests_structured_mathjax_formatting(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "math"}).json()

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Solve this algebra problem step by step."},
    )
    assert response.status_code == 200
    system = FakeChatLLM.last_system[0]["content"]
    assert "step-by-step tutoring format" in system
    assert "\\(" in system and "\\[" in system
    assert "\\frac" in system
    assert "\\boxed" in system
    assert "### Step 1" in system


def test_project_chat_reports_model_configuration_problem(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: "LLM is not configured")
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Please answer this question"})
    assert response.status_code == 503
    assert response.json()["detail"] == "LLM is not configured"
