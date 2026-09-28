from __future__ import annotations

from fastapi.testclient import TestClient

from app.server import create_app
from app.api.routes import math_response_needs_repair, unified_capability_plan


class FakeChatLLM:
    last_system = None

    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
        FakeChatLLM.last_system = system_msgs
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        return f"Gemini reply to: {user_messages[-1]} (turns={len(user_messages)})"


class RepairingChatLLM(FakeChatLLM):
    calls = 0

    async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
        RepairingChatLLM.calls += 1
        if RepairingChatLLM.calls == 1:
            return r"### Step 1\n\[ y = \frac{2m + 15}{2"
        return r"### Step 1\n\[ y = \frac{2m + 15}{2} \]\n\n\boxed{m+7}"


class MultiPassRepairingChatLLM(FakeChatLLM):
    calls = 0

    async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
        MultiPassRepairingChatLLM.calls += 1
        if MultiPassRepairingChatLLM.calls < 4:
            return r"### Step 2\n\[ \frac{x^2+5x+6}{"
        return r"### Step 1\n\[ y = \frac{2m + 15}{2} \]\n\n### Step 2\n\[ \frac{x^2+5x+6}{2x+5} \]\n\n\boxed{B}"


class StreamingChatLLM(FakeChatLLM):
    async def ask(self, messages, system_msgs=None, stream=False, on_token=None, on_reset=None, **kwargs):
        response = "<think>private scratch text</think>Visible answer"
        for chunk in ("<think>private", " scratch text</think>", "Visible ", "answer"):
            if on_token is not None:
                result = on_token(chunk)
                if hasattr(result, "__await__"):
                    await result
        return response


def test_unified_capability_plan_combines_browser_vision_and_engineering():
    plan = unified_capability_plan(
        "Browse this website, inspect the screenshot, update the README, and run tests: https://example.com",
        has_attachments=True,
    )
    assert plan["intent"] == "multi_capability_task"
    assert set(plan["capabilities"]) == {"vision", "research_browser", "engineering"}
    assert plan["requires_task"] is True
    assert len(plan["steps"]) >= 4


def test_project_chat_persists_messages_and_context(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()

    first = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Tell me something"})
    assert first.status_code == 200
    assert first.json()["assistant"]["content"] == "Gemini reply to: Tell me something (turns=1)"
    assert isinstance(first.json()["assistant"]["response_time_ms"], int)
    assert first.json()["assistant"]["response_time_ms"] >= 0

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
    assert isinstance(history.json()[1]["response_time_ms"], int)

    # A fresh application instance reads the same SQLite conversation.
    fresh = TestClient(create_app(tmp_path, check_llm=False))
    assert len(fresh.get(f"/api/projects/{project['id']}/chat").json()) == 4


def test_project_chat_can_stream_visible_deltas_and_persist_final_reply(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", StreamingChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "stream-chat"}).json()

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Tell me something"},
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: delta" in response.text
    assert "private scratch text" not in response.text
    history = client.get(f"/api/projects/{project['id']}/chat").json()
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "Visible answer"
    assert isinstance(history[-1]["response_time_ms"], int)


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
    assert payload["plan"]["requires_task"] is True
    assert "engineering" in payload["plan"]["capabilities"]
    assert payload["task"]["prompt"] == "Please update the project README with setup instructions."
    assert payload["task"]["plan"]["intent"] == "engineering_task"
    assert isinstance(payload["assistant"]["response_time_ms"], int)
    assert payload["assistant"]["response_time_ms"] >= 0
    assert started and started[0].id == payload["task"]["id"]
    events = client.get(f"/api/tasks/{payload['task']['id']}/events/history").json()
    assert events[0]["type"] == "task.planned"


def test_project_chat_acknowledges_browser_research_without_promising_changes(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "browser-research"}).json()
    client.app.state.orchestrator.start = lambda task: None

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Go through this website and tell me what is in there https://example.com"},
    )
    assert response.status_code == 200
    acknowledgement = response.json()["assistant"]["content"]
    assert "summarize what I find" in acknowledgement
    assert "won’t modify the project" in acknowledgement


def test_project_chat_confirms_browsing_capability(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "browse"}).json()

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Can you browse on the internet?"},
    )
    assert response.status_code == 200
    answer = response.json()["assistant"]["content"]
    assert answer.startswith("Yes.")
    assert "shared browser" in answer
    assert "do not browse automatically" in answer


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
    assert "exact polished tutoring style" in system
    assert "### Given:" in system
    assert "### Step 1:" in system
    assert "$$...$$" in system
    assert "---" in system
    assert "Absolutely!" in system
    assert "\\frac" in system
    assert "\\boxed" in system
    assert "Do not begin with generic wording" in system


def test_math_response_repair_detects_unclosed_latex():
    assert math_response_needs_repair(r"\[ y = \frac{2m+15}{2") is True
    assert math_response_needs_repair(r"\[ y = \frac{2m+15}{2} \]") is False


def test_project_chat_replaces_incomplete_math_answer(tmp_path, monkeypatch):
    RepairingChatLLM.calls = 0
    monkeypatch.setattr("app.api.routes.LLM", RepairingChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "math-repair"}).json()

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Solve this algebra equation step by step."},
    )
    assert response.status_code == 200
    answer = response.json()["assistant"]["content"]
    assert RepairingChatLLM.calls == 2
    assert answer.startswith("### Step 1")
    assert r"\frac{2m + 15}{2}" in answer
    assert r"\boxed{m+7}" in answer
    assert answer.count(r"\[ y") == 1


def test_project_chat_allows_multiple_math_repair_passes(tmp_path, monkeypatch):
    MultiPassRepairingChatLLM.calls = 0
    monkeypatch.setattr("app.api.routes.LLM", MultiPassRepairingChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "long-math"}).json()

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Solve both algebra questions in the attached images step by step."},
    )
    assert response.status_code == 200
    assert MultiPassRepairingChatLLM.calls == 4
    answer = response.json()["assistant"]["content"]
    assert r"\frac{x^2+5x+6}{2x+5}" in answer
    assert r"\boxed{B}" in answer


def test_project_chat_reports_model_configuration_problem(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: "LLM is not configured")
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Please answer this question"})
    assert response.status_code == 503
    assert response.json()["detail"] == "LLM is not configured"
