from __future__ import annotations

from fastapi.testclient import TestClient

from app.server import create_app
from test_support import install_fake_controller


class FakeChatLLM:
    last_system = None
    last_messages = None
    model = "deepseek-r1:7b"

    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
        FakeChatLLM.last_system = system_msgs
        FakeChatLLM.last_messages = messages
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        return f"DeepSeek reply to: {user_messages[-1]} (turns={len(user_messages)})"


class StreamingChatLLM(FakeChatLLM):
    async def ask(self, messages, system_msgs=None, stream=False, on_token=None, on_reset=None, **kwargs):
        response = "<think>private scratch text</think>Visible answer"
        for chunk in ("<think>private", " scratch text</think>", "Visible ", "answer"):
            if on_token is not None:
                result = on_token(chunk)
                if hasattr(result, "__await__"):
                    await result
        return response


def test_project_chat_persists_messages_and_sends_verbatim_user_text_to_deepseek(tmp_path, monkeypatch):
    controller_calls = install_fake_controller(monkeypatch)
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()

    first_text = "Hi OpenManus! Can you explain my app?"
    first = client.post(f"/api/projects/{project['id']}/chat", json={"message": first_text})
    assert first.status_code == 200
    assert controller_calls[0]["prompt"] == first_text
    assert first.json()["assistant"]["content"] == f"DeepSeek reply to: {first_text} (turns=1)"
    assert isinstance(first.json()["assistant"]["response_time_ms"], int)
    assert first.json()["assistant"]["response_time_ms"] >= 0

    second_text = "What did I say?"
    second = client.post(f"/api/projects/{project['id']}/chat", json={"message": second_text})
    assert second.status_code == 200
    assert controller_calls[1]["prompt"] == second_text
    assert second.json()["assistant"]["content"] == f"DeepSeek reply to: {second_text} (turns=2)"

    history = client.get(f"/api/projects/{project['id']}/chat")
    assert history.status_code == 200
    assert [(m["role"], m["content"]) for m in history.json()] == [
        ("user", first_text),
        ("assistant", f"DeepSeek reply to: {first_text} (turns=1)"),
        ("user", second_text),
        ("assistant", f"DeepSeek reply to: {second_text} (turns=2)"),
    ]
    assert isinstance(history.json()[1]["response_time_ms"], int)
    fresh = TestClient(create_app(tmp_path, check_llm=False))
    assert len(fresh.get(f"/api/projects/{project['id']}/chat").json()) == 4


def test_greeting_uses_real_controller_path_and_deepseek_answer(tmp_path, monkeypatch):
    from app.config import config

    monkeypatch.setitem(config.llm, "reasoning", config.llm["default"])
    monkeypatch.setattr("app.api.routes.reasoning_enabled", lambda: True)

    class DeepSeekGreeting:
        model = "deepseek-r1:7b"
        base_url = "http://host.docker.internal:11434/v1"
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
            self.calls.append((messages, system_msgs, kwargs))
            if kwargs.get("response_format"):
                return '{"route":"respond","needs_workspace_context":false,"summary":"Greet the user","intent":"conversation","capability_ids":[166],"handlers":[],"rationale":[]}'
            return "Hello! How can I help?"

    monkeypatch.setattr("app.api.routes.LLM", DeepSeekGreeting)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "greeting"}).json()
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Hellooo"})
    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "chat"
    assert response.json()["assistant"]["content"] == "Hello! How can I help?"
    assert len(DeepSeekGreeting.calls) == 2
    assert DeepSeekGreeting.calls[0][0] == [{"role": "user", "content": "Hellooo"}]
    assert "Executable capability directory" not in DeepSeekGreeting.calls[0][1][0]["content"]


def test_project_chat_can_stream_visible_deltas_and_persist_final_reply(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch)
    monkeypatch.setattr("app.api.routes.LLM", StreamingChatLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "stream-chat"}).json()
    text = "Tell me something"

    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": text},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert calls[0]["prompt"] == text
    assert "event: delta" in response.text
    assert "private scratch text" not in response.text
    history = client.get(f"/api/projects/{project['id']}/chat").json()
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "Visible answer"
    assert isinstance(history[-1]["response_time_ms"], int)


def test_project_chat_replay_after_stream_disconnect_is_idempotent(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch)
    generated = []

    class CountingChatLLM(FakeChatLLM):
        async def ask(self, messages, system_msgs=None, stream=False, **kwargs):
            generated.append(1)
            return "A durable reply"

    monkeypatch.setattr("app.api.routes.LLM", CountingChatLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "replay-chat"}).json()
    headers = {"Accept": "text/event-stream", "Idempotency-Key": "chat-replay-1"}
    text = "Tell me a short greeting"
    first = client.post(f"/api/projects/{project['id']}/chat", json={"message": text}, headers=headers)
    assert first.status_code == 200
    second = client.post(f"/api/projects/{project['id']}/chat", json={"message": text}, headers={"Idempotency-Key": "chat-replay-1"})
    assert second.status_code == 200
    assert second.json()["assistant"]["content"] == "A durable reply"
    assert len(calls) == 1
    assert len(generated) == 1
    history = client.get(f"/api/projects/{project['id']}/chat").json()
    assert [(item["role"], item["content"]) for item in history] == [("user", text), ("assistant", "A durable reply")]


def test_project_chat_rejects_unknown_project(tmp_path, monkeypatch):
    install_fake_controller(monkeypatch)
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    assert client.get("/api/projects/missing/chat").status_code == 404
    assert client.post("/api/projects/missing/chat", json={"message": "Hi"}).status_code == 404


def test_project_chat_includes_workspace_context_only_when_deepseek_selects_it(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch, needs_workspace_context=True)
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "repo"}).json()
    workspace = tmp_path / "projects" / project["id"]
    (workspace / "README.md").write_text("Repository architecture overview", encoding="utf-8")
    (workspace / "app.py").write_text("print('hello')", encoding="utf-8")
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Explain the python architecture"})
    assert response.status_code == 200
    assert calls[0]["prompt"] == "Explain the python architecture"
    system = FakeChatLLM.last_system[0]["content"]
    assert "README.md" in system
    assert "Repository architecture overview" in system
    assert "app.py" in system


def test_deepseek_routes_explicit_execution_request_to_the_capability_agent(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch, capability_ids=[1], route="execute")
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "unified"}).json()
    started = []
    client.app.state.orchestrator.start = lambda task: started.append(task)
    text = "Please update the project README with setup instructions."

    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": text})
    assert response.status_code == 200
    payload = response.json()
    assert calls[0]["prompt"] == text
    assert payload["kind"] == "task"
    assert payload["plan"]["route"] == "execute"
    assert "requires_task" not in payload["plan"]
    assert payload["plan"]["route"] == "execute"
    assert "qwen_coder" in {step["handler"] for step in payload["task"]["plan"]["control_unit_plan"]["steps"]}
    assert payload["task"]["prompt"] == text
    assert payload["task"]["plan"]["intent"] == "test"
    assert isinstance(payload["assistant"]["response_time_ms"], int)
    assert started and started[0].id == payload["task"]["id"]
    events = client.get(f"/api/tasks/{payload['task']['id']}/events/history").json()
    assert events[0]["type"] == "task.planned"


def test_deepseek_selected_research_task_acknowledges_no_project_mutation(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch, capability_ids=[147], route="execute")
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "browser-research"}).json()
    client.app.state.orchestrator.start = lambda task: None
    text = "Go through this website and tell me what is in there https://example.com"
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": text})
    assert response.status_code == 200
    assert calls[0]["prompt"] == text
    acknowledgement = response.json()["assistant"]["content"]
    assert "summarize what I find" in acknowledgement
    assert "won’t modify the project" in acknowledgement


def test_capability_question_is_not_answered_by_a_deterministic_route_shortcut(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch, capability_ids=[167], route="respond")
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "browse"}).json()
    text = "Can you browse on the internet?"
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": text})
    assert response.status_code == 200
    assert calls[0]["prompt"] == text
    assert response.json()["assistant"]["content"].startswith("DeepSeek reply to:")


def test_project_chat_passes_math_questions_without_a_keyword_specific_prompt_or_retry(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch)
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "math"}).json()
    text = "  Solve this algebra problem step by step.  "
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": text})
    assert response.status_code == 200
    assert calls[0]["prompt"] == text
    assert FakeChatLLM.last_messages[-1]["content"] == text
    system = FakeChatLLM.last_system[0]["content"]
    assert "exact polished tutoring style" not in system
    assert "math classifier" not in system.lower()


def test_project_chat_fails_closed_when_deepseek_is_unavailable(tmp_path, monkeypatch):
    install_fake_controller(monkeypatch)
    monkeypatch.delitem(__import__("app.config", fromlist=["config"]).config.llm, "reasoning", raising=False)
    monkeypatch.setattr("app.api.routes.reasoning_enabled", lambda: False)
    client = TestClient(create_app(tmp_path, check_llm=False))
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Please answer this question"})
    assert response.status_code == 503
    assert "DeepSeek is required" in response.json()["detail"]
