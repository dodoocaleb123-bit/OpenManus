from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import config
from app.server import create_app
from test_support import install_fake_controller, stub_orchestrator_start


def _client(tmp_path, monkeypatch, **controller_options):
    calls = install_fake_controller(monkeypatch, **controller_options)
    client = TestClient(create_app(tmp_path, check_llm=False))
    started = stub_orchestrator_start(client)
    return client, calls, started


def test_every_project_message_creates_a_task_and_is_sent_verbatim_to_deepseek(tmp_path, monkeypatch):
    client, controller_calls, started = _client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()

    first_text = "Hi OpenManus! Can you explain my app?"
    first = client.post(f"/api/projects/{project['id']}/chat", json={"message": first_text})
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["kind"] == "task"
    assert first_payload["task"]["prompt"] == first_text
    assert controller_calls[0]["prompt"] == first_text
    assert "route" not in first_payload["plan"]
    assert first_payload["task"]["plan"]["control_unit_plan"]["steps"] == []
    assert first_payload["assistant"]["content"]

    second_text = "What did I say?"
    second = client.post(f"/api/projects/{project['id']}/chat", json={"message": second_text})
    assert second.status_code == 200
    assert second.json()["kind"] == "task"
    assert second.json()["task"]["prompt"] == second_text
    assert controller_calls[1]["prompt"] == second_text
    assert len(started) == 2

    history = client.get(f"/api/projects/{project['id']}/chat")
    assert history.status_code == 200
    assert [(m["role"], m["content"]) for m in history.json()] == [
        ("user", first_text),
        ("assistant", first_payload["assistant"]["content"]),
        ("user", second_text),
        ("assistant", second.json()["assistant"]["content"]),
    ]
    fresh = TestClient(create_app(tmp_path, check_llm=False))
    assert len(fresh.get(f"/api/projects/{project['id']}/chat").json()) == 4


def test_greeting_uses_deepseek_capability_plan_then_becomes_a_task(tmp_path, monkeypatch):
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
            system = system_msgs[0]["content"]
            return '{"summary":"Respond to the greeting","intent":"greeting","needs_workspace_context":false,"workflow":[],"requires_confirmation":false,"rationale":[]}'

    monkeypatch.setattr("app.api.routes.LLM", DeepSeekGreeting)
    client = TestClient(create_app(tmp_path, check_llm=False))
    started = stub_orchestrator_start(client)
    project = client.post("/api/projects", json={"name": "greeting"}).json()
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Hellooo"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["kind"] == "task"
    assert payload["task"]["prompt"] == "Hellooo"
    assert "route" not in payload["plan"]
    assert payload["plan"]["control_unit_plan"]["steps"] == []
    assert len(DeepSeekGreeting.calls) == 1
    assert all(messages == [{"role": "user", "content": "Hellooo"}] for messages, _, _ in DeepSeekGreeting.calls)
    planner_context = DeepSeekGreeting.calls[0][1][0]["content"]
    assert "MODEL AND TOOL DESCRIPTIONS" in planner_context
    assert "Executable capability directory" not in planner_context
    assert len(started) == 1


def test_accept_event_stream_still_creates_task_not_separate_chat_response(tmp_path, monkeypatch):
    client, calls, started = _client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "stream-chat"}).json()
    text = "Tell me something"
    response = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": text},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["kind"] == "task"
    assert calls[0]["prompt"] == text
    task_id = response.json()["task"]["id"]
    assert started[0].id == task_id
    events = client.get(f"/api/tasks/{task_id}/events/history").json()
    assert any(event["type"] == "task.planned" for event in events)
    assert not any(event["type"] == "chat.delta" for event in events)


def test_project_chat_replay_after_disconnect_is_idempotent(tmp_path, monkeypatch):
    client, calls, started = _client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "replay-chat"}).json()
    headers = {"Accept": "text/event-stream", "Idempotency-Key": "chat-replay-1"}
    text = "Tell me a short greeting"
    first = client.post(f"/api/projects/{project['id']}/chat", json={"message": text}, headers=headers)
    second = client.post(f"/api/projects/{project['id']}/chat", json={"message": text}, headers={"Idempotency-Key": "chat-replay-1"})
    assert first.status_code == second.status_code == 200
    assert first.json()["kind"] == second.json()["kind"] == "task"
    assert first.json()["task"]["id"] == second.json()["task"]["id"]
    assert len(calls) == 1
    assert len(started) == 1
    history = client.get(f"/api/projects/{project['id']}/chat").json()
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[0]["content"] == text


def test_project_chat_rejects_unknown_project(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    assert client.get("/api/projects/missing/chat").status_code == 404
    assert client.post("/api/projects/missing/chat", json={"message": "Hi"}).status_code == 404


def test_project_chat_workspace_context_is_a_deepseek_plan_choice(tmp_path, monkeypatch):
    client, calls, _ = _client(tmp_path, monkeypatch, needs_workspace_context=True)
    project = client.post("/api/projects", json={"name": "repo"}).json()
    workspace = tmp_path / "projects" / project["id"]
    (workspace / "README.md").write_text("Repository architecture overview", encoding="utf-8")
    (workspace / "app.py").write_text("print('hello')", encoding="utf-8")
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Explain the python architecture"})
    assert response.status_code == 200
    assert calls[0]["prompt"] == "Explain the python architecture"
    assert response.json()["plan"]["needs_workspace_context"] is True
    # The first controller call gets the exact request and metadata, not a
    # preselected/truncated workspace excerpt before it asks for one.
    assert "Repository architecture overview" not in str(calls[0]["context"])


def test_deepseek_plan_can_combine_answer_and_project_execution(tmp_path, monkeypatch):
    client, calls, started = _client(tmp_path, monkeypatch, handlers=["qwen_coder"])
    project = client.post("/api/projects", json={"name": "unified"}).json()
    text = "Please update the project README with setup instructions, then explain what changed."
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": text})
    assert response.status_code == 200
    payload = response.json()
    assert calls[0]["prompt"] == text
    assert payload["kind"] == "task"
    assert "route" not in payload["plan"]
    steps = payload["task"]["plan"]["control_unit_plan"]["steps"]
    assert {step["handler"] for step in steps} == {"qwen_coder"}
    assert steps[0]["tools"]
    assert payload["task"]["prompt"] == text
    assert started and started[0].id == payload["task"]["id"]


def test_research_and_final_answer_are_one_task_workflow(tmp_path, monkeypatch):
    client, calls, _ = _client(tmp_path, monkeypatch, handlers=["qwen2.5_3b"])
    project = client.post("/api/projects", json={"name": "browser-research"}).json()
    text = "Go through this website and tell me what is in there https://example.com"
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": text})
    assert response.status_code == 200
    assert calls[0]["prompt"] == text
    plan = response.json()["task"]["plan"]["control_unit_plan"]
    assert {step["handler"] for step in plan["steps"]} == {"qwen2.5_3b"}
    assert response.json()["task"]["prompt"] == text


def test_direct_question_is_still_a_deepseek_planned_task(tmp_path, monkeypatch):
    client, calls, _ = _client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "browse"}).json()
    text = "Can you browse on the internet?"
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": text})
    assert response.status_code == 200
    assert calls[0]["prompt"] == text
    assert response.json()["kind"] == "task"
    assert response.json()["task"]["prompt"] == text


def test_project_chat_passes_math_questions_without_keyword_specific_prompt_or_rewrite(tmp_path, monkeypatch):
    client, calls, _ = _client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "math"}).json()
    text = "  Solve this algebra problem step by step.  "
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": text})
    assert response.status_code == 200
    assert calls[0]["prompt"] == text
    assert response.json()["task"]["prompt"] == text
    assert "route" not in response.json()["plan"]


def test_project_chat_fails_closed_when_deepseek_is_unavailable(tmp_path, monkeypatch):
    client, _, _ = _client(tmp_path, monkeypatch)
    monkeypatch.delitem(config.llm, "reasoning", raising=False)
    monkeypatch.setattr("app.api.routes.reasoning_enabled", lambda: False)
    project = client.post("/api/projects", json={"name": "chat-demo"}).json()
    response = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Please answer this question"})
    assert response.status_code == 503
    assert "DeepSeek is required" in response.json()["detail"]
