from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.platform.models import TaskStatus
from app.server import create_app


class FakeChatLLM:
    last_system = None
    last_messages = None

    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, **kwargs):
        FakeChatLLM.last_system = system_msgs
        FakeChatLLM.last_messages = messages
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        return f"Local reply: {user_messages[-1]}"


def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    monkeypatch.setattr("app.api.routes.llm_problem", lambda: None)
    return TestClient(create_app(tmp_path, check_llm=False))


def test_answer_and_plan_modes_do_not_launch_tasks_or_load_repository(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "modes"}).json()
    workspace = tmp_path / "projects" / project["id"]
    (workspace / "secret-source.py").write_text("important source", encoding="utf-8")

    answer = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Explain this source.py file", "mode": "answer"},
    )
    assert answer.status_code == 200
    assert answer.json()["kind"] == "chat"
    assert "Answer only" in FakeChatLLM.last_system[0]["content"]
    assert "important source" not in FakeChatLLM.last_system[0]["content"]

    plan = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Create a calculator app", "mode": "plan"},
    )
    assert plan.status_code == 200
    assert plan.json()["kind"] == "chat"
    assert plan.json()["plan"]["intent"] == "plan_only"
    assert "Plan only" in FakeChatLLM.last_system[0]["content"]
    assert client.get(f"/api/projects/{project['id']}/tasks").json() == []


def test_math_request_requires_current_problem_and_does_not_reuse_prior_solution(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "math-grounding"}).json()

    first = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Solve 2+2", "mode": "answer"},
    )
    assert first.status_code == 200
    assert len(FakeChatLLM.last_messages) == 1

    previous_call = FakeChatLLM.last_messages
    missing = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "I want you to answer a mathematics question", "mode": "answer"},
    )
    assert missing.status_code == 200
    assert "complete mathematics question" in missing.json()["assistant"]["content"]
    assert FakeChatLLM.last_messages is previous_call

    second = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Solve 5+5", "mode": "answer"},
    )
    assert second.status_code == 200
    assert len(FakeChatLLM.last_messages) == 1
    assert FakeChatLLM.last_messages[0]["content"] == "Solve 5+5"
    assert "Never invent a problem" in FakeChatLLM.last_system[0]["content"]
    assert "Files explicitly attached to this message: none" in FakeChatLLM.last_system[0]["content"]


def test_project_memory_is_persisted_scoped_and_rejects_credentials(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "memory"}).json()
    saved = client.put(
        f"/api/projects/{project['id']}/memory",
        json={"content": "Prefer Python 3.11 and keep functions small."},
    )
    assert saved.status_code == 200
    assert client.get(f"/api/projects/{project['id']}/memory").json()["content"] == "Prefer Python 3.11 and keep functions small."

    chat = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Explain the project architecture", "mode": "inspect"},
    )
    assert chat.status_code == 200
    assert "Prefer Python 3.11" in FakeChatLLM.last_system[0]["content"]
    assert "untrusted reference data" in FakeChatLLM.last_system[0]["content"]

    rejected = client.put(
        f"/api/projects/{project['id']}/memory",
        json={"content": "api_key=do-not-save"},
    )
    assert rejected.status_code == 400


def test_task_launch_idempotency_returns_existing_task_without_duplicate_chat(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "idempotency"}).json()
    started = []
    client.app.state.orchestrator.start = lambda task: started.append(task.id)
    payload = {"message": "Create a README with setup instructions"}
    headers = {"Idempotency-Key": "retry-safe-request-1"}

    first = client.post(f"/api/projects/{project['id']}/chat", json=payload, headers=headers)
    second = client.post(f"/api/projects/{project['id']}/chat", json=payload, headers=headers)
    assert first.status_code == second.status_code == 200
    assert first.json()["task"]["id"] == second.json()["task"]["id"]
    assert len(started) == 1
    history = client.get(f"/api/projects/{project['id']}/chat").json()
    assert len(history) == 2
    assert history[0]["task_id"] == first.json()["task"]["id"]

    conflict = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "A different task"},
        headers=headers,
    )
    assert conflict.status_code == 409


def test_task_feedback_and_latency_metrics_are_persisted_locally(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "metrics"}).json()
    store = client.app.state.store
    task = store.create_task(project["id"], "run verification")
    task.status = TaskStatus.SUCCEEDED
    task.first_token_ms = 420
    asyncio.run(store.save_task(task))
    store.add_chat_message(
        project["id"], "assistant", "Verified output", response_time_ms=2400,
        time_to_first_token_ms=420, task_id=task.id,
    )

    feedback = client.post(f"/api/tasks/{task.id}/feedback", json={"rating": 1})
    assert feedback.status_code == 200
    assert feedback.json()["rating"] == 1
    metrics = client.get("/api/metrics").json()
    assert metrics["feedback"]["helpful"] == 1
    assert metrics["latency"]["response_time"]["average_ms"] == 2400
    assert metrics["latency"]["time_to_first_token"]["average_ms"] == 420
    assert client.get(f"/api/projects/{project['id']}/chat").json()[-1]["task_id"] == task.id


def test_github_push_and_pull_request_routes_require_confirmation(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "git-confirm"}).json()
    push = client.post(f"/api/projects/{project['id']}/git/push")
    pr = client.post(f"/api/projects/{project['id']}/github/pull-request", json={"title": "Review"})
    assert push.status_code == 428
    assert pr.status_code == 428


def test_git_tool_allows_delegated_external_changes_but_honors_prohibition():
    from app.platform.git_tool import PlatformGitTool

    explicit = PlatformGitTool(user_request="Finish the README changes and push them to GitHub")
    assert explicit._explicitly_requested("push")
    assert not explicit._explicitly_requested("publish_repository")
    assert asyncio.run(explicit._approve_external_action("push", "push changes")) is None

    blocked = PlatformGitTool(user_request="Update the README; do not push anything")
    assert not blocked._explicitly_requested("push")
    assert "prohibited" in asyncio.run(blocked._approve_external_action("push", "push changes"))

    delegated = PlatformGitTool(user_request="Update the README")
    assert asyncio.run(delegated._approve_external_action("push", "push the current branch")) is None


def test_agent_redacts_secrets_and_blocks_secret_config_files(tmp_path):
    from app.platform.agent import WorkspaceEditor, redact_sensitive_text

    text = redact_sensitive_text("api_key=abc123 and token ghp_12345678901234567890")
    assert "abc123" not in text
    assert "ghp_12345678901234567890" not in text
    editor = WorkspaceEditor(workspace=tmp_path)
    message = asyncio.run(editor.execute(command="view", path=".env"))
    assert "blocked" in message.lower()


def test_project_delete_cleans_memory_and_feedback_before_foreign_key_parents(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    project = client.post("/api/projects", json={"name": "cleanup"}).json()
    store = client.app.state.store
    task = store.create_task(project["id"], "completed task")
    task.status = TaskStatus.SUCCEEDED
    asyncio.run(store.save_task(task))
    store.set_project_memory(project["id"], "Keep the tests focused.")
    store.set_task_feedback(task.id, 1)

    response = client.delete(f"/api/projects/{project['id']}", headers={"X-OpenManus-Confirm": "true"})
    assert response.status_code == 200
    assert store.get_project_memory(project["id"])["content"] == ""
    assert store.feedback_metrics()["rated_tasks"] == 0
