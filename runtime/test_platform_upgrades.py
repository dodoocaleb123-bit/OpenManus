from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.platform.models import TaskStatus
from app.server import create_app
from test_support import install_fake_controller


class FakeChatLLM:
    last_system = None
    last_messages = None
    model = "deepseek-r1:7b"

    def __init__(self, *args, **kwargs):
        pass

    async def ask(self, messages, system_msgs=None, **kwargs):
        FakeChatLLM.last_system = system_msgs
        FakeChatLLM.last_messages = messages
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        return f"Local reply: {user_messages[-1]}"


def make_client(tmp_path, monkeypatch):
    install_fake_controller(monkeypatch)
    return TestClient(create_app(tmp_path, check_llm=False))


def test_deepseek_selects_workflow_and_legacy_mode_field_does_not_gate_it(tmp_path, monkeypatch):
    def choose(prompt, context):
        return [1] if prompt == "Create a calculator app" else [168]

    calls = install_fake_controller(monkeypatch, selector=choose)
    client = TestClient(create_app(tmp_path, check_llm=False))
    started = []
    client.app.state.orchestrator.start = lambda task: started.append(task)
    project = client.post("/api/projects", json={"name": "modes"}).json()
    workspace = tmp_path / "projects" / project["id"]
    (workspace / "secret-source.py").write_text("important source", encoding="utf-8")

    answer = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Explain this source.py file", "mode": "implement"})
    assert answer.status_code == 200
    assert answer.json()["kind"] == "task"
    assert "route" not in answer.json()["plan"]
    assert answer.json()["task"]["plan"]["control_unit_plan"]["selected_capabilities"] == [168]
    assert calls[0]["prompt"] == "Explain this source.py file"

    build = client.post(f"/api/projects/{project['id']}/chat", json={"message": "Create a calculator app", "mode": "answer"})
    assert build.status_code == 200
    assert calls[1]["prompt"] == "Create a calculator app"
    assert build.json()["kind"] == "task"
    assert "route" not in build.json()["plan"]
    assert client.get(f"/api/projects/{project['id']}/tasks").json()
    assert len(started) == 2


def test_project_memory_is_persisted_scoped_and_rejects_credentials(tmp_path, monkeypatch):
    calls = install_fake_controller(monkeypatch)
    client = TestClient(create_app(tmp_path, check_llm=False))
    client.app.state.orchestrator.start = lambda task: None
    project = client.post("/api/projects", json={"name": "memory"}).json()
    saved = client.put(
        f"/api/projects/{project['id']}/memory",
        json={"content": "Prefer Python 3.11 and keep functions small."},
    )
    assert saved.status_code == 200
    assert client.get(f"/api/projects/{project['id']}/memory").json()["content"] == "Prefer Python 3.11 and keep functions small."

    chat = client.post(
        f"/api/projects/{project['id']}/chat",
        json={"message": "Explain the project architecture"},
    )
    assert chat.status_code == 200
    assert chat.json()["kind"] == "task"
    assert "Prefer Python 3.11" in calls[0]["context"]["project_memory"]

    rejected = client.put(
        f"/api/projects/{project['id']}/memory",
        json={"content": "api_key=do-not-save"},
    )
    assert rejected.status_code == 400


def test_task_launch_idempotency_returns_existing_task_without_duplicate_chat(tmp_path, monkeypatch):
    install_fake_controller(monkeypatch, capability_ids=[1])
    monkeypatch.setattr("app.api.routes.LLM", FakeChatLLM)
    client = TestClient(create_app(tmp_path, check_llm=False))
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


def test_git_tool_requires_explicit_request_or_user_approval_and_honors_prohibition():
    from app.platform.git_tool import PlatformGitTool

    explicit = PlatformGitTool(user_request="Finish the README changes and push them to GitHub")
    assert explicit._explicitly_requested("push")
    assert not explicit._explicitly_requested("publish_repository")
    assert asyncio.run(explicit._approve_external_action("push", "push changes")) is None

    blocked = PlatformGitTool(user_request="Update the README; do not push anything")
    assert not blocked._explicitly_requested("push")
    assert "prohibited" in asyncio.run(blocked._approve_external_action("push", "push changes"))

    delegated = PlatformGitTool(user_request="Update the README")
    assert "not requested" in asyncio.run(delegated._approve_external_action("push", "push the current branch"))

    question = PlatformGitTool(user_request="How do I push this project to GitHub?")
    assert not question._explicitly_requested("push")
    assert "not requested" in asyncio.run(question._approve_external_action("push", "push the current branch"))

    prompts = []

    async def approve(question):
        prompts.append(question)
        return "yes"

    asked = PlatformGitTool(user_request="Update the README", request_approval=approve)
    assert asyncio.run(asked._approve_external_action("push", "push the current branch")) is None
    assert len(prompts) == 1 and "push the current branch" in prompts[0]

    async def decline(question):
        return "no"

    declined = PlatformGitTool(user_request="Update the README", request_approval=decline)
    assert "did not approve" in asyncio.run(declined._approve_external_action("publish_repository", "create a new GitHub repository"))


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
