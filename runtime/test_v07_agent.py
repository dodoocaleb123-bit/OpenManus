"""v0.7 — the agent can actually do the job it was designed for.

Covers the workspace tools, git/GitHub workflow, the orchestrator's live
channels (events, questions, messages, cancel) and the real PlatformManus
agent class driven by a scripted LLM.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.platform.agent import WorkspaceBash, WorkspaceEditor, WorkspacePython
from app.platform.git import GitWorkspace
from app.platform.git_service import ProjectGit
from app.platform.store import PlatformStore
from app.platform.orchestrator import _is_browser_research_task, _research_url
from app.server import create_app


# ------------------------------------------------------------------ helpers


def run(coro):
    return asyncio.run(coro)


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def wait_for(client: TestClient, task_id: str, predicate, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = client.get(f"/api/tasks/{task_id}/events/history").json()
        if predicate(events):
            return events
        time.sleep(0.05)
    raise AssertionError(f"timed out; events: {[e['type'] for e in events]}")


def has(type_):
    return lambda events: any(e["type"] == type_ for e in events)


def test_browser_research_intent_does_not_enter_coding_mode():
    assert _is_browser_research_task(
        "Go through this website and tell me what is in there https://example.com/search?q=university"
    )
    assert _is_browser_research_task("Can you browse https://example.com and summarize the page?")
    assert not _is_browser_research_task("Browse the website and update the project README with its findings")
    assert _research_url("Please inspect https://example.com/page?q=1.") == "https://example.com/page?q=1"


# -------------------------------------------------------- workspace tools


def test_bash_runs_in_workspace_without_platform_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("PLATFORM_PASSWORD", "hunter2")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tok")

    async def go():
        bash = WorkspaceBash(workspace=tmp_path)
        try:
            out = await bash.execute(command="pwd; env | grep -E 'ghp_secret|sk-secret|hunter2|=tok$' || echo CLEAN")
            await bash.execute(command="cd /tmp")
            again = await bash.execute(command="pwd")  # persistent session
            return out, again
        finally:
            bash.close()

    out, again = run(go())
    assert out.output.splitlines()[0] == str(tmp_path)
    assert "CLEAN" in out.output and "sk-secret" not in out.output
    assert again.output == "/tmp"


def test_bash_close_kills_background_servers(tmp_path):
    async def go():
        bash = WorkspaceBash(workspace=tmp_path)
        await bash.execute(command="sleep 300 > /dev/null 2>&1 & echo $! > pid")
        pid = int((tmp_path / "pid").read_text())
        bash.close()
        await asyncio.sleep(0.3)
        return pid

    pid = run(go())
    assert subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode != 0


def test_python_tool_cwd_timeout_and_errors(tmp_path):
    tool = WorkspacePython(workspace=tmp_path)
    ok = run(tool.execute(code="import os; print(os.getcwd())"))
    assert ok.output.strip() == str(tmp_path)
    slow = run(tool.execute(code="import time; time.sleep(5)", timeout=1))
    assert "timed out" in slow.error
    bad = run(tool.execute(code="raise SystemExit(3)"))
    assert bad.error.startswith("exit code 3")


def test_editor_accepts_workspace_relative_paths(tmp_path):
    editor = WorkspaceEditor(workspace=tmp_path)
    run(editor.execute(command="create", path="src/app.py", file_text="print('hi')\n"))
    assert (tmp_path / "src" / "app.py").read_text() == "print('hi')\n"


# ------------------------------------------------------------- git + github


@pytest.fixture
def bare_remote(tmp_path):
    remote = tmp_path / "remote.git"
    git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    return remote


def test_nested_folder_is_not_treated_as_its_own_repo(tmp_path):
    outer = tmp_path / "platform"
    outer.mkdir()
    git("init", cwd=outer)
    project = GitWorkspace(outer / "workspace" / "projects" / "p1")
    assert not run(project.is_repo())
    with pytest.raises(Exception, match="no git repository"):
        run(project.status())


def test_project_git_flow_excludes_dependencies_and_persists_state(tmp_path, bare_remote):
    store = PlatformStore(tmp_path / "data")
    project = store.create_project("demo")
    ws = Path(project.workspace)
    svc = ProjectGit(store, project)

    async def go():
        await svc.init()
        (ws / "app.py").write_text("print('v1')\n")
        (ws / "node_modules" / "pkg").mkdir(parents=True)
        (ws / "node_modules" / "pkg" / "index.js").write_text("x")
        (ws / ".venv").mkdir()
        (ws / ".venv" / "pyvenv.cfg").write_text("x")
        (ws / ".env").write_text("SECRET=1")
        await svc.commit("initial")
        await svc.git.set_remote(str(bare_remote))
        await svc.push()
        branch = await svc.create_branch("feature/greeting")
        (ws / "app.py").write_text("print('v2')\n")
        await svc.commit("change greeting")
        await svc.push()
        return branch

    branch = run(go())
    assert branch == "feature/greeting"
    tracked = git("ls-tree", "-r", "--name-only", "main", cwd=bare_remote).splitlines()
    assert tracked == ["app.py"]  # no node_modules / .venv / .env
    assert "feature/greeting" in git("branch", cwd=bare_remote)
    reloaded = store.get_project(project.id)
    assert reloaded.branch == "feature/greeting" and reloaded.git_ready
    with pytest.raises(Exception, match="Nothing to commit"):
        run(svc.commit("empty"))


def _mock_github(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr("app.platform.github.httpx.AsyncClient", factory)


def test_create_pull_request_uses_default_branch_and_reuses_existing(monkeypatch):
    from app.platform.github import GitHubClient

    calls = []

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path, request.url.params.get("head")))
        assert request.headers["authorization"] == "Bearer t0k"
        if request.method == "GET" and request.url.path == "/repos/me/app":
            return httpx.Response(200, json={"default_branch": "develop"})
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["base"] == "develop" and body["head"] == "feat"
            if len([c for c in calls if c[0] == "POST"]) > 1:
                return httpx.Response(422, json={"message": "Validation Failed", "errors": [{"message": "A pull request already exists for me:feat."}]})
            return httpx.Response(201, json={"number": 7, "html_url": "https://github.com/me/app/pull/7", "title": "T", "base": {"ref": "develop"}})
        if request.method == "GET" and request.url.path == "/repos/me/app/pulls":
            return httpx.Response(200, json=[{"number": 7, "html_url": "https://github.com/me/app/pull/7"}])
        return httpx.Response(404)

    _mock_github(monkeypatch, handler)
    client = GitHubClient("t0k")
    pr = run(client.create_pull_request("me", "app", head="feat", title="T"))
    assert pr["number"] == 7
    again = run(client.create_pull_request("me", "app", head="feat", title="T"))
    assert again["already_existed"] and again["number"] == 7
    assert ("GET", "/repos/me/app/pulls", "me:feat") in calls


def test_agent_git_tool_publishes_new_project(tmp_path, bare_remote, monkeypatch):
    from app.platform.git_tool import PlatformGitTool

    def handler(request: httpx.Request):
        assert request.method == "POST" and request.url.path == "/user/repos"
        body = json.loads(request.content)
        assert body["private"] is True
        return httpx.Response(201, json={"name": body["name"], "full_name": f"me/{body['name']}", "html_url": f"https://github.com/me/{body['name']}", "private": True, "owner": {"login": "me"}})

    _mock_github(monkeypatch, handler)
    monkeypatch.setenv("GITHUB_TOKEN", "t0k")
    monkeypatch.setattr("app.platform.git_service.github_remote", lambda owner, repo: str(bare_remote))
    store = PlatformStore(tmp_path / "data")
    project = store.create_project("fresh")
    (Path(project.workspace) / "index.html").write_text("<h1>hi</h1>")
    events = []

    async def on_event(t, m, d):
        events.append((t, d))

    tool = PlatformGitTool(store=store, project_id=project.id, on_event=on_event)
    result = run(tool.execute(action="publish_repository", repo_name="sneaker-shop"))
    assert result.error is None, result.error
    assert git("ls-tree", "-r", "--name-only", "main", cwd=bare_remote) == "index.html"
    assert events and events[0][0] == "github.repository"
    assert store.get_project(project.id).git_ready


# --------------------------------------------------------- orchestrator/API


class FakeAgent:
    """Stands in for PlatformManus: exercises the orchestrator's channels."""

    def __init__(self, script, **channels):
        self.script = script
        self.channels = channels
        self.current_step = 0
        self.runs = 0
        self.closed = False

    async def run(self, prompt):
        self.runs += 1
        await self.script(self, prompt)

    def final_summary(self):
        return f"done after {self.runs} run(s)"

    def close_processes(self):
        self.closed = True


def make_client(tmp_path, script):
    agents = []

    async def factory(*, project, task, emit, inbox, ask, extra_tools):
        agent = FakeAgent(script, project=project, emit=emit, inbox=inbox, ask=ask, tools={t.name: t for t in extra_tools})
        agents.append(agent)
        return agent

    app = create_app(tmp_path, agent_factory=factory, check_llm=False)
    return TestClient(app), agents


def test_agent_question_is_answered_from_the_ui(tmp_path):
    async def script(agent, prompt):
        await agent.channels["emit"]("agent.thought", "planning", {})
        answer = await agent.channels["ask"]("Which colour scheme?")
        Path(agent.channels["project"].workspace, "choice.txt").write_text(answer)

    client, agents = make_client(tmp_path, script)
    with client:
        p = client.post("/api/projects", json={"name": "q"}).json()
        t = client.post("/api/tasks", json={"project_id": p["id"], "prompt": "build it"}).json()
        wait_for(client, t["id"], has("agent.question"))
        assert client.get(f"/api/tasks/{t['id']}").json()["pending_question"] == "Which colour scheme?"
        r = client.post(f"/api/tasks/{t['id']}/messages", json={"message": "dark"}).json()
        assert r["delivery"] == "answered"
        events = wait_for(client, t["id"], has("task.succeeded"))
        task = client.get(f"/api/tasks/{t['id']}").json()
    assert (Path(p["workspace"]) / "choice.txt").read_text() == "dark"
    assert task["status"] == "succeeded" and task["result"] == "done after 1 run(s)"
    assert {"agent.thought", "human.reply", "coding.validation"} <= {e["type"] for e in events}
    assert agents[0].closed
    assert set(agents[0].channels["tools"]) == {"platform_browser", "platform_git"}


def test_messages_during_a_task_are_queued_for_the_agent(tmp_path):
    received = []

    async def script(agent, prompt):
        inbox = agent.channels["inbox"]
        received.append(await asyncio.wait_for(inbox.get(), timeout=10))

    client, _ = make_client(tmp_path, script)
    with client:
        p = client.post("/api/projects", json={"name": "m"}).json()
        t = client.post("/api/tasks", json={"project_id": p["id"], "prompt": "x"}).json()
        wait_for(client, t["id"], has("agent.running"))
        assert client.post(f"/api/tasks/{t['id']}/messages", json={"message": "use Tailwind"}).json()["delivery"] == "queued"
        wait_for(client, t["id"], has("task.succeeded"))
        # finished tasks can't receive messages
        assert client.post(f"/api/tasks/{t['id']}/messages", json={"message": "late"}).status_code == 409
    assert received == ["use Tailwind"]


def test_cancel_stops_a_running_task_and_one_task_per_project(tmp_path):
    async def script(agent, prompt):
        await asyncio.sleep(60)

    client, agents = make_client(tmp_path, script)
    with client:
        p = client.post("/api/projects", json={"name": "c"}).json()
        t = client.post("/api/tasks", json={"project_id": p["id"], "prompt": "x"}).json()
        wait_for(client, t["id"], has("agent.running"))
        second = client.post("/api/tasks", json={"project_id": p["id"], "prompt": "y"})
        assert second.status_code == 409
        r = client.post(f"/api/tasks/{t['id']}/cancel")
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        wait_for(client, t["id"], has("task.cancelled"))
        assert client.post(f"/api/tasks/{t['id']}/cancel").status_code == 409
        # project is free again
        assert client.post("/api/tasks", json={"project_id": p["id"], "prompt": "y"}).status_code == 200
    assert agents[0].closed


def test_failing_validation_triggers_repair_cycle(tmp_path):
    async def script(agent, prompt):
        ws = Path(agent.channels["project"].workspace)
        if agent.runs == 1:
            (ws / "package.json").write_text(json.dumps({"scripts": {"test": "node check.js"}}))
            (ws / "check.js").write_text("process.exit(require('fs').existsSync('fixed') ? 0 : 1)")
            (ws / "node_modules").mkdir(exist_ok=True)
            (ws / "node_modules" / ".openmanus-install").write_text("stale")
        else:
            assert "independent validation failed" in prompt
            (ws / "fixed").write_text("1")

    client, agents = make_client(tmp_path, script)
    import shutil
    if not shutil.which("node"):
        pytest.skip("node not installed")
    with client:
        p = client.post("/api/projects", json={"name": "r"}).json()
        t = client.post("/api/tasks", json={"project_id": p["id"], "prompt": "x"}).json()
        events = wait_for(client, t["id"], lambda ev: any(e["type"] in ("task.succeeded", "task.failed") for e in ev), timeout=120)
    types = [e["type"] for e in events]
    assert "coding.repair" in types and types[-1] == "task.succeeded", types
    assert agents[0].runs == 2


def test_unconfigured_llm_fails_fast_with_setup_hint(tmp_path):
    client = TestClient(create_app(tmp_path))  # example config in use
    with client:
        status = client.get("/api/status").json()
        assert status["llm"]["configured"] is False
        p = client.post("/api/projects", json={"name": "n"}).json()
        t = client.post("/api/tasks", json={"project_id": p["id"], "prompt": "x"}).json()
        events = wait_for(client, t["id"], has("task.failed"), timeout=10)
    assert "LLM_API_KEY" in events[-1]["message"]


def test_sse_stream_resumes_after_last_event_id(tmp_path):
    async def script(agent, prompt):
        for i in range(3):
            await agent.channels["emit"]("agent.thought", f"t{i}", {})

    client, _ = make_client(tmp_path, script)
    with client:
        p = client.post("/api/projects", json={"name": "s"}).json()
        t = client.post("/api/tasks", json={"project_id": p["id"], "prompt": "x"}).json()
        history = wait_for(client, t["id"], has("task.succeeded"))
        body = client.get(f"/api/tasks/{t['id']}/events").text
        ids = [line[4:] for line in body.splitlines() if line.startswith("id: ")]
        assert ids == [e["id"] for e in history]
        resumed = client.get(f"/api/tasks/{t['id']}/events", headers={"Last-Event-ID": ids[2]}).text
        assert [line[4:] for line in resumed.splitlines() if line.startswith("id: ")] == ids[3:]


# ------------------------------------------------ real PlatformManus + fake LLM


def test_platform_manus_runs_tools_and_emits_live_events(tmp_path, monkeypatch):
    """The real agent loop (BaseAgent -> ToolCallAgent -> Manus -> PlatformManus)."""
    import tiktoken

    monkeypatch.setenv("OPENMANUS_DISABLE_BROWSER_USE", "1")
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: SimpleNamespace(encode=lambda s: s.split()))
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda name: (_ for _ in ()).throw(KeyError(name)))
    from app.llm import LLM
    from app.platform.agent import PlatformManus
    from app.schema import Function, ToolCall

    LLM._instances.clear()
    script = [
        ("I'll create the file.", [("bash", {"command": "echo hello > hello.txt && cat hello.txt"})]),
        ("Verified.", [("str_replace_editor", {"command": "view", "path": "hello.txt"})]),
        ("Created hello.txt containing 'hello' and verified it.", [("terminate", {"status": "success"})]),
    ]

    async def fake_ask_tool(*args, **kwargs):
        content, calls = script.pop(0)
        return SimpleNamespace(
            content=content,
            tool_calls=[ToolCall(id=f"c{i}", function=Function(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(calls)],
        )

    events = []

    async def emit(t, m, d):
        events.append((t, m, d))

    async def go():
        agent = await PlatformManus.create_for_project(
            project_name="demo", workspace=tmp_path, repository=None, branch=None, emit=emit, inbox=asyncio.Queue(),
        )
        monkeypatch.setattr(agent.llm, "ask_tool", fake_ask_tool)
        agent.inbox.put_nowait("please also be quick")
        try:
            await agent.run("create hello.txt")
        finally:
            agent.close_processes()
        return agent

    agent = run(go())
    LLM._instances.clear()
    assert (tmp_path / "hello.txt").read_text() == "hello\n"
    types = [e[0] for e in events]
    assert types.count("agent.tool_call") == 3 and types.count("agent.tool_result") == 3
    assert "agent.user_message" in types
    bash_result = next(e for e in events if e[0] == "agent.tool_result" and e[2]["tool"] == "bash")
    assert bash_result[2]["ok"] and "hello" in bash_result[2]["output"]
    assert agent.final_summary().startswith("Created hello.txt")
    assert any("please also be quick" in (m.content or "") for m in agent.memory.messages)
    assert str(tmp_path) in agent.system_prompt


# ------------------------------------------------------------- model adapter


@pytest.mark.parametrize(
    "model,vision,reasoning",
    [
        ("claude-sonnet-5", True, False),
        ("claude-opus-5-5", True, False),
        ("claude-fable-5-1", True, False),
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", True, False),
        ("gpt-4o", True, False),
        ("gpt-4.1-mini", True, False),
        ("gpt-5", True, True),  # rejects max_tokens / temperature
        ("o3-mini", False, True),
        ("gemini-2.5-pro", True, False),
        ("deepseek-chat", False, False),
    ],
)
def test_model_capability_detection(model, vision, reasoning, monkeypatch):
    from app.llm import is_reasoning_model, model_supports_images

    monkeypatch.delenv("LLM_SUPPORTS_IMAGES", raising=False)
    monkeypatch.delenv("LLM_REASONING_MODEL", raising=False)
    assert model_supports_images(model) is vision
    assert is_reasoning_model(model) is reasoning


def test_capability_env_overrides(monkeypatch):
    from app.llm import is_reasoning_model, model_supports_images

    monkeypatch.setenv("LLM_SUPPORTS_IMAGES", "true")
    monkeypatch.setenv("LLM_REASONING_MODEL", "false")
    assert model_supports_images("my-custom-model") and not is_reasoning_model("gpt-5")


def test_permanent_llm_errors_are_not_retried():
    from openai import AuthenticationError, NotFoundError

    from app.llm import _is_retryable

    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    auth = AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
    missing = NotFoundError("no such model", response=httpx.Response(404, request=request), body=None)
    assert not _is_retryable(auth) and not _is_retryable(missing)
    assert _is_retryable(TimeoutError())
