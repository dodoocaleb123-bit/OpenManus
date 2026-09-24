import asyncio
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

from app.server import create_app


def test_git_ops_refuse_foreign_repositories(tmp_path: Path):
    """Project git operations must never touch a repository above the workspace."""
    import subprocess

    from app.platform.git import GitError, GitWorkspace

    def git(*args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    # Parent directory is a git repo (like the platform's own checkout).
    git("init", "-q", "-b", "main", cwd=tmp_path)
    git("config", "user.email", "t@e.com", cwd=tmp_path)
    git("config", "user.name", "T", cwd=tmp_path)
    (tmp_path / "platform_code.py").write_text("print('platform')\n")
    git("add", ".", cwd=tmp_path)
    git("commit", "-qm", "platform source", cwd=tmp_path)

    workspace = tmp_path / "runtime" / "workspace" / "projects" / "abc123"
    workspace.mkdir(parents=True)
    git_ws = GitWorkspace(workspace)

    async def run():
        assert await git_ws.is_repo() is False
        for op in (git_ws.status, git_ws.diff, lambda: git_ws.create_branch("x")):
            try:
                await op()
            except GitError:
                continue
            raise AssertionError(f"{op} accepted a non-repo workspace")
        try:
            await git_ws.commit("should not happen")
        except GitError:
            pass
        else:
            raise AssertionError("commit accepted a non-repo workspace")

    asyncio.run(run())
    # The parent repository is untouched: still on main, no new commits.
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert len(log.strip().splitlines()) == 1 and "platform source" in log


def test_git_ops_work_on_own_repository(tmp_path: Path):
    from app.platform.git import GitWorkspace

    async def run():
        g = GitWorkspace(tmp_path / "repo")
        await g._run("init")
        await g._run("config", "user.email", "t@e.com")
        await g._run("config", "user.name", "T")
        (g.path / "README.md").write_text("hello")
        await g._run("add", ".")
        await g._run("commit", "-m", "init")
        assert await g.is_repo() is True
        assert await g.current_branch()
        branch = await g.create_branch("feature-x")
        assert branch == "feature-x"  # the name comes back for the UI
        status = await g.status()
        assert status["branch"] == "feature-x"

    asyncio.run(run())


def test_github_connection_and_branch_persist(tmp_path: Path, monkeypatch):
    from app.platform.git import GitWorkspace

    async def fake_clone(self, remote, branch=None):
        # Stand-in for cloning from GitHub (no network in tests).
        await self._run("init", "-b", "main")
        await self._run("config", "user.email", "t@e.com")
        await self._run("config", "user.name", "T")
        (self.path / "README.md").write_text("hi")
        await self._run("add", ".")
        await self._run("commit", "-m", "init")

    monkeypatch.setattr(GitWorkspace, "clone", fake_clone)
    client = TestClient(create_app(tmp_path))
    project = client.post("/api/projects", json={"name": "demo"}).json()
    connected = client.post(
        f"/api/projects/{project['id']}/github/connect",
        json={"owner": "o", "repo": "r"},
    ).json()["project"]
    assert connected["git_ready"] is True

    # After a "reload" the connection must still be there.
    again = [p for p in client.get("/api/projects").json() if p["id"] == project["id"]][0]
    assert again["git_ready"] is True
    assert again["repository"] == "o/r"

    branched = client.post(
        f"/api/projects/{project['id']}/git/branch", json={"name": "feature-x"}
    ).json()
    assert branched["branch"] == "feature-x"
    again = [p for p in client.get("/api/projects").json() if p["id"] == project["id"]][0]
    assert again["branch"] == "feature-x"


def test_git_status_refuses_project_without_repository(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    project = client.post("/api/projects", json={"name": "norepo"}).json()
    response = client.get(f"/api/projects/{project['id']}/git/status")
    assert response.status_code == 400
    assert "not a git repository" in response.json()["detail"]


def test_browser_rejection_leaves_no_orphan_task(tmp_path: Path):
    from types import SimpleNamespace

    from app.platform.browser import BrowserManager
    from app.platform.orchestrator import AgentOrchestrator
    from app.platform.store import PlatformStore

    store = PlatformStore(tmp_path)
    browsers = BrowserManager(tmp_path)
    from fastapi import FastAPI

    from app.api.routes import build_router

    inner = FastAPI()
    inner.include_router(build_router(store, AgentOrchestrator(store, browsers)))
    client = TestClient(inner)
    a = client.post("/api/projects", json={"name": "A"}).json()
    b = client.post("/api/projects", json={"name": "B"}).json()
    browsers.sessions["sess1"] = SimpleNamespace(id="sess1", project_id=a["id"])

    response = client.post(
        "/api/tasks",
        json={"project_id": b["id"], "prompt": "build it", "browser_session_id": "sess1"},
    )
    assert response.status_code == 400
    assert client.get(f"/api/projects/{b['id']}/tasks").json() == []


def test_duplicate_project_names_get_distinct_workspaces(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    one = client.post("/api/projects", json={"name": "dup"}).json()
    two = client.post("/api/projects", json={"name": "dup"}).json()
    assert one["workspace"] != two["workspace"]
    assert one["workspace"].endswith(one["id"])
    assert two["workspace"].endswith(two["id"])


def test_repair_runs_get_full_step_budget(tmp_path: Path, monkeypatch):
    """Every implementation/repair run must start with a fresh Manus step budget."""
    from app.platform.models import TaskStatus
    from app.platform.orchestrator import AgentOrchestrator
    from app.platform.store import PlatformStore

    run_starts = []

    class StubManus:
        def __init__(self):
            self.current_step = 0
            self.max_steps = 20

            class _Tools:
                def add_tools(self, *args, **kwargs):
                    pass

            self.available_tools = _Tools()

        @classmethod
        async def create(cls):
            return cls()

        async def run(self, request=None):
            run_starts.append(self.current_step)
            self.current_step += 7  # simulate finishing mid-budget
            return "stub result"

        async def cleanup(self):
            pass

    stub = types.ModuleType("app.agent.manus")
    stub.Manus = StubManus
    monkeypatch.setitem(sys.modules, "app.agent.manus", stub)

    async def main():
        store = PlatformStore(tmp_path)
        project = store.create_project("loop")
        (Path(project.workspace) / "package.json").write_text(
            '{"scripts": {"test": "node -e \\"process.exit(1)\\""}}', encoding="utf-8"
        )
        task = store.create_task(project.id, "fix it")
        await AgentOrchestrator(store).run_task(task)
        return store.get_task(task.id)

    task = asyncio.run(main())
    # Initial run + 3 repair runs, each starting from step 0 (was: 0, 7, 14, 1).
    assert run_starts == [0, 0, 0, 0], run_starts
    assert task.status == TaskStatus.FAILED
    assert task.checkpoint == "validation_exhausted"
