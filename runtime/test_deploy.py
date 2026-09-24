import asyncio
import base64
from pathlib import Path

from fastapi.testclient import TestClient

from app.server import create_app


def _basic(user: str, password: str) -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_auth_required_when_password_set(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PLATFORM_ADMIN_PASSWORD", "s3cret-password")
    monkeypatch.setenv("PLATFORM_ADMIN_USER", "caleb")
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/health").status_code == 200  # health check stays open
    assert client.get("/api/projects").status_code == 401
    assert client.get("/", headers=_basic("caleb", "wrong")).status_code == 401
    good = _basic("caleb", "s3cret-password")
    assert client.get("/api/projects", headers=good).status_code == 200
    assert client.get("/", headers=good).status_code == 200


def test_auth_off_without_password(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("PLATFORM_ADMIN_PASSWORD", raising=False)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/projects").status_code == 200


def test_data_dir_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PLATFORM_DATA_DIR", str(tmp_path / "data"))
    client = TestClient(create_app())
    project = client.post("/api/projects", json={"name": "envproj"}).json()
    assert project["workspace"].startswith(str(tmp_path / "data"))


def test_scrubbed_env_drops_tokens(monkeypatch):
    from app.utils.env import scrubbed_env

    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("GH_TOKEN", "y")
    monkeypatch.setenv("KEEP_ME", "z")
    env = scrubbed_env()
    assert "GITHUB_TOKEN" not in env
    assert "GH_TOKEN" not in env
    assert env["KEEP_ME"] == "z"


def test_coding_loop_hides_github_token(tmp_path: Path, monkeypatch):
    """Workspace test commands must not see the platform's GitHub token."""
    from app.platform.coding_loop import CodingLoop

    monkeypatch.setenv("GITHUB_TOKEN", "fake-secret-token")
    (tmp_path / "package.json").write_text(
        '{"scripts": {"test": "node leak.js"}}', encoding="utf-8"
    )
    (tmp_path / "leak.js").write_text(
        "process.exit(process.env.GITHUB_TOKEN ? 1 : 0)", encoding="utf-8"
    )
    results = asyncio.run(CodingLoop(tmp_path).validate())
    assert results[0].ok, results[0].output


def test_daytona_optional_and_env_overrides(monkeypatch):
    from app.config import DaytonaSettings, _env_overrides

    assert DaytonaSettings().daytona_api_key == ""  # no crash without a key
    monkeypatch.setenv("OPENMANUS_LLM_API_KEY", "k-123")
    monkeypatch.setenv("OPENMANUS_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("OPENMANUS_LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("OPENMANUS_LLM_TEMPERATURE", "0.0")
    values = _env_overrides("OPENMANUS_LLM_")
    assert values == {
        "model": "gpt-test",
        "api_key": "k-123",
        "max_tokens": 4096,
        "temperature": 0.0,
    }


def test_sse_stream_emits_heartbeats_and_ends(tmp_path: Path):
    from app.api.routes import sse_event_stream
    from app.platform.models import Event
    from app.platform.store import PlatformStore

    async def main():
        store = PlatformStore(tmp_path)
        project = store.create_project("sse")
        task = store.create_task(project.id, "job")
        await store.emit(Event(task_id=task.id, type="task.started", message="start"))

        async def finish_later():
            await asyncio.sleep(0.25)  # idle long enough for heartbeats
            await store.emit(Event(task_id=task.id, type="task.failed", message="boom"))

        feeder = asyncio.create_task(finish_later())
        frames = []
        async for chunk in sse_event_stream(
            store, task.id, heartbeat_interval=0.05, max_lifetime=5, poll_interval=0.02
        ):
            frames.append(chunk)
        await feeder
        text = "".join(frames)
        assert "task.started" in text and "task.failed" in text
        assert ": ping" in text  # idle heartbeat keeps proxies happy

    asyncio.run(main())


def test_sse_stream_resumes_after_last_event_id(tmp_path: Path):
    from app.api.routes import sse_event_stream
    from app.platform.models import Event
    from app.platform.store import PlatformStore

    async def main():
        store = PlatformStore(tmp_path)
        project = store.create_project("sse2")
        task = store.create_task(project.id, "job")
        first = Event(task_id=task.id, type="task.started", message="start")
        await store.emit(first)
        await store.emit(Event(task_id=task.id, type="coding.implementation", message="done"))

        frames = []
        async for chunk in sse_event_stream(
            store,
            task.id,
            after_id=first.id,
            heartbeat_interval=5,
            max_lifetime=0.15,
            poll_interval=0.02,
        ):
            frames.append(chunk)
        text = "".join(frames)
        assert first.id not in text  # already-seen event is skipped on reconnect
        assert "coding.implementation" in text

    asyncio.run(main())
