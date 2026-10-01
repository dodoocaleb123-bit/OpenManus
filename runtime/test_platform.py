from pathlib import Path
from fastapi.testclient import TestClient

from app.server import create_app
from test_support import install_fake_controller


def test_health(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    assert client.get('/api/health').json()['status'] == 'ok'


def test_project_and_task(tmp_path: Path, monkeypatch):
    install_fake_controller(monkeypatch, capability_ids=[1], route="execute")
    client = TestClient(create_app(tmp_path))
    client.app.state.orchestrator.start = lambda task: None
    p = client.post('/api/projects', json={'name': 'demo'}).json()
    assert p['git_ready'] is False
    task = client.post('/api/tasks', json={'project_id': p['id'], 'prompt': 'create a hello world app'}).json()
    assert task['project_id'] == p['id']


def test_git_workspace_branch_validation(tmp_path: Path):
    from app.platform.git import GitWorkspace, GitError
    import asyncio
    async def run():
        g = GitWorkspace(tmp_path / 'repo')
        await g._run('init')
        await g._run('config', 'user.email', 'test@example.com')
        await g._run('config', 'user.name', 'Test')
        (g.path / 'README.md').write_text('hello')
        await g._run('add', 'README.md')
        await g._run('commit', '-m', 'init')
        assert await g.current_branch()
        try:
            await g.create_branch('../bad')
        except GitError:
            return
        raise AssertionError('invalid branch accepted')
    asyncio.run(run())


def test_persistence_and_resume(tmp_path: Path):
    from app.platform.store import PlatformStore
    from app.platform.models import TaskStatus
    import asyncio

    store1 = PlatformStore(tmp_path)
    p = store1.create_project('persistent')
    t = store1.create_task(p.id, 'do work')
    t.status = TaskStatus.RUNNING
    asyncio.run(store1.save_task(t))
    asyncio.run(store1.emit(__import__('app.platform.models', fromlist=['Event']).Event(task_id=t.id, type='checkpoint', message='halfway')))

    store2 = PlatformStore(tmp_path)
    assert store2.get_project(p.id).name == 'persistent'
    assert store2.get_task(t.id).status == TaskStatus.RUNNING
    recovered = asyncio.run(store2.recover_interrupted())
    assert len(recovered) == 1
    assert store2.get_task(t.id).status == TaskStatus.QUEUED
    assert len(asyncio.run(store2.list_events(t.id))) == 1


def test_chat_response_time_migrates_and_persists(tmp_path: Path):
    import sqlite3
    from app.platform.store import PlatformStore

    root = tmp_path / "legacy"
    root.mkdir()
    with sqlite3.connect(root / "platform.db") as db:
        db.execute(
            "CREATE TABLE chat_messages (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO chat_messages(id, project_id, role, content, created_at) VALUES(?,?,?,?,?)",
            ("legacy-message", "legacy-project", "assistant", "An older reply", "2026-01-01T00:00:00+00:00"),
        )

    store = PlatformStore(root)
    old = store.list_chat_messages("legacy-project")[0]
    assert old.content == "An older reply"
    assert old.response_time_ms is None

    project = store.create_project("timed-chat")
    saved = store.add_chat_message(project.id, "assistant", "New reply", response_time_ms=2345)
    restored = PlatformStore(root).list_chat_messages(project.id)[0]
    assert saved.response_time_ms == 2345
    assert restored.response_time_ms == 2345


def test_task_reply_duration_uses_task_creation_time():
    from datetime import datetime, timedelta, timezone
    from app.platform.models import Task
    from app.platform.orchestrator import _task_response_time_ms

    task = Task(
        project_id="project",
        prompt="measure this task",
        created_at=datetime.now(timezone.utc) - timedelta(seconds=3),
    )
    assert _task_response_time_ms(task) >= 2900


def test_auth_disabled_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv('PLATFORM_PASSWORD', raising=False)
    client = TestClient(create_app(tmp_path))
    assert client.get('/').status_code == 200
    assert client.get('/api/projects').status_code == 200


def test_basic_auth_protects_everything_except_health(tmp_path: Path):
    client = TestClient(create_app(tmp_path, password='s3cret'))
    # Health stays open so hosting platforms can poll it without credentials.
    assert client.get('/api/health').json()['status'] == 'ok'
    for path in ('/', '/api/projects', '/static/index.html'):
        r = client.get(path)
        assert r.status_code == 401, path
        assert r.headers['www-authenticate'].startswith('Basic realm=')
    assert client.get('/api/projects', auth=('admin', 'wrong')).status_code == 401
    assert client.get('/api/projects', auth=('someone', 's3cret')).status_code == 401
    assert client.get('/api/projects', auth=('admin', 's3cret')).status_code == 200
    assert client.get('/', auth=('admin', 's3cret')).status_code == 200


def test_basic_auth_reads_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv('PLATFORM_PASSWORD', 'from-env')
    monkeypatch.setenv('PLATFORM_USERNAME', 'ops')
    client = TestClient(create_app(tmp_path))
    assert client.get('/api/projects').status_code == 401
    assert client.get('/api/projects', auth=('ops', 'from-env')).status_code == 200


def test_basic_auth_sse_stream_requires_credentials(tmp_path: Path):
    client = TestClient(create_app(tmp_path, password='s3cret'))
    p = client.post('/api/projects', json={'name': 'demo'}, auth=('admin', 's3cret')).json()
    assert client.get(f"/api/projects/{p['id']}/tasks").status_code == 401
    assert client.get(f"/api/projects/{p['id']}/tasks", auth=('admin', 's3cret')).status_code == 200
