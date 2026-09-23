from pathlib import Path
from fastapi.testclient import TestClient

from app.server import create_app


def test_health(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    assert client.get('/api/health').json()['status'] == 'ok'


def test_project_and_task(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
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
