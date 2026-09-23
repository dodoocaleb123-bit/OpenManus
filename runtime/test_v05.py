from pathlib import Path
from app.platform.store import PlatformStore

def test_task_browser_session_persistence(tmp_path: Path):
    s=PlatformStore(tmp_path); p=s.create_project('Demo')
    t=s.create_task(p.id,'browse'); t.browser_session_id='abc123'
    import asyncio; asyncio.run(s.save_task(t))
    got=s.get_task(t.id)
    assert got.browser_session_id=='abc123'
