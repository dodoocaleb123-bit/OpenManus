import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from app.platform.browser import BrowserSession
from app.server import create_app


def test_browser_api_requires_project():
    with tempfile.TemporaryDirectory() as d:
        app = create_app(Path(d))
        with TestClient(app) as client:
            response = client.post('/api/browser/sessions', json={'project_id': 'missing', 'url': 'about:blank'})
            assert response.status_code == 404


def test_browser_session_status_endpoint_404():
    with tempfile.TemporaryDirectory() as d:
        app = create_app(Path(d))
        with TestClient(app) as client:
            response = client.get('/api/browser/sessions/nope')
            assert response.status_code == 404


# --- human takeover endpoints, driven through a fake Playwright page ---------


class _FakeMouse:
    def __init__(self, log):
        self.log = log

    async def click(self, x, y):
        self.log.append(("click", x, y))


class _FakeKeyboard:
    def __init__(self, log):
        self.log = log

    async def type(self, text):
        self.log.append(("type", text))

    async def press(self, key):
        self.log.append(("press", key))


class _FakePage:
    def __init__(self):
        self.log = []
        self.url = "http://localhost:3000/"
        self.mouse = _FakeMouse(self.log)
        self.keyboard = _FakeKeyboard(self.log)

    def is_closed(self):
        return False

    async def title(self):
        return "Dev server"

    async def goto(self, url, **_):
        self.log.append(("goto", url))
        self.url = url


def _attach_fake_session(app, project_id):
    browsers = app.state.browsers
    session = BrowserSession("fake01", project_id, browsers.root)
    session.page = _FakePage()
    browsers.sessions[session.id] = session
    return session


def test_human_takeover_click_keyboard_and_project_session():
    with tempfile.TemporaryDirectory() as d:
        app = create_app(Path(d), check_llm=False)
        with TestClient(app) as client:
            project = client.post("/api/projects", json={"name": "Shop"}).json()
            assert client.get(f"/api/browser/projects/{project['id']}/session").status_code == 404

            session = _attach_fake_session(app, project["id"])
            found = client.get(f"/api/browser/projects/{project['id']}/session").json()
            assert found["id"] == "fake01" and found["running"] and found["title"] == "Dev server"

            assert client.post("/api/browser/sessions/fake01/click_at", json={"x": 120.5, "y": 40}).status_code == 200
            r = client.post("/api/browser/sessions/fake01/keyboard", json={"text": "hello", "key": "Enter"})
            assert r.status_code == 200
            assert session.page.log == [("click", 120.5, 40), ("type", "hello"), ("press", "Enter")]

            # Out-of-range coordinates are rejected before touching the page.
            assert client.post("/api/browser/sessions/fake01/click_at", json={"x": -1, "y": 5}).status_code == 422
            assert client.post("/api/browser/sessions/nope/keyboard", json={"key": "Tab"}).status_code == 404

            # Opening "a browser" for the project reuses the shared session the agent uses.
            r = client.post("/api/browser/sessions", json={"project_id": project["id"], "url": "http://localhost:3000/cart"})
            assert r.status_code == 200 and r.json()["id"] == "fake01"
            assert session.page.log[-1] == ("goto", "http://localhost:3000/cart")
            assert len(app.state.browsers.sessions) == 1
