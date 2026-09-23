import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
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
