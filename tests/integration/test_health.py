from fastapi.testclient import TestClient

from papertrace.api.app import app


def test_health_returns_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_the_frontend() -> None:
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert '<form id="ask"' in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/app.css").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
