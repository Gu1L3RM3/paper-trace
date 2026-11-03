from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from papertrace.api.app import app
from papertrace.api.dependencies import rewire
from papertrace.llm.chatgpt_auth import AUTHORIZE_URL, DISCOVERY_URL, TOKEN_URL, ChatGPTAuth
from papertrace.settings import Settings
from tests.unit.test_chatgpt_auth import _tokens

REVOKE_URL = "https://auth.openai.com/api/accounts/oauth/revoke"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    http = httpx.AsyncClient()
    app.state.settings = Settings(_env_file=None, llm_model="m")
    app.state.http = http
    app.state.chatgpt = ChatGPTAuth(http=http, path=tmp_path / "chatgpt.json")
    rewire(app.state)
    with respx.mock:
        yield TestClient(app, base_url="http://localhost:8000", follow_redirects=False)
    for name in ("settings", "http", "chatgpt", "pipeline", "chatgpt_error", "chatgpt_wired"):
        if hasattr(app.state, name):
            delattr(app.state, name)


def _sign_in(client: TestClient) -> None:
    login = client.get("/auth/chatgpt/login")
    params = {k: v[0] for k, v in parse_qs(urlsplit(login.headers["location"]).query).items()}
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json=_tokens(params["nonce"])))
    callback = client.get(
        "/auth/callback", params={"code": "c", "state": params["state"], "client_id": "oaiapp_1"}
    )
    assert callback.status_code == 303 and callback.headers["location"] == "/"


def test_research_without_credentials_asks_to_connect(client: TestClient) -> None:
    response = client.post("/api/research", json={"text": "q"})

    assert response.status_code == 503
    assert "Connect ChatGPT" in response.json()["detail"]


def test_login_redirects_to_openai_with_the_loopback_callback(client: TestClient) -> None:
    response = client.get("/auth/chatgpt/login")

    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith(AUTHORIZE_URL)
    # SIWC only accepts 127.0.0.1, even when the page was opened on localhost.
    assert parse_qs(urlsplit(location).query)["redirect_uri"] == [
        "http://127.0.0.1:8000/auth/callback"
    ]


def test_callback_connects_and_wires_the_plan_pipeline(client: TestClient) -> None:
    _sign_in(client)

    assert client.get("/api/auth/chatgpt").json() == {
        "connected": True,
        "email": "ana@example.com",
        "error": None,
    }
    assert app.state.pipeline is not None


@pytest.mark.parametrize(
    ("params", "hint"),
    [({"code": "c", "state": "forged"}, "did not start here"), ({"error": "access_denied"}, "")],
)
def test_failed_callback_is_reported_once(
    client: TestClient, params: dict[str, str], hint: str
) -> None:
    client.get("/auth/chatgpt/login")

    callback = client.get("/auth/callback", params=params)

    assert callback.status_code == 303
    status = client.get("/api/auth/chatgpt").json()
    assert status["connected"] is False and hint in status["error"]
    assert client.get("/api/auth/chatgpt").json()["error"] is None


def test_logout_revokes_and_unwires_the_pipeline(client: TestClient) -> None:
    _sign_in(client)
    respx.get(DISCOVERY_URL).mock(
        return_value=httpx.Response(200, json={"revocation_endpoint": REVOKE_URL})
    )
    respx.post(REVOKE_URL).mock(return_value=httpx.Response(200))

    assert client.post("/api/auth/chatgpt/logout").json() == {"revoked": True}
    assert client.get("/api/auth/chatgpt").json()["connected"] is False
    assert client.post("/api/research", json={"text": "q"}).status_code == 503


def test_session_that_ends_elsewhere_falls_back_on_next_request(client: TestClient) -> None:
    _sign_in(client)
    app.state.chatgpt._login.tokens = None  # e.g. a terminal refresh error

    assert client.post("/api/research", json={"text": "q"}).status_code == 503


def test_refused_code_sends_the_browser_to_a_fresh_sign_in(client: TestClient) -> None:
    login = client.get("/auth/chatgpt/login")
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(400, json={"error": "invalid_grant"}))

    callback = client.get(
        "/auth/callback", params={"code": "c", "state": state, "client_id": "oaiapp_1"}
    )

    assert callback.status_code == 303
    location = callback.headers["location"]
    assert location.startswith(AUTHORIZE_URL)
    assert parse_qs(urlsplit(location).query)["client_id"] == ["oaiapp_1"]
