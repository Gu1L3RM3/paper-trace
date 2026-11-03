import base64
import json
import stat
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from papertrace.domain.errors import LLMError
from papertrace.llm.chatgpt_auth import (
    AUTHORIZE_URL,
    DISCOVERY_URL,
    SCOPE,
    TOKEN_URL,
    ChatGPTAuth,
    SignInRetry,
)

REDIRECT = "http://127.0.0.1:8000/auth/callback"
REVOKE_URL = "https://auth.openai.com/api/accounts/oauth/revoke"


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _jwt(claims: dict[str, Any]) -> str:
    def part(data: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256'})}.{part(claims)}.sig"


def _tokens(nonce: str, *, access: str = "at-1", refresh: str = "rt-1", **extra: Any) -> Any:
    return {
        "access_token": access,
        "refresh_token": refresh,
        "id_token": _jwt({"nonce": nonce, "aud": "oaiapp_1", "email": "ana@example.com"}),
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": SCOPE,
        **extra,
    }


def _query(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def _form(route: respx.Route, call: int = -1) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(route.calls[call].request.content.decode()).items()}


async def _connect(auth: ChatGPTAuth, route: respx.Route | None = None) -> None:
    params = _query(auth.authorize_url(REDIRECT))
    (route or respx.post(TOKEN_URL)).mock(
        return_value=httpx.Response(200, json=_tokens(params["nonce"]))
    )
    await auth.finish_login(code="c", state=params["state"], client_id="oaiapp_1")


def _auth(http: httpx.AsyncClient, path: Path, clock: Clock | None = None) -> ChatGPTAuth:
    return ChatGPTAuth(http=http, path=path, clock=clock or Clock())


async def test_first_authorize_url_registers_with_pkce_and_a_persisted_host(tmp_path: Path) -> None:
    path = tmp_path / "chatgpt.json"
    async with httpx.AsyncClient() as http:
        url = _auth(http, path).authorize_url(REDIRECT)

    assert url.startswith(AUTHORIZE_URL + "?")
    params = _query(url)
    assert params["client_id"] == "dynamic_agent_client"
    assert params["agent_name_hint"] == "PaperTrace"
    assert params["response_type"] == "code"
    assert params["redirect_uri"] == REDIRECT
    assert params["scope"] == SCOPE
    assert params["resource"] == "https://api.openai.com/v1"
    assert params["code_challenge_method"] == "S256"
    assert len(params["code_challenge"]) == 43  # base64url SHA-256, no padding
    assert params["state"] and params["nonce"] and params["state"] != params["nonce"]
    assert params["ext_agent_host_id"].startswith("urn:uuid:")
    # The host ID is chosen once, before the first sign-in, and reused.
    assert json.loads(path.read_text())["host_id"] == params["ext_agent_host_id"]
    async with httpx.AsyncClient() as http:
        again = _query(_auth(http, path).authorize_url(REDIRECT))
    assert again["ext_agent_host_id"] == params["ext_agent_host_id"]


@respx.mock
async def test_login_exchanges_the_code_and_saves_a_private_file(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "chatgpt.json"
    route = respx.post(TOKEN_URL)
    async with httpx.AsyncClient() as http:
        auth = _auth(http, path)
        await _connect(auth, route)

        assert auth.connected and auth.email == "ana@example.com"
        assert await auth.access_token() == "at-1"

    form = _form(route)
    assert form["grant_type"] == "authorization_code"
    assert form["client_id"] == "oaiapp_1"
    assert form["code"] == "c"
    assert form["redirect_uri"] == REDIRECT
    assert form["resource"] == "https://api.openai.com/v1"
    assert form["code_verifier"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    saved = json.loads(path.read_text())
    assert saved["client_id"] == "oaiapp_1" and saved["tokens"]["refresh_token"] == "rt-1"


@respx.mock
async def test_reauthorization_reuses_the_issued_client_id(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json")
        await _connect(auth)
        params = _query(auth.authorize_url(REDIRECT))

    assert params["client_id"] == "oaiapp_1"
    assert "agent_name_hint" not in params


async def test_unknown_state_is_rejected_without_calling_openai(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json")
        auth.authorize_url(REDIRECT)
        with pytest.raises(LLMError, match="Connect ChatGPT"):
            await auth.finish_login(code="c", state="forged", client_id="oaiapp_1")
    assert not auth.connected


@respx.mock
async def test_wrong_nonce_is_rejected(tmp_path: Path) -> None:
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json=_tokens("other")))
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json")
        state = _query(auth.authorize_url(REDIRECT))["state"]
        with pytest.raises(LLMError):
            await auth.finish_login(code="c", state=state, client_id="oaiapp_1")
    assert not auth.connected


@respx.mock
async def test_missing_plan_scope_is_reported_and_not_saved(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json")
        params = _query(auth.authorize_url(REDIRECT))
        body = _tokens(params["nonce"], scope="openid email profile")
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(LLMError, match="plan usage"):
            await auth.finish_login(code="c", state=params["state"], client_id="oaiapp_1")
    assert not auth.connected


@respx.mock
async def test_token_endpoint_error_relays_only_the_error_code(tmp_path: Path) -> None:
    respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(400, json={"error": "invalid_request", "detail": "rt-secret"})
    )
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json")
        state = _query(auth.authorize_url(REDIRECT))["state"]
        with pytest.raises(LLMError, match="invalid_request") as caught:
            await auth.finish_login(code="c", state=state, client_id="oaiapp_1")
    assert "rt-secret" not in str(caught.value)


@respx.mock
async def test_rejected_code_keeps_the_issued_client_and_retries_once(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(400, json={"error": "invalid_grant"}))
    async with httpx.AsyncClient() as http:
        auth = _auth(http, path)
        state = _query(auth.authorize_url(REDIRECT))["state"]
        with pytest.raises(SignInRetry) as retry:
            await auth.finish_login(code="c", state=state, client_id="oaiapp_1")

        # A fresh authorization with the same registration, never a new one.
        params = _query(retry.value.url)
        assert retry.value.url.startswith(AUTHORIZE_URL)
        assert params["client_id"] == "oaiapp_1" and "agent_name_hint" not in params
        assert params["redirect_uri"] == REDIRECT
        assert json.loads(path.read_text())["client_id"] == "oaiapp_1"

        with pytest.raises(LLMError, match="invalid_grant") as second:
            await auth.finish_login(code="c2", state=params["state"], client_id=None)
        assert not isinstance(second.value, SignInRetry)
    assert not auth.connected


async def test_pkce_verifier_matches_the_devkit_size(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json")
        auth.authorize_url(REDIRECT)
    assert auth._attempt is not None and len(auth._attempt.verifier) == 43


@respx.mock
async def test_network_failure_on_login_becomes_llm_error(tmp_path: Path) -> None:
    respx.post(TOKEN_URL).mock(side_effect=httpx.ConnectError("down"))
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json")
        state = _query(auth.authorize_url(REDIRECT))["state"]
        with pytest.raises(LLMError, match=r"auth\.openai\.com"):
            await auth.finish_login(code="c", state=state, client_id="oaiapp_1")


@respx.mock
async def test_expiring_token_is_refreshed_and_the_rotated_refresh_token_saved(
    tmp_path: Path,
) -> None:
    clock = Clock()
    path = tmp_path / "c.json"
    async with httpx.AsyncClient() as http:
        auth = _auth(http, path, clock)
        await _connect(auth)
        clock.now += 3590  # inside the refresh margin
        route = respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(200, json=_tokens("n", access="at-2", refresh="rt-2"))
        )
        assert await auth.access_token() == "at-2"
        assert await auth.access_token() == "at-2"  # fresh now: no second refresh

    assert route.call_count == 2  # login exchange + one refresh, same mocked route
    form = _form(route)
    assert form == {
        "grant_type": "refresh_token",
        "client_id": "oaiapp_1",
        "refresh_token": "rt-1",
        "resource": "https://api.openai.com/v1",
    }
    saved = json.loads(path.read_text())["tokens"]
    assert saved["refresh_token"] == "rt-2" and saved["email"] == "ana@example.com"


@respx.mock
async def test_terminal_refresh_error_clears_tokens_but_keeps_the_client(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "c.json"
    async with httpx.AsyncClient() as http:
        auth = _auth(http, path, clock)
        await _connect(auth)
        clock.now += 4000
        respx.post(TOKEN_URL).mock(
            return_value=httpx.Response(400, json={"error": "refresh_token_reused"})
        )
        with pytest.raises(LLMError, match="sign in again"):
            await auth.access_token()

    assert not auth.connected
    saved = json.loads(path.read_text())
    assert saved["tokens"] is None and saved["client_id"] == "oaiapp_1"


@respx.mock
async def test_temporary_refresh_failure_keeps_the_tokens(tmp_path: Path) -> None:
    clock = Clock()
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json", clock)
        await _connect(auth)
        clock.now += 4000
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(503))
        with pytest.raises(LLMError):
            await auth.access_token()
    assert auth.connected


async def test_access_token_without_login_asks_to_connect(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match="Connect ChatGPT"):
            await _auth(http, tmp_path / "c.json").access_token()


@respx.mock
async def test_saved_login_is_loaded_on_start(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    async with httpx.AsyncClient() as http:
        await _connect(_auth(http, path))
        reloaded = _auth(http, path)
        assert reloaded.connected and await reloaded.access_token() == "at-1"


async def test_corrupt_file_starts_disconnected(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("{not json")
    async with httpx.AsyncClient() as http:
        assert not _auth(http, path).connected


@respx.mock
async def test_logout_revokes_the_refresh_token_and_keeps_the_registration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "c.json"
    respx.get(DISCOVERY_URL).mock(
        return_value=httpx.Response(200, json={"revocation_endpoint": REVOKE_URL})
    )
    revoke = respx.post(REVOKE_URL).mock(return_value=httpx.Response(200))
    async with httpx.AsyncClient() as http:
        auth = _auth(http, path)
        await _connect(auth)
        assert await auth.logout() is True

    assert _form(revoke) == {
        "token": "rt-1",
        "token_type_hint": "refresh_token",
        "client_id": "oaiapp_1",
    }
    assert not auth.connected
    saved = json.loads(path.read_text())
    assert saved["tokens"] is None and saved["client_id"] == "oaiapp_1" and saved["host_id"]


@respx.mock
@pytest.mark.parametrize(
    "discovery",
    [
        httpx.Response(200, json={"revocation_endpoint": "https://evil.test/revoke"}),
        httpx.Response(500),
    ],
)
async def test_logout_clears_locally_when_revocation_is_not_confirmed(
    tmp_path: Path, discovery: httpx.Response
) -> None:
    respx.get(DISCOVERY_URL).mock(return_value=discovery)
    evil = respx.post("https://evil.test/revoke")
    async with httpx.AsyncClient() as http:
        auth = _auth(http, tmp_path / "c.json")
        await _connect(auth)
        assert await auth.logout() is False
    assert not auth.connected and not evil.called


async def test_logout_when_disconnected_is_a_no_op(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        assert await _auth(http, tmp_path / "c.json").logout() is True
