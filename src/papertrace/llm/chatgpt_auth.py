"""Sign in with ChatGPT: OAuth (PKCE) so a local user spends their ChatGPT plan, not API credits.

Follows OpenAI's documented flow for open-source, locally hosted apps:
https://developers.openai.com/siwc/token-sharing-open-source
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field, ValidationError

from papertrace.domain.errors import LLMError

_AUTH = "https://auth.openai.com"
AUTHORIZE_URL = f"{_AUTH}/api/accounts/authorize"
TOKEN_URL = f"{_AUTH}/api/accounts/oauth/token"
DISCOVERY_URL = f"{_AUTH}/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
_PLAN_SCOPE = "chatgpt.tokens.use.direct"
_REGISTRATION = "dynamic_agent_client"  # first sign-in only; OpenAI then issues our own ID
_APP_NAME = "PaperTrace"
_REFRESH_MARGIN = 60.0  # seconds before expiry when the access token is renewed
# Refresh errors after which the token set is dead and the user must sign in again.
_TERMINAL = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}
_CONNECT = "Click Connect ChatGPT to try again."


class _Tokens(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: float
    email: str | None = None


class _Login(BaseModel):
    host_id: str = Field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    client_id: str | None = None
    tokens: _Tokens | None = None


@dataclass(frozen=True)
class _Attempt:
    state: str
    verifier: str
    nonce: str
    redirect_uri: str


class _TokenError(LLMError):
    def __init__(self, code: str) -> None:
        super().__init__(f"ChatGPT sign-in failed ({code}).")
        self.code = code


class ChatGPTAuth:
    """Holds one local ChatGPT login: the OAuth dance, token refresh and sign-out."""

    def __init__(
        self, *, http: httpx.AsyncClient, path: Path, clock: Callable[[], float] = time.time
    ) -> None:
        self._http = http
        self._path = path
        self._clock = clock
        self._login = self._load()
        self._attempt: _Attempt | None = None
        self._lock = asyncio.Lock()  # refresh tokens rotate: never refresh twice at once

    @property
    def connected(self) -> bool:
        return self._login.tokens is not None

    @property
    def email(self) -> str | None:
        return self._login.tokens.email if self._login.tokens else None

    def authorize_url(self, redirect_uri: str) -> str:
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        # shortcut: only the latest attempt is kept, so an older open sign-in tab fails.
        self._attempt = _Attempt(
            secrets.token_urlsafe(32), verifier, secrets.token_urlsafe(32), redirect_uri
        )
        params = {
            "client_id": self._login.client_id or _REGISTRATION,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPE,
            "resource": RESOURCE,
            "state": self._attempt.state,
            "nonce": self._attempt.nonce,
            "code_challenge": challenge.rstrip(b"=").decode(),
            "code_challenge_method": "S256",
            "ext_agent_host_id": self._login.host_id,
        }
        if self._login.client_id is None:
            params["agent_name_hint"] = _APP_NAME
        self._save()  # the host ID must be persisted before its first sign-in
        return f"{AUTHORIZE_URL}?{urlencode(params)}"

    async def finish_login(self, *, code: str, state: str, client_id: str | None) -> None:
        attempt, self._attempt = self._attempt, None
        if attempt is None or not secrets.compare_digest(attempt.state, state):
            raise LLMError(f"This ChatGPT sign-in expired or did not start here. {_CONNECT}")
        # A new registration returns its issued ID on the callback; a reauthorization may not.
        client = client_id or self._login.client_id
        if client is None:
            raise LLMError(f"ChatGPT did not return a client ID. {_CONNECT}")
        try:
            body = await self._token(
                grant_type="authorization_code",
                client_id=client,
                code=code,
                code_verifier=attempt.verifier,
                redirect_uri=attempt.redirect_uri,
            )
        except _TokenError as exc:
            raise LLMError(f"{exc} {_CONNECT}") from exc
        claims = _claims(body.get("id_token"))
        if claims.get("nonce") != attempt.nonce or claims.get("aud") not in (client, [client]):
            raise LLMError(f"ChatGPT returned an ID token for another sign-in. {_CONNECT}")
        self._login.client_id = client
        if _PLAN_SCOPE not in str(body.get("scope", "")).split():
            self._save()
            raise LLMError(
                "ChatGPT plan usage was not granted for PaperTrace. Allow it when signing in, "
                "or set LLM_API_KEY."
            )
        email = claims.get("email")
        self._login.tokens = self._tokens(body, email if isinstance(email, str) else None)
        self._save()

    async def access_token(self) -> str:
        """Current access token, refreshed shortly before it expires."""
        async with self._lock:
            tokens = self._login.tokens
            if tokens is None:
                raise LLMError(
                    "ChatGPT is not connected. Click Connect ChatGPT or set LLM_API_KEY."
                )
            if tokens.expires_at - _REFRESH_MARGIN > self._clock():
                return tokens.access_token
            try:
                body = await self._token(
                    grant_type="refresh_token",
                    client_id=self._login.client_id or "",
                    refresh_token=tokens.refresh_token,
                )
            except _TokenError as exc:
                if exc.code not in _TERMINAL and exc.code != "401":
                    raise LLMError(
                        f"Could not renew the ChatGPT session ({exc.code}). Try again shortly."
                    ) from exc
                self._login.tokens = None
                self._save()
                raise LLMError(
                    "The ChatGPT session ended. Click Connect ChatGPT to sign in again."
                ) from exc
            self._login.tokens = self._tokens(body, tokens.email)
            self._save()
            return self._login.tokens.access_token

    async def logout(self) -> bool:
        """Revoke and forget the tokens; False when OpenAI did not confirm the revocation."""
        tokens = self._login.tokens
        if tokens is None:
            return True
        revoked = False
        # shortcut: one revocation attempt, no backoff; the user can also disconnect in ChatGPT.
        try:
            endpoint = (await self._http.get(DISCOVERY_URL)).json()["revocation_endpoint"]
            # The refresh token only ever goes to OpenAI's own auth host.
            if isinstance(endpoint, str) and endpoint.startswith(f"{_AUTH}/"):
                response = await self._http.post(
                    endpoint,
                    data={
                        "token": tokens.refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": self._login.client_id or "",
                    },
                )
                revoked = response.status_code == 200
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
        self._login.tokens = None
        self._save()
        return revoked

    async def _token(self, **form: str) -> dict[str, Any]:
        try:
            response = await self._http.post(TOKEN_URL, data={**form, "resource": RESOURCE})
        except httpx.HTTPError as exc:
            raise LLMError(
                "Cannot reach auth.openai.com. Check the connection and try again."
            ) from exc
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code == 200 and isinstance(body, dict) and "access_token" in body:
            return body
        # Only the error code is relayed: token responses can echo secrets.
        code = body.get("error") if isinstance(body, dict) else None
        raise _TokenError(code if isinstance(code, str) else str(response.status_code))

    def _tokens(self, body: dict[str, Any], email: str | None) -> _Tokens:
        try:
            return _Tokens(
                access_token=body["access_token"],
                refresh_token=body["refresh_token"],
                expires_at=self._clock() + float(body["expires_in"]),
                email=email,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise LLMError(f"ChatGPT returned an incomplete token set. {_CONNECT}") from exc

    def _load(self) -> _Login:
        try:
            return _Login.model_validate_json(self._path.read_text())
        except (OSError, ValidationError):
            return _Login()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp = self._path.with_suffix(".tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write(self._login.model_dump_json())
        os.replace(temp, self._path)  # atomic: a rotated refresh token is never half-written


def _claims(id_token: object) -> dict[str, Any]:
    # The ID token comes straight from the token endpoint over TLS, so the TLS check stands in
    # for the signature (OIDC Core 3.1.3.7); nonce and audience are still checked by the caller.
    try:
        payload = str(id_token).split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}
