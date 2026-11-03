import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from papertrace import server
from papertrace.settings import Settings


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[list[str]]:
    """No env file, no real browser, no real server."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for var in ("LLM_MODEL", "LLM_API_KEY", "LLM_MAX_CONCURRENCY"):
        monkeypatch.delenv(var, raising=False)
    opened: list[str] = []
    monkeypatch.setattr(server, "_open_when_ready", lambda url, port: opened.append(url))
    yield opened


def _run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append({"app": app, **kw}))
    return calls


def test_invalid_configuration_prints_friendly_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "many")

    assert server.main() == 1
    assert "LLM_MAX_CONCURRENCY" in capsys.readouterr().err


def test_starts_with_no_configuration_and_opens_the_browser(
    monkeypatch: pytest.MonkeyPatch, isolated: list[str]
) -> None:
    calls = _run(monkeypatch)
    monkeypatch.setattr(server, "_PORT", _free())

    assert server.main() == 0
    port = server._PORT
    assert calls == [{"app": "papertrace.api.app:app", "host": "127.0.0.1", "port": port}]
    assert isolated == [f"http://127.0.0.1:{port}/"]


def test_busy_default_port_falls_back_to_a_free_one(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _run(monkeypatch)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        monkeypatch.setattr(server, "_PORT", busy.getsockname()[1])

        assert server.main() == 0
    assert calls[0]["port"] not in (server._PORT, 0)


def test_browser_opens_once_the_server_listens(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    with socket.socket() as listening:
        listening.bind(("127.0.0.1", 0))
        listening.listen()
        port = listening.getsockname()[1]
        _real_open_when_ready("http://x/", port)
    assert opened == ["http://x/"]


def _free() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# Captured before the autouse fixture replaces it.
_real_open_when_ready = server._open_when_ready
