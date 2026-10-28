from pathlib import Path

import pytest
import uvicorn

from papertrace import server


def test_missing_configuration_prints_friendly_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for var in ("LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no env file to pick up

    assert server.main() == 1
    assert "LLM_API_KEY" in capsys.readouterr().err


def test_main_starts_uvicorn_on_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))

    assert server.main() == 0
    assert calls == [("papertrace.api.app:app", {"host": "127.0.0.1", "port": 8000})]
