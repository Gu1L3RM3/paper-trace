import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from papertrace.api.app import app
from papertrace.api.dependencies import get_history, get_pipeline
from papertrace.domain.errors import SourceError
from papertrace.history import SQLiteHistory
from papertrace.pipeline import Pipeline
from tests.fakes import FakeVerifier
from tests.integration.test_research_api import _Planner, _Researcher, _stream, _Synthesizer


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    store = SQLiteHistory(tmp_path / "history.sqlite3")
    app.dependency_overrides[get_history] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def _pipeline(error: Exception | None = None) -> Pipeline:
    return Pipeline(_Planner(), _Researcher(error), _Synthesizer(), FakeVerifier())


def test_finished_research_is_saved_and_replayable(client: TestClient) -> None:
    app.dependency_overrides[get_pipeline] = lambda: _pipeline()
    streamed = _stream(client)

    [entry] = client.get("/api/history").json()
    assert entry["question"] == "q" and entry["status"] == "complete"
    saved = client.get(f"/api/history/{entry['id']}").json()
    assert [(e["event"], json.loads(e["data"])) for e in saved] == streamed


def test_failed_research_is_saved_as_an_error(client: TestClient) -> None:
    app.dependency_overrides[get_pipeline] = lambda: _pipeline(SourceError("OpenAlex down"))
    _stream(client)

    assert client.get("/api/history").json()[0]["status"] == "error"


def test_research_can_be_deleted(client: TestClient) -> None:
    app.dependency_overrides[get_pipeline] = lambda: _pipeline()
    _stream(client)
    entry = client.get("/api/history").json()[0]["id"]

    assert client.delete(f"/api/history/{entry}").status_code == 204
    assert client.get(f"/api/history/{entry}").status_code == 404
    assert client.delete(f"/api/history/{entry}").status_code == 404
    assert client.get("/api/history").json() == []
