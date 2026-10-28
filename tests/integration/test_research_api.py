import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from papertrace.api.app import app
from papertrace.api.dependencies import get_pipeline
from papertrace.domain.errors import SourceError
from papertrace.domain.models import (
    Paper,
    ResearchQuestion,
    ResearchResult,
    SearchPlan,
    SearchQuery,
    SynthesisResult,
)
from papertrace.domain.verification import VerificationResult
from papertrace.pipeline import Pipeline
from tests.fakes import FakeVerifier

PAPER = Paper(title="T", authors=["A B"], year=2020, doi=None, abstract=None, url="https://x")


class _Planner:
    async def plan(self, question: ResearchQuestion) -> SearchPlan:
        return SearchPlan(queries=[SearchQuery(query=q, angle=q) for q in ("a", "b")])


class _Researcher:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def research(self, plan: SearchPlan) -> ResearchResult:
        if self._error:
            raise self._error
        return ResearchResult(papers=[PAPER], failed_queries=[])


class _Synthesizer:
    async def revise(
        self,
        question: ResearchQuestion,
        papers: list[Paper],
        synthesis: SynthesisResult,
        verification: VerificationResult,
    ) -> SynthesisResult:
        return synthesis

    async def synthesize(self, question: ResearchQuestion, papers: list[Paper]) -> SynthesisResult:
        return SynthesisResult(
            markdown="ok (A, 2020) (Ghost, 1999)",
            unmatched_citations=["Ghost, 1999"],
            insufficient_evidence=False,
        )


def _stream(client: TestClient) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    with client.stream("POST", "/api/research", json={"text": "q"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        name = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1])))
    return events


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_streams_stages_in_order_with_typed_payloads(client: TestClient) -> None:
    app.dependency_overrides[get_pipeline] = lambda: Pipeline(
        _Planner(), _Researcher(), _Synthesizer(), FakeVerifier()
    )

    events = _stream(client)

    assert [name for name, _ in events] == ["plan", "papers", "synthesis", "verification", "done"]
    assert len(events[0][1]["queries"]) == 2  # type: ignore[arg-type]
    assert events[1][1]["papers"][0]["title"] == "T"  # type: ignore[index]
    assert events[2][1]["unmatched_citations"] == ["Ghost, 1999"]


def test_stage_failure_arrives_as_error_event_not_500(client: TestClient) -> None:
    app.dependency_overrides[get_pipeline] = lambda: Pipeline(
        _Planner(), _Researcher(SourceError("OpenAlex down")), _Synthesizer(), FakeVerifier()
    )

    events = _stream(client)

    assert [name for name, _ in events] == ["plan", "error"]
    assert events[1][1] == {"message": "OpenAlex down"}


def test_blank_question_is_rejected(client: TestClient) -> None:
    app.dependency_overrides[get_pipeline] = lambda: Pipeline(
        _Planner(), _Researcher(), _Synthesizer(), FakeVerifier()
    )
    response = client.post("/api/research", json={"text": "  "})

    assert response.status_code == 422
