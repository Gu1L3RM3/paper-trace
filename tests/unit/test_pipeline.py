from papertrace.domain.errors import LLMError, SourceError
from papertrace.domain.models import (
    Paper,
    ResearchQuestion,
    ResearchResult,
    SearchPlan,
    SearchQuery,
    SynthesisResult,
)
from papertrace.pipeline import Pipeline, StageEvent

QUESTION = ResearchQuestion(text="q")
PLAN = SearchPlan(queries=[SearchQuery(query="a", angle="x"), SearchQuery(query="b", angle="y")])
PAPER = Paper(title="T", authors=["A B"], year=2020, doi=None, abstract=None, url="https://x")
RESULT = ResearchResult(papers=[PAPER], failed_queries=["b"])
SYNTHESIS = SynthesisResult(markdown="m", unmatched_citations=[], insufficient_evidence=False)


class _Planner:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def plan(self, question: ResearchQuestion) -> SearchPlan:
        if self._error:
            raise self._error
        return PLAN


class _Researcher:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def research(self, plan: SearchPlan) -> ResearchResult:
        if self._error:
            raise self._error
        return RESULT


class _Synthesizer:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.received: list[Paper] | None = None

    async def synthesize(self, question: ResearchQuestion, papers: list[Paper]) -> SynthesisResult:
        if self._error:
            raise self._error
        self.received = papers
        return SYNTHESIS


async def _run(pipeline: Pipeline) -> list[StageEvent]:
    return [event async for event in pipeline.run(QUESTION)]


async def test_happy_path_yields_stages_in_order_with_payloads() -> None:
    synthesizer = _Synthesizer()

    events = await _run(Pipeline(_Planner(), _Researcher(), synthesizer))

    assert [e.name for e in events] == ["plan", "papers", "synthesis", "done"]
    assert events[0].payload == PLAN
    assert events[1].payload == RESULT
    assert events[2].payload == SYNTHESIS
    assert events[3].payload is None
    assert synthesizer.received == [PAPER]


async def test_failure_becomes_error_event_and_stops_the_run() -> None:
    cases = [
        (Pipeline(_Planner(LLMError("llm down")), _Researcher(), _Synthesizer()), ["error"]),
        (
            Pipeline(_Planner(), _Researcher(SourceError("oa down")), _Synthesizer()),
            ["plan", "error"],
        ),
        (
            Pipeline(_Planner(), _Researcher(), _Synthesizer(LLMError("boom"))),
            ["plan", "papers", "error"],
        ),
    ]
    for pipeline, expected in cases:
        events = await _run(pipeline)
        assert [e.name for e in events] == expected
        assert events[-1].payload is not None


async def test_unexpected_exception_still_becomes_error_event() -> None:
    events = await _run(
        Pipeline(_Planner(ValueError("secret detail")), _Researcher(), _Synthesizer())
    )

    assert [e.name for e in events] == ["error"]
    assert "secret detail" not in str(events[0].payload)


async def test_zero_papers_with_failed_searches_is_an_error_not_missing_evidence() -> None:
    class _Partial(_Researcher):
        async def research(self, plan: SearchPlan) -> ResearchResult:
            return ResearchResult(papers=[], failed_queries=["a"])

    synthesizer = _Synthesizer()
    events = await _run(Pipeline(_Planner(), _Partial(), synthesizer))

    assert [e.name for e in events] == ["plan", "papers", "error"]
    assert synthesizer.received is None
