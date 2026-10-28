from typing import Literal

from papertrace.domain.errors import LLMError, SourceError
from papertrace.domain.models import (
    Paper,
    ResearchQuestion,
    ResearchResult,
    SearchPlan,
    SearchQuery,
    SynthesisResult,
)
from papertrace.domain.verification import VerificationResult
from papertrace.pipeline import Pipeline, StageEvent
from tests.fakes import FakeVerifier

QUESTION = ResearchQuestion(text="q")
PLAN = SearchPlan(queries=[SearchQuery(query="a", angle="x"), SearchQuery(query="b", angle="y")])
PAPER = Paper(
    title="T",
    authors=["A B"],
    year=2020,
    doi=None,
    abstract=None,
    url="https://x",
)
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

    async def revise(
        self,
        question: ResearchQuestion,
        papers: list[Paper],
        synthesis: SynthesisResult,
        verification: VerificationResult,
    ) -> SynthesisResult:
        return synthesis

    async def synthesize(self, question: ResearchQuestion, papers: list[Paper]) -> SynthesisResult:
        if self._error:
            raise self._error
        self.received = papers
        return SYNTHESIS


async def _run(pipeline: Pipeline) -> list[StageEvent]:
    return [event async for event in pipeline.run(QUESTION)]


async def test_happy_path_yields_stages_in_order_with_payloads() -> None:
    synthesizer = _Synthesizer()

    events = await _run(Pipeline(_Planner(), _Researcher(), synthesizer, FakeVerifier()))

    assert [e.name for e in events] == ["plan", "papers", "synthesis", "verification", "done"]
    assert events[0].payload == PLAN
    assert events[1].payload == RESULT
    assert events[2].payload == SYNTHESIS
    assert events[4].payload is None
    assert synthesizer.received == [PAPER]


async def test_failure_becomes_error_event_and_stops_the_run() -> None:
    cases = [
        (
            Pipeline(_Planner(LLMError("llm down")), _Researcher(), _Synthesizer(), FakeVerifier()),
            ["error"],
        ),
        (
            Pipeline(
                _Planner(), _Researcher(SourceError("oa down")), _Synthesizer(), FakeVerifier()
            ),
            ["plan", "error"],
        ),
        (
            Pipeline(_Planner(), _Researcher(), _Synthesizer(LLMError("boom")), FakeVerifier()),
            ["plan", "papers", "error"],
        ),
    ]
    for pipeline, expected in cases:
        events = await _run(pipeline)
        assert [e.name for e in events] == expected
        assert events[-1].payload is not None


async def test_unexpected_exception_still_becomes_error_event() -> None:
    events = await _run(
        Pipeline(
            _Planner(ValueError("secret detail")), _Researcher(), _Synthesizer(), FakeVerifier()
        )
    )

    assert [e.name for e in events] == ["error"]
    assert "secret detail" not in str(events[0].payload)


async def test_zero_papers_with_failed_searches_is_an_error_not_missing_evidence() -> None:
    class _Partial(_Researcher):
        async def research(self, plan: SearchPlan) -> ResearchResult:
            return ResearchResult(papers=[], failed_queries=["a"])

    synthesizer = _Synthesizer()
    events = await _run(Pipeline(_Planner(), _Partial(), synthesizer, FakeVerifier()))

    assert [e.name for e in events] == ["plan", "papers", "error"]
    assert synthesizer.received is None


class _Verifier:
    def __init__(
        self,
        status: Literal["no_issues_found", "issues_found", "not_verifiable"] = "no_issues_found",
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.error = error
        self.calls = 0

    async def verify(
        self, question: ResearchQuestion, synthesis: SynthesisResult, papers: list[Paper]
    ) -> VerificationResult:
        from papertrace.domain.verification import VerificationResult

        self.calls += 1
        if self.error:
            raise self.error
        return VerificationResult(claims=[], findings=[], status=self.status, reason="test")


async def test_revision_is_bounded_and_only_final_is_emitted() -> None:
    class Revising(_Synthesizer):
        calls = 0

        async def revise(
            self,
            question: ResearchQuestion,
            papers: list[Paper],
            synthesis: SynthesisResult,
            verification: VerificationResult,
        ) -> SynthesisResult:
            self.calls += 1
            return synthesis.model_copy(update={"markdown": "final"})

    synth = Revising()
    from tests.fakes import papers

    class RealResearcher(_Researcher):
        async def research(self, plan: SearchPlan) -> ResearchResult:
            return ResearchResult(papers=papers(), failed_queries=[])

    verifier = _Verifier("issues_found")
    events = await _run(Pipeline(_Planner(), RealResearcher(), synth, verifier))
    assert [e.name for e in events] == ["plan", "papers", "synthesis", "verification", "done"]
    assert isinstance(events[2].payload, SynthesisResult)
    assert isinstance(events[3].payload, VerificationResult)
    assert events[2].payload.markdown == "final"
    assert events[3].payload.revision_count == 1
    assert synth.calls == 1 and verifier.calls == 2


async def test_verifier_error_hides_draft() -> None:
    events = await _run(
        Pipeline(_Planner(), _Researcher(), _Synthesizer(), _Verifier(error=LLMError("down")))
    )
    assert [e.name for e in events] == ["plan", "papers", "error"]


async def test_second_verification_and_revision_failures_hide_final() -> None:
    from tests.fakes import papers

    class RealResearcher(_Researcher):
        async def research(self, plan: SearchPlan) -> ResearchResult:
            return ResearchResult(papers=papers(), failed_queries=[])

    class SecondFailure(_Verifier):
        async def verify(
            self, question: ResearchQuestion, synthesis: SynthesisResult, papers: list[Paper]
        ) -> VerificationResult:
            if self.calls:
                raise LLMError("second verification failed")
            return await super().verify(question, synthesis, papers)

    class RevisionFailure(_Synthesizer):
        async def revise(
            self,
            question: ResearchQuestion,
            papers: list[Paper],
            synthesis: SynthesisResult,
            verification: VerificationResult,
        ) -> SynthesisResult:
            raise LLMError("revision failed")

    for synth, verifier in [
        (_Synthesizer(), SecondFailure("issues_found")),
        (RevisionFailure(), _Verifier("issues_found")),
    ]:
        events = await _run(Pipeline(_Planner(), RealResearcher(), synth, verifier))
        assert [e.name for e in events] == ["plan", "papers", "error"]
