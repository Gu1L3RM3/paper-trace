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
        needs_revision: bool | None = None,
    ) -> None:
        self.status = status
        self.needs_revision = status == "issues_found" if needs_revision is None else needs_revision
        self.error = error
        self.calls = 0

    async def verify(
        self, question: ResearchQuestion, synthesis: SynthesisResult, papers: list[Paper]
    ) -> VerificationResult:
        from papertrace.domain.verification import VerificationResult

        self.calls += 1
        if self.error:
            raise self.error
        return VerificationResult(
            claims=[],
            findings=[],
            status=self.status,
            reason="test",
            needs_revision=self.needs_revision,
        )


async def test_draft_streams_first_then_one_revised_pair() -> None:
    synth = _CountingRevisions([])
    verifier = _Verifier("issues_found")
    events = await _run(Pipeline(_Planner(), _RealResearcher(), synth, verifier))
    assert [e.name for e in events] == [
        "plan",
        "papers",
        "synthesis",
        "verification",
        "synthesis",
        "verification",
        "done",
    ]
    draft, revised = events[2].payload, events[4].payload
    assert isinstance(draft, SynthesisResult) and isinstance(revised, SynthesisResult)
    assert draft.markdown != "revised" and not draft.revision
    assert revised.markdown == "revised" and revised.revision
    assert isinstance(events[3].payload, VerificationResult)
    assert isinstance(events[5].payload, VerificationResult)
    assert events[3].payload.revision_count == 0 and events[5].payload.revision_count == 1
    assert synth.calls == 1 and verifier.calls == 2


async def test_verifier_error_follows_the_streamed_draft() -> None:
    events = await _run(
        Pipeline(_Planner(), _Researcher(), _Synthesizer(), _Verifier(error=LLMError("down")))
    )
    assert [e.name for e in events] == ["plan", "papers", "synthesis", "error"]


async def test_revision_failures_keep_the_draft_and_finish() -> None:
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
        events = await _run(Pipeline(_Planner(), _RealResearcher(), synth, verifier))
        assert [e.name for e in events] == ["plan", "papers", "synthesis", "verification", "done"]


class _CountingRevisions(_Synthesizer):
    def __init__(self, unmatched: list[str]) -> None:
        super().__init__()
        self.calls = 0
        self.unmatched = unmatched

    async def revise(
        self,
        question: ResearchQuestion,
        papers: list[Paper],
        synthesis: SynthesisResult,
        verification: VerificationResult,
    ) -> SynthesisResult:
        self.calls += 1
        return synthesis.model_copy(
            update={"markdown": "revised", "unmatched_citations": self.unmatched}
        )


class _RealResearcher(_Researcher):
    async def research(self, plan: SearchPlan) -> ResearchResult:
        from tests.fakes import papers

        return ResearchResult(papers=papers(), failed_queries=[])


async def test_issues_a_rewrite_cannot_fix_skip_the_revision() -> None:
    synth = _CountingRevisions([])
    verifier = _Verifier("issues_found", needs_revision=False)
    events = await _run(Pipeline(_Planner(), _RealResearcher(), synth, verifier))
    assert [e.name for e in events][-1] == "done"
    assert synth.calls == 0 and verifier.calls == 1


async def test_revision_that_adds_unmatched_citations_is_discarded() -> None:
    synth = _CountingRevisions(["Ghost, 1900"])
    verifier = _Verifier("issues_found")
    events = await _run(Pipeline(_Planner(), _RealResearcher(), synth, verifier))
    assert isinstance(events[2].payload, SynthesisResult)
    assert events[2].payload.markdown != "revised"
    assert synth.calls == 1 and verifier.calls == 1
