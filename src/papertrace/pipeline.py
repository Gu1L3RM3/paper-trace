import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from papertrace.domain.errors import PaperTraceError, SourceError
from papertrace.domain.models import (
    Paper,
    ResearchQuestion,
    ResearchResult,
    SearchPlan,
    SynthesisResult,
)
from papertrace.domain.verification import VerificationResult

logger = logging.getLogger(__name__)


class ErrorPayload(BaseModel):
    message: str


@dataclass(frozen=True)
class StageEvent:
    name: Literal["plan", "papers", "synthesis", "verification", "error", "done"]
    payload: BaseModel | None = None


class Planner(Protocol):
    async def plan(self, question: ResearchQuestion) -> SearchPlan: ...


class Researcher(Protocol):
    async def research(self, plan: SearchPlan) -> ResearchResult: ...


class Synthesizer(Protocol):
    async def synthesize(
        self, question: ResearchQuestion, papers: list[Paper]
    ) -> SynthesisResult: ...

    async def revise(
        self,
        question: ResearchQuestion,
        papers: list[Paper],
        synthesis: SynthesisResult,
        verification: VerificationResult,
    ) -> SynthesisResult: ...


class Verifier(Protocol):
    async def verify(
        self, question: ResearchQuestion, synthesis: SynthesisResult, papers: list[Paper]
    ) -> VerificationResult: ...


class Pipeline:
    """Only coordinates: each stage's decisions live in its agent."""

    def __init__(
        self, planner: Planner, researcher: Researcher, synthesizer: Synthesizer, verifier: Verifier
    ) -> None:
        self._planner = planner
        self._researcher = researcher
        self._synthesizer = synthesizer
        self._verifier = verifier

    async def run(self, question: ResearchQuestion) -> AsyncIterator[StageEvent]:
        try:
            plan = await self._planner.plan(question)
            yield StageEvent("plan", plan)
            result = await self._researcher.research(plan)
            yield StageEvent("papers", result)
            if not result.papers and result.failed_queries:
                # Nothing came back and part of the search broke: that is not "no evidence".
                raise SourceError(f"Search failed for: {', '.join(result.failed_queries)}")
            synthesis = await self._synthesizer.synthesize(question, result.papers)
            verification = await self._verifier.verify(question, synthesis, result.papers)
            if verification.needs_revision:
                revised = await self._synthesizer.revise(
                    question, result.papers, synthesis, verification
                )
                # A revision that cites more unlisted papers is worse than the draft it fixes.
                if len(revised.unmatched_citations) <= len(synthesis.unmatched_citations):
                    synthesis = revised
                    verification = await self._verifier.verify(question, synthesis, result.papers)
                    verification = verification.model_copy(update={"revision_count": 1})
            yield StageEvent("synthesis", synthesis)
            yield StageEvent("verification", verification)
            yield StageEvent("done")
        except PaperTraceError as exc:
            yield StageEvent("error", ErrorPayload(message=str(exc)))
        except Exception:
            logger.exception("Unexpected pipeline failure")
            yield StageEvent("error", ErrorPayload(message="Unexpected internal error"))
