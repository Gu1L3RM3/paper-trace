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


_INTERNAL_ERROR = "Unexpected internal error. Try again; if it persists, check the server logs."


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
                raise SourceError(
                    "Every search failed, so there is nothing to cite: "
                    f"{', '.join(result.failed_queries)}. Try again in a moment."
                )
            synthesis = await self._synthesizer.synthesize(question, result.papers)
            yield StageEvent("synthesis", synthesis)
            verification = await self._verifier.verify(question, synthesis, result.papers)
            yield StageEvent("verification", verification)
            if verification.needs_revision:
                revision = await self._revise(question, result.papers, synthesis, verification)
                if revision:
                    yield StageEvent("synthesis", revision[0])
                    yield StageEvent("verification", revision[1])
            yield StageEvent("done")
        except PaperTraceError as exc:
            yield StageEvent("error", ErrorPayload(message=str(exc)))
        except Exception:
            logger.exception("Unexpected pipeline failure")
            yield StageEvent(
                "error",
                ErrorPayload(message=_INTERNAL_ERROR),
            )

    async def _revise(
        self,
        question: ResearchQuestion,
        papers: list[Paper],
        draft: SynthesisResult,
        verification: VerificationResult,
    ) -> tuple[SynthesisResult, VerificationResult] | None:
        """One bounded revision; the draft is already streamed, so any failure just keeps it."""
        try:
            revised = await self._synthesizer.revise(question, papers, draft, verification)
            # A revision that cites more unlisted papers is worse than the draft it fixes.
            if len(revised.unmatched_citations) > len(draft.unmatched_citations):
                return None
            revised = revised.model_copy(update={"revision": True})
            checked = await self._verifier.verify(question, revised, papers)
        except PaperTraceError:
            logger.warning("Revision failed; keeping the draft", exc_info=True)
            return None
        return revised, checked.model_copy(update={"revision_count": 1})
