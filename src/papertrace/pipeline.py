from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from papertrace.domain.errors import PaperTraceError
from papertrace.domain.models import (
    Paper,
    ResearchQuestion,
    ResearchResult,
    SearchPlan,
    SynthesisResult,
)


class ErrorPayload(BaseModel):
    message: str


@dataclass(frozen=True)
class StageEvent:
    name: Literal["plan", "papers", "synthesis", "error", "done"]
    payload: BaseModel | None = None


class Planner(Protocol):
    async def plan(self, question: ResearchQuestion) -> SearchPlan: ...


class Researcher(Protocol):
    async def research(self, plan: SearchPlan) -> ResearchResult: ...


class Synthesizer(Protocol):
    async def synthesize(
        self, question: ResearchQuestion, papers: list[Paper]
    ) -> SynthesisResult: ...


class Pipeline:
    """Only coordinates: each stage's decisions live in its agent."""

    def __init__(self, planner: Planner, researcher: Researcher, synthesizer: Synthesizer) -> None:
        self._planner = planner
        self._researcher = researcher
        self._synthesizer = synthesizer

    async def run(self, question: ResearchQuestion) -> AsyncIterator[StageEvent]:
        try:
            plan = await self._planner.plan(question)
            yield StageEvent("plan", plan)
            result = await self._researcher.research(plan)
            yield StageEvent("papers", result)
            yield StageEvent(
                "synthesis", await self._synthesizer.synthesize(question, result.papers)
            )
            yield StageEvent("done")
        except PaperTraceError as exc:
            yield StageEvent("error", ErrorPayload(message=str(exc)))
