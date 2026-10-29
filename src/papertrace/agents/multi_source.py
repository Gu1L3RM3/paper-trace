import asyncio

from papertrace.agents.research import ResearchAgent
from papertrace.domain.errors import SourceError
from papertrace.domain.models import Paper, ResearchResult, SearchPlan
from papertrace.domain.papers import rank_papers
from papertrace.sources.base import PaperSource

# Every paper goes into the synthesis and verification prompts, so this bounds tokens and latency.
MAX_PAPERS = 15


class MultiSourceResearcher:
    def __init__(self, sources: dict[str, PaperSource]) -> None:
        if not sources:
            raise ValueError("At least one paper source is required")
        self._sources = sources

    async def research(self, plan: SearchPlan) -> ResearchResult:
        outcomes = await asyncio.gather(
            *(ResearchAgent(source).research(plan) for source in self._sources.values()),
            return_exceptions=True,
        )
        ranked: list[list[Paper]] = []
        failed: list[str] = []
        errors: list[SourceError] = []
        for name, outcome in zip(self._sources, outcomes, strict=True):
            if isinstance(outcome, SourceError):
                errors.append(outcome)
                failed.extend(f"{name}: {q.query}" for q in plan.queries)
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                ranked.append(outcome.papers)
                failed.extend(f"{name}: {q}" for q in outcome.failed_queries)
        if len(errors) == len(outcomes):
            raise errors[0]
        return ResearchResult(papers=rank_papers(ranked, MAX_PAPERS), failed_queries=failed)
