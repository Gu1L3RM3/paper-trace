import asyncio

from papertrace.domain.errors import SourceError
from papertrace.domain.models import Paper, ResearchResult, SearchPlan
from papertrace.domain.papers import dedup_papers
from papertrace.sources.base import PaperSource


class ResearchAgent:
    def __init__(self, source: PaperSource) -> None:
        self._source = source

    async def research(self, plan: SearchPlan) -> ResearchResult:
        queries = [q.query for q in plan.queries]
        outcomes = await asyncio.gather(
            *(self._source.search(q) for q in queries), return_exceptions=True
        )
        papers: list[Paper] = []
        failed: list[tuple[str, SourceError]] = []
        for query, outcome in zip(queries, outcomes, strict=True):
            if isinstance(outcome, SourceError):
                failed.append((query, outcome))
            elif isinstance(outcome, BaseException):
                raise outcome  # not a source failure: a bug that must not be hidden
            else:
                papers += outcome
        if len(failed) == len(queries):
            raise failed[0][1]
        return ResearchResult(papers=dedup_papers(papers), failed_queries=[q for q, _ in failed])
