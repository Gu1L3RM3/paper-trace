import asyncio

from papertrace.domain.errors import SourceError
from papertrace.domain.models import Paper, SearchPlan
from papertrace.domain.papers import dedup_papers
from papertrace.sources.base import PaperSource


class ResearchAgent:
    def __init__(self, source: PaperSource) -> None:
        self._source = source

    async def research(self, plan: SearchPlan) -> list[Paper]:
        outcomes = await asyncio.gather(
            *(self._source.search(q.query) for q in plan.queries), return_exceptions=True
        )
        results = [o for o in outcomes if not isinstance(o, BaseException)]
        failures = [o for o in outcomes if isinstance(o, SourceError)]
        for outcome in outcomes:
            # Only source failures are tolerated; anything else is a bug.
            if isinstance(outcome, BaseException) and not isinstance(outcome, SourceError):
                raise outcome
        if not results:
            raise failures[0]
        return dedup_papers([paper for papers in results for paper in papers])
