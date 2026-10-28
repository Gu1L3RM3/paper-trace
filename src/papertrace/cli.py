import asyncio
import sys

import httpx

from papertrace.agents.orchestrator import OrchestratorAgent
from papertrace.agents.research import ResearchAgent
from papertrace.domain.errors import PaperTraceError
from papertrace.domain.models import Paper, ResearchQuestion
from papertrace.llm.factory import build_llm_client
from papertrace.settings import Settings
from papertrace.sources.openalex import OpenAlexSource


def format_papers(papers: list[Paper]) -> str:
    if not papers:
        return "No papers found."
    return "\n".join(
        f"{i}. {p.title} ({_author(p)}, {p.year or 'n.d.'})\n   {p.url}"
        for i, p in enumerate(papers, 1)
    )


def _author(paper: Paper) -> str:
    if not paper.authors:
        return "Unknown"
    surname = paper.authors[0].split()[-1]
    return surname if len(paper.authors) == 1 else f"{surname} et al."


async def _run(question: str, settings: Settings) -> str:
    semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
    llm = build_llm_client(settings, model=settings.model_for("orchestrator"), semaphore=semaphore)
    async with httpx.AsyncClient(timeout=30) as http:
        source = OpenAlexSource(client=http, mailto=settings.openalex_mailto)
        plan = await OrchestratorAgent(llm).plan(ResearchQuestion(text=question))
        papers = await ResearchAgent(source).research(plan)
    queries = "\n".join(f"- {q.query} ({q.angle})" for q in plan.queries)
    return f"Queries:\n{queries}\n\n{format_papers(papers)}"


def main() -> int:
    if len(sys.argv) != 2:
        print('usage: python -m papertrace.cli "<question>"', file=sys.stderr)
        return 2
    try:
        print(asyncio.run(_run(sys.argv[1], Settings())))
    except PaperTraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
