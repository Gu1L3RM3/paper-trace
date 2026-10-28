import time

import pytest

from papertrace.agents.research import ResearchAgent
from papertrace.domain.errors import SourceError
from papertrace.domain.models import Paper, SearchPlan, SearchQuery
from tests.fakes import FakePaperSource


def _paper(title: str, doi: str | None) -> Paper:
    return Paper(title=title, authors=["A"], year=2020, doi=doi, abstract=None, url="https://x")


def _plan(*queries: str) -> SearchPlan:
    return SearchPlan(queries=[SearchQuery(query=q, angle=q) for q in queries])


async def test_queries_run_concurrently() -> None:
    delay = 0.2
    source = FakePaperSource({q: [_paper(q, f"10.1/{q}")] for q in "abcd"}, delay=delay)

    start = time.perf_counter()
    papers = (await ResearchAgent(source).research(_plan("a", "b", "c", "d"))).papers
    elapsed = time.perf_counter() - start

    assert len(papers) == 4
    assert elapsed < delay * 2  # sequential would take 4 * delay


async def test_papers_shared_across_queries_are_deduplicated() -> None:
    shared = _paper("Shared", "10.1/shared")
    source = FakePaperSource({"a": [shared, _paper("A", "10.1/a")], "b": [shared]})

    result = await ResearchAgent(source).research(_plan("a", "b"))
    papers = result.papers

    assert [p.doi for p in papers] == ["10.1/shared", "10.1/a"]


async def test_failing_query_does_not_kill_the_others() -> None:
    source = FakePaperSource({"a": SourceError("boom"), "b": [_paper("B", "10.1/b")]})

    result = await ResearchAgent(source).research(_plan("a", "b"))
    papers = result.papers

    assert [p.doi for p in papers] == ["10.1/b"]
    assert result.failed_queries == ["a"]


async def test_all_queries_failing_raises_source_error() -> None:
    source = FakePaperSource({"a": SourceError("x"), "b": SourceError("y")})

    with pytest.raises(SourceError):
        await ResearchAgent(source).research(_plan("a", "b"))


async def test_no_results_without_failures_is_an_empty_list() -> None:
    source = FakePaperSource({"a": [], "b": []})

    result = await ResearchAgent(source).research(_plan("a", "b"))
    assert result.papers == []
    assert result.failed_queries == []


async def test_unexpected_errors_are_not_swallowed() -> None:
    source = FakePaperSource({"a": RuntimeError("bug"), "b": [_paper("B", "10.1/b")]})

    with pytest.raises(RuntimeError):
        await ResearchAgent(source).research(_plan("a", "b"))
