import pytest

from papertrace.agents.multi_source import MAX_PAPERS, MultiSourceResearcher
from papertrace.domain.errors import SourceError
from papertrace.domain.models import SearchPlan, SearchQuery
from tests.fakes import FakePaperSource, papers


def plan() -> SearchPlan:
    return SearchPlan(queries=[SearchQuery(query=q, angle=q) for q in ("a", "b")])


async def test_combines_and_preserves_abstract() -> None:
    paper = papers()[0]
    empty = paper.model_copy(update={"abstract": None})
    result = await MultiSourceResearcher(
        {
            "OpenAlex": FakePaperSource({"a": [empty], "b": []}),
            "Semantic Scholar": FakePaperSource({"a": [paper], "b": []}),
        }
    ).research(plan())
    assert result.papers == [paper]
    assert result.failed_queries == []


async def test_partial_failure_is_visible() -> None:
    result = await MultiSourceResearcher(
        {
            "OpenAlex": FakePaperSource({"a": papers(), "b": []}),
            "Semantic Scholar": FakePaperSource({"a": SourceError("bad"), "b": []}),
        }
    ).research(plan())
    assert result.papers
    assert result.failed_queries == ["Semantic Scholar: a"]


async def test_all_sources_fail() -> None:
    with pytest.raises(SourceError):
        await MultiSourceResearcher(
            {
                "OpenAlex": FakePaperSource({"a": SourceError("bad"), "b": SourceError("bad")}),
            }
        ).research(plan())


async def test_unexpected_exception_propagates() -> None:
    with pytest.raises(RuntimeError):
        await MultiSourceResearcher(
            {
                "OpenAlex": FakePaperSource({"a": RuntimeError("bug"), "b": []}),
            }
        ).research(plan())


async def test_complete_source_failure_and_empty_success() -> None:
    result = await MultiSourceResearcher(
        {
            "OpenAlex": FakePaperSource({"a": [], "b": []}),
            "Semantic Scholar": FakePaperSource({"a": SourceError("bad"), "b": SourceError("bad")}),
        }
    ).research(plan())
    assert result.papers == []
    assert result.failed_queries == ["Semantic Scholar: a", "Semantic Scholar: b"]


async def test_sources_run_concurrently() -> None:
    import time

    start = time.monotonic()
    source = FakePaperSource({"a": papers(), "b": []}, delay=0.2)
    result = await MultiSourceResearcher({"one": source, "two": source}).research(plan())
    assert result.papers
    assert time.monotonic() - start < 0.35


def test_requires_source() -> None:
    with pytest.raises(ValueError):
        MultiSourceResearcher({})


async def test_keeps_only_the_best_ranked_papers() -> None:
    many = [
        paper.model_copy(update={"title": f"T{i}", "doi": f"10.1/{i}", "abstract": "text"})
        for i, paper in enumerate(papers() * 8)
    ]
    result = await MultiSourceResearcher(
        {"OpenAlex": FakePaperSource({"a": many, "b": []})}
    ).research(plan())
    assert result.papers == many[:MAX_PAPERS]
