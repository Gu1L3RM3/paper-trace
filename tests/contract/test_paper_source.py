"""Every PaperSource adapter must pass this suite unchanged (Liskov)."""

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx

from papertrace.domain.errors import SourceError
from papertrace.sources.base import PaperSource
from papertrace.sources.openalex import OpenAlexSource
from papertrace.sources.semantic_scholar import SemanticScholarSource

FIXTURES = Path(__file__).parent.parent / "fixtures"
OPENALEX_URL = "https://api.openalex.org/works"


def _openalex(client: httpx.AsyncClient) -> PaperSource:
    return OpenAlexSource(client=client, mailto="me@example.com")


@pytest.fixture(
    params=[
        (_openalex, OPENALEX_URL, "openalex_works.json"),
        (
            lambda client: SemanticScholarSource(client=client),
            "https://api.semanticscholar.org/graph/v1/paper/search",
            "semantic_scholar.json",
        ),
    ],
    ids=["openalex", "semantic-scholar"],
)
def source_case(
    request: pytest.FixtureRequest,
) -> tuple[Callable[[httpx.AsyncClient], PaperSource], str, dict[str, object]]:
    build, url, fixture = request.param
    return build, url, json.loads((FIXTURES / fixture).read_text())


async def test_search_returns_papers_with_title_and_url(
    source_case: tuple[Callable[[httpx.AsyncClient], PaperSource], str, dict[str, object]],
) -> None:
    build, url, payload = source_case
    with respx.mock:
        respx.get(url).mock(return_value=httpx.Response(200, json=payload))
        async with httpx.AsyncClient() as client:
            papers = await build(client).search("intermittent fasting cognitive performance")
    assert papers
    assert all(p.title and p.url for p in papers)
    assert {p.doi for p in papers if p.doi} <= {p.doi for p in papers}
    assert any(p.abstract for p in papers)
    assert all(p.doi is None or p.doi.startswith("10.") for p in papers)


async def test_search_failure_becomes_source_error(
    source_case: tuple[Callable[[httpx.AsyncClient], PaperSource], str, dict[str, object]],
) -> None:
    build, url, _ = source_case
    with respx.mock:
        respx.get(url).mock(return_value=httpx.Response(500))
        async with httpx.AsyncClient() as client:
            with pytest.raises(SourceError):
                await build(client).search("anything")
