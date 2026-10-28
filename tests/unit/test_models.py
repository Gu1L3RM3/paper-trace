import pytest
from pydantic import ValidationError

from papertrace.domain.models import Paper, SearchPlan, SearchQuery


def _queries(n: int) -> list[SearchQuery]:
    return [SearchQuery(query=f"q{i}", angle=f"a{i}") for i in range(n)]


@pytest.mark.parametrize("n", [0, 1, 5])
def test_search_plan_rejects_out_of_range_query_count(n: int) -> None:
    with pytest.raises(ValidationError):
        SearchPlan(queries=_queries(n))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_search_plan_accepts_two_to_four_queries(n: int) -> None:
    assert len(SearchPlan(queries=_queries(n)).queries) == n


@pytest.mark.parametrize(
    "raw",
    [
        "https://doi.org/10.1000/ABC",
        "http://dx.doi.org/10.1000/abc",
        "doi:10.1000/abc",
        " 10.1000/ABC ",
    ],
)
def test_paper_normalizes_doi(raw: str) -> None:
    paper = Paper(title="t", authors=[], year=None, doi=raw, abstract=None, url="https://x")
    assert paper.doi == "10.1000/abc"


def test_paper_keeps_missing_doi_none() -> None:
    paper = Paper(title="t", authors=[], year=None, doi=None, abstract=None, url="https://x")
    assert paper.doi is None
