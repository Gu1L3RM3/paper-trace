from papertrace.domain.citations import find_unmatched_citations
from papertrace.domain.models import Paper


def _paper(authors: list[str], year: int | None) -> Paper:
    return Paper(title="T", authors=authors, year=year, doi=None, abstract=None, url="https://x")


PAPERS = [_paper(["Ana Silva", "Bo Chen"], 2020), _paper(["José Álvarez"], 2018)]


def test_matching_citations_are_not_flagged() -> None:
    text = "Foo (Silva et al., 2020) and bar (Alvarez, 2018)."

    assert find_unmatched_citations(text, PAPERS) == []


def test_unlisted_author_or_wrong_year_is_flagged() -> None:
    text = "Foo (Silva, 2021) and (Nobody et al., 2020)."

    assert find_unmatched_citations(text, PAPERS) == ["Silva, 2021", "Nobody et al., 2020"]


def test_grouped_and_pair_citations_are_checked_each() -> None:
    text = "Both (Silva and Chen, 2020; Ghost, 2019)."

    assert find_unmatched_citations(text, PAPERS) == ["Ghost, 2019"]


def test_any_listed_author_matches_and_duplicates_collapse() -> None:
    text = "(Chen, 2020) (Ghost, 2019) (Ghost, 2019) (not a citation) (see Fig. 3)"

    assert find_unmatched_citations(text, PAPERS) == ["Ghost, 2019"]
