from papertrace.domain.citations import citation_candidates, find_unmatched_citations
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


def test_duplicates_collapse_and_non_citations_are_ignored() -> None:
    text = "(Ghost, 2019) (Ghost, 2019) (not a citation) (see Fig. 3) (see Table 2020)"

    assert find_unmatched_citations(text, PAPERS) == ["Ghost, 2019"]


def test_only_the_first_author_of_the_same_paper_matches() -> None:
    papers = [_paper(["Ann Smith", "Bob Jones"], 2020), _paper(["Cy Lee"], 2018)]

    assert find_unmatched_citations("(Jones, 2020) (Smith, 2018) (Smith, 2020)", papers) == [
        "Jones, 2020",
        "Smith, 2018",
    ]


def test_narrative_citations_are_checked() -> None:
    text = "Silva (2020) found X. Silva et al. (2020), Silva and Chen (2020) and Nakamura (2019)."

    assert find_unmatched_citations(text, PAPERS) == ["Nakamura (2019)"]


def test_citation_format_variants() -> None:
    ok = "(Silva 2020) (Silva et al, 2020) (Silva, Chen, 2020) (Silva et al., 2020, p. 5)"
    bad = "(Fabricated 2021) (Fabricated, 2021, p. 4) (Fabricated, n.d.) (Fabricated, 2021a)"

    assert find_unmatched_citations(ok, PAPERS) == []
    assert find_unmatched_citations(bad, PAPERS) == [
        "Fabricated 2021",
        "Fabricated, 2021, p. 4",
        "Fabricated, n.d.",
        "Fabricated, 2021a",
    ]


def test_undated_paper_can_be_cited_as_nd() -> None:
    papers = [_paper(["Eve Stone"], None)]

    assert find_unmatched_citations("(Stone, n.d.) (Stone, 2020)", papers) == ["Stone, 2020"]


def test_suffix_and_last_first_author_names() -> None:
    papers = [_paper(["Martin Luther King Jr."], 2020), _paper(["Doe, John"], 2019)]

    assert find_unmatched_citations("(King, 2020) (Doe, 2019)", papers) == []


def test_candidates_use_the_surname_of_multi_word_cited_names() -> None:
    papers = [_paper(["Jan van der Berg"], 2020)]

    assert citation_candidates("(Van der Berg, 2020)", papers) == [[1]]
