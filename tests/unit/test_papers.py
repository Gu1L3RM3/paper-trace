from papertrace.domain.models import Paper
from papertrace.domain.papers import abstract_from_inverted_index, dedup_papers


def _paper(title: str, doi: str | None = None) -> Paper:
    return Paper(title=title, authors=[], year=2020, doi=doi, abstract=None, url="https://x")


def test_abstract_rebuilt_in_word_order() -> None:
    index = {"fasting": [1], "Intermittent": [0], "works": [2, 3], "well": [4]}
    assert abstract_from_inverted_index(index) == "Intermittent fasting works works well"


def test_abstract_none_when_index_missing_or_empty() -> None:
    assert abstract_from_inverted_index(None) is None
    assert abstract_from_inverted_index({}) is None


def test_dedup_by_doi_keeps_first() -> None:
    first, dup = _paper("A", "10.1/a"), _paper("Different title", "10.1/a")
    assert dedup_papers([first, dup, _paper("B", "10.1/b")]) == [first, _paper("B", "10.1/b")]


def test_dedup_falls_back_to_normalized_title_without_doi() -> None:
    first = _paper("Fasting & Cognition!")
    dup = _paper("  fasting cognition ")
    assert dedup_papers([first, dup]) == [first]


def test_dedup_does_not_merge_doi_less_with_different_title() -> None:
    papers = [_paper("One"), _paper("Two")]
    assert dedup_papers(papers) == papers


def test_dedup_keeps_distinct_non_latin_titles() -> None:
    papers = [_paper("间歇性禁食与认知"), _paper("Прерывистое голодание"), _paper("الصيام المتقطع")]
    assert dedup_papers(papers) == papers


def test_dedup_merges_titles_differing_only_by_accents_and_case() -> None:
    assert len(dedup_papers([_paper("Étude du jeûne"), _paper("ETUDE DU JEUNE")])) == 1
