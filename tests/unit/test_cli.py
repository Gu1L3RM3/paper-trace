from papertrace.cli import format_papers
from papertrace.domain.models import Paper


def test_format_papers_lists_authors_year_and_link() -> None:
    paper = Paper(
        title="Fasting and cognition",
        authors=["Ana Silva", "Bo Chen", "Cy Doe", "Di Eve"],
        year=2021,
        doi="10.1/x",
        abstract=None,
        url="https://doi.org/10.1/x",
    )
    assert format_papers([paper]) == (
        "1. Fasting and cognition (Silva et al., 2021)\n   https://doi.org/10.1/x"
    )


def test_format_papers_handles_empty_and_unknown_fields() -> None:
    assert format_papers([]) == "No papers found."
    anon = Paper(title="T", authors=[], year=None, doi=None, abstract=None, url="https://u")
    assert format_papers([anon]) == "1. T (Unknown, n.d.)\n   https://u"
