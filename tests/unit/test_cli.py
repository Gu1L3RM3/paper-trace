import sys
from pathlib import Path

import pytest

from papertrace.cli import format_papers, main
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


def test_blank_author_name_does_not_crash() -> None:
    paper = Paper(title="T", authors=[" "], year=2020, doi=None, abstract=None, url="https://u")
    assert format_papers([paper]) == "1. T (Unknown, 2020)\n   https://u"


def test_missing_configuration_prints_friendly_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for var in ("LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no env file to pick up
    monkeypatch.setattr(sys, "argv", ["papertrace", "question"])

    assert main() == 1
    assert "LLM_MODEL" in capsys.readouterr().err.upper()
