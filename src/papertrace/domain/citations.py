import re
import unicodedata

from papertrace.domain.models import Paper
from papertrace.domain.papers import surname

_YEAR = r"(?:\d{4}[a-z]?|n\.d\.)"
# "(Smith, 2020)", "(Smith 2020)", "(Smith et al., 2020, p. 5)"; one group may hold several.
_GROUP = re.compile(r"\(([^()]*)\)")
_PARENTHETICAL = re.compile(rf"^(?P<authors>.+?),?\s+(?P<year>{_YEAR})(?:,.*)?$")
# "Smith (2020)", "Smith et al. (2020)", "Smith and Jones (2020)".
# shortcut: the name must start with an ASCII capital, so narrative "Álvarez (2020)" is
# not checked; upgrade by testing str.isupper() on a generic word match.
_TAIL = r"[^\W\d_]*(?:['\u2019-][^\W\d_]+)*"
_NARRATIVE = re.compile(
    rf"(?P<authors>(?<![\w])[A-Z]{_TAIL}(?:\s+et al\.?|\s+(?:and|&)\s+[A-Z]{_TAIL})?)"
    rf"\s+\((?P<year>{_YEAR})\)"
)
_ET_AL = re.compile(r"\s+et al\.?.*$")
_AUTHOR_SEPARATOR = re.compile(r"\s*(?:,|&|\band\b)\s*")


def _fold(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    return "".join(c for c in decomposed if c.isalpha())


def _year(token: str) -> int | None:
    return None if token == "n.d." else int(token[:4])


def _first_author(authors: str) -> str:
    cited = _ET_AL.sub("", authors.strip())
    return next((p for p in _AUTHOR_SEPARATOR.split(cited) if p), "")


def _cited_key(authors: str) -> str:
    return _fold(surname(_first_author(authors)))


def _mentions(text: str) -> list[tuple[str, str, str]]:
    """(as written, authors, year token) for every parenthetical and narrative citation."""
    found: list[tuple[str, str, str]] = []
    for group in _GROUP.findall(text):
        for part in (p.strip() for p in group.split(";")):
            match = _PARENTHETICAL.match(part)
            # Capitalised start filters out asides such as "(see Fig. 2020)".
            if match and match["authors"][0].isupper():
                found.append((part, match["authors"], match["year"]))
    for match in _NARRATIVE.finditer(text):
        found.append((f"{match['authors']} ({match['year']})", match["authors"], match["year"]))
    return found


def find_unmatched_citations(text: str, papers: list[Paper]) -> list[str]:
    """Citations in `text` that no paper in `papers` backs, in first-seen order.

    A citation matches when its first author's surname and year are those of the same paper's
    first author, as the synthesis prompt requires.

    shortcut: matches surname + year only (not title), so two papers sharing both are
    indistinguishable; upgrade if the prompt ever needs disambiguating suffixes.
    """
    known = {(_fold(surname(p.authors[0])), p.year) for p in papers if p.authors}
    unmatched: dict[str, None] = {}
    for written, authors, year in _mentions(text):
        if (_cited_key(authors), _year(year)) not in known:
            unmatched[written] = None
    return list(unmatched)


def citation_candidates(text: str, papers: list[Paper]) -> list[list[int]]:
    """Stable one-based candidates for each citation, retaining ambiguity."""
    return [
        [
            i
            for i, paper in enumerate(papers, 1)
            if paper.authors
            and _fold(surname(paper.authors[0])) == _cited_key(authors)
            and paper.year == _year(year)
        ]
        for _, authors, year in _mentions(text)
    ]
