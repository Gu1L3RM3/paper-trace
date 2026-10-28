import re
import unicodedata

from papertrace.domain.models import Paper
from papertrace.domain.papers import surname

_GROUP = re.compile(r"\(([^()]*\d{4}[^()]*)\)")
_CITATION = re.compile(r"^(?P<authors>.+?),\s*(?P<year>\d{4})[a-z]?$")
_AUTHOR_SEPARATOR = re.compile(r"\s+(?:and|&)\s+")


def _fold(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    return "".join(c for c in decomposed if c.isalpha())


def find_unmatched_citations(text: str, papers: list[Paper]) -> list[str]:
    """(Author, Year) citations in `text` that no paper in `papers` backs, in first-seen order.

    shortcut: matches surname + year only (not title), so two papers sharing both are
    indistinguishable; upgrade if the prompt ever needs disambiguating suffixes.
    """
    known = {(_fold(surname(a)), p.year) for p in papers for a in p.authors}
    unmatched: dict[str, None] = {}
    for group in _GROUP.findall(text):
        for part in group.split(";"):
            match = _CITATION.match(part.strip())
            if not match:
                continue  # parenthetical that is not a citation, e.g. "(see Fig. 2020)"
            first = _AUTHOR_SEPARATOR.split(match["authors"])[0].removesuffix(" et al.")
            if (_fold(surname(first)), int(match["year"])) not in known:
                unmatched[part.strip()] = None
    return list(unmatched)
