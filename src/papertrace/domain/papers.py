import re

from papertrace.domain.models import Paper

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex ships abstracts as word -> positions; rebuild the plain text."""
    if not index:
        return None
    words = {pos: word for word, positions in index.items() for pos in positions}
    return " ".join(words[pos] for pos in sorted(words))


def _key(paper: Paper) -> str:
    return paper.doi or _NON_ALNUM.sub("", paper.title.lower())


def dedup_papers(papers: list[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    for paper in papers:
        seen.setdefault(_key(paper), paper)
    return list(seen.values())
