import unicodedata

from papertrace.domain.models import Paper


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex ships abstracts as word -> positions; rebuild the plain text."""
    if not index:
        return None
    words = {pos: word for word, positions in index.items() for pos in positions}
    return " ".join(words[pos] for pos in sorted(words))


def _normalize_title(title: str) -> str:
    """Case, accent and punctuation insensitive, but keeps non-Latin scripts distinct."""
    decomposed = unicodedata.normalize("NFKD", title.casefold())
    return "".join(c for c in decomposed if c.isalnum())


def _key(paper: Paper) -> str:
    return paper.doi or _normalize_title(paper.title) or paper.title


def dedup_papers(papers: list[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    for paper in papers:
        seen.setdefault(_key(paper), paper)
    return list(seen.values())


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def surname(author: str) -> str:
    if "," in author:  # "Last, First"
        return author.split(",")[0].strip()
    words = author.split()
    while len(words) > 1 and words[-1].rstrip(".").casefold() in _NAME_SUFFIXES:
        words.pop()
    return words[-1] if words else ""
