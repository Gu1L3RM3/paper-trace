import unicodedata

from papertrace.domain.models import Paper


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex ships abstracts as word -> positions; rebuild the plain text."""
    if not index:
        return None
    words = {pos: word for word, positions in index.items() for pos in positions}
    return " ".join(words[pos] for pos in sorted(words))


_RRF_K = 60  # standard damping constant for reciprocal-rank fusion


def _normalize_title(title: str) -> str:
    """Case, accent and punctuation insensitive, but keeps non-Latin scripts distinct."""
    decomposed = unicodedata.normalize("NFKD", title.casefold())
    return "".join(c for c in decomposed if c.isalnum())


def _key(paper: Paper) -> str:
    return paper.doi or _normalize_title(paper.title) or paper.title


def dedup_papers(papers: list[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    for paper in papers:
        key = _key(paper)
        previous = seen.get(key)
        if previous is None or (
            not (previous.abstract or "").strip() and (paper.abstract or "").strip()
        ):
            seen[key] = paper
    return list(seen.values())


def rank_papers(ranked_lists: list[list[Paper]], limit: int | None = None) -> list[Paper]:
    """Reciprocal-rank fusion: papers many searches agree on come first.

    Papers without an abstract go last because neither synthesis nor verification can use them.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, paper in enumerate(ranked):
            scores[_key(paper)] = scores.get(_key(paper), 0.0) + 1 / (_RRF_K + rank)
    merged = dedup_papers([paper for ranked in ranked_lists for paper in ranked])
    merged.sort(key=lambda p: (not (p.abstract or "").strip(), -scores[_key(p)]))
    return merged[:limit]


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def surname(author: str) -> str:
    if "," in author:  # "Last, First"
        return author.split(",")[0].strip()
    words = author.split()
    while len(words) > 1 and words[-1].rstrip(".").casefold() in _NAME_SUFFIXES:
        words.pop()
    return words[-1] if words else ""


def prompt_view(index: int, paper: Paper) -> dict[str, object]:
    """The only fields a model needs about a paper; ids and URLs would just cost tokens."""
    return {
        "index": index,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "abstract": paper.abstract,
    }
