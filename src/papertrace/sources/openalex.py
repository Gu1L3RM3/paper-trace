from typing import Any

import httpx
from pydantic import ValidationError

from papertrace.domain.errors import SourceError
from papertrace.domain.models import Paper
from papertrace.domain.papers import abstract_from_inverted_index

_URL = "https://api.openalex.org/works"
_FIELDS = "id,doi,title,publication_year,authorships,abstract_inverted_index,primary_location"


class OpenAlexSource:
    def __init__(
        self, *, client: httpx.AsyncClient, mailto: str | None = None, per_page: int = 15
    ) -> None:
        self._client = client
        self._mailto = mailto
        self._per_page = per_page

    async def search(self, query: str) -> list[Paper]:
        params: dict[str, str | int] = {
            "search": query,
            "per-page": self._per_page,
            "select": _FIELDS,
        }
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            response = await self._client.get(_URL, params=params)
            response.raise_for_status()
            works = response.json()["results"]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise SourceError(f"OpenAlex search failed: {exc}") from exc
        return [paper for work in works if (paper := _to_paper(work)) is not None]


def _to_paper(work: dict[str, Any]) -> Paper | None:
    """Map one record; a malformed one is skipped so it cannot sink the other results."""
    title = work.get("title")
    if not title:
        return None
    location = work.get("primary_location") or {}
    names = ((a.get("author") or {}).get("display_name") for a in work.get("authorships") or [])
    try:
        return Paper(
            title=title,
            authors=[name for name in names if name],
            year=work.get("publication_year"),
            doi=work.get("doi"),
            abstract=abstract_from_inverted_index(work.get("abstract_inverted_index")),
            url=location.get("landing_page_url") or work.get("doi") or work["id"],
        )
    except (ValidationError, KeyError, AttributeError):
        return None
