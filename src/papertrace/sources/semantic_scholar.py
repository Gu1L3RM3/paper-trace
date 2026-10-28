import asyncio
from typing import Any

import httpx
from pydantic import ValidationError

from papertrace.domain.errors import SourceError
from papertrace.domain.models import Paper

_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarSource:
    def __init__(self, *, client: httpx.AsyncClient, api_key: str | None = None) -> None:
        self._client = client
        self._api_key = api_key
        self._lock = asyncio.Lock()
        self._next_request = 0.0

    async def search(self, query: str) -> list[Paper]:
        # Share pacing across queries and concurrent research requests in this process.
        async with self._lock:
            loop = asyncio.get_running_loop()
            await asyncio.sleep(max(0, self._next_request - loop.time()))
            self._next_request = loop.time() + 1
        headers = {"x-api-key": self._api_key} if self._api_key else {}
        try:
            response = await self._client.get(
                _URL,
                params={
                    "query": query,
                    "limit": 15,
                    "fields": "title,authors,year,externalIds,abstract,url",
                },
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()["data"]
            if not isinstance(data, list):
                raise ValueError("data must be a list")
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            hint = " Check SEMANTIC_SCHOLAR_API_KEY or try again later."
            raise SourceError(f"Semantic Scholar answered HTTP {code}.{hint}") from exc
        except httpx.HTTPError as exc:
            raise SourceError(
                "Cannot reach Semantic Scholar. Check the connection, then try again."
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise SourceError(
                "Semantic Scholar returned an unexpected response. Try again later."
            ) from exc
        return [paper for record in data if (paper := _to_paper(record)) is not None]


def _to_paper(record: Any) -> Paper | None:
    if not isinstance(record, dict) or not record.get("title"):
        return None
    try:
        doi = (record.get("externalIds") or {}).get("DOI")
        return Paper(
            title=record["title"],
            authors=[a["name"] for a in record.get("authors") or [] if a.get("name")],
            year=record.get("year"),
            doi=doi,
            abstract=record.get("abstract"),
            url=record.get("url")
            or (
                f"https://doi.org/{doi}"
                if doi
                else f"https://www.semanticscholar.org/paper/{record['paperId']}"
            ),
        )
    except (ValidationError, KeyError, TypeError, AttributeError):
        return None
