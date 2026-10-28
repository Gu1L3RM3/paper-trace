import json
from pathlib import Path

import httpx
import pytest
import respx

from papertrace.domain.errors import SourceError
from papertrace.sources.semantic_scholar import SemanticScholarSource

URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PAYLOAD = json.loads(Path("tests/fixtures/semantic_scholar.json").read_text())


async def test_mapping_and_optional_key() -> None:
    with respx.mock:
        route = respx.get(URL).respond(200, json=PAYLOAD)
        async with httpx.AsyncClient() as client:
            papers = await SemanticScholarSource(client=client, api_key="private").search("control")
        assert papers and papers[0].abstract
        assert papers[0].title == PAYLOAD["data"][0]["title"]
        assert route.calls[0].request.headers["x-api-key"] == "private"


@pytest.mark.parametrize("payload", [{"data": []}, {"data": [None, {}, {"title": 123}]}])
async def test_empty_and_invalid_records(payload: object) -> None:
    with respx.mock:
        route = respx.get(URL).respond(200, json=payload)
        async with httpx.AsyncClient() as client:
            assert await SemanticScholarSource(client=client).search("control") == []
        assert "x-api-key" not in route.calls[0].request.headers


@pytest.mark.parametrize("status,payload", [(429, {}), (500, {}), (200, {}), (200, {"data": None})])
async def test_failures(status: int, payload: object) -> None:
    with respx.mock:
        respx.get(URL).respond(status, json=payload)
        async with httpx.AsyncClient() as client:
            with pytest.raises(SourceError):
                await SemanticScholarSource(client=client).search("control")


async def test_timeout() -> None:
    with respx.mock:
        respx.get(URL).mock(side_effect=httpx.ReadTimeout("secret"))
        async with httpx.AsyncClient() as client:
            with pytest.raises(SourceError, match="connection"):
                await SemanticScholarSource(client=client).search("control")


async def test_pacing_is_shared_across_queries() -> None:
    import asyncio
    import time

    starts: list[float] = []

    def response(request: httpx.Request) -> httpx.Response:
        starts.append(time.monotonic())
        return httpx.Response(200, json={"data": []})

    with respx.mock:
        respx.get(URL).mock(side_effect=response)
        async with httpx.AsyncClient() as client:
            source = SemanticScholarSource(client=client)
            await asyncio.gather(source.search("a"), source.search("b"))
    assert starts[1] - starts[0] >= 0.99


async def test_missing_optional_fields_and_malformed_response() -> None:
    record = dict(PAYLOAD["data"][0])
    for field in ("abstract", "authors", "year", "externalIds", "url"):
        record.pop(field, None)
    with respx.mock:
        route = respx.get(URL).respond(200, json={"data": [record]})
        async with httpx.AsyncClient() as client:
            source = SemanticScholarSource(client=client)
            paper = (await source.search("control"))[0]
            assert paper.doi is None and paper.abstract is None
            assert paper.authors == [] and paper.year is None
            route.respond(200, text="not json")
            with pytest.raises(SourceError):
                await source.search("control")
