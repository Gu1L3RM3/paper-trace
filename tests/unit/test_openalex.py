import json
from pathlib import Path

import httpx
import respx

from papertrace.sources.openalex import OpenAlexSource

WORKS = json.loads((Path(__file__).parent.parent / "fixtures" / "openalex_works.json").read_text())


async def test_maps_recorded_work_and_sends_polite_params() -> None:
    with respx.mock:
        route = respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json=WORKS)
        )
        async with httpx.AsyncClient() as client:
            papers = await OpenAlexSource(client=client, mailto="me@example.com").search("fasting")

    params = route.calls.last.request.url.params
    assert params["search"] == "fasting"
    assert params["mailto"] == "me@example.com"
    first = papers[0]
    assert first.doi == "10.1007/s40279-015-0408-6"
    assert first.year == 2015
    assert first.authors[0] == "Anissa Cherif"
    assert first.url == "https://doi.org/10.1007/s40279-015-0408-6"
    assert first.abstract is None
    assert papers[2].abstract
    assert papers[2].abstract.startswith("This study aims to investigate the effect of Ramadan")


async def test_skips_works_without_title() -> None:
    payload = {"results": [{"id": "https://openalex.org/W1", "title": None}]}
    with respx.mock:
        respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json=payload)
        )
        async with httpx.AsyncClient() as client:
            assert await OpenAlexSource(client=client).search("x") == []
