from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse, ServerSentEvent

from papertrace.api.dependencies import build_pipeline, get_pipeline
from papertrace.domain.models import ResearchQuestion
from papertrace.pipeline import Pipeline
from papertrace.settings import Settings


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with httpx.AsyncClient(timeout=30) as http:
        app.state.pipeline = build_pipeline(Settings(), http)
        yield


app = FastAPI(title="PaperTrace AI", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research")
async def research(
    question: ResearchQuestion, pipeline: Annotated[Pipeline, Depends(get_pipeline)]
) -> EventSourceResponse:
    async def events() -> AsyncIterator[ServerSentEvent]:
        async for event in pipeline.run(question):
            data = event.payload.model_dump_json() if event.payload is not None else "{}"
            yield ServerSentEvent(data=data, event=event.name)

    return EventSourceResponse(events())


# Mounted last so it never shadows the API routes.
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web" / "static", html=True))
