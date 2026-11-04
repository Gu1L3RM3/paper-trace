from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse, ServerSentEvent

from papertrace.api.dependencies import get_chatgpt, get_history, get_pipeline, rewire, wire
from papertrace.domain.errors import LLMError
from papertrace.domain.models import ResearchQuestion
from papertrace.history import HistoryEntry, SavedEvent, SQLiteHistory
from papertrace.llm.chatgpt_auth import ChatGPTAuth, SignInRetry
from papertrace.pipeline import Pipeline
from papertrace.settings import Settings


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with httpx.AsyncClient(timeout=30) as http:
        wire(app.state, Settings(), http)
        yield


app = FastAPI(title="PaperTrace AI", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research")
async def research(
    question: ResearchQuestion,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    history: Annotated[SQLiteHistory, Depends(get_history)],
) -> EventSourceResponse:
    async def events() -> AsyncIterator[ServerSentEvent]:
        saved: list[SavedEvent] = []
        try:
            async for event in pipeline.run(question):
                data = event.payload.model_dump_json() if event.payload is not None else "{}"
                saved.append(SavedEvent(event=event.name, data=data))
                yield ServerSentEvent(data=data, event=event.name)
        finally:
            # Also runs when the browser leaves mid-run; one local insert, so a sync call is fine.
            if saved:
                history.save(question.text, saved)

    return EventSourceResponse(events())


@app.get("/api/history")
def history_entries(history: Annotated[SQLiteHistory, Depends(get_history)]) -> list[HistoryEntry]:
    return history.entries()


@app.get("/api/history/{research_id}")
def history_events(
    research_id: int, history: Annotated[SQLiteHistory, Depends(get_history)]
) -> list[SavedEvent]:
    events = history.events(research_id)
    if events is None:
        raise HTTPException(status_code=404, detail="This research is not in the history.")
    return events


@app.delete("/api/history/{research_id}", status_code=204)
def history_delete(
    research_id: int, history: Annotated[SQLiteHistory, Depends(get_history)]
) -> None:
    if not history.delete(research_id):
        raise HTTPException(status_code=404, detail="This research is not in the history.")


@app.get("/auth/chatgpt/login")
def chatgpt_login(
    request: Request, auth: Annotated[ChatGPTAuth, Depends(get_chatgpt)]
) -> RedirectResponse:
    # Sign in with ChatGPT only accepts the 127.0.0.1 loopback; keep the port we answer on.
    callback = f"http://127.0.0.1:{request.url.port or 80}/auth/callback"
    return RedirectResponse(auth.authorize_url(callback), status_code=303)


@app.get("/auth/callback")
async def chatgpt_callback(
    request: Request,
    auth: Annotated[ChatGPTAuth, Depends(get_chatgpt)],
    code: str | None = None,
    state: str = "",
    client_id: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    problem = None
    if error or not code:
        problem = (
            "ChatGPT sign-in was cancelled or not allowed. Click Connect ChatGPT to try again."
        )
    else:
        try:
            await auth.finish_login(code=code, state=state, client_id=client_id)
        except SignInRetry as retry:
            return RedirectResponse(retry.url, status_code=303)
        except LLMError as exc:
            problem = str(exc)
        rewire(request.app.state)
    request.app.state.chatgpt_error = problem
    return RedirectResponse("/", status_code=303)


@app.get("/api/auth/chatgpt")
def chatgpt_status(
    request: Request, auth: Annotated[ChatGPTAuth, Depends(get_chatgpt)]
) -> dict[str, object]:
    # A failed sign-in is reported once, on the page load that follows the callback.
    error, request.app.state.chatgpt_error = request.app.state.chatgpt_error, None
    return {"connected": auth.connected, "email": auth.email, "error": error}


@app.post("/api/auth/chatgpt/logout")
async def chatgpt_logout(
    request: Request, auth: Annotated[ChatGPTAuth, Depends(get_chatgpt)]
) -> dict[str, bool]:
    revoked = await auth.logout()
    rewire(request.app.state)
    return {"revoked": revoked}


# Mounted last so it never shadows the API routes.
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web" / "static", html=True))
