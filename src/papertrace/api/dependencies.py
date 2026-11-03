"""Composition root: the only place that instantiates concrete adapters and agents."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

import httpx
from fastapi import HTTPException, Request
from starlette.datastructures import State

from papertrace.agents.multi_source import MultiSourceResearcher
from papertrace.agents.orchestrator import OrchestratorAgent
from papertrace.agents.synthesis import SynthesisAgent
from papertrace.agents.verification import VerificationAgent
from papertrace.llm.chatgpt_auth import ChatGPTAuth
from papertrace.llm.openai_client import NO_CREDENTIALS, build_llm_client
from papertrace.pipeline import Pipeline
from papertrace.settings import Settings
from papertrace.sources.openalex import OpenAlexSource
from papertrace.sources.semantic_scholar import SemanticScholarSource

_CHATGPT_FILE = Path.home() / ".config" / "papertrace" / "chatgpt.json"


def build_pipeline(
    settings: Settings,
    http: httpx.AsyncClient,
    chatgpt_token: Callable[[], Awaitable[str]] | None = None,
) -> Pipeline:
    # One semaphore for every agent, so LLM_MAX_CONCURRENCY caps the whole app.
    semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
    llm = partial(build_llm_client, settings, semaphore=semaphore, chatgpt_token=chatgpt_token)

    return Pipeline(
        OrchestratorAgent(llm(model=settings.model_for("orchestrator"))),
        build_researcher(settings, http),
        SynthesisAgent(llm(model=settings.model_for("synthesis"))),
        VerificationAgent(
            llm(
                model=settings.model_for("verification"),
                api=settings.verification_api,
                reasoning_effort=settings.verification_reasoning_effort,
            )
        ),
    )


def wire(state: State, settings: Settings, http: httpx.AsyncClient) -> None:
    state.settings = settings
    state.http = http
    state.chatgpt = ChatGPTAuth(http=http, path=_CHATGPT_FILE)
    state.chatgpt_error = None
    rewire(state)


def rewire(state: State) -> None:
    """Rebuild the pipeline for the current credential: a connected ChatGPT wins over the key."""
    auth: ChatGPTAuth = state.chatgpt
    state.chatgpt_wired = auth.connected
    if auth.connected:
        state.pipeline = build_pipeline(state.settings, state.http, auth.access_token)
    elif state.settings.llm_api_key is not None:
        state.pipeline = build_pipeline(state.settings, state.http)
    else:
        state.pipeline = None


def get_chatgpt(request: Request) -> ChatGPTAuth:
    auth: ChatGPTAuth = request.app.state.chatgpt
    return auth


def get_pipeline(request: Request) -> Pipeline:
    state = request.app.state
    # The session can end on its own (a dead refresh token), so follow it on the next request.
    if state.chatgpt_wired != state.chatgpt.connected:
        rewire(state)
    pipeline: Pipeline | None = state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail=NO_CREDENTIALS)
    return pipeline


def build_researcher(settings: Settings, http: httpx.AsyncClient) -> MultiSourceResearcher:
    key = settings.semantic_scholar_api_key
    return MultiSourceResearcher(
        {
            "OpenAlex": OpenAlexSource(client=http, mailto=settings.openalex_mailto),
            "Semantic Scholar": SemanticScholarSource(
                client=http, api_key=key.get_secret_value() if key else None
            ),
        }
    )
