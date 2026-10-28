"""Composition root: the only place that instantiates concrete adapters and agents."""

import asyncio

import httpx
from fastapi import Request

from papertrace.agents.orchestrator import OrchestratorAgent
from papertrace.agents.research import ResearchAgent
from papertrace.agents.synthesis import SynthesisAgent
from papertrace.agents.verification import VerificationAgent
from papertrace.llm.openai_client import build_llm_client
from papertrace.pipeline import Pipeline
from papertrace.settings import Settings
from papertrace.sources.openalex import OpenAlexSource


def build_pipeline(settings: Settings, http: httpx.AsyncClient) -> Pipeline:
    # One semaphore for every agent, so LLM_MAX_CONCURRENCY caps the whole app.
    semaphore = asyncio.Semaphore(settings.llm_max_concurrency)

    return Pipeline(
        OrchestratorAgent(
            build_llm_client(
                settings, model=settings.model_for("orchestrator"), semaphore=semaphore
            )
        ),
        ResearchAgent(OpenAlexSource(client=http, mailto=settings.openalex_mailto)),
        SynthesisAgent(
            build_llm_client(settings, model=settings.model_for("synthesis"), semaphore=semaphore)
        ),
        VerificationAgent(
            build_llm_client(
                settings, model=settings.model_for("verification"), semaphore=semaphore
            )
        ),
    )


def get_pipeline(request: Request) -> Pipeline:
    pipeline: Pipeline = request.app.state.pipeline
    return pipeline
