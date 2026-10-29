"""Composition root: the only place that instantiates concrete adapters and agents."""

import asyncio

import httpx
from fastapi import Request

from papertrace.agents.multi_source import MultiSourceResearcher
from papertrace.agents.orchestrator import OrchestratorAgent
from papertrace.agents.synthesis import SynthesisAgent
from papertrace.agents.verification import VerificationAgent
from papertrace.llm.openai_client import build_llm_client
from papertrace.pipeline import Pipeline
from papertrace.settings import Settings
from papertrace.sources.openalex import OpenAlexSource
from papertrace.sources.semantic_scholar import SemanticScholarSource


def build_pipeline(settings: Settings, http: httpx.AsyncClient) -> Pipeline:
    # One semaphore for every agent, so LLM_MAX_CONCURRENCY caps the whole app.
    semaphore = asyncio.Semaphore(settings.llm_max_concurrency)

    return Pipeline(
        OrchestratorAgent(
            build_llm_client(
                settings, model=settings.model_for("orchestrator"), semaphore=semaphore
            )
        ),
        build_researcher(settings, http),
        SynthesisAgent(
            build_llm_client(settings, model=settings.model_for("synthesis"), semaphore=semaphore)
        ),
        VerificationAgent(
            build_llm_client(
                settings,
                model=settings.model_for("verification"),
                semaphore=semaphore,
                api=settings.verification_api,
                reasoning_effort=settings.verification_reasoning_effort,
            )
        ),
    )


def get_pipeline(request: Request) -> Pipeline:
    pipeline: Pipeline = request.app.state.pipeline
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
