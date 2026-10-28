"""Opt-in against real APIs: `uv run pytest -m live`. LLM tests need credentials configured."""

import httpx
import pytest
from pydantic import ValidationError

from papertrace.agents.orchestrator import OrchestratorAgent
from papertrace.agents.research import ResearchAgent
from papertrace.domain.models import ResearchQuestion
from papertrace.llm.factory import build_llm_client
from papertrace.settings import Settings
from papertrace.sources.openalex import OpenAlexSource

pytestmark = pytest.mark.live


async def test_openalex_returns_real_papers() -> None:
    async with httpx.AsyncClient(timeout=30) as http:
        papers = await OpenAlexSource(client=http).search("intermittent fasting cognition")
    assert papers
    assert all(p.title and p.url for p in papers)


async def test_question_in_portuguese_yields_real_papers() -> None:
    try:
        settings = Settings()
    except ValidationError:
        pytest.skip("LLM credentials not configured")
    llm = build_llm_client(settings, model=settings.model_for("orchestrator"))
    plan = await OrchestratorAgent(llm).plan(
        ResearchQuestion(text="O jejum intermitente melhora a memória?")
    )
    async with httpx.AsyncClient(timeout=30) as http:
        result = await ResearchAgent(OpenAlexSource(client=http)).research(plan)
    assert 2 <= len(plan.queries) <= 4
    assert result.papers


async def test_live_verification_report() -> None:
    from papertrace.agents.synthesis import SynthesisAgent
    from papertrace.agents.verification import VerificationAgent
    from tests.fakes import papers

    try:
        settings = Settings()
    except ValidationError:
        pytest.skip("LLM credentials not configured")
    question = ResearchQuestion(
        text="What do the retrieved abstracts say about fasting and cognition?"
    )
    evidence = [p for p in papers() if p.abstract]
    synthesis = await SynthesisAgent(
        build_llm_client(settings, model=settings.model_for("synthesis"))
    ).synthesize(question, evidence)
    report = await VerificationAgent(
        build_llm_client(settings, model=settings.model_for("verification"))
    ).verify(question, synthesis, evidence)
    assert report.claims
    assert report.reason
    print(report.model_dump_json(indent=2))
