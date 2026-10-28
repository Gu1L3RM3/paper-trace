import asyncio

import anthropic
import openai

from papertrace.llm.anthropic_client import AnthropicClient
from papertrace.llm.base import LLMClient
from papertrace.llm.openai_compatible import OpenAICompatibleClient
from papertrace.settings import Settings


def build_llm_client(
    settings: Settings, *, model: str | None = None, semaphore: asyncio.Semaphore | None = None
) -> LLMClient:
    """Pick the adapter from LLM_PROVIDER; `model` overrides LLM_MODEL for one agent."""
    model = model or settings.llm_model
    api_key = settings.llm_api_key.get_secret_value()
    if settings.llm_provider == "anthropic":
        sdk = anthropic.AsyncAnthropic(api_key=api_key)
        return AnthropicClient(client=sdk, model=model, semaphore=semaphore)
    oai = openai.AsyncOpenAI(api_key=api_key, base_url=settings.llm_base_url)
    return OpenAICompatibleClient(client=oai, model=model, semaphore=semaphore)
