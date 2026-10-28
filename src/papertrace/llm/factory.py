import asyncio
import importlib
from collections.abc import Callable

from papertrace.domain.errors import LLMError
from papertrace.llm.base import LLMClient
from papertrace.settings import Settings

# (settings, model, semaphore) -> client. The only contract a provider has to meet.
LLMProviderBuilder = Callable[[Settings, str, asyncio.Semaphore | None], LLMClient]


def _anthropic(settings: Settings, model: str, semaphore: asyncio.Semaphore | None) -> LLMClient:
    import anthropic  # lazy: the SDK is only needed when this provider is selected

    from papertrace.llm.anthropic_client import AnthropicClient

    if settings.llm_api_key is None:
        raise LLMError("LLM_API_KEY is required for the anthropic provider")
    sdk = anthropic.AsyncAnthropic(api_key=settings.llm_api_key.get_secret_value())
    return AnthropicClient(client=sdk, model=model, semaphore=semaphore)


def _openai_compatible(
    settings: Settings, model: str, semaphore: asyncio.Semaphore | None
) -> LLMClient:
    import openai

    from papertrace.llm.openai_compatible import OpenAICompatibleClient

    # Local servers (Ollama, vLLM) ignore the key but the SDK insists on one.
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
    sdk = openai.AsyncOpenAI(api_key=key, base_url=settings.llm_base_url)
    return OpenAICompatibleClient(client=sdk, model=model, semaphore=semaphore)


_PROVIDERS: dict[str, LLMProviderBuilder] = {
    "anthropic": _anthropic,
    "openai_compatible": _openai_compatible,
}


def register_llm_provider(name: str, builder: LLMProviderBuilder) -> None:
    """Add a provider without touching this module's existing code (Open/Closed)."""
    _PROVIDERS[name] = builder


def _resolve(provider: str) -> LLMProviderBuilder:
    if provider in _PROVIDERS:
        return _PROVIDERS[provider]
    if ":" in provider:  # "package.module:function" lets users plug in their own adapter
        module_name, _, attr = provider.partition(":")
        try:
            return getattr(importlib.import_module(module_name), attr)  # type: ignore[no-any-return]
        except (ImportError, AttributeError) as exc:
            raise LLMError(f"Cannot load LLM provider {provider!r}: {exc}") from exc
    raise LLMError(f"Unknown LLM_PROVIDER {provider!r}; available: {', '.join(sorted(_PROVIDERS))}")


def build_llm_client(
    settings: Settings, *, model: str | None = None, semaphore: asyncio.Semaphore | None = None
) -> LLMClient:
    """Pick the adapter from LLM_PROVIDER; `model` overrides LLM_MODEL for one agent."""
    builder = _resolve(settings.llm_provider)
    return builder(settings, model or settings.llm_model, semaphore)
