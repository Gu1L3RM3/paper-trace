import asyncio
import importlib
from collections.abc import Callable

from papertrace.domain.errors import LLMError
from papertrace.llm.base import LLMClient
from papertrace.settings import Settings

# (settings, model, semaphore) -> client. The only contract a provider has to meet.
LLMProviderBuilder = Callable[[Settings, str, asyncio.Semaphore | None], LLMClient]


# Built-ins import their adapter lazily, so a vendor SDK is loaded only when selected.
def _anthropic(settings: Settings, model: str, semaphore: asyncio.Semaphore | None) -> LLMClient:
    from papertrace.llm.anthropic_client import build_anthropic_client

    return build_anthropic_client(settings, model, semaphore)


def _openai_compatible(
    settings: Settings, model: str, semaphore: asyncio.Semaphore | None
) -> LLMClient:
    from papertrace.llm.openai_compatible import build_openai_compatible_client

    return build_openai_compatible_client(settings, model, semaphore)


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
            builder = getattr(importlib.import_module(module_name), attr)
        except Exception as exc:  # a user module may fail at import in any way
            raise LLMError(f"Cannot load LLM provider {provider!r}: {exc}") from exc
        if not callable(builder):
            raise LLMError(f"LLM provider {provider!r} is not callable")
        return builder  # type: ignore[no-any-return]
    raise LLMError(f"Unknown LLM_PROVIDER {provider!r}; available: {', '.join(sorted(_PROVIDERS))}")


def build_llm_client(
    settings: Settings, *, model: str | None = None, semaphore: asyncio.Semaphore | None = None
) -> LLMClient:
    """Pick the adapter from LLM_PROVIDER; `model` overrides LLM_MODEL for one agent."""
    builder = _resolve(settings.llm_provider)
    return builder(settings, model or settings.llm_model, semaphore)
