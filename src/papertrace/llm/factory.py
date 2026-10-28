import asyncio
import importlib
from collections.abc import Callable

from papertrace.domain.errors import LLMError
from papertrace.llm.base import LLMClient
from papertrace.settings import Settings

# (settings, model, semaphore) -> client. The only contract a provider has to meet.
LLMProviderBuilder = Callable[[Settings, str, asyncio.Semaphore | None], LLMClient]


def _chat_completions(
    settings: Settings, model: str, semaphore: asyncio.Semaphore | None
) -> LLMClient:
    from papertrace.llm.http_client import build_http_llm_client

    return build_http_llm_client(settings, model, semaphore)


_PROVIDERS: dict[str, LLMProviderBuilder] = {"chat_completions": _chat_completions}


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
