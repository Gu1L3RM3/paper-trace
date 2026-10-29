import asyncio

import httpx
import pytest

from papertrace.api import dependencies
from papertrace.settings import Settings


async def test_all_agents_share_one_semaphore_and_their_own_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str | None, asyncio.Semaphore | None]] = []

    def recording(
        settings: Settings,
        *,
        model: str | None = None,
        semaphore: asyncio.Semaphore | None = None,
        **_: object,
    ) -> object:
        seen.append((model, semaphore))
        return object()

    monkeypatch.setattr(dependencies, "build_llm_client", recording)
    settings = Settings(
        _env_file=None,
        llm_api_key="k",
        llm_model="base",
        synthesis_model="big",
    )

    async with httpx.AsyncClient() as http:
        dependencies.build_pipeline(settings, http)

    assert [model for model, _ in seen] == ["base", "big", "big"]
    assert seen[0][1] is not None and all(sem is seen[0][1] for _, sem in seen)


async def test_only_the_verifier_gets_the_api_and_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[object, object]] = []

    def recording(settings: Settings, **kwargs: object) -> object:
        seen.append((kwargs.get("api", "chat"), kwargs.get("reasoning_effort")))
        return object()

    monkeypatch.setattr(dependencies, "build_llm_client", recording)
    settings = Settings(
        _env_file=None,
        llm_api_key="k",
        llm_model="base",
        verification_api="responses",
        verification_reasoning_effort="low",
    )

    async with httpx.AsyncClient() as http:
        dependencies.build_pipeline(settings, http)

    assert seen == [("chat", None), ("chat", None), ("responses", "low")]
