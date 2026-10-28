import asyncio

import httpx

from papertrace.api.dependencies import build_pipeline
from papertrace.llm.factory import register_llm_provider
from papertrace.settings import Settings
from tests.fakes import build_fake_llm


async def test_all_agents_share_one_semaphore_and_their_own_model() -> None:
    seen: list[tuple[str, asyncio.Semaphore | None]] = []

    def recording(settings: Settings, model: str, semaphore: asyncio.Semaphore | None) -> object:
        seen.append((model, semaphore))
        return build_fake_llm(settings, model, semaphore)

    register_llm_provider("recording", recording)  # type: ignore[arg-type]
    settings = Settings(
        llm_provider="recording",
        llm_model="base",
        llm_base_url="http://x",
        synthesis_model="big",
    )

    async with httpx.AsyncClient() as http:
        build_pipeline(settings, http)

    assert [model for model, _ in seen] == ["base", "big"]
    assert seen[0][1] is not None and seen[0][1] is seen[1][1]
