import asyncio

from pydantic import BaseModel

from papertrace.domain.models import Paper, ResearchQuestion
from papertrace.llm.base import Message, ToolSpec


class FakeLLMClient:
    """Returns a canned model for call_tool and records what it was asked."""

    def __init__(self, result: BaseModel, text: str = "") -> None:
        self._result = result
        self._text = text
        self.calls: list[tuple[str, list[Message], str]] = []

    async def call_tool[T: BaseModel](
        self, *, system: str, messages: list[Message], tool: ToolSpec[T]
    ) -> T:
        self.calls.append((system, messages, tool.name))
        return tool.output.model_validate(self._result.model_dump())

    async def complete(self, *, system: str, messages: list[Message]) -> str:
        self.calls.append((system, messages, "complete"))
        return self._text


class FakePaperSource:
    """Maps query -> papers (or an exception) after an async delay."""

    def __init__(self, results: dict[str, list[Paper] | Exception], delay: float = 0.0) -> None:
        self._results = results
        self._delay = delay

    async def search(self, query: str) -> list[Paper]:
        await asyncio.sleep(self._delay)
        result = self._results[query]
        if isinstance(result, Exception):
            raise result
        return result


def build_fake_llm(
    settings: object, model: str, semaphore: asyncio.Semaphore | None
) -> FakeLLMClient:
    """Plugin entry point used to prove providers can live outside the package."""
    return FakeLLMClient(ResearchQuestion(text="x"), text=f"model={model}")
