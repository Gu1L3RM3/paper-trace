import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from papertrace.domain.models import Paper, ResearchQuestion, SynthesisResult
from papertrace.domain.papers import abstract_from_inverted_index
from papertrace.domain.verification import VerificationResult
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


class FakeVerifier:
    async def verify(
        self, question: ResearchQuestion, synthesis: "SynthesisResult", papers: list[Paper]
    ) -> "VerificationResult":
        return VerificationResult(claims=[], findings=[], status="no_issues_found", reason="fake")


def papers() -> list[Paper]:
    works = json.loads(Path("tests/fixtures/openalex_works.json").read_text())["results"]
    return [
        Paper(
            title=w["title"],
            authors=[a["author"]["display_name"] for a in w["authorships"]],
            year=w["publication_year"],
            doi=w["doi"],
            abstract=abstract_from_inverted_index(w.get("abstract_inverted_index")),
            url=w["id"],
        )
        for w in works
    ]
