from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ToolSpec[T: BaseModel]:
    """Forced tool call whose arguments are validated into `output`."""

    name: str
    description: str
    output: type[T]


class LLMClient(Protocol):
    async def call_tool[T: BaseModel](
        self, *, system: str, messages: list[Message], tool: ToolSpec[T]
    ) -> T: ...

    async def complete(self, *, system: str, messages: list[Message]) -> str: ...
