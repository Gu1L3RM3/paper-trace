import asyncio
from contextlib import nullcontext

import anthropic
from anthropic.types import (
    AssistantMessageParam,
    MessageParam,
    ToolChoiceToolParam,
    ToolParam,
    UserMessageParam,
)
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec

_MAX_TOKENS = 4096


class AnthropicClient:
    def __init__(
        self,
        *,
        client: anthropic.AsyncAnthropic,
        model: str,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._semaphore = semaphore

    async def call_tool[T: BaseModel](
        self, *, system: str, messages: list[Message], tool: ToolSpec[T]
    ) -> T:
        spec: ToolParam = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.output.model_json_schema(),
        }
        choice: ToolChoiceToolParam = {"type": "tool", "name": tool.name}
        response = await self._create(system, messages, tools=[spec], tool_choice=choice)
        for block in response.content:
            if block.type == "tool_use" and block.name == tool.name:
                try:
                    return tool.output.model_validate(block.input)
                except ValidationError as exc:
                    raise LLMError(f"Invalid arguments for tool {tool.name}: {exc}") from exc
        raise LLMError(f"Model did not call tool {tool.name}")

    async def complete(self, *, system: str, messages: list[Message]) -> str:
        response = await self._create(system, messages)
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise LLMError("Model returned an empty response")
        return text

    async def _create(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolParam] | anthropic.Omit = anthropic.omit,
        tool_choice: ToolChoiceToolParam | anthropic.Omit = anthropic.omit,
    ) -> anthropic.types.Message:
        params: list[MessageParam] = [
            UserMessageParam(role="user", content=m.content)
            if m.role == "user"
            else AssistantMessageParam(role="assistant", content=m.content)
            for m in messages
        ]
        try:
            async with self._semaphore or nullcontext():
                return await self._client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    system=system,
                    messages=params,
                    tools=tools,
                    tool_choice=tool_choice,
                )
        except anthropic.AnthropicError as exc:
            raise LLMError(str(exc)) from exc
