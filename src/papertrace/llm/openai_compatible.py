import asyncio
import json
from contextlib import nullcontext

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec


class OpenAICompatibleClient:
    """Any endpoint speaking the OpenAI chat API with function calling."""

    def __init__(
        self,
        *,
        client: openai.AsyncOpenAI,
        model: str,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._semaphore = semaphore

    async def call_tool[T: BaseModel](
        self, *, system: str, messages: list[Message], tool: ToolSpec[T]
    ) -> T:
        spec: ChatCompletionFunctionToolParam = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.output.model_json_schema(),
            },
        }
        choice: ChatCompletionNamedToolChoiceParam = {
            "type": "function",
            "function": {"name": tool.name},
        }
        response = await self._create(system, messages, tools=[spec], tool_choice=choice)
        for call in response.choices[0].message.tool_calls or []:
            if call.type == "function" and call.function.name == tool.name:
                try:
                    return tool.output.model_validate(json.loads(call.function.arguments))
                except (ValidationError, ValueError) as exc:
                    raise LLMError(f"Invalid arguments for tool {tool.name}: {exc}") from exc
        raise LLMError(f"Model did not call tool {tool.name}")

    async def complete(self, *, system: str, messages: list[Message]) -> str:
        response = await self._create(system, messages)
        return response.choices[0].message.content or ""

    async def _create(
        self,
        system: str,
        messages: list[Message],
        tools: list[ChatCompletionFunctionToolParam] | openai.Omit = openai.omit,
        tool_choice: ChatCompletionNamedToolChoiceParam | openai.Omit = openai.omit,
    ) -> ChatCompletion:
        params: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
        params += [
            ChatCompletionUserMessageParam(role="user", content=m.content)
            if m.role == "user"
            else ChatCompletionAssistantMessageParam(role="assistant", content=m.content)
            for m in messages
        ]
        try:
            async with self._semaphore or nullcontext():
                return await self._client.chat.completions.create(
                    model=self._model, messages=params, tools=tools, tool_choice=tool_choice
                )
        except openai.OpenAIError as exc:
            raise LLMError(str(exc)) from exc
