import asyncio
import json
from contextlib import nullcontext

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec
from papertrace.settings import Settings


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
        for call in self._message(response).tool_calls or []:
            if call.type == "function" and call.function.name == tool.name:
                try:
                    return tool.output.model_validate(json.loads(call.function.arguments))
                except (ValidationError, ValueError) as exc:
                    raise LLMError(f"Invalid arguments for tool {tool.name}: {exc}") from exc
        raise LLMError(f"Model did not call tool {tool.name}")

    async def complete(self, *, system: str, messages: list[Message]) -> str:
        response = await self._create(system, messages)
        text = self._message(response).content
        if not text:
            raise LLMError("Model returned an empty response")
        return text

    @staticmethod
    def _message(response: ChatCompletion) -> ChatCompletionMessage:
        if not response.choices:
            raise LLMError("Model returned no choices")
        return response.choices[0].message

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


def build_openai_compatible_client(
    settings: Settings, model: str, semaphore: asyncio.Semaphore | None
) -> OpenAICompatibleClient:
    # Local servers (Ollama, vLLM) ignore the key but the SDK insists on one.
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
    sdk = openai.AsyncOpenAI(api_key=key, base_url=settings.llm_base_url)
    return OpenAICompatibleClient(client=sdk, model=model, semaphore=semaphore)
