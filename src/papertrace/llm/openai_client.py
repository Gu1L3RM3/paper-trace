import asyncio
from contextlib import nullcontext
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec
from papertrace.settings import Settings

_TIMEOUT = 120.0
_MAX_RETRIES = 2  # SDK retries 408/409/429/5xx and network errors with backoff
# Keywords that strict decoders reject; ToolSpec.output still enforces them locally.
_UNSUPPORTED = {"minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "maxItems"}


def _clean(node: Any) -> Any:
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned = {
        key: _clean(value)
        for key, value in node.items()
        if key not in _UNSUPPORTED and not (key == "minItems" and value > 1)
    }
    if "properties" in cleaned:
        cleaned["additionalProperties"] = False
    return cleaned


def tool_parameters(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema sent to the API: portable across strict decoders, validated locally."""
    return _clean(model.model_json_schema())  # type: ignore[no-any-return]


class OpenAIClient:
    """`LLMClient` over the official OpenAI SDK (chat completions with a forced tool call)."""

    def __init__(
        self, *, client: AsyncOpenAI, model: str, semaphore: asyncio.Semaphore | None = None
    ) -> None:
        self._client = client
        self._model = model
        self._semaphore = semaphore

    async def call_tool[T: BaseModel](
        self, *, system: str, messages: list[Message], tool: ToolSpec[T]
    ) -> T:
        message = await self._chat(
            system,
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool_parameters(tool.output),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool.name}},
        )
        for call in message.tool_calls or []:
            if call.type == "function" and call.function.name == tool.name:
                try:
                    return tool.output.model_validate_json(call.function.arguments)
                except ValidationError as exc:
                    raise LLMError(f"Invalid arguments for tool {tool.name}: {exc}") from exc
        raise LLMError(f"Model did not call tool {tool.name}")

    async def complete(self, *, system: str, messages: list[Message]) -> str:
        text = (await self._chat(system, messages)).content
        if not text:
            raise LLMError("Model returned an empty response")
        return text

    async def _chat(
        self, system: str, messages: list[Message], **extra: Any
    ) -> ChatCompletionMessage:
        payload = cast(
            list[ChatCompletionMessageParam],
            [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
        )
        try:
            async with self._semaphore or nullcontext():
                response = await self._client.chat.completions.create(
                    model=self._model, messages=payload, **extra
                )
            message: ChatCompletionMessage = response.choices[0].message
            return message
        except (OpenAIError, IndexError, AttributeError, TypeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc


def build_llm_client(
    settings: Settings, *, model: str | None = None, semaphore: asyncio.Semaphore | None = None
) -> OpenAIClient:
    """`model` overrides LLM_MODEL for one agent; LLM_BASE_URL only matters for gateways."""
    client = AsyncOpenAI(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )
    return OpenAIClient(client=client, model=model or settings.llm_model, semaphore=semaphore)
