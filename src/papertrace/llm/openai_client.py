import asyncio
from contextlib import nullcontext
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec
from papertrace.settings import ReasoningEffort, Settings

_TIMEOUT = 120.0
_CORRECTIONS = 1  # times a bad tool call goes back to the model
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


def _explain(exc: Exception, model: str) -> str:
    """Cause plus next step. Provider texts are relayed only for 400s: auth errors can echo keys."""
    if isinstance(exc, APITimeoutError):
        return "The OpenAI request timed out. Try again, or use a faster model in LLM_MODEL."
    if isinstance(exc, APIConnectionError):
        return (
            "Cannot reach the OpenAI API. Check the network connection and, if set, LLM_BASE_URL."
        )
    if isinstance(exc, AuthenticationError | PermissionDeniedError):
        return "OpenAI rejected the credentials. Check that LLM_API_KEY is valid and has access."
    if isinstance(exc, NotFoundError):
        return (
            f"Model {model!r} was not found. Set LLM_MODEL (or the per-agent override) to a "
            "model your account can use."
        )
    if isinstance(exc, RateLimitError):
        return (
            "OpenAI rate limit or quota exceeded. Wait a moment and try again, lower "
            "LLM_MAX_CONCURRENCY, or check the account's billing."
        )
    if isinstance(exc, BadRequestError):
        return (
            f"OpenAI rejected the request: {exc.message}. Check that {model!r} supports "
            "function calling."
        )
    if isinstance(exc, APIStatusError):
        return f"OpenAI returned a server error (HTTP {exc.status_code}). Try again shortly."
    return f"The OpenAI call failed ({type(exc).__name__}). Try again; check the server logs."


class OpenAIClient:
    """`LLMClient` over the official OpenAI SDK (chat completions with a forced tool call)."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        semaphore: asyncio.Semaphore | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._client = client
        self._reasoning_effort = reasoning_effort
        self._model = model
        self._semaphore = semaphore

    async def call_tool[T: BaseModel](
        self, *, system: str, messages: list[Message], tool: ToolSpec[T]
    ) -> T:
        """Forced tool call; a bad call goes back to the model, which gets one chance to fix it."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool_parameters(tool.output),
                },
            }
        ]
        choice = {"type": "function", "function": {"name": tool.name}}
        feedback: list[dict[str, Any]] = []
        for attempt in range(_CORRECTIONS + 1):
            message = await self._chat(system, messages, feedback, tools=tools, tool_choice=choice)
            call = next(
                (
                    c
                    for c in message.tool_calls or []
                    if c.type == "function" and c.function.name == tool.name
                ),
                None,
            )
            cause: Exception | None = None
            turn: list[dict[str, Any]]
            if call is None:
                problem = (
                    f"Model {self._model!r} did not call {tool.name}. Try again; if it keeps "
                    "happening, set LLM_MODEL to a model that supports function calling."
                )
                # An assistant turn without text or tool calls is invalid for some providers.
                turn = (
                    [{"role": "assistant", "content": message.content}] if message.content else []
                )
                turn.append({"role": "user", "content": f"Call the tool {tool.name} to answer."})
            else:
                try:
                    return tool.output.model_validate_json(call.function.arguments)
                except ValidationError as exc:
                    cause = exc
                    fields = "; ".join(
                        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                    )
                    problem = (
                        f"The model's arguments for {tool.name} do not match its schema "
                        f"({fields}). Try again; if it keeps happening, use a stronger model."
                    )
                    turn = [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool.name,
                                        "arguments": call.function.arguments,
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": (
                                f"Your arguments do not match the schema of {tool.name}: "
                                f"{fields}. Call {tool.name} again with arguments that satisfy "
                                "the schema."
                            ),
                        },
                    ]
            if attempt == _CORRECTIONS:
                raise LLMError(problem) from cause
            feedback += turn
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete(self, *, system: str, messages: list[Message]) -> str:
        text = (await self._chat(system, messages)).content
        if not text:
            raise LLMError(
                "The model returned an empty answer (filtered or cut off). Try again or rephrase."
            )
        return text

    async def _chat(
        self,
        system: str,
        messages: list[Message],
        feedback: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> ChatCompletionMessage:
        payload = cast(
            list[ChatCompletionMessageParam],
            [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages]
            + (feedback or []),
        )
        if self._reasoning_effort:
            extra["reasoning_effort"] = self._reasoning_effort
        try:
            async with self._semaphore or nullcontext():
                response = await self._client.chat.completions.create(
                    model=self._model, messages=payload, **extra
                )
            message: ChatCompletionMessage = response.choices[0].message
            return message
        except (OpenAIError, IndexError, AttributeError, TypeError) as exc:
            raise LLMError(_explain(exc, self._model)) from exc


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
    return OpenAIClient(
        client=client,
        model=model or settings.llm_model,
        semaphore=semaphore,
        reasoning_effort=settings.llm_reasoning_effort,
    )
