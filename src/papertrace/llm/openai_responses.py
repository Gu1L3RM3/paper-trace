import asyncio
from contextlib import nullcontext
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import Response, ResponseInputParam
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec
from papertrace.llm.openai_client import _CORRECTIONS, _explain, tool_parameters
from papertrace.settings import ReasoningEffort


class OpenAIResponsesClient:
    """`LLMClient` over the Responses API, where some models (gpt-6.1-terra) take tools."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        semaphore: asyncio.Semaphore | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._semaphore = semaphore
        self._reasoning_effort = reasoning_effort

    async def call_tool[T: BaseModel](
        self, *, system: str, messages: list[Message], tool: ToolSpec[T]
    ) -> T:
        """Forced tool call; a bad call goes back to the model, which gets one chance to fix it."""
        tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool_parameters(tool.output),
                "strict": False,
            }
        ]
        choice = {"type": "function", "name": tool.name}
        feedback: list[dict[str, str]] = []
        for attempt in range(_CORRECTIONS + 1):
            response = await self._create(
                system, messages, feedback, tools=tools, tool_choice=choice
            )
            call = next(
                (
                    item
                    for item in response.output
                    if item.type == "function_call" and item.name == tool.name
                ),
                None,
            )
            cause: Exception | None = None
            if call is None:
                problem = (
                    f"Model {self._model!r} did not call {tool.name}. Try again; if it keeps "
                    "happening, set the model to one that supports function calling."
                )
                correction = f"Call the tool {tool.name} to answer."
            else:
                try:
                    return tool.output.model_validate_json(call.arguments)
                except ValidationError as exc:
                    cause = exc
                    fields = "; ".join(
                        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                    )
                    problem = (
                        f"The model's arguments for {tool.name} do not match its schema "
                        f"({fields}). Try again; if it keeps happening, use a stronger model."
                    )
                    correction = (
                        f"Your arguments do not match the schema of {tool.name}: {fields}. "
                        f"Call {tool.name} again with arguments that satisfy the schema."
                    )
            if attempt == _CORRECTIONS:
                raise LLMError(problem) from cause
            feedback.append({"role": "user", "content": correction})
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete(self, *, system: str, messages: list[Message]) -> str:
        text = (await self._create(system, messages)).output_text
        if not text:
            raise LLMError(
                "The model returned an empty answer (filtered or cut off). Try again or rephrase."
            )
        return text

    async def _create(
        self,
        system: str,
        messages: list[Message],
        feedback: list[dict[str, str]] | None = None,
        **extra: Any,
    ) -> Response:
        if self._reasoning_effort:
            extra["reasoning"] = {"effort": self._reasoning_effort}
        items = cast(
            ResponseInputParam,
            [{"role": m.role, "content": m.content} for m in messages] + (feedback or []),
        )
        try:
            async with self._semaphore or nullcontext():
                response: Response = await self._client.responses.create(
                    model=self._model, instructions=system, input=items, **extra
                )
                return response
        except (OpenAIError, AttributeError, TypeError) as exc:
            raise LLMError(_explain(exc, self._model)) from exc
