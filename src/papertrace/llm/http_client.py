import asyncio
from contextlib import nullcontext
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec
from papertrace.settings import Settings

_TIMEOUT = 120.0  # local models can be slow


class HttpLLMClient:
    """Any server speaking the chat/completions protocol with function calling.

    Vendor-neutral: Ollama, vLLM, LM Studio, OpenRouter, Groq, gateways, hosted APIs.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        api_key: str | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._client = client
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
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
                        "parameters": tool.output.model_json_schema(),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool.name}},
        )
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == tool.name:
                try:
                    return tool.output.model_validate_json(function.get("arguments") or "")
                except ValidationError as exc:
                    raise LLMError(f"Invalid arguments for tool {tool.name}: {exc}") from exc
        raise LLMError(f"Model did not call tool {tool.name}")

    async def complete(self, *, system: str, messages: list[Message]) -> str:
        text = (await self._chat(system, messages)).get("content")
        if not isinstance(text, str) or not text:
            raise LLMError("Model returned an empty response")
        return text

    async def _chat(self, system: str, messages: list[Message], **extra: object) -> dict[str, Any]:
        body = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            **extra,
        }
        try:
            async with self._semaphore or nullcontext():
                response = await self._client.post(self._url, json=body, headers=self._headers)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if not isinstance(message, dict):
            raise LLMError("Unexpected response shape from the LLM")
        return message


def build_http_llm_client(
    settings: Settings, model: str, semaphore: asyncio.Semaphore | None
) -> HttpLLMClient:
    return HttpLLMClient(
        client=httpx.AsyncClient(timeout=_TIMEOUT),
        base_url=settings.llm_base_url,
        model=model,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        semaphore=semaphore,
    )
