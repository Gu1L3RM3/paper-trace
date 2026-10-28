import asyncio
from contextlib import nullcontext
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec
from papertrace.settings import Settings

_TIMEOUT = 120.0  # local models can be slow
_ATTEMPTS = 3
_BACKOFF = 1.0  # seconds, doubled after each failed attempt
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Keywords that strict / structured-output decoders reject; ToolSpec.output still enforces them.
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
    """JSON Schema sent to the provider: portable across strict decoders, validated locally."""
    return _clean(model.model_json_schema())  # type: ignore[no-any-return]


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
                        "parameters": tool_parameters(tool.output),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool.name}},
        )
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == tool.name:
                arguments = function.get("arguments") or ""
                try:
                    # Some local servers return the arguments as an object, not a JSON string.
                    if isinstance(arguments, dict):
                        return tool.output.model_validate(arguments)
                    return tool.output.model_validate_json(arguments)
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
            message = (await self._post(body)).json()["choices"][0]["message"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if not isinstance(message, dict):
            raise LLMError("Unexpected response shape from the LLM")
        return message

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        """POST with bounded retries on 429/5xx/network errors; the slot is free while waiting."""
        for attempt in range(_ATTEMPTS):
            last = attempt == _ATTEMPTS - 1
            try:
                async with self._semaphore or nullcontext():
                    response = await self._client.post(self._url, json=body, headers=self._headers)
                if response.status_code not in _RETRYABLE_STATUS or last:
                    return response.raise_for_status()
            except httpx.TransportError:
                if last:
                    raise
            await asyncio.sleep(_BACKOFF * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover


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
