"""Every LLMClient adapter must pass this suite unchanged (Liskov)."""

import json
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest
import respx
from openai import AsyncOpenAI
from pydantic import BaseModel

from papertrace.domain.errors import LLMError
from papertrace.llm.base import LLMClient, Message, ToolSpec
from papertrace.llm.openai_client import OpenAIClient


class Answer(BaseModel):
    value: int


TOOL = ToolSpec(name="answer", description="Give the answer", output=Answer)
MESSAGES = [Message(role="user", content="What is 6 times 7?")]
BASE_URL = "http://llm.test/v1"


@dataclass
class Provider:
    """How to build one adapter and fake the API it talks to."""

    build: Callable[[httpx.AsyncClient, str | None], LLMClient]
    url: str
    tool_reply: Callable[[dict[str, object]], dict[str, object]]
    text_reply: Callable[[str], dict[str, object]]
    empty_reply: dict[str, object]
    no_content_reply: dict[str, object]


def _chat_completion(message: dict[str, object]) -> dict[str, object]:
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": message}]}


HTTP = Provider(
    build=lambda client, key: OpenAIClient(
        client=AsyncOpenAI(
            api_key=key or "k",
            base_url=BASE_URL,
            http_client=client,  # type: ignore[arg-type]
            max_retries=2,
        ),
        model="m",
    ),
    url=f"{BASE_URL}/chat/completions",
    tool_reply=lambda args: _chat_completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "answer", "arguments": json.dumps(args)},
                }
            ],
        }
    ),
    text_reply=lambda text: _chat_completion({"role": "assistant", "content": text}),
    empty_reply=_chat_completion({"role": "assistant", "content": "no tool"}),
    no_content_reply={"choices": []},
)


@pytest.fixture(params=[HTTP], ids=["http"])
def provider(request: pytest.FixtureRequest) -> Provider:
    return request.param  # type: ignore[no-any-return]


def _mock(provider: Provider, status: int, payload: dict[str, object]) -> respx.Route:
    return respx.post(provider.url).mock(return_value=httpx.Response(status, json=payload))


@respx.mock
async def test_call_tool_returns_validated_model(provider: Provider) -> None:
    _mock(provider, 200, provider.tool_reply({"value": 42}))
    async with httpx.AsyncClient() as http:
        result = await provider.build(http, None).call_tool(
            system="s", messages=MESSAGES, tool=TOOL
        )
    assert result == Answer(value=42)


@respx.mock
async def test_call_tool_rejects_arguments_that_fail_validation(provider: Provider) -> None:
    _mock(provider, 200, provider.tool_reply({"value": "nope"}))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError):
            await provider.build(http, None).call_tool(system="s", messages=MESSAGES, tool=TOOL)


@respx.mock
async def test_call_tool_raises_when_model_skips_the_tool(provider: Provider) -> None:
    _mock(provider, 200, provider.empty_reply)
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError):
            await provider.build(http, None).call_tool(system="s", messages=MESSAGES, tool=TOOL)


@respx.mock
async def test_complete_returns_text(provider: Provider) -> None:
    _mock(provider, 200, provider.text_reply("forty-two"))
    async with httpx.AsyncClient() as http:
        assert await provider.build(http, None).complete(system="s", messages=MESSAGES) == (
            "forty-two"
        )


@respx.mock
async def test_vendor_http_error_becomes_llm_error(provider: Provider) -> None:
    _mock(provider, 400, {"error": "bad"})
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError):
            await provider.build(http, None).complete(system="s", messages=MESSAGES)


@respx.mock
async def test_network_failure_becomes_llm_error(provider: Provider) -> None:
    respx.post(provider.url).mock(side_effect=httpx.ConnectError("refused"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError):
            await provider.build(http, None).complete(system="s", messages=MESSAGES)


@respx.mock
async def test_response_without_content_becomes_llm_error(provider: Provider) -> None:
    _mock(provider, 200, provider.no_content_reply)
    async with httpx.AsyncClient() as http:
        client = provider.build(http, None)
        with pytest.raises(LLMError):
            await client.complete(system="s", messages=MESSAGES)
        with pytest.raises(LLMError):
            await client.call_tool(system="s", messages=MESSAGES, tool=TOOL)


@respx.mock
async def test_non_json_body_becomes_llm_error(provider: Provider) -> None:
    respx.post(provider.url).mock(return_value=httpx.Response(200, text="<html>oops</html>"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError):
            await provider.build(http, None).complete(system="s", messages=MESSAGES)


@respx.mock
async def test_api_key_is_sent_as_bearer(provider: Provider) -> None:
    route = _mock(provider, 200, provider.text_reply("ok"))
    async with httpx.AsyncClient() as http:
        await provider.build(http, "secret").complete(system="s", messages=MESSAGES)
    assert route.calls.last.request.headers["authorization"] == "Bearer secret"


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "openai._base_client.BaseClient._calculate_retry_timeout", lambda *_, **__: 0
    )


@respx.mock
@pytest.mark.usefixtures("no_backoff")
@pytest.mark.parametrize(
    "failure", [httpx.Response(429), httpx.Response(503), httpx.ConnectError("x")]
)
async def test_transient_failure_is_retried(
    provider: Provider, failure: httpx.Response | Exception
) -> None:
    ok = httpx.Response(200, json=provider.text_reply("ok"))
    route = respx.post(provider.url).mock(side_effect=[failure, ok])
    async with httpx.AsyncClient() as http:
        assert await provider.build(http, None).complete(system="s", messages=MESSAGES) == "ok"
    assert route.call_count == 2


@respx.mock
@pytest.mark.usefixtures("no_backoff")
async def test_retries_are_bounded(provider: Provider) -> None:
    route = respx.post(provider.url).mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError):
            await provider.build(http, None).complete(system="s", messages=MESSAGES)
    assert route.call_count == 3


@respx.mock
async def test_client_error_is_not_retried(provider: Provider) -> None:
    route = _mock(provider, 400, {"error": "bad"})
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError):
            await provider.build(http, None).complete(system="s", messages=MESSAGES)
    assert route.call_count == 1


def test_tool_parameters_drop_keywords_strict_decoders_reject() -> None:
    from papertrace.domain.models import SearchPlan
    from papertrace.llm.openai_client import tool_parameters

    schema = tool_parameters(SearchPlan)
    text = str(schema)

    assert "minLength" not in text and "maxItems" not in text and "minItems" not in text
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["SearchQuery"]["additionalProperties"] is False
    assert "description" in schema["properties"]["queries"]
