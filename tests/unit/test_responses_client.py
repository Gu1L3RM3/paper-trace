import json
from typing import Any

import httpx
import pytest
import respx
from openai import AsyncOpenAI
from pydantic import BaseModel

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec
from papertrace.llm.openai_responses import OpenAIResponsesClient

URL = "http://llm.test/v1/responses"
MESSAGES = [Message(role="user", content="hi")]


class Answer(BaseModel):
    value: int


TOOL = ToolSpec(name="answer", description="Give the answer", output=Answer)


def _call(arguments: str, name: str = "answer") -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _reply(*output: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, json={"id": "r", "object": "response", "status": "completed", "output": list(output)}
    )


def _text(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "m",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _sent(route: respx.Route, call: int = 0) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls[call].request.content)
    return body


async def _client(http: httpx.AsyncClient, **kwargs: Any) -> OpenAIResponsesClient:
    sdk = AsyncOpenAI(
        api_key="k",
        base_url="http://llm.test/v1",
        http_client=http,  # type: ignore[arg-type]
        max_retries=0,
    )
    return OpenAIResponsesClient(client=sdk, model="gpt-x", **kwargs)


@respx.mock
async def test_forced_tool_call_is_parsed_and_effort_is_sent() -> None:
    route = respx.post(URL).mock(return_value=_reply(_call('{"value": 7}')))
    async with httpx.AsyncClient() as http:
        client = await _client(http, reasoning_effort="low")
        result = await client.call_tool(system="be brief", messages=MESSAGES, tool=TOOL)

    assert result == Answer(value=7)
    sent = _sent(route)
    assert sent["instructions"] == "be brief"
    assert sent["input"] == [{"role": "user", "content": "hi"}]
    assert sent["tool_choice"] == {"type": "function", "name": "answer"}
    assert sent["tools"][0]["name"] == "answer"
    assert sent["tools"][0]["parameters"]["properties"]["value"]
    assert sent["reasoning"] == {"effort": "low"}


@respx.mock
async def test_no_reasoning_is_sent_when_effort_is_unset() -> None:
    route = respx.post(URL).mock(return_value=_reply(_text("hello")))
    async with httpx.AsyncClient() as http:
        assert await (await _client(http)).complete(system="s", messages=MESSAGES) == "hello"
    assert "reasoning" not in _sent(route)


@respx.mock
async def test_bad_arguments_go_back_to_the_model_once() -> None:
    route = respx.post(URL).mock(
        side_effect=[_reply(_call('{"value": "x"}')), _reply(_call('{"value": 2}'))]
    )
    async with httpx.AsyncClient() as http:
        result = await (await _client(http)).call_tool(system="s", messages=MESSAGES, tool=TOOL)

    assert result == Answer(value=2)
    feedback = _sent(route, 1)["input"][-1]
    assert feedback["role"] == "user"
    assert "value" in feedback["content"]


@respx.mock
async def test_second_bad_call_and_skipped_tool_surface_as_llm_errors() -> None:
    respx.post(URL).mock(return_value=_reply(_call('{"value": "x"}')))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match="do not match"):
            await (await _client(http)).call_tool(system="s", messages=MESSAGES, tool=TOOL)

    respx.post(URL).mock(return_value=_reply(_text("no tool")))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match="did not call answer"):
            await (await _client(http)).call_tool(system="s", messages=MESSAGES, tool=TOOL)


@respx.mock
async def test_empty_answer_and_api_failures_become_llm_errors() -> None:
    respx.post(URL).mock(return_value=_reply())
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match="empty answer"):
            await (await _client(http)).complete(system="s", messages=MESSAGES)

    respx.post(URL).mock(return_value=httpx.Response(429, json={"error": {"message": "slow"}}))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match="rate limit"):
            await (await _client(http)).complete(system="s", messages=MESSAGES)
