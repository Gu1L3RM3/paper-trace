"""Every LLMClient adapter must pass this suite unchanged (Liskov)."""

import json
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
import httpx2
import openai
import pytest
from pydantic import BaseModel

from papertrace.domain.errors import LLMError
from papertrace.llm.anthropic_client import AnthropicClient
from papertrace.llm.base import LLMClient, Message, ToolSpec
from papertrace.llm.openai_compatible import OpenAICompatibleClient


class Answer(BaseModel):
    value: int


TOOL = ToolSpec(name="answer", description="Give the answer", output=Answer)
MESSAGES = [Message(role="user", content="What is 6 times 7?")]


@dataclass
class Provider:
    """How to build one adapter and fake its vendor HTTP API."""

    build: Callable[[httpx2.MockTransport], LLMClient]
    tool_reply: Callable[[dict[str, object]], dict[str, object]]
    text_reply: Callable[[str], dict[str, object]]
    empty_reply: dict[str, object]
    no_content_reply: dict[str, object]


def _anthropic_message(content: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": content,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _anthropic(transport: httpx2.MockTransport) -> LLMClient:
    sdk = anthropic.AsyncAnthropic(
        api_key="k", http_client=httpx2.AsyncClient(transport=transport), max_retries=0
    )
    return AnthropicClient(client=sdk, model="m")


ANTHROPIC = Provider(
    build=_anthropic,
    tool_reply=lambda args: _anthropic_message(
        [{"type": "tool_use", "id": "t1", "name": "answer", "input": args}]
    ),
    text_reply=lambda text: _anthropic_message([{"type": "text", "text": text}]),
    empty_reply=_anthropic_message([{"type": "text", "text": "no tool"}]),
    no_content_reply=_anthropic_message([]),
)


def _openai(transport: httpx2.MockTransport) -> LLMClient:
    sdk = openai.AsyncOpenAI(
        api_key="k",
        base_url="http://llm.test/v1",
        http_client=httpx2.AsyncClient(transport=transport),
        max_retries=0,
    )
    return OpenAICompatibleClient(client=sdk, model="m")


def _chat_completion(message: dict[str, object]) -> dict[str, object]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }


OPENAI = Provider(
    build=_openai,
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
    no_content_reply={**_chat_completion({}), "choices": []},
)


@pytest.fixture(params=[ANTHROPIC, OPENAI], ids=["anthropic", "openai_compatible"])
def provider(request: pytest.FixtureRequest) -> Provider:
    return request.param  # type: ignore[no-any-return]


def _replying(status: int, payload: dict[str, object]) -> httpx2.MockTransport:
    return httpx2.MockTransport(lambda request: httpx2.Response(status, json=payload))


async def test_call_tool_returns_validated_model(provider: Provider) -> None:
    client = provider.build(_replying(200, provider.tool_reply({"value": 42})))
    result = await client.call_tool(system="s", messages=MESSAGES, tool=TOOL)
    assert result == Answer(value=42)


async def test_call_tool_rejects_arguments_that_fail_validation(provider: Provider) -> None:
    client = provider.build(_replying(200, provider.tool_reply({"value": "nope"})))
    with pytest.raises(LLMError):
        await client.call_tool(system="s", messages=MESSAGES, tool=TOOL)


async def test_call_tool_raises_when_model_skips_the_tool(provider: Provider) -> None:
    client = provider.build(_replying(200, provider.empty_reply))
    with pytest.raises(LLMError):
        await client.call_tool(system="s", messages=MESSAGES, tool=TOOL)


async def test_complete_returns_text(provider: Provider) -> None:
    client = provider.build(_replying(200, provider.text_reply("forty-two")))
    assert await client.complete(system="s", messages=MESSAGES) == "forty-two"


async def test_vendor_http_error_becomes_llm_error(provider: Provider) -> None:
    client = provider.build(_replying(400, {"error": "bad"}))
    with pytest.raises(LLMError):
        await client.complete(system="s", messages=MESSAGES)


async def test_response_without_content_becomes_llm_error(provider: Provider) -> None:
    client = provider.build(_replying(200, provider.no_content_reply))
    with pytest.raises(LLMError):
        await client.complete(system="s", messages=MESSAGES)
    with pytest.raises(LLMError):
        await client.call_tool(system="s", messages=MESSAGES, tool=TOOL)
