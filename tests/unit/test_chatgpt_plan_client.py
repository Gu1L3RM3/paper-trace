import json
from typing import Any

import httpx
import pytest
import respx
from openai import AsyncOpenAI
from pydantic import BaseModel

from papertrace.domain.errors import LLMError
from papertrace.llm.base import Message, ToolSpec
from papertrace.llm.chatgpt_plan import ChatGPTPlanClient

URL = "http://llm.test/v1/responses"
MESSAGES = [Message(role="user", content="hi")]


class Answer(BaseModel):
    value: int


TOOL = ToolSpec(name="answer", description="Give the answer", output=Answer)


def sse(*events: dict[str, Any]) -> httpx.Response:
    """A streamed Responses API reply, one SSE frame per event."""
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def finished(status: str, output: list[dict[str, Any]], **response: Any) -> dict[str, Any]:
    return {
        "type": f"response.{status}",
        "response": {"id": "r", "object": "response", "status": status, "output": output}
        | response,
    }


TEXT = {
    "type": "message",
    "id": "m",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "forty-two", "annotations": []}],
}
CALL = {
    "type": "function_call",
    "id": "fc",
    "call_id": "c",
    "name": "answer",
    "arguments": '{"value": 7}',
    "status": "completed",
}


async def _client(http: httpx.AsyncClient) -> ChatGPTPlanClient:
    async def token() -> str:
        return "oauth-token"

    sdk = AsyncOpenAI(
        api_key=token,
        base_url="http://llm.test/v1",
        http_client=http,  # type: ignore[arg-type]
        max_retries=0,
    )
    return ChatGPTPlanClient(client=sdk, model="gpt-x")


@respx.mock
async def test_streams_without_storing_and_sends_the_oauth_token() -> None:
    route = respx.post(URL).mock(
        return_value=sse(
            {"type": "response.output_text.delta", "delta": "forty"}, finished("completed", [TEXT])
        )
    )
    async with httpx.AsyncClient() as http:
        assert await (await _client(http)).complete(system="s", messages=MESSAGES) == "forty-two"

    sent = json.loads(route.calls.last.request.content)
    assert sent["stream"] is True and sent["store"] is False
    assert sent["instructions"] == "s"
    assert route.calls.last.request.headers["authorization"] == "Bearer oauth-token"


@respx.mock
async def test_tools_are_grouped_in_a_namespace_and_forced() -> None:
    route = respx.post(URL).mock(return_value=sse(finished("completed", [CALL])))
    async with httpx.AsyncClient() as http:
        result = await (await _client(http)).call_tool(system="s", messages=MESSAGES, tool=TOOL)

    assert result == Answer(value=7)
    sent = json.loads(route.calls.last.request.content)
    [namespace] = sent["tools"]
    assert namespace["type"] == "namespace"
    assert [t["name"] for t in namespace["tools"]] == ["answer"]
    assert sent["tool_choice"] == {"type": "function", "name": "answer"}


@respx.mock
@pytest.mark.parametrize(
    ("status", "code", "hint"),
    [
        ("failed", "subscription_sharing_usage_limit_exceeded", "Usage"),
        ("failed", "subscription_sharing_usage_unavailable", "temporarily"),
        ("failed", "something_new", "something_new"),
        ("incomplete", None, "cut off"),
    ],
)
async def test_failed_stream_explains_the_plan_error(
    status: str, code: str | None, hint: str
) -> None:
    error = {"code": code, "message": "x"} if code else None
    respx.post(URL).mock(return_value=sse(finished(status, [], error=error)))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match=hint):
            await (await _client(http)).complete(system="s", messages=MESSAGES)


@respx.mock
async def test_stream_that_ends_early_is_an_error() -> None:
    respx.post(URL).mock(return_value=sse({"type": "response.output_text.delta", "delta": "x"}))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match="ended"):
            await (await _client(http)).complete(system="s", messages=MESSAGES)


@respx.mock
@pytest.mark.parametrize(
    ("status", "body", "hint"),
    [
        (401, {"error": {"message": "no", "code": None}}, "connect ChatGPT again"),
        (
            403,
            {"error": {"message": "no", "code": "subscription_sharing_user_not_eligible"}},
            "cannot be used",
        ),
        (403, {"detail": "region"}, "refused"),
        (
            429,
            {"error": {"message": "no", "code": "subscription_sharing_usage_limit_exceeded"}},
            "Usage",
        ),
    ],
)
async def test_rejected_request_explains_the_plan_error(
    status: int, body: dict[str, Any], hint: str
) -> None:
    respx.post(URL).mock(return_value=httpx.Response(status, json=body))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match=hint) as caught:
            await (await _client(http)).complete(system="s", messages=MESSAGES)
    assert "LLM_API_KEY is valid" not in str(caught.value)
