"""Every error a user or operator sees must say what went wrong and what to try next."""

import httpx
import pytest
import respx
from openai import AsyncOpenAI
from pydantic import BaseModel

from papertrace.domain.errors import LLMError, SourceError
from papertrace.llm.base import Message, ToolSpec
from papertrace.llm.openai_client import OpenAIClient
from papertrace.sources.openalex import OpenAlexSource

URL = "http://llm.test/v1/chat/completions"
MESSAGES = [Message(role="user", content="hi")]
SECRET = "sk-super-secret"


class Answer(BaseModel):
    value: int


TOOL = ToolSpec(name="answer", description="Give the answer", output=Answer)


def _client(http: httpx.AsyncClient) -> OpenAIClient:
    sdk = AsyncOpenAI(
        api_key=SECRET,
        base_url="http://llm.test/v1",
        http_client=http,  # type: ignore[arg-type]
        max_retries=0,
    )
    return OpenAIClient(client=sdk, model="gpt-x")


async def _failure(reply: httpx.Response | Exception) -> str:
    route = respx.post(URL)
    if isinstance(reply, Exception):
        route.mock(side_effect=reply)
    else:
        route.mock(return_value=reply)
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError) as caught:
            await _client(http).complete(system="s", messages=MESSAGES)
    return str(caught.value)


@respx.mock
@pytest.mark.parametrize(
    ("reply", "cause", "fix"),
    [
        (
            httpx.Response(401, json={"error": {"message": f"bad key {SECRET}"}}),
            "rejected",
            "LLM_API_KEY",
        ),
        (httpx.Response(404, json={"error": {"message": "no model"}}), "gpt-x", "LLM_MODEL"),
        (
            httpx.Response(429, json={"error": {"message": "slow"}}),
            "rate limit",
            "LLM_MAX_CONCURRENCY",
        ),
        (httpx.Response(503, json={"error": {"message": "down"}}), "server error", "try again"),
        (httpx.ConnectError("refused"), "reach", "LLM_BASE_URL"),
        (httpx.ReadTimeout("slow"), "timed out", "try again"),
    ],
    ids=["auth", "model", "rate", "server", "network", "timeout"],
)
async def test_api_failures_name_the_cause_and_the_fix(
    reply: httpx.Response | Exception, cause: str, fix: str
) -> None:
    message = await _failure(reply)

    assert cause in message.lower() or cause in message
    assert fix.lower() in message.lower()
    assert SECRET not in message


@respx.mock
async def test_bad_request_relays_the_providers_reason_and_hints_at_the_model() -> None:
    message = await _failure(httpx.Response(400, json={"error": {"message": "tools unsupported"}}))

    assert "tools unsupported" in message and "function calling" in message


def _tool_reply(arguments: str) -> httpx.Response:
    call = {"id": "t", "type": "function", "function": {"name": "answer", "arguments": arguments}}
    message = {"role": "assistant", "content": None, "tool_calls": [call]}
    return httpx.Response(200, json={"choices": [{"index": 0, "message": message}]})


@respx.mock
async def test_invalid_tool_arguments_name_the_field_not_the_payload() -> None:
    respx.post(URL).mock(return_value=_tool_reply('{"value": "private text"}'))
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError) as caught:
            await _client(http).call_tool(system="s", messages=MESSAGES, tool=TOOL)

    message = str(caught.value)
    assert "value" in message and "schema" in message and "try again" in message.lower()
    assert "private text" not in message


@respx.mock
async def test_skipped_tool_and_empty_reply_suggest_a_next_step() -> None:
    no_tool = httpx.Response(
        200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}
    )
    respx.post(URL).mock(return_value=no_tool)
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match="function calling"):
            await _client(http).call_tool(system="s", messages=MESSAGES, tool=TOOL)
    assert "try again" in (await _failure(no_tool_empty())).lower()


def no_tool_empty() -> httpx.Response:
    message = {"role": "assistant", "content": ""}
    return httpx.Response(200, json={"choices": [{"index": 0, "message": message}]})


@respx.mock
@pytest.mark.parametrize("status", [429, 500, 404])
async def test_openalex_failure_explains_and_never_leaks_the_mailto(status: int) -> None:
    respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(status))
    async with httpx.AsyncClient() as http:
        with pytest.raises(SourceError) as caught:
            await OpenAlexSource(client=http, mailto="me@example.com").search("x")

    message = str(caught.value)
    assert str(status) in message and "try again" in message.lower()
    assert "me@example.com" not in message


@respx.mock
async def test_openalex_network_failure_is_explained() -> None:
    respx.get("https://api.openalex.org/works").mock(side_effect=httpx.ConnectError("refused"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(SourceError, match="network"):
            await OpenAlexSource(client=http).search("x")


def test_every_report_rejection_says_what_to_correct() -> None:
    import inspect
    import re

    from papertrace.domain import verification

    messages = re.findall(r'InvalidReportError\(\s*"([^"]+)"', inspect.getsource(verification))
    assert len(messages) == 7
    assert all(". " in m for m in messages), messages  # "<cause>. <what to fix>"


async def test_degraded_verification_tells_the_reader_the_claims_are_unchecked() -> None:
    from papertrace.agents.verification import REJECTED_REASON

    assert "unchecked" in REJECTED_REASON and "try again" in REJECTED_REASON.lower()
