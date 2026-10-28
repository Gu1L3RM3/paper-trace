"""A tool error goes back to the model once, so it can fix its own call."""

import json

import httpx
import pytest
import respx

from papertrace.domain.errors import LLMError
from tests.unit.test_error_messages import MESSAGES, TOOL, URL, Answer, _client, _tool_reply

BAD = '{"value": "nope"}'
GOOD = '{"value": 7}'


def _text_reply() -> httpx.Response:
    message = {"role": "assistant", "content": "I would rather chat."}
    return httpx.Response(200, json={"choices": [{"index": 0, "message": message}]})


def _sent(route: respx.Route, call: int) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = json.loads(route.calls[call].request.content)["messages"]
    return messages


@respx.mock
async def test_invalid_arguments_are_returned_to_the_model_and_the_fix_is_used() -> None:
    route = respx.post(URL).mock(side_effect=[_tool_reply(BAD), _tool_reply(GOOD)])
    async with httpx.AsyncClient() as http:
        result = await _client(http).call_tool(system="s", messages=MESSAGES, tool=TOOL)

    assert result == Answer(value=7)
    assert route.call_count == 2
    assistant, feedback = _sent(route, 1)[-2:]
    assert assistant["role"] == "assistant" and assistant["tool_calls"]
    assert feedback["role"] == "tool" and feedback["tool_call_id"] == "t"
    assert "value" in str(feedback["content"]) and "answer" in str(feedback["content"])


@respx.mock
async def test_only_one_correction_is_attempted() -> None:
    route = respx.post(URL).mock(side_effect=[_tool_reply(BAD), _tool_reply(BAD)])
    async with httpx.AsyncClient() as http:
        with pytest.raises(LLMError, match="schema"):
            await _client(http).call_tool(system="s", messages=MESSAGES, tool=TOOL)

    assert route.call_count == 2


@respx.mock
async def test_a_skipped_tool_is_requested_again() -> None:
    route = respx.post(URL).mock(side_effect=[_text_reply(), _tool_reply(GOOD)])
    async with httpx.AsyncClient() as http:
        result = await _client(http).call_tool(system="s", messages=MESSAGES, tool=TOOL)

    assert result == Answer(value=7)
    assert _sent(route, 1)[-1]["role"] == "user"
    assert "answer" in str(_sent(route, 1)[-1]["content"])
