"""`LLMClient` that spends the signed-in user's ChatGPT plan through the Responses API.

The route requires `store=false`, `stream=true` and namespaced tools:
https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
"""

from typing import Any

from openai import APIStatusError
from openai.types.responses import Response

from papertrace.domain.errors import LLMError
from papertrace.llm.base import ToolSpec
from papertrace.llm.openai_responses import OpenAIResponsesClient

_USAGE = (
    "Your ChatGPT plan's usage limit is reached. Check ChatGPT Settings → Usage, then try "
    "again later."
)
_UNAVAILABLE = "ChatGPT plan usage is temporarily unavailable. Try again in a few minutes."
_RECONNECT = "ChatGPT did not accept this session. Disconnect, then connect ChatGPT again."
_ERRORS = {
    "subscription_sharing_usage_limit_exceeded": _USAGE,
    "subscription_sharing_usage_unavailable": _UNAVAILABLE,
    "subscription_sharing_user_unavailable": _UNAVAILABLE,
    "subscription_sharing_invalid_user": _RECONNECT,
    "subscription_sharing_user_not_eligible": (
        "This ChatGPT account's plan cannot be used by other apps. Disconnect ChatGPT or set "
        "LLM_API_KEY."
    ),
}

# Rejections without a plan error code; the generic texts would blame LLM_API_KEY.
_STATUS = {401: _RECONNECT, 403: "ChatGPT refused plan usage for this account or region."}


class ChatGPTPlanClient(OpenAIResponsesClient):
    def _tools(self, tool: ToolSpec[Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "namespace",
                "name": "papertrace",
                "description": "PaperTrace research tools.",
                "tools": super()._tools(tool),
            }
        ]

    async def _send(self, **request: Any) -> Response:
        try:
            stream = await self._client.responses.create(**request, store=False, stream=True)
            async with stream:
                async for event in stream:
                    if event.type == "response.completed":
                        completed: Response = event.response
                        return completed
                    if event.type in ("response.failed", "response.incomplete"):
                        raise LLMError(_explain_failure(event.response))
        except APIStatusError as exc:
            message = _ERRORS.get(str(exc.code)) or _STATUS.get(exc.status_code)
            if message is None:
                raise  # the shared Responses handler explains generic API errors
            raise LLMError(message) from exc
        raise LLMError("The ChatGPT stream ended before the answer was complete. Try again.")


def _explain_failure(response: Response) -> str:
    if response.status == "incomplete":
        return "The ChatGPT answer was cut off before it finished. Try again or rephrase."
    code = response.error.code if response.error else "unknown_error"
    return _ERRORS.get(code, f"ChatGPT could not answer ({code}). Try again.")
