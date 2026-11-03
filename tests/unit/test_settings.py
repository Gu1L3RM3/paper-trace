from typing import Any

import pytest

from papertrace.domain.errors import LLMError
from papertrace.llm.chatgpt_plan import ChatGPTPlanClient
from papertrace.llm.openai_client import OpenAIClient, build_llm_client
from papertrace.llm.openai_responses import OpenAIResponsesClient
from papertrace.settings import CONFIG_DIR, Settings


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"llm_api_key": "k", "llm_model": "m", **overrides}
    return Settings(_env_file=None, **values)


def test_nothing_is_required_to_start() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_api_key is None
    assert settings.llm_model == "gpt-5.4-mini"


def test_user_config_file_is_read_before_the_local_env_file() -> None:
    assert Settings.model_config["env_file"] == (CONFIG_DIR / "config.env", ".env")


def test_per_agent_model_overrides_default() -> None:
    settings = _settings(orchestrator_model="small")
    assert settings.model_for("orchestrator") == "small"
    assert settings.model_for("synthesis") == "m"


def test_verification_model_defaults_to_the_effective_synthesis_model() -> None:
    assert _settings(synthesis_model="big").model_for("verification") == "big"


def test_builder_returns_an_openai_client_and_base_url_is_optional() -> None:
    assert isinstance(build_llm_client(_settings()), OpenAIClient)
    assert isinstance(
        build_llm_client(_settings(llm_base_url="http://gw/v1"), model="x"), OpenAIClient
    )


def test_verification_effort_and_api_default_to_unset_chat() -> None:
    settings = _settings()
    assert settings.verification_reasoning_effort is None
    assert settings.verification_api == "chat"


def test_builder_picks_the_adapter_and_passes_the_effort() -> None:
    chat = build_llm_client(_settings(), reasoning_effort="low")
    assert isinstance(chat, OpenAIClient)
    assert chat._reasoning_effort == "low"
    assert isinstance(build_llm_client(_settings(), api="responses"), OpenAIResponsesClient)


def test_builder_uses_the_plan_adapter_when_given_a_chatgpt_token() -> None:
    async def token() -> str:
        return "t"

    client = build_llm_client(_settings(llm_api_key=None), chatgpt_token=token, api="chat")
    assert isinstance(client, ChatGPTPlanClient)


def test_builder_without_any_credential_asks_to_connect() -> None:
    with pytest.raises(LLMError, match="Connect ChatGPT"):
        build_llm_client(_settings(llm_api_key=None))
