import pytest
from pydantic import BaseModel

from papertrace.domain.errors import LLMError
from papertrace.llm.anthropic_client import AnthropicClient
from papertrace.llm.factory import build_llm_client, register_llm_provider
from papertrace.llm.openai_compatible import OpenAICompatibleClient
from papertrace.settings import Settings
from tests.fakes import FakeLLMClient


class Answer(BaseModel):
    pass


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, llm_api_key="k", llm_model="m", **overrides)  # type: ignore[arg-type]


def test_anthropic_provider_builds_anthropic_client() -> None:
    assert isinstance(build_llm_client(_settings(llm_provider="anthropic")), AnthropicClient)


def test_openai_compatible_provider_builds_openai_client() -> None:
    settings = _settings(llm_provider="openai_compatible", llm_base_url="http://localhost:11434/v1")
    assert isinstance(build_llm_client(settings), OpenAICompatibleClient)


def test_empty_env_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "")
    settings = Settings(_env_file=None, llm_api_key="k", llm_model="m")
    assert settings.llm_base_url is None
    assert settings.model_for("orchestrator") == "m"


def test_per_agent_model_overrides_default() -> None:
    settings = _settings(orchestrator_model="small")
    assert settings.model_for("orchestrator") == "small"
    assert settings.model_for("synthesis") == "m"


async def test_provider_can_be_a_dotted_path_outside_the_package() -> None:
    settings = _settings(llm_provider="tests.fakes:build_fake_llm")
    client = build_llm_client(settings, model="custom")
    assert await client.complete(system="s", messages=[]) == "model=custom"


def test_registered_provider_is_selectable_by_name() -> None:
    register_llm_provider("mine", lambda settings, model, semaphore: FakeLLMClient(Answer(), "x"))
    assert isinstance(build_llm_client(_settings(llm_provider="mine")), FakeLLMClient)


def test_unknown_provider_lists_the_available_ones() -> None:
    with pytest.raises(LLMError, match="anthropic"):
        build_llm_client(_settings(llm_provider="nope"))


def test_api_key_is_optional_for_keyless_local_endpoints() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compatible",
        llm_model="llama3",
        llm_base_url="http://localhost:11434/v1",
    )
    assert isinstance(build_llm_client(settings), OpenAICompatibleClient)


def test_dotted_path_to_a_non_callable_is_an_llm_error() -> None:
    with pytest.raises(LLMError, match="callable"):
        build_llm_client(_settings(llm_provider="os:environ"))


def test_dotted_path_to_a_missing_module_is_an_llm_error() -> None:
    with pytest.raises(LLMError):
        build_llm_client(_settings(llm_provider="no_such_pkg_xyz:build"))


def test_anthropic_provider_honors_base_url_for_gateways() -> None:
    settings = _settings(llm_provider="anthropic", llm_base_url="http://gateway.test")
    client = build_llm_client(settings)
    assert str(client._client.base_url).startswith("http://gateway.test")  # type: ignore[attr-defined]


def test_anthropic_without_key_is_an_llm_error() -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", llm_model="m")
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        build_llm_client(settings)
