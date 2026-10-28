import pytest
from pydantic import BaseModel, ValidationError

from papertrace.domain.errors import LLMError
from papertrace.llm.factory import build_llm_client, register_llm_provider
from papertrace.llm.http_client import HttpLLMClient
from papertrace.settings import Settings
from tests.fakes import FakeLLMClient


class Answer(BaseModel):
    pass


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_model": "m",
        "llm_base_url": "http://localhost:11434/v1",
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_default_provider_is_the_generic_http_client_and_needs_no_key() -> None:
    assert isinstance(build_llm_client(_settings()), HttpLLMClient)


def test_base_url_and_model_are_required() -> None:
    with pytest.raises(ValidationError, match="llm_base_url"):
        Settings(_env_file=None, llm_model="m")
    with pytest.raises(ValidationError, match="llm_model"):
        Settings(_env_file=None, llm_base_url="http://x/v1")


def test_empty_env_values_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "")
    settings = _settings()
    assert settings.llm_api_key is None
    assert settings.model_for("orchestrator") == "m"


def test_per_agent_model_overrides_default() -> None:
    settings = _settings(orchestrator_model="small")
    assert settings.model_for("orchestrator") == "small"
    assert settings.model_for("synthesis") == "m"


async def test_provider_can_be_a_dotted_path_outside_the_package() -> None:
    client = build_llm_client(_settings(llm_provider="tests.fakes:build_fake_llm"), model="custom")
    assert await client.complete(system="s", messages=[]) == "model=custom"


def test_registered_provider_is_selectable_by_name() -> None:
    register_llm_provider("mine", lambda settings, model, semaphore: FakeLLMClient(Answer(), "x"))
    assert isinstance(build_llm_client(_settings(llm_provider="mine")), FakeLLMClient)


def test_unknown_provider_lists_the_available_ones() -> None:
    with pytest.raises(LLMError, match="chat_completions"):
        build_llm_client(_settings(llm_provider="nope"))


def test_dotted_path_to_a_non_callable_is_an_llm_error() -> None:
    with pytest.raises(LLMError, match="callable"):
        build_llm_client(_settings(llm_provider="os:environ"))


def test_dotted_path_to_a_missing_module_is_an_llm_error() -> None:
    with pytest.raises(LLMError):
        build_llm_client(_settings(llm_provider="no_such_pkg_xyz:build"))
