import pytest

from papertrace.llm.anthropic_client import AnthropicClient
from papertrace.llm.factory import build_llm_client
from papertrace.llm.openai_compatible import OpenAICompatibleClient
from papertrace.settings import Settings


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
