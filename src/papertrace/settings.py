from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class Settings(BaseSettings):
    # The example env file ships empty values; treat them as unset.
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    llm_api_key: SecretStr
    llm_model: str
    # Only for an OpenAI-compatible gateway or proxy; None means the OpenAI API.
    llm_base_url: str | None = None
    orchestrator_model: str | None = None
    synthesis_model: str | None = None
    verification_model: str | None = None
    llm_max_concurrency: int = 4
    # Some models (e.g. gpt-6-luna) accept function tools on chat completions only with "none".
    llm_reasoning_effort: ReasoningEffort | None = None
    openalex_mailto: str | None = None
    semantic_scholar_api_key: SecretStr | None = None

    def model_for(self, agent: Literal["orchestrator", "synthesis", "verification"]) -> str:
        if agent == "verification":
            return self.verification_model or self.model_for("synthesis")
        override = self.orchestrator_model if agent == "orchestrator" else self.synthesis_model
        return override or self.llm_model
