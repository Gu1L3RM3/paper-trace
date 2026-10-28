from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # The example env file ships empty values; treat them as unset.
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    # Built-in name or "package.module:function" for a custom provider (see llm/factory.py).
    llm_provider: str = "chat_completions"
    llm_api_key: SecretStr | None = None
    llm_model: str
    llm_base_url: str
    orchestrator_model: str | None = None
    synthesis_model: str | None = None
    llm_max_concurrency: int = 4
    openalex_mailto: str | None = None

    def model_for(self, agent: Literal["orchestrator", "synthesis"]) -> str:
        override = self.orchestrator_model if agent == "orchestrator" else self.synthesis_model
        return override or self.llm_model
