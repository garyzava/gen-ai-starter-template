import logging
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the root of the project to help with absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application Settings configuration.

    Following 12-factor app principles:
    - SECRETS (from .env): API keys, credentials - must be kept secure
    - CONFIGURATION (code defaults): Behavioral settings with sensible defaults

    All LLM configuration is consolidated here with full validation.
    """

    # ==========================================================================
    # SECRETS - From environment variables / .env file
    # ==========================================================================

    OPENAI_API_KEY: SecretStr = Field(..., description="Required OpenAI API Key")

    # Add other secrets here as needed:
    # ANTHROPIC_API_KEY: Optional[SecretStr] = None
    # DATABASE_URL: Optional[SecretStr] = None

    # ==========================================================================
    # APPLICATION CONFIG - Defaults defined in code
    # ==========================================================================

    APP_NAME: str = "GenAI Project"
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = False

    # ==========================================================================
    # LLM CONFIG - All LLM settings with validation
    # ==========================================================================

    # Model selection
    LLM_MODEL: str = "gpt-4-turbo"

    # Standard parameters (portable across providers)
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=1000, ge=1)
    LLM_TOP_P: float = Field(default=1.0, ge=0.0, le=1.0)
    LLM_STOP: Optional[List[str]] = None

    # Advanced parameters (OpenAI-specific)
    LLM_SEED: Optional[int] = None
    LLM_FREQUENCY_PENALTY: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    LLM_PRESENCE_PENALTY: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    LLM_USER: Optional[str] = None

    # ==========================================================================
    # VECTOR DATABASE CONFIG
    # ==========================================================================

    VECTOR_DB_PATH: Path = Field(default=PROJECT_ROOT / "data" / "chroma_db")

    # ==========================================================================
    # CONFIG BOILERPLATE
    # ==========================================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra keys in .env that aren't defined here
    )

    @computed_field
    def is_production(self) -> bool:
        """Helper property to check environment."""
        return self.ENVIRONMENT == "production"

    def get_llm_params(
        self,
        portable_only: bool = False,
        **overrides: Any
    ) -> Dict[str, Any]:
        """
        Get LLM parameters as a dictionary for API calls.

        Args:
            portable_only: If True, only include standard parameters safe for
                          any LLM provider (excludes seed, penalties, etc.)
            **overrides: Override any parameter (e.g., temperature=0.5)

        Returns:
            Dictionary of non-None parameters for API calls

        Example:
            params = settings.get_llm_params(temperature=0.5)
            response = await client.chat.completions.create(**params)
        """
        # Standard parameters (portable)
        params: Dict[str, Any] = {
            "temperature": self.LLM_TEMPERATURE,
            "max_tokens": self.LLM_MAX_TOKENS,
            "top_p": self.LLM_TOP_P,
        }

        if self.LLM_STOP is not None:
            params["stop"] = self.LLM_STOP

        # Advanced parameters (OpenAI-specific)
        if not portable_only:
            if self.LLM_SEED is not None:
                params["seed"] = self.LLM_SEED
            if self.LLM_FREQUENCY_PENALTY is not None:
                params["frequency_penalty"] = self.LLM_FREQUENCY_PENALTY
            if self.LLM_PRESENCE_PENALTY is not None:
                params["presence_penalty"] = self.LLM_PRESENCE_PENALTY
            if self.LLM_USER is not None:
                params["user"] = self.LLM_USER

        # Apply overrides (with validation warning for unknown keys)
        if overrides:
            standard_keys = {"temperature", "max_tokens", "top_p", "stop", "stream"}
            advanced_keys = {"seed", "frequency_penalty", "presence_penalty", "user"}
            known_keys = standard_keys | advanced_keys

            unknown = set(overrides.keys()) - known_keys
            if unknown:
                logger.warning(f"Unknown LLM parameters ignored: {unknown}")

            for key, value in overrides.items():
                if key not in known_keys or value is None:
                    continue
                # Skip advanced keys if portable_only
                if portable_only and key in advanced_keys:
                    continue
                params[key] = value

        return params


# Singleton instantiation
# Importing this 'settings' object elsewhere ensures we only validate once.
try:
    settings = Settings()
except Exception as e:
    # Fail fast if configuration is invalid
    print(f"Configuration Error: {e}")
    raise
