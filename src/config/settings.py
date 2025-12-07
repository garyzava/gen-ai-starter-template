from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the root of the project to help with absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """
    Application Settings configuration.

    Following 12-factor app principles:
    - SECRETS (from .env): API keys, credentials - things that vary between deploys
      and must be kept secure
    - CONFIGURATION (code defaults): Behavioral settings with sensible defaults
      that can be overridden programmatically

    Secrets are loaded from environment variables / .env file.
    Configuration defaults are defined in code (LLMConfig) and can be
    overridden at runtime.
    """

    # ===========================================================================
    # SECRETS - Loaded from environment variables / .env file
    # ===========================================================================

    # LLM Provider API Keys
    # SecretStr prevents the value from being displayed in logs/reprs
    OPENAI_API_KEY: SecretStr = Field(..., description="Required OpenAI API Key")

    # Add other secrets here as needed:
    # ANTHROPIC_API_KEY: Optional[SecretStr] = None
    # DATABASE_URL: Optional[SecretStr] = None

    # ===========================================================================
    # APPLICATION CONFIG - Defaults defined in code, not in .env
    # ===========================================================================

    # General App Config
    APP_NAME: str = "GenAI Project"
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = False

    # Default LLM model (can be overridden per-request via LLMConfig)
    DEFAULT_LLM_MODEL: str = "gpt-4-turbo"

    # Vector Database Config
    VECTOR_DB_PATH: Path = Field(default=PROJECT_ROOT / "data" / "chroma_db")

    # ===========================================================================
    # Config Boilerplate
    # ===========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra keys in .env that aren't defined here
    )

    @computed_field
    def is_production(self) -> bool:
        """Helper property to check environment"""
        return self.ENVIRONMENT == "production"

# Singleton instantiation
# Importing this 'settings' object elsewhere ensures we only validate once.
try:
    settings = Settings()
except Exception as e:
    # Fail fast if configuration is invalid
    print(f"❌ Configuration Error: {e}")
    raise
