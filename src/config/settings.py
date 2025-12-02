import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the root of the project to help with absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    """
    Application Settings configuration.
    Validates environment variables on startup.
    """
    
    # --- General App Config ---
    APP_NAME: str = "GenAI Project"
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = False
    
    # --- LLM Configuration ---
    # SecretStr prevents the value from being displayed in logs/reprs
    OPENAI_API_KEY: SecretStr = Field(..., description="Required OpenAI API Key")
    
    # Validation: Temperature must be between 0.0 and 2.0
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    LLM_MODEL: str = "gpt-4-turbo"
    MAX_TOKENS: int = 1000

    # --- Vector Database Config ---
    VECTOR_DB_PATH: Path = Field(default=PROJECT_ROOT / "data" / "chroma_db")

    # --- Config Boilerplate ---
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