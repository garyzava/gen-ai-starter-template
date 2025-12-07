from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Union, Dict, Any

from src.config.settings import settings
from src.llm.config import LLMConfig
from src.schemas.chat import LLMResponse, Message


class BaseLLMClient(ABC):
    """
    Abstract Base Class for all LLM providers (OpenAI, Anthropic, etc.).
    Enforces a standard interface so the rest of the app doesn't care which model is used.

    Configuration follows 12-factor principles:
    - Secrets (API keys) come from environment via settings
    - Behavioral config (temperature, etc.) comes from LLMConfig with code defaults
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: Override API key (defaults to settings.OPENAI_API_KEY)
            model: Override model name (defaults to settings.DEFAULT_LLM_MODEL)
            default_config: Default LLMConfig for all requests (can be overridden per-call)
        """
        # Secrets from settings
        self.api_key = api_key or settings.OPENAI_API_KEY.get_secret_value()
        self.model = model or settings.DEFAULT_LLM_MODEL

        # Default configuration (behavioral settings)
        if default_config is None:
            self._default_config = LLMConfig()
        elif isinstance(default_config, dict):
            self._default_config = LLMConfig.from_dict(default_config)
        else:
            self._default_config = default_config

    @property
    def default_config(self) -> LLMConfig:
        """Get the default configuration for this client."""
        return self._default_config

    @abstractmethod
    async def achat(
        self,
        messages: list[Message],
        config: Optional[Union[LLMConfig, Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Send a chat request to the LLM and get a complete response.

        Args:
            messages: List of conversation messages
            config: Optional config override for this request

        Returns:
            LLMResponse with content and metadata
        """
        pass

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        config: Optional[Union[LLMConfig, Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream the response chunk by chunk (critical for good UX).

        Args:
            messages: List of conversation messages
            config: Optional config override for this request

        Yields:
            String chunks of the response
        """
        pass

    def _resolve_config(
        self,
        config: Optional[Union[LLMConfig, Dict[str, Any]]] = None
    ) -> LLMConfig:
        """
        Resolve the configuration to use for a request.

        Priority: per-request config > client default config > LLMConfig defaults

        Args:
            config: Optional per-request configuration override

        Returns:
            Resolved LLMConfig instance
        """
        if config is None:
            return self._default_config

        if isinstance(config, dict):
            # Merge with defaults
            return self._default_config.merge_with(config)

        return config

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """
        Helper to convert Pydantic models to standard dicts usually expected by APIs.
        Can be overridden if a provider needs a weird format.
        """
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
