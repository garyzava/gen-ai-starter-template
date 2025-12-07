from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Any

from src.config.settings import settings
from src.schemas.chat import LLMResponse, Message


class BaseLLMClient(ABC):
    """
    Abstract Base Class for all LLM providers (OpenAI, Anthropic, etc.).
    Enforces a standard interface so the rest of the app doesn't care which model is used.

    Configuration is centralized in settings with validation.
    Use **overrides for per-request parameter changes.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Allow overrides, but default to settings
        self.api_key = api_key or settings.OPENAI_API_KEY.get_secret_value()
        self.model = model or settings.LLM_MODEL

    @abstractmethod
    async def achat(
        self,
        messages: list[Message],
        **overrides: Any
    ) -> LLMResponse:
        """
        Send a chat request to the LLM and get a complete response.

        Args:
            messages: List of conversation messages
            **overrides: Override any LLM parameter (e.g., temperature=0.5)

        Returns:
            LLMResponse with content and metadata
        """
        pass

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        **overrides: Any
    ) -> AsyncGenerator[str, None]:
        """
        Stream the response chunk by chunk (critical for good UX).

        Args:
            messages: List of conversation messages
            **overrides: Override any LLM parameter (e.g., temperature=0.5)

        Yields:
            String chunks of the response
        """
        pass

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """
        Helper to convert Pydantic models to standard dicts usually expected by APIs.
        Can be overridden if a provider needs a weird format.
        """
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
