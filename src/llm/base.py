from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from src.schemas.chat import Message, LLMResponse
from src.config.settings import settings

class BaseLLMClient(ABC):
    """
    Abstract Base Class for all LLM providers (OpenAI, Anthropic, etc.).
    Enforces a standard interface so the rest of the app doesn't care which model is used.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Allow overrides, but default to settings
        self.api_key = api_key or settings.OPENAI_API_KEY.get_secret_value()
        self.model = model or settings.LLM_MODEL
    
    @abstractmethod
    async def achat(self, messages: list[Message], temperature: float = 0.7) -> LLMResponse:
        """
        Send a chat request to the LLM and get a complete response.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def astream(self, messages: list[Message], temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """
        Stream the response chunk by chunk (critical for good UX).
        Must be implemented by subclasses.
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