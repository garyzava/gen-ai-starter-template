import logging
from typing import AsyncGenerator, Optional, Union, Dict, Any

from openai import APIError, AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.base import BaseLLMClient
from src.llm.config import LLMConfig
from src.schemas.chat import LLMResponse, Message, Role

# Configure logger
logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    Concrete implementation of BaseLLMClient for OpenAI.
    Handles retries, authentication, and response parsing.

    Configuration follows 12-factor principles:
    - API key (secret) comes from environment via settings
    - Behavioral config (temperature, etc.) uses LLMConfig with validation

    Usage:
        # Basic usage with defaults
        client = OpenAIClient()
        response = await client.achat(messages)

        # With custom config
        config = LLMConfig(temperature=0.5, max_tokens=500)
        response = await client.achat(messages, config=config)

        # With advanced features (OpenAI-specific)
        config = LLMConfig(temperature=0.5, seed=12345, frequency_penalty=0.5)
        response = await client.achat(messages, config=config)

        # Override config per-request
        response = await client.achat(messages, config={"temperature": 0.9})
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None
    ):
        super().__init__(api_key=api_key, model=model, default_config=default_config)
        self.client = AsyncOpenAI(api_key=self.api_key)

    @retry(
        retry=retry_if_exception_type(APIError),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def achat(
        self,
        messages: list[Message],
        config: Optional[Union[LLMConfig, Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Non-streaming chat completion with automatic retries.

        Args:
            messages: List of conversation messages
            config: Optional LLMConfig or dict to override defaults

        Returns:
            LLMResponse with content and token usage stats
        """
        resolved_config = self._resolve_config(config)
        formatted_msgs = self._format_messages(messages)
        api_params = resolved_config.to_api_params()

        try:
            logger.debug(f"Sending request to {self.model} with config: {api_params}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_msgs,
                **api_params
            )

            # Extract content
            raw_choice = response.choices[0]
            content = raw_choice.message.content or ""

            # Extract usage stats
            usage = response.usage
            token_stats = {
                "input": usage.prompt_tokens if usage else 0,
                "output": usage.completion_tokens if usage else 0,
                "total": usage.total_tokens if usage else 0
            }

            return LLMResponse(
                content=content,
                role=Role.ASSISTANT,
                token_usage=token_stats,
            )

        except Exception as e:
            logger.error(f"OpenAI Chat Error: {e}")
            raise RuntimeError(f"OpenAI Chat Error: {e}") from e

    async def astream(
        self,
        messages: list[Message],
        config: Optional[Union[LLMConfig, Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat completion.

        Args:
            messages: List of conversation messages
            config: Optional LLMConfig or dict to override defaults

        Yields:
            String chunks of the response
        """
        resolved_config = self._resolve_config(config)
        formatted_msgs = self._format_messages(messages)

        # Force stream=True for streaming
        api_params = resolved_config.to_api_params()
        api_params["stream"] = True

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_msgs,
            **api_params
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
