import logging
from typing import AsyncGenerator, Any

from openai import APIError, AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.llm.base import BaseLLMClient
from src.schemas.chat import LLMResponse, Message, Role

# Configure logger
logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    Concrete implementation of BaseLLMClient for OpenAI.
    Handles retries, authentication, and response parsing.

    All configuration comes from settings with per-request overrides via **kwargs.

    Usage:
        client = OpenAIClient()

        # Use settings defaults
        response = await client.achat(messages)

        # Override specific params
        response = await client.achat(messages, temperature=0.5, max_tokens=500)

        # Stream with overrides
        async for chunk in client.astream(messages, temperature=0.9):
            print(chunk, end="")
    """

    def __init__(self):
        super().__init__()
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
        **overrides: Any
    ) -> LLMResponse:
        """
        Non-streaming chat completion with automatic retries.

        Args:
            messages: List of conversation messages
            **overrides: Override any LLM parameter (e.g., temperature=0.5)

        Returns:
            LLMResponse with content and token usage stats
        """
        formatted_msgs = self._format_messages(messages)
        params = settings.get_llm_params(**overrides)

        try:
            logger.debug(f"Sending request to {self.model} with params: {params}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_msgs,
                **params
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
        **overrides: Any
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat completion.

        Args:
            messages: List of conversation messages
            **overrides: Override any LLM parameter (e.g., temperature=0.5)

        Yields:
            String chunks of the response
        """
        formatted_msgs = self._format_messages(messages)
        params = settings.get_llm_params(stream=True, **overrides)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_msgs,
            **params
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
