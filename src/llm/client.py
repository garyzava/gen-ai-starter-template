import logging
from typing import AsyncGenerator

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
    """

    def __init__(self):
        super().__init__()
        # AsyncOpenAI automatically finds OPENAI_API_KEY in env,
        # but passing it explicitly from settings is safer/clearer.
        self.client = AsyncOpenAI(api_key=self.api_key)

    @retry(
        retry=retry_if_exception_type(APIError), # Retry on 500s, 429s
        wait=wait_exponential(multiplier=1, min=4, max=10), # Wait 4s, 8s, 10s...
        stop=stop_after_attempt(3), # Stop after 3 tries
        reraise=True
    )
    async def achat(self, messages: list[Message], temperature: float = None) -> LLMResponse:
        """
        Non-streaming chat completion with automatic retries.
        """
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        formatted_msgs = self._format_messages(messages)

        try:
            logger.debug(f"Sending request to {self.model}...")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_msgs,
                temperature=temp,
                max_tokens=settings.MAX_TOKENS
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
                # Store the raw response if you need deep debugging later
                # raw_response=response
            )

        except Exception as e:
            logger.error(f"OpenAI Chat Error: {e}")
            raise

    async def astream(
        self, messages: list[Message], temperature: float = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat completion.
        Note: Retries on streams are trickier; usually handled at the connection level.
        """
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        formatted_msgs = self._format_messages(messages)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_msgs,
            temperature=temp,
            max_tokens=settings.MAX_TOKENS,
            stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
