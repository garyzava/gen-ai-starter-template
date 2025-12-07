"""Tests for the OpenAI client using mocked responses."""

from unittest.mock import AsyncMock, patch

import pytest

from src.llm.client import OpenAIClient
from src.llm.config import LLMConfig
from src.schemas.chat import Role


class TestOpenAIClient:
    """Tests for OpenAIClient with mocked API calls."""

    @pytest.mark.asyncio
    async def test_achat_returns_response(
        self, mock_openai_response, sample_system_message, sample_user_message
    ):
        """Test that achat returns a properly formatted LLMResponse."""
        messages = [sample_system_message, sample_user_message]

        with patch("src.llm.client.AsyncOpenAI") as mock_class:
            # Setup mock - use AsyncMock for async method
            mock_client = mock_class.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )

            # Create client and call
            client = OpenAIClient()
            response = await client.achat(messages)

            # Verify response
            assert response.content == "Mocked response content"
            assert response.role == Role.ASSISTANT
            assert response.token_usage["input"] == 10
            assert response.token_usage["output"] == 20
            assert response.token_usage["total"] == 30

    @pytest.mark.asyncio
    async def test_achat_formats_messages_correctly(
        self, mock_openai_response, sample_system_message, sample_user_message
    ):
        """Test that messages are formatted correctly for the API."""
        messages = [sample_system_message, sample_user_message]

        with patch("src.llm.client.AsyncOpenAI") as mock_class:
            mock_client = mock_class.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )

            client = OpenAIClient()
            await client.achat(messages)

            # Check how the API was called
            call_args = mock_client.chat.completions.create.call_args
            messages_sent = call_args.kwargs["messages"]

            assert messages_sent[0]["role"] == "system"
            assert messages_sent[0]["content"] == sample_system_message.content
            assert messages_sent[1]["role"] == "user"
            assert messages_sent[1]["content"] == sample_user_message.content

    @pytest.mark.asyncio
    async def test_achat_uses_config_params(
        self, mock_openai_response, sample_system_message, sample_user_message
    ):
        """Test that config parameters are passed to API."""
        messages = [sample_system_message, sample_user_message]
        config = LLMConfig(temperature=0.5, max_tokens=500)

        with patch("src.llm.client.AsyncOpenAI") as mock_class:
            mock_client = mock_class.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )

            client = OpenAIClient()
            await client.achat(messages, config=config)

            # Check config was passed
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["temperature"] == 0.5
            assert call_args.kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_achat_uses_dict_config(
        self, mock_openai_response, sample_system_message, sample_user_message
    ):
        """Test that dict config overrides work."""
        messages = [sample_system_message, sample_user_message]

        with patch("src.llm.client.AsyncOpenAI") as mock_class:
            mock_client = mock_class.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )

            client = OpenAIClient()
            await client.achat(messages, config={"temperature": 0.9})

            # Check config was passed
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_achat_with_advanced_config(
        self, mock_openai_response, sample_system_message, sample_user_message
    ):
        """Test that advanced config parameters are passed."""
        messages = [sample_system_message, sample_user_message]
        config = LLMConfig(temperature=0.5, seed=12345, frequency_penalty=0.5)

        with patch("src.llm.client.AsyncOpenAI") as mock_class:
            mock_client = mock_class.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )

            client = OpenAIClient()
            await client.achat(messages, config=config)

            # Check advanced params were passed
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["seed"] == 12345
            assert call_args.kwargs["frequency_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_astream_yields_chunks(
        self, sample_system_message, sample_user_message
    ):
        """Test that astream yields response chunks correctly."""
        messages = [sample_system_message, sample_user_message]

        async def mock_stream():
            for chunk_text in ["Hello", " ", "world", "!"]:
                chunk = AsyncMock()
                chunk.choices = [AsyncMock(delta=AsyncMock(content=chunk_text))]
                yield chunk

        with patch("src.llm.client.AsyncOpenAI") as mock_class:
            mock_client = mock_class.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_stream()
            )

            client = OpenAIClient()
            chunks = [chunk async for chunk in client.astream(messages)]

            assert chunks == ["Hello", " ", "world", "!"]
            assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_astream_skips_empty_chunks(
        self, sample_system_message, sample_user_message
    ):
        """Test that astream skips chunks with None or empty content."""
        messages = [sample_system_message, sample_user_message]

        async def mock_stream():
            contents = ["Hello", None, "", "world"]
            for content in contents:
                chunk = AsyncMock()
                chunk.choices = [AsyncMock(delta=AsyncMock(content=content))]
                yield chunk

        with patch("src.llm.client.AsyncOpenAI") as mock_class:
            mock_client = mock_class.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_stream()
            )

            client = OpenAIClient()
            chunks = [chunk async for chunk in client.astream(messages)]

            # Should only have non-empty chunks
            assert chunks == ["Hello", "world"]

    @pytest.mark.asyncio
    async def test_astream_forces_stream_true(
        self, sample_system_message, sample_user_message
    ):
        """Test that astream always sets stream=True regardless of config."""
        messages = [sample_system_message, sample_user_message]
        config = LLMConfig(stream=False)  # Explicitly set to False

        async def mock_stream():
            yield AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="Hi"))])

        with patch("src.llm.client.AsyncOpenAI") as mock_class:
            mock_client = mock_class.return_value
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_stream()
            )

            client = OpenAIClient()
            _ = [chunk async for chunk in client.astream(messages, config=config)]

            # Check stream was forced to True
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["stream"] is True


class TestOpenAIClientInit:
    """Tests for OpenAIClient initialization."""

    def test_init_with_default_config(self):
        """Test client initializes with default config."""
        with patch("src.llm.client.AsyncOpenAI"):
            client = OpenAIClient()
            assert client.default_config.temperature == 0.7
            assert client.default_config.max_tokens == 1000

    def test_init_with_custom_config(self):
        """Test client initializes with custom config."""
        custom_config = LLMConfig(temperature=0.5, max_tokens=500)

        with patch("src.llm.client.AsyncOpenAI"):
            client = OpenAIClient(default_config=custom_config)
            assert client.default_config.temperature == 0.5
            assert client.default_config.max_tokens == 500

    def test_init_with_dict_config(self):
        """Test client initializes with dict config."""
        with patch("src.llm.client.AsyncOpenAI"):
            client = OpenAIClient(default_config={"temperature": 0.3})
            assert client.default_config.temperature == 0.3
