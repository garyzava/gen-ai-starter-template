"""
Pytest fixtures for the GenAI project test suite.

Fixtures defined here are automatically available to all test files
without needing to import them.

The conftest.py file is a special pytest configuration file that serves
several purposes:
- Shared fixtures: Define fixtures automatically used by all test files
  in the same directory and subdirectories
- Plugins and hooks: Register custom pytest plugins or hooks
- Configuration: Set up custom markers, command-line options, etc.
- Automatic discovery: pytest auto-discovers conftest.py files
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.schemas.chat import LLMResponse, Message, Role

# ---------------------------------------------------------------------------
# Environment Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """Set environment to testing mode for the entire test session."""
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["OPENAI_API_KEY"] = "test-api-key-not-real"
    yield
    # Cleanup after all tests
    os.environ.pop("ENVIRONMENT", None)
    os.environ.pop("OPENAI_API_KEY", None)


# ---------------------------------------------------------------------------
# Schema Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_user_message() -> Message:
    """A sample user message for testing."""
    return Message(role=Role.USER, content="Hello, how are you?")


@pytest.fixture
def sample_system_message() -> Message:
    """A sample system message for testing."""
    return Message(
        role=Role.SYSTEM,
        content="You are a helpful assistant."
    )


@pytest.fixture
def sample_assistant_message() -> Message:
    """A sample assistant message for testing."""
    return Message(role=Role.ASSISTANT, content="I'm doing well, thank you!")


@pytest.fixture
def sample_conversation(
    sample_system_message,
    sample_user_message,
    sample_assistant_message
) -> list[Message]:
    """A sample multi-turn conversation."""
    return [sample_system_message, sample_user_message, sample_assistant_message]


@pytest.fixture
def sample_llm_response() -> LLMResponse:
    """A sample LLM response for testing."""
    return LLMResponse(
        content="This is a test response from the LLM.",
        role=Role.ASSISTANT,
        token_usage={"input": 10, "output": 20, "total": 30}
    )


# ---------------------------------------------------------------------------
# Mock Client Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response structure."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Mocked response content"),
            finish_reason="stop"
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    return mock_response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """
    Mock AsyncOpenAI client to avoid real API calls in tests.

    Usage:
        async def test_something(mock_openai_client):
            with patch('src.llm.client.AsyncOpenAI', return_value=mock_openai_client):
                # Your test code here
    """
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    return client


@pytest.fixture
def mock_streaming_openai_client():
    """
    Mock AsyncOpenAI client for streaming responses.

    Usage:
        async def test_streaming(mock_streaming_openai_client):
            with patch('src.llm.client.AsyncOpenAI', return_value=mock_streaming_openai_client):
                # Your test code here
    """
    async def mock_stream():
        chunks = ["Hello", " ", "world", "!"]
        for chunk_text in chunks:
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content=chunk_text))]
            yield chunk

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=mock_stream())
    return client


# ---------------------------------------------------------------------------
# Settings Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(tmp_path):
    """
    Mock settings object for testing main module.

    Uses real settings values but overrides VECTOR_DB_PATH to use tmp_path.

    Usage:
        def test_something(mock_settings):
            with patch('src.main.settings', mock_settings):
                # Your test code here
    """
    from src.config.settings import settings as real_settings

    mock = MagicMock()
    # Copy real values from settings
    mock.APP_NAME = real_settings.APP_NAME
    mock.ENVIRONMENT = real_settings.ENVIRONMENT
    mock.LLM_MODEL = real_settings.LLM_MODEL
    mock.LLM_TEMPERATURE = real_settings.LLM_TEMPERATURE
    mock.OPENAI_API_KEY = real_settings.OPENAI_API_KEY
    # Override path to use temp directory for tests
    mock.VECTOR_DB_PATH = tmp_path / "test_db"
    return mock
