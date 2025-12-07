"""Tests for chat schema models."""

import pytest
from pydantic import ValidationError

from src.schemas.chat import LLMResponse, Message, Role


class TestMessage:
    """Tests for the Message model."""

    def test_message_creation(self):
        """Test creating a valid message."""
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.metadata == {}

    def test_message_with_optional_fields(self):
        """Test message with name and metadata."""
        msg = Message(
            role=Role.ASSISTANT,
            content="Hi there",
            name="assistant_1",
            metadata={"source": "test"},
        )
        assert msg.name == "assistant_1"
        assert msg.metadata["source"] == "test"

    def test_message_is_immutable(self):
        """Test that Message instances are frozen (immutable)."""
        msg = Message(role=Role.USER, content="Hello")
        with pytest.raises(ValidationError):
            msg.content = "Changed"

    def test_message_requires_role_and_content(self):
        """Test that role and content are required."""
        with pytest.raises(ValidationError):
            Message(role=Role.USER)  # Missing content

        with pytest.raises(ValidationError):
            Message(content="Hello")  # Missing role


class TestRole:
    """Tests for the Role enum."""

    def test_role_values(self):
        """Test that Role enum has expected values."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"


class TestLLMResponse:
    """Tests for the LLMResponse model."""

    def test_llm_response_defaults(self):
        """Test LLMResponse with default values."""
        response = LLMResponse(content="Hello")
        assert response.content == "Hello"
        assert response.role == Role.ASSISTANT
        assert response.token_usage == {"input": 0, "output": 0, "total": 0}
        assert response.raw_response is None

    def test_llm_response_with_token_usage(self):
        """Test LLMResponse with custom token usage."""
        response = LLMResponse(
            content="Response",
            token_usage={"input": 10, "output": 20, "total": 30},
        )
        assert response.token_usage["input"] == 10
        assert response.token_usage["total"] == 30

    def test_llm_response_excludes_raw_response_from_dict(self):
        """Test that raw_response is excluded when serializing."""
        response = LLMResponse(
            content="Hello",
            raw_response={"some": "data"},
        )
        response_dict = response.model_dump()
        assert "raw_response" not in response_dict
