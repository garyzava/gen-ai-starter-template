from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """Standardized roles for conversation history."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message(BaseModel):
    """
    A single message in the chat history.
    """
    role: Role
    content: str
    # Optional name (useful for multi-agent or tool responses)
    name: Optional[str] = None
    # Flexible dict for storing provider-specific metadata (e.g., token counts)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True) # Makes instances immutable (safer for caching)

class LLMResponse(BaseModel):
    """
    Standardized response from any LLM provider.
    """
    content: str
    role: Role = Role.ASSISTANT
    # Usage stats (useful for cost tracking)
    token_usage: dict[str, int] = Field(
        default_factory=lambda: {"input": 0, "output": 0, "total": 0}
    )
    # Keep original API response for debugging
    raw_response: Any = Field(default=None, exclude=True)
