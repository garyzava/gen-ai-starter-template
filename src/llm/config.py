"""
LLM Configuration with validation.

This module provides a unified configuration dataclass for LLM clients,
combining standard (portable) and advanced (provider-specific) parameters.
"""

import logging
from dataclasses import dataclass, fields, field, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Unified LLM configuration with validation.

    Standard parameters (portable across providers):
        - temperature: Controls randomness (0.0-2.0)
        - max_tokens: Maximum tokens in response
        - top_p: Nucleus sampling parameter (0.0-1.0)
        - stream: Whether to stream the response
        - stop: Stop sequences

    Advanced parameters (provider-specific, use portable_only=True to exclude):
        - seed: For reproducible outputs
        - frequency_penalty: Penalize frequent tokens (-2.0 to 2.0)
        - presence_penalty: Penalize tokens already present (-2.0 to 2.0)
        - logit_bias: Modify likelihood of specific tokens
        - user: End-user identifier for abuse detection
        - response_format: Output format specification

    Usage:
        # Standard config (safe for any provider)
        config = LLMConfig(temperature=0.5)

        # With advanced features (OpenAI-specific)
        config = LLMConfig(temperature=0.5, seed=12345, frequency_penalty=0.5)

        # Get only portable params
        params = config.to_api_params(portable_only=True)
    """

    # --- Standard Parameters (Portable) ---
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = field(default=None)

    # --- Advanced Parameters (Provider-Specific) ---
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = field(default=None)
    user: Optional[str] = None
    response_format: Optional[Dict[str, str]] = field(default=None)

    # Class-level constants for validation
    _STANDARD_FIELDS: frozenset = frozenset({
        'temperature', 'max_tokens', 'top_p', 'stream', 'stop'
    })
    _ADVANCED_FIELDS: frozenset = frozenset({
        'seed', 'frequency_penalty', 'presence_penalty',
        'logit_bias', 'user', 'response_format'
    })

    def __post_init__(self) -> None:
        """Validate parameter bounds after initialization."""
        # Temperature validation
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        # top_p validation
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be between 0.0 and 1.0, got {self.top_p}"
            )

        # max_tokens validation
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be positive, got {self.max_tokens}"
            )

        # frequency_penalty validation
        if self.frequency_penalty is not None:
            if not -2.0 <= self.frequency_penalty <= 2.0:
                raise ValueError(
                    f"frequency_penalty must be between -2.0 and 2.0, "
                    f"got {self.frequency_penalty}"
                )

        # presence_penalty validation
        if self.presence_penalty is not None:
            if not -2.0 <= self.presence_penalty <= 2.0:
                raise ValueError(
                    f"presence_penalty must be between -2.0 and 2.0, "
                    f"got {self.presence_penalty}"
                )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """
        Create LLMConfig from a dictionary, filtering unknown keys with warnings.

        Args:
            data: Dictionary of configuration parameters

        Returns:
            LLMConfig instance with validated parameters
        """
        valid_keys = {f.name for f in fields(cls) if not f.name.startswith('_')}
        unknown_keys = set(data.keys()) - valid_keys

        if unknown_keys:
            logger.warning(
                f"Unknown LLMConfig keys ignored: {unknown_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_api_params(self, portable_only: bool = False) -> Dict[str, Any]:
        """
        Convert config to API parameters dictionary.

        Args:
            portable_only: If True, only include standard parameters that are
                          safe to use with any LLM provider.

        Returns:
            Dictionary of non-None parameters for API calls
        """
        result = {}

        for key, value in asdict(self).items():
            # Skip private fields
            if key.startswith('_'):
                continue

            # Skip None values
            if value is None:
                continue

            # If portable_only, skip advanced fields
            if portable_only and key in self._ADVANCED_FIELDS:
                continue

            result[key] = value

        return result

    def merge_with(self, overrides: Optional[Dict[str, Any]]) -> "LLMConfig":
        """
        Create a new config with overrides applied.

        Args:
            overrides: Dictionary of parameters to override

        Returns:
            New LLMConfig instance with merged parameters
        """
        if not overrides:
            return self

        current = asdict(self)
        # Remove private fields
        current = {k: v for k, v in current.items() if not k.startswith('_')}
        current.update(overrides)
        return self.from_dict(current)
