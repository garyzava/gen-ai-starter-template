"""Tests for the LLMConfig dataclass."""

import logging
import pytest

from src.llm.config import LLMConfig


class TestLLMConfigValidation:
    """Tests for LLMConfig validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = LLMConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 1.0
        assert config.stream is False
        assert config.stop is None
        assert config.seed is None
        assert config.frequency_penalty is None
        assert config.presence_penalty is None

    def test_temperature_validation_valid(self):
        """Test valid temperature values."""
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = LLMConfig(temperature=2.0)
        assert config.temperature == 2.0

        config = LLMConfig(temperature=1.0)
        assert config.temperature == 1.0

    def test_temperature_validation_invalid(self):
        """Test invalid temperature values raise errors."""
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be between"):
            LLMConfig(temperature=2.1)

    def test_top_p_validation_valid(self):
        """Test valid top_p values."""
        config = LLMConfig(top_p=0.0)
        assert config.top_p == 0.0

        config = LLMConfig(top_p=1.0)
        assert config.top_p == 1.0

    def test_top_p_validation_invalid(self):
        """Test invalid top_p values raise errors."""
        with pytest.raises(ValueError, match="top_p must be between"):
            LLMConfig(top_p=-0.1)

        with pytest.raises(ValueError, match="top_p must be between"):
            LLMConfig(top_p=1.1)

    def test_max_tokens_validation_valid(self):
        """Test valid max_tokens values."""
        config = LLMConfig(max_tokens=1)
        assert config.max_tokens == 1

        config = LLMConfig(max_tokens=10000)
        assert config.max_tokens == 10000

    def test_max_tokens_validation_invalid(self):
        """Test invalid max_tokens values raise errors."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            LLMConfig(max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            LLMConfig(max_tokens=-1)

    def test_frequency_penalty_validation_valid(self):
        """Test valid frequency_penalty values."""
        config = LLMConfig(frequency_penalty=-2.0)
        assert config.frequency_penalty == -2.0

        config = LLMConfig(frequency_penalty=2.0)
        assert config.frequency_penalty == 2.0

        config = LLMConfig(frequency_penalty=None)
        assert config.frequency_penalty is None

    def test_frequency_penalty_validation_invalid(self):
        """Test invalid frequency_penalty values raise errors."""
        with pytest.raises(ValueError, match="frequency_penalty must be between"):
            LLMConfig(frequency_penalty=-2.1)

        with pytest.raises(ValueError, match="frequency_penalty must be between"):
            LLMConfig(frequency_penalty=2.1)

    def test_presence_penalty_validation_valid(self):
        """Test valid presence_penalty values."""
        config = LLMConfig(presence_penalty=-2.0)
        assert config.presence_penalty == -2.0

        config = LLMConfig(presence_penalty=2.0)
        assert config.presence_penalty == 2.0

    def test_presence_penalty_validation_invalid(self):
        """Test invalid presence_penalty values raise errors."""
        with pytest.raises(ValueError, match="presence_penalty must be between"):
            LLMConfig(presence_penalty=-2.1)

        with pytest.raises(ValueError, match="presence_penalty must be between"):
            LLMConfig(presence_penalty=2.1)


class TestLLMConfigFromDict:
    """Tests for LLMConfig.from_dict() method."""

    def test_from_dict_valid_keys(self):
        """Test creating config from dict with valid keys."""
        data = {"temperature": 0.5, "max_tokens": 500}
        config = LLMConfig.from_dict(data)
        assert config.temperature == 0.5
        assert config.max_tokens == 500

    def test_from_dict_unknown_keys_warning(self, caplog):
        """Test that unknown keys produce a warning."""
        data = {"temperature": 0.5, "unknown_param": 123, "typo_key": "value"}

        with caplog.at_level(logging.WARNING):
            config = LLMConfig.from_dict(data)

        assert config.temperature == 0.5
        assert "Unknown LLMConfig keys ignored" in caplog.text
        assert "unknown_param" in caplog.text
        assert "typo_key" in caplog.text

    def test_from_dict_empty(self):
        """Test creating config from empty dict uses defaults."""
        config = LLMConfig.from_dict({})
        assert config.temperature == 0.7
        assert config.max_tokens == 1000


class TestLLMConfigToApiParams:
    """Tests for LLMConfig.to_api_params() method."""

    def test_to_api_params_excludes_none(self):
        """Test that None values are excluded from API params."""
        config = LLMConfig(temperature=0.5)
        params = config.to_api_params()

        assert "temperature" in params
        assert "seed" not in params
        assert "frequency_penalty" not in params

    def test_to_api_params_includes_all_set_values(self):
        """Test that all set values are included."""
        config = LLMConfig(
            temperature=0.5,
            max_tokens=500,
            seed=12345,
            frequency_penalty=0.5
        )
        params = config.to_api_params()

        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 500
        assert params["seed"] == 12345
        assert params["frequency_penalty"] == 0.5

    def test_to_api_params_portable_only(self):
        """Test portable_only excludes advanced parameters."""
        config = LLMConfig(
            temperature=0.5,
            max_tokens=500,
            seed=12345,
            frequency_penalty=0.5
        )
        params = config.to_api_params(portable_only=True)

        # Standard params should be included
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 500

        # Advanced params should be excluded
        assert "seed" not in params
        assert "frequency_penalty" not in params


class TestLLMConfigMerge:
    """Tests for LLMConfig.merge_with() method."""

    def test_merge_with_overrides(self):
        """Test merging with override values."""
        config = LLMConfig(temperature=0.5, max_tokens=500)
        merged = config.merge_with({"temperature": 0.9, "seed": 12345})

        assert merged.temperature == 0.9
        assert merged.max_tokens == 500  # Not overridden
        assert merged.seed == 12345

    def test_merge_with_none(self):
        """Test merging with None returns same config values."""
        config = LLMConfig(temperature=0.5)
        merged = config.merge_with(None)

        assert merged.temperature == 0.5

    def test_merge_with_empty_dict(self):
        """Test merging with empty dict returns same config values."""
        config = LLMConfig(temperature=0.5)
        merged = config.merge_with({})

        assert merged.temperature == 0.5
