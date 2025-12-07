"""Tests for the Settings configuration."""

import logging
import pytest

from src.config.settings import settings


class TestSettingsGetLlmParams:
    """Tests for Settings.get_llm_params() method."""

    def test_get_llm_params_returns_defaults(self):
        """Test that get_llm_params returns default values."""
        params = settings.get_llm_params()

        assert "temperature" in params
        assert "max_tokens" in params
        assert "top_p" in params
        assert params["temperature"] == settings.LLM_TEMPERATURE
        assert params["max_tokens"] == settings.LLM_MAX_TOKENS

    def test_get_llm_params_with_overrides(self):
        """Test that overrides are applied."""
        params = settings.get_llm_params(temperature=0.5, max_tokens=500)

        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 500

    def test_get_llm_params_portable_only(self):
        """Test portable_only excludes advanced parameters."""
        # First, get all params
        all_params = settings.get_llm_params(seed=12345)

        # With portable_only, seed should be excluded
        portable_params = settings.get_llm_params(portable_only=True, seed=12345)

        # Standard params should be present in both
        assert "temperature" in portable_params
        assert "max_tokens" in portable_params

        # Advanced params should only be in all_params (if set)
        # Note: seed is passed as override, so it should be in all_params
        assert "seed" in all_params
        assert "seed" not in portable_params

    def test_get_llm_params_warns_on_unknown_keys(self, caplog):
        """Test that unknown keys produce a warning."""
        with caplog.at_level(logging.WARNING):
            params = settings.get_llm_params(unknown_param=123, typo_key="value")

        assert "Unknown LLM parameters ignored" in caplog.text
        assert "unknown_param" in caplog.text
        # Unknown keys should not be in the result
        assert "unknown_param" not in params

    def test_get_llm_params_stream_override(self):
        """Test that stream can be set via override."""
        params = settings.get_llm_params(stream=True)
        assert params["stream"] is True


class TestSettingsValidation:
    """Tests for Settings validation (via pydantic)."""

    def test_settings_has_required_fields(self):
        """Test that settings has all required fields."""
        assert hasattr(settings, "OPENAI_API_KEY")
        assert hasattr(settings, "LLM_MODEL")
        assert hasattr(settings, "LLM_TEMPERATURE")
        assert hasattr(settings, "LLM_MAX_TOKENS")

    def test_settings_temperature_in_range(self):
        """Test that temperature is within valid range."""
        assert 0.0 <= settings.LLM_TEMPERATURE <= 2.0

    def test_settings_max_tokens_positive(self):
        """Test that max_tokens is positive."""
        assert settings.LLM_MAX_TOKENS >= 1

    def test_settings_top_p_in_range(self):
        """Test that top_p is within valid range."""
        assert 0.0 <= settings.LLM_TOP_P <= 1.0
