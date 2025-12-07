"""Tests for the main module."""

from unittest.mock import patch

from src.config.settings import settings
from src.main import main


class TestMain:
    """Tests for the main function."""

    def test_main_prints_startup_info(self, capsys):
        """Test that main prints app name, model, and temperature."""
        main()

        captured = capsys.readouterr()
        assert settings.APP_NAME in captured.out
        assert settings.LLM_MODEL in captured.out
        assert "Temp:" in captured.out

    def test_main_creates_vector_db_directory(self, mock_settings):
        """Test that main creates the vector DB directory if it doesn't exist."""
        with patch("src.main.settings", mock_settings):
            main()
            assert mock_settings.VECTOR_DB_PATH.exists()

    def test_main_skips_directory_creation_if_exists(self, mock_settings, capsys):
        """Test that main doesn't recreate the directory if it exists."""
        # Pre-create the directory
        mock_settings.VECTOR_DB_PATH.mkdir(parents=True)

        with patch("src.main.settings", mock_settings):
            main()

            captured = capsys.readouterr()
            # Should NOT print "Created database at" since it already exists
            assert "Created database at" not in captured.out
