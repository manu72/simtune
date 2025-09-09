from pathlib import Path
from unittest.mock import patch

from core.config import Settings


class TestSettings:
    """Test Settings configuration class."""

    def test_settings_defaults(self, tmp_path, monkeypatch):
        """Test Settings default values."""
        # Clear environment variables to test defaults
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("DEBUG", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)

        with patch("core.config.Path") as mock_path:
            # Mock Path to use tmp_path
            mock_path.return_value = tmp_path
            settings = Settings(
                data_dir=tmp_path / "data",
                authors_dir=tmp_path / "data" / "authors",
                _env_file=None,  # Disable .env file loading
            )

            assert settings.openai_api_key is None
            assert settings.openai_org_id is None
            assert settings.gemini_api_key is None
            assert settings.debug is False
            assert settings.log_level == "info"
            assert str(settings.data_dir).endswith("data")
            assert str(settings.authors_dir).endswith("authors")

    def test_settings_with_env_vars(self, tmp_path, monkeypatch):
        """Test Settings with environment variables."""
        # Set environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("OPENAI_ORG_ID", "test-org-id")
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "debug")

        settings = Settings(
            data_dir=tmp_path / "data", authors_dir=tmp_path / "data" / "authors"
        )

        assert settings.openai_api_key == "test-openai-key"
        assert settings.openai_org_id == "test-org-id"
        assert settings.gemini_api_key == "test-gemini-key"
        assert settings.debug is True
        assert settings.log_level == "debug"

    def test_settings_case_insensitive(self, tmp_path, monkeypatch):
        """Test Settings case insensitive environment variables."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("DEBUG", "True")

        settings = Settings(
            data_dir=tmp_path / "data", authors_dir=tmp_path / "data" / "authors"
        )

        assert settings.openai_api_key == "test-key"
        assert settings.debug is True

    def test_ensure_directories(self, tmp_path):
        """Test directory creation."""
        data_dir = tmp_path / "test_data"
        authors_dir = tmp_path / "test_data" / "test_authors"

        # Directories shouldn't exist initially
        assert not data_dir.exists()
        assert not authors_dir.exists()

        Settings(data_dir=data_dir, authors_dir=authors_dir, _env_file=None)

        # Directories should be created
        assert data_dir.exists()
        assert authors_dir.exists()

    def test_ensure_directories_existing(self, tmp_path):
        """Test directory creation when directories already exist."""
        data_dir = tmp_path / "existing_data"
        authors_dir = tmp_path / "existing_data" / "existing_authors"

        # Create directories first
        data_dir.mkdir(parents=True)
        authors_dir.mkdir(parents=True)

        # Should not raise error when directories already exist
        Settings(data_dir=data_dir, authors_dir=authors_dir, _env_file=None)

        assert data_dir.exists()
        assert authors_dir.exists()

    def test_has_openai_key(self, tmp_path, monkeypatch):
        """Test has_openai_key method."""
        # Clear environment variables to test defaults
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Without API key
        settings = Settings(
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )
        assert settings.has_openai_key() is False

        # With API key
        settings_with_key = Settings(
            openai_api_key="test-key",
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )
        assert settings_with_key.has_openai_key() is True

        # With empty string (should be False)
        settings_empty = Settings(
            openai_api_key="",
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )
        assert settings_empty.has_openai_key() is False

    def test_has_gemini_key(self, tmp_path, monkeypatch):
        """Test has_gemini_key method."""
        # Clear environment variables to test defaults
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Without API key
        settings = Settings(
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )
        assert settings.has_gemini_key() is False

        # With API key
        settings_with_key = Settings(
            gemini_api_key="test-key",
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )
        assert settings_with_key.has_gemini_key() is True

        # With empty string (should be False)
        settings_empty = Settings(
            gemini_api_key="",
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )
        assert settings_empty.has_gemini_key() is False

    def test_validate_provider_access(self, tmp_path, monkeypatch):
        """Test validate_provider_access method."""
        # Clear environment variables to test defaults
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        settings = Settings(
            openai_api_key="openai-key",
            gemini_api_key="gemini-key",
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )

        # Valid providers with keys
        assert settings.validate_provider_access("openai") is True
        assert settings.validate_provider_access("gemini") is True

        # Invalid provider
        assert settings.validate_provider_access("invalid") is False

        # Valid providers without keys
        settings_no_keys = Settings(
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )
        assert settings_no_keys.validate_provider_access("openai") is False
        assert settings_no_keys.validate_provider_access("gemini") is False

    def test_validate_provider_access_partial_keys(self, tmp_path, monkeypatch):
        """Test validate_provider_access with partial API keys."""
        # Clear environment variables to test defaults
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        settings = Settings(
            openai_api_key="openai-key",
            # No gemini key
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )

        assert settings.validate_provider_access("openai") is True
        assert settings.validate_provider_access("gemini") is False

    def test_dotenv_loading(self, tmp_path):
        """Test that dotenv functionality works."""
        # Since load_dotenv() is called at module level,
        # we just verify it exists and doesn't error
        import core.config

        assert hasattr(core.config, "load_dotenv")

    def test_config_class_settings(self, tmp_path):
        """Test model configuration settings."""
        settings = Settings(
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )

        # Test that config is properly set
        config = settings.model_config
        assert config.get("env_file") == ".env"
        assert config.get("case_sensitive") is False
        assert config.get("extra") == "allow"

    def test_field_descriptions(self, tmp_path):
        """Test field descriptions are properly set."""
        settings = Settings(
            data_dir=tmp_path / "data",
            authors_dir=tmp_path / "data" / "authors",
            _env_file=None,
        )

        # Check that fields have proper descriptions
        fields = Settings.model_fields
        assert "Base directory for all data storage" in str(fields["data_dir"])
        assert "Directory for author profiles" in str(fields["authors_dir"])

    def test_path_handling(self, tmp_path):
        """Test Path object handling."""
        data_dir = tmp_path / "custom_data"
        authors_dir = tmp_path / "custom_data" / "custom_authors"

        settings = Settings(data_dir=data_dir, authors_dir=authors_dir, _env_file=None)

        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.authors_dir, Path)
        assert settings.data_dir == data_dir
        assert settings.authors_dir == authors_dir
