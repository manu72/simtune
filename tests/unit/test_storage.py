import json
from unittest.mock import Mock, patch

import jsonlines
import pytest
import yaml
from pydantic import ValidationError

from core.models import (
    AuthorProfile,
    Dataset,
    FineTuneJob,
    JobStatus,
    ModelMetadata,
    Provider,
    StyleGuide,
    TrainingExample,
)
from core.storage import AuthorStorage, get_author_profile, list_authors


class TestAuthorStorage:
    """Test AuthorStorage class."""

    def test_init(self, temp_data_dir, mock_settings):
        """Test AuthorStorage initialization."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            assert storage.author_id == "test_author"
            assert storage.author_dir == mock_settings.authors_dir / "test_author"
            assert storage.author_dir.exists()

    def test_init_creates_directory(self, temp_data_dir, mock_settings):
        """Test that AuthorStorage creates author directory."""
        author_dir = mock_settings.authors_dir / "new_author"
        assert not author_dir.exists()

        with patch("core.storage.settings", mock_settings):
            AuthorStorage("new_author")
            assert author_dir.exists()

    def test_save_profile(self, temp_data_dir, mock_settings, sample_author_profile):
        """Test saving author profile."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")
            storage.save_profile(sample_author_profile)

            # Check profile.json exists
            profile_path = storage.author_dir / "profile.json"
            assert profile_path.exists()

            # Check style_guide.yml exists
            style_guide_path = storage.author_dir / "style_guide.yml"
            assert style_guide_path.exists()

            # Verify profile content
            with open(profile_path, "r") as f:
                profile_data = json.load(f)

            assert profile_data["author_id"] == sample_author_profile.author_id
            assert profile_data["name"] == sample_author_profile.name
            assert profile_data["description"] == sample_author_profile.description

            # Verify style guide content
            with open(style_guide_path, "r") as f:
                style_data = yaml.safe_load(f)

            assert style_data["tone"] == sample_author_profile.style_guide.tone
            assert style_data["voice"] == sample_author_profile.style_guide.voice

    def test_load_profile(self, temp_data_dir, mock_settings, sample_author_profile):
        """Test loading author profile."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Save then load profile
            storage.save_profile(sample_author_profile)
            loaded_profile = storage.load_profile()

            assert loaded_profile is not None
            assert loaded_profile.author_id == sample_author_profile.author_id
            assert loaded_profile.name == sample_author_profile.name
            assert loaded_profile.description == sample_author_profile.description
            assert (
                loaded_profile.style_guide.tone
                == sample_author_profile.style_guide.tone
            )

    def test_load_profile_not_exists(self, temp_data_dir, mock_settings):
        """Test loading profile when file doesn't exist."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("nonexistent_author")
            profile = storage.load_profile()

            assert profile is None

    def test_load_profile_invalid_json(self, temp_data_dir, mock_settings):
        """Test loading profile with invalid JSON."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create invalid JSON file
            profile_path = storage.author_dir / "profile.json"
            with open(profile_path, "w") as f:
                f.write("invalid json content")

            with pytest.raises(json.JSONDecodeError):
                storage.load_profile()

    def test_save_dataset(self, temp_data_dir, mock_settings, sample_dataset):
        """Test saving dataset."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")
            storage.save_dataset(sample_dataset)

            dataset_path = storage.author_dir / "train.jsonl"
            assert dataset_path.exists()

            # Verify content
            examples = []
            with jsonlines.open(dataset_path, "r") as reader:
                for obj in reader:
                    examples.append(obj)

            assert len(examples) == sample_dataset.size

            # Check first example structure
            first_example = examples[0]
            assert "messages" in first_example
            assert len(first_example["messages"]) >= 2

    def test_load_dataset(self, temp_data_dir, mock_settings, sample_dataset):
        """Test loading dataset."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Save then load dataset
            storage.save_dataset(sample_dataset)
            loaded_dataset = storage.load_dataset()

            assert loaded_dataset is not None
            assert loaded_dataset.author_id == sample_dataset.author_id
            assert loaded_dataset.size == sample_dataset.size
            assert len(loaded_dataset.examples) == len(sample_dataset.examples)

            # Compare first example
            original_first = sample_dataset.examples[0]
            loaded_first = loaded_dataset.examples[0]
            assert original_first.messages == loaded_first.messages

    def test_load_dataset_not_exists(self, temp_data_dir, mock_settings):
        """Test loading dataset when file doesn't exist."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")
            dataset = storage.load_dataset()

            # Should return empty dataset
            assert dataset is not None
            assert dataset.author_id == "test_author"
            assert dataset.size == 0
            assert dataset.examples == []

    def test_load_dataset_empty_file(self, temp_data_dir, mock_settings):
        """Test loading dataset from empty file."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create empty dataset file
            dataset_path = storage.author_dir / "train.jsonl"
            dataset_path.touch()

            dataset = storage.load_dataset()
            assert dataset is not None
            assert dataset.size == 0

    def test_load_dataset_invalid_jsonl(self, temp_data_dir, mock_settings):
        """Test loading dataset with invalid JSONL."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create invalid JSONL file
            dataset_path = storage.author_dir / "train.jsonl"
            with open(dataset_path, "w") as f:
                f.write("invalid jsonl line\n")
                f.write('{"valid": "json"}\n')

            with pytest.raises((json.JSONDecodeError, jsonlines.InvalidLineError)):
                storage.load_dataset()

    def test_save_model_metadata(
        self, temp_data_dir, mock_settings, sample_model_metadata
    ):
        """Test saving model metadata."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")
            storage.save_model_metadata(sample_model_metadata)

            models_path = storage.author_dir / "models.json"
            assert models_path.exists()

            # Verify content
            with open(models_path, "r") as f:
                metadata_data = json.load(f)

            assert len(metadata_data["fine_tune_jobs"]) == len(
                sample_model_metadata.fine_tune_jobs
            )
            assert metadata_data["active_model"] == sample_model_metadata.active_model

    def test_load_model_metadata(
        self, temp_data_dir, mock_settings, sample_model_metadata
    ):
        """Test loading model metadata."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Save then load metadata
            storage.save_model_metadata(sample_model_metadata)
            loaded_metadata = storage.load_model_metadata()

            assert loaded_metadata is not None
            assert len(loaded_metadata.fine_tune_jobs) == len(
                sample_model_metadata.fine_tune_jobs
            )

            # Compare first job
            if sample_model_metadata.fine_tune_jobs:
                original_job = sample_model_metadata.fine_tune_jobs[0]
                loaded_job = loaded_metadata.fine_tune_jobs[0]
                assert original_job.job_id == loaded_job.job_id
                assert original_job.provider == loaded_job.provider

    def test_load_model_metadata_not_exists(self, temp_data_dir, mock_settings):
        """Test loading metadata when file doesn't exist."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")
            metadata = storage.load_model_metadata()

            # Should return empty metadata
            assert metadata is not None
            assert metadata.fine_tune_jobs == []
            assert metadata.active_model is None

    def test_exists(self, temp_data_dir, mock_settings, sample_author_profile):
        """Test exists method."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Should not exist initially
            assert not storage.exists()

            # Should exist after saving profile
            storage.save_profile(sample_author_profile)
            assert storage.exists()

    def test_exists_directory_only(self, temp_data_dir, mock_settings):
        """Test exists when directory exists but profile doesn't."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Directory is created by init, but no profile saved
            assert storage.author_dir.exists()
            assert not storage.exists()

    def test_file_permissions_error(
        self, temp_data_dir, mock_settings, sample_author_profile
    ):
        """Test handling file permission errors."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Mock file operations to raise PermissionError
            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                with pytest.raises(PermissionError):
                    storage.save_profile(sample_author_profile)

    def test_json_serialization_error(
        self, temp_data_dir, mock_settings, sample_author_profile
    ):
        """Test handling JSON serialization errors."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Mock json.dump to raise an error
            with patch("json.dump", side_effect=TypeError("Object not serializable")):
                with pytest.raises(TypeError):
                    storage.save_profile(sample_author_profile)


class TestModuleFunctions:
    """Test module-level functions."""

    def test_list_authors(self, temp_data_dir, mock_settings, sample_author_profile):
        """Test listing authors."""
        with patch("core.storage.settings", mock_settings):
            # Initially empty
            authors = list_authors()
            assert authors == []

            # Create some authors
            storage1 = AuthorStorage("author1")
            profile1 = sample_author_profile.model_copy()
            profile1.author_id = "author1"
            profile1.name = "Author 1"
            storage1.save_profile(profile1)

            storage2 = AuthorStorage("author2")
            profile2 = sample_author_profile.model_copy()
            profile2.author_id = "author2"
            profile2.name = "Author 2"
            storage2.save_profile(profile2)

            authors = list_authors()
            assert sorted(authors) == ["author1", "author2"]

    def test_list_authors_no_directory(self, temp_data_dir):
        """Test listing authors when authors directory doesn't exist."""
        non_existent_dir = temp_data_dir / "nonexistent"
        mock_settings_no_dir = Mock()
        mock_settings_no_dir.authors_dir = non_existent_dir

        with patch("core.storage.settings", mock_settings_no_dir):
            authors = list_authors()
            assert authors == []

    def test_list_authors_directory_without_profiles(
        self, temp_data_dir, mock_settings
    ):
        """Test listing authors with directories but no profiles."""
        with patch("core.storage.settings", mock_settings):
            # Create directory without profile.json
            author_dir = mock_settings.authors_dir / "incomplete_author"
            author_dir.mkdir()

            authors = list_authors()
            assert authors == []

    def test_list_authors_mixed(
        self, temp_data_dir, mock_settings, sample_author_profile
    ):
        """Test listing authors with mix of valid and invalid directories."""
        with patch("core.storage.settings", mock_settings):
            # Valid author
            storage_valid = AuthorStorage("valid_author")
            profile_valid = sample_author_profile.model_copy()
            profile_valid.author_id = "valid_author"
            profile_valid.name = "Valid Author"
            storage_valid.save_profile(profile_valid)

            # Directory without profile
            incomplete_dir = mock_settings.authors_dir / "incomplete_author"
            incomplete_dir.mkdir()

            # File instead of directory (edge case)
            file_path = mock_settings.authors_dir / "not_a_directory"
            file_path.touch()

            authors = list_authors()
            assert authors == ["valid_author"]

    def test_get_author_profile(
        self, temp_data_dir, mock_settings, sample_author_profile
    ):
        """Test getting author profile."""
        with patch("core.storage.settings", mock_settings):
            # Save profile
            storage = AuthorStorage("test_author")
            storage.save_profile(sample_author_profile)

            # Get profile
            profile = get_author_profile("test_author")
            assert profile is not None
            assert profile.author_id == sample_author_profile.author_id
            assert profile.name == sample_author_profile.name

    def test_get_author_profile_not_exists(self, temp_data_dir, mock_settings):
        """Test getting profile that doesn't exist."""
        with patch("core.storage.settings", mock_settings):
            profile = get_author_profile("nonexistent_author")
            assert profile is None

    def test_get_author_profile_invalid_data(self, temp_data_dir, mock_settings):
        """Test getting profile with invalid data."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create invalid profile data
            profile_path = storage.author_dir / "profile.json"
            with open(profile_path, "w") as f:
                json.dump({"invalid": "profile_data"}, f)

            with pytest.raises(ValidationError):
                get_author_profile("test_author")


class TestStorageIntegration:
    """Integration tests for storage operations."""

    def test_full_author_workflow(
        self,
        temp_data_dir,
        mock_settings,
        sample_author_profile,
        sample_dataset,
        sample_model_metadata,
    ):
        """Test complete author data workflow."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("integration_author")

            # Save all data
            storage.save_profile(sample_author_profile)
            storage.save_dataset(sample_dataset)
            storage.save_model_metadata(sample_model_metadata)

            # Verify author exists
            assert storage.exists()
            assert "integration_author" in list_authors()

            # Load all data
            loaded_profile = storage.load_profile()
            loaded_dataset = storage.load_dataset()
            loaded_metadata = storage.load_model_metadata()

            # Verify data integrity
            assert loaded_profile.author_id == sample_author_profile.author_id
            assert loaded_dataset.size == sample_dataset.size
            assert len(loaded_metadata.fine_tune_jobs) == len(
                sample_model_metadata.fine_tune_jobs
            )

    def test_concurrent_access(
        self, temp_data_dir, mock_settings, sample_author_profile
    ):
        """Test concurrent access to storage."""
        with patch("core.storage.settings", mock_settings):
            # Multiple storage instances for same author
            storage1 = AuthorStorage("concurrent_author")
            storage2 = AuthorStorage("concurrent_author")

            # Both should see the same directory
            assert storage1.author_dir == storage2.author_dir

            # Save from one instance
            storage1.save_profile(sample_author_profile)

            # Load from another instance
            loaded_profile = storage2.load_profile()
            assert loaded_profile.name == sample_author_profile.name

    def test_data_persistence_across_instances(
        self, temp_data_dir, mock_settings, sample_author_profile
    ):
        """Test data persists across storage instances."""
        with patch("core.storage.settings", mock_settings):
            # Create and save with first instance
            storage1 = AuthorStorage("persistent_author")
            storage1.save_profile(sample_author_profile)

            # Create new instance and load
            storage2 = AuthorStorage("persistent_author")
            loaded_profile = storage2.load_profile()

            assert loaded_profile.author_id == sample_author_profile.author_id
            assert loaded_profile.name == sample_author_profile.name
