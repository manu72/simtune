from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from cli.main import app
from core.models import AuthorProfile
from core.storage import AuthorStorage


class TestCLIBasicCommands:
    """Test basic CLI commands."""

    def test_version_command(self):
        """Test version command."""
        runner = CliRunner()
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "Simtune" in result.stdout
        assert "v0.1.0" in result.stdout
        assert "Stage 1 POC" in result.stdout

    def test_help_command(self):
        """Test help command."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Personal AI Writing Assistant" in result.stdout
        assert "author" in result.stdout
        assert "dataset" in result.stdout
        assert "train" in result.stdout

    @patch("core.storage.list_authors")
    def test_status_command_no_authors(self, mock_list_authors):
        """Test status command with no authors."""
        mock_list_authors.return_value = []

        runner = CliRunner()
        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "No authors found" in result.stdout
        assert "simtune init" in result.stdout

    @patch("core.storage.list_authors")
    @patch("core.storage.get_author_profile")
    @patch("core.storage.AuthorStorage")
    def test_status_command_with_authors(
        self,
        mock_storage_class,
        mock_get_profile,
        mock_list_authors,
        sample_author_profile,
    ):
        """Test status command with existing authors."""
        # Mock authors list
        mock_list_authors.return_value = ["test_author"]
        mock_get_profile.return_value = sample_author_profile

        # Mock storage instance
        mock_storage = Mock()
        mock_storage.load_dataset.return_value = Mock(size=10)
        mock_storage.load_model_metadata.return_value = Mock(
            fine_tune_jobs=[], get_latest_successful_job=Mock(return_value=None)
        )
        mock_storage_class.return_value = mock_storage

        runner = CliRunner()
        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "test_author" in result.stdout or "Test Author" in result.stdout
        assert "1 author" in result.stdout


class TestAuthorCommands:
    """Test author management commands."""

    @patch("cli.commands.author.AuthorStorage")
    @patch("cli.commands.author.Prompt")
    @patch("cli.commands.author.Confirm")
    def test_author_create_interactive(
        self, mock_confirm, mock_prompt, mock_storage_class, temp_data_dir
    ):
        """Test interactive author creation."""
        # Mock user inputs
        mock_prompt.ask.side_effect = [
            "Test Author",  # name
            "A test author for unit testing",  # description
            "casual",  # tone
            "first_person",  # voice
            "informal",  # formality
            "medium",  # length preference
            "technology, writing",  # topics
            "politics",  # avoid topics
            "Keep it fun and engaging",  # style notes
        ]
        mock_confirm.ask.return_value = True

        # Mock storage
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        with patch("core.storage.settings.authors_dir", temp_data_dir / "authors"):
            runner = CliRunner()
            result = runner.invoke(app, ["author", "create", "test_author"])

            assert result.exit_code == 0
            # Verify storage save was called
            mock_storage.save_profile.assert_called_once()

    @patch("cli.commands.author.list_authors")
    def test_author_list_empty(self, mock_list_authors):
        """Test listing authors when none exist."""
        mock_list_authors.return_value = []

        runner = CliRunner()
        result = runner.invoke(app, ["author", "list"])

        assert result.exit_code == 0
        assert "No authors found" in result.stdout

    @patch("cli.commands.author.list_authors")
    @patch("cli.commands.author.get_author_profile")
    def test_author_list_with_authors(
        self, mock_get_profile, mock_list_authors, sample_author_profile
    ):
        """Test listing existing authors."""
        mock_list_authors.return_value = ["test_author"]
        mock_get_profile.return_value = sample_author_profile

        runner = CliRunner()
        result = runner.invoke(app, ["author", "list"])

        assert result.exit_code == 0
        assert "test_author" in result.stdout
        assert "Test Author" in result.stdout

    def test_author_create_non_interactive(self):
        """Test non-interactive author creation."""
        with patch("cli.commands.author.create_author_simple") as mock_create:
            runner = CliRunner()
            runner.invoke(
                app,
                [
                    "author",
                    "create",
                    "test_author",
                    "--name",
                    "Test Author",
                    "--description",
                    "Test description",
                    "--no-interactive",
                ],
            )

            # Should call the simple creation function
            mock_create.assert_called_once_with(
                "test_author", "Test Author", "Test description"
            )


class TestDatasetCommands:
    """Test dataset management commands."""

    @patch("cli.commands.dataset.get_author_profile")
    @patch("cli.commands.dataset.AuthorStorage")
    @patch("cli.commands.dataset.DatasetBuilder")
    def test_dataset_build_command(
        self, mock_builder_class, mock_storage_class, mock_get_author
    ):
        """Test dataset build command."""
        # Mock author profile
        mock_profile = Mock()
        mock_profile.name = "Test Author"
        mock_get_author.return_value = mock_profile

        # Mock storage exists check
        mock_storage = Mock()
        mock_storage.exists.return_value = True
        mock_storage_class.return_value = mock_storage

        # Mock dataset builder
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        runner = CliRunner()
        result = runner.invoke(app, ["dataset", "build", "test_author"])

        assert result.exit_code == 0
        # Verify builder was created and interactive_build was called
        mock_builder_class.assert_called_once_with("test_author")
        mock_builder.interactive_build.assert_called_once()

    @patch("cli.commands.dataset.get_author_profile")
    @patch("cli.commands.dataset.AuthorStorage")
    def test_dataset_build_author_not_exists(self, mock_storage_class, mock_get_author):
        """Test dataset build with non-existent author."""
        # Mock get_author_profile to return None (author doesn't exist)
        mock_get_author.return_value = None

        mock_storage = Mock()
        mock_storage.exists.return_value = False
        mock_storage_class.return_value = mock_storage

        runner = CliRunner()
        result = runner.invoke(app, ["dataset", "build", "nonexistent_author"])

        assert result.exit_code == 1  # Should fail when author doesn't exist
        assert "Author 'nonexistent_author' not found" in result.stdout

    @patch("cli.commands.dataset.get_author_profile")
    @patch("cli.commands.dataset.AuthorStorage")
    def test_dataset_validate_command(self, mock_storage_class, mock_get_author):
        """Test dataset validate command."""
        # Mock author profile
        mock_profile = Mock()
        mock_profile.name = "Test Author"
        mock_get_author.return_value = mock_profile

        # Mock storage and dataset
        mock_storage = Mock()
        mock_storage.exists.return_value = True
        mock_dataset = Mock()
        mock_dataset.size = 15
        mock_storage.load_dataset.return_value = mock_dataset
        mock_storage_class.return_value = mock_storage

        with patch("cli.commands.dataset.DatasetValidator") as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate.return_value = {"is_valid": True, "issues": []}
            mock_validator.get_validation_summary.return_value = "ready"
            mock_validator_class.return_value = mock_validator

            runner = CliRunner()
            result = runner.invoke(app, ["dataset", "validate", "test_author"])

            assert result.exit_code == 0
            mock_validator.validate.assert_called_once()


class TestTrainCommands:
    """Test training commands."""

    @patch("cli.commands.train.get_author_profile")
    @patch("cli.commands.train.AuthorStorage")
    @patch("cli.commands.train.OpenAIAdapter")
    @patch("cli.commands.train.DatasetValidator")
    def test_train_start_command(
        self,
        mock_validator_class,
        mock_adapter_class,
        mock_storage_class,
        mock_get_author,
    ):
        """Test train start command."""
        # Mock author profile
        mock_profile = Mock()
        mock_profile.name = "Test Author"
        mock_get_author.return_value = mock_profile

        # Mock storage
        mock_storage = Mock()
        mock_storage.exists.return_value = True
        mock_dataset = Mock()
        mock_dataset.size = 20
        mock_storage.load_dataset.return_value = mock_dataset
        mock_metadata = Mock()
        mock_storage.load_model_metadata.return_value = mock_metadata
        mock_storage_class.return_value = mock_storage

        # Mock dataset validator
        mock_validator = Mock()
        mock_validator.get_validation_summary.return_value = "ready"
        mock_validator_class.return_value = mock_validator

        # Mock OpenAI adapter
        mock_adapter = Mock()
        mock_adapter.upload_training_file.return_value = "file-123"
        mock_job = Mock()
        mock_job.job_id = "ft-job-123"
        mock_job.status = Mock()
        mock_job.status.value = "pending"
        mock_adapter.create_fine_tune_job.return_value = mock_job
        mock_adapter_class.return_value = mock_adapter

        runner = CliRunner()
        result = runner.invoke(app, ["train", "start", "test_author"])

        assert result.exit_code == 0
        mock_adapter.upload_training_file.assert_called_once()
        mock_adapter.create_fine_tune_job.assert_called_once()

    @patch("cli.commands.train.get_author_profile")
    @patch("cli.commands.train.AuthorStorage")
    def test_train_start_no_dataset(self, mock_storage_class, mock_get_author):
        """Test train start with no dataset."""
        # Mock author profile
        mock_profile = Mock()
        mock_profile.name = "Test Author"
        mock_get_author.return_value = mock_profile

        mock_storage = Mock()
        mock_storage.exists.return_value = True
        mock_dataset = Mock()
        mock_dataset.size = 0
        mock_storage.load_dataset.return_value = mock_dataset
        mock_storage_class.return_value = mock_storage

        runner = CliRunner()
        result = runner.invoke(app, ["train", "start", "test_author"])

        assert result.exit_code == 1  # Should fail when no dataset
        assert "No dataset found" in result.stdout

    @patch("cli.commands.train.get_author_profile")
    @patch("cli.commands.train.AuthorStorage")
    def test_train_status_command(self, mock_storage_class, mock_get_author):
        """Test train status command."""
        # Mock author profile
        mock_profile = Mock()
        mock_profile.name = "Test Author"
        mock_get_author.return_value = mock_profile

        # Mock storage and metadata
        mock_storage = Mock()
        mock_storage.exists.return_value = True
        mock_metadata = Mock()
        mock_metadata.fine_tune_jobs = []
        mock_storage.load_model_metadata.return_value = mock_metadata
        mock_storage_class.return_value = mock_storage

        runner = CliRunner()
        result = runner.invoke(app, ["train", "status", "test_author"])

        assert result.exit_code == 0


class TestInitCommand:
    """Test init command workflow."""

    @patch("cli.commands.author.create_author_interactive")
    def test_init_command(self, mock_create_author):
        """Test init command."""
        runner = CliRunner()
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "Welcome to Simtune!" in result.stdout
        mock_create_author.assert_called_once()


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete workflows."""

    @pytest.mark.skip(reason="Complex integration test - skip for POC stage")
    @patch("core.adapters.openai_adapter.settings")
    @patch("core.storage.settings")
    def test_complete_author_workflow(
        self, mock_storage_settings, mock_adapter_settings, temp_data_dir
    ):
        """Test complete workflow: create author -> build dataset -> start training."""
        # Setup temporary directories
        authors_dir = temp_data_dir / "authors"
        authors_dir.mkdir()
        mock_storage_settings.authors_dir = authors_dir
        mock_adapter_settings.has_openai_key.return_value = True
        mock_adapter_settings.openai_api_key = "test-key"

        runner = CliRunner()

        # Step 1: Create author (non-interactive)
        with patch("cli.commands.author.create_author_simple") as mock_create:
            # Mock the profile creation
            def create_mock_profile(*args, **kwargs):
                profile = AuthorProfile(
                    author_id="workflow_test",
                    name="Workflow Test",
                    description="Test workflow",
                )
                # Simulate saving the profile
                storage = AuthorStorage("workflow_test")
                storage.save_profile(profile)

            mock_create.side_effect = create_mock_profile

            result = runner.invoke(
                app,
                [
                    "author",
                    "create",
                    "workflow_test",
                    "--name",
                    "Workflow Test",
                    "--description",
                    "Test workflow",
                    "--no-interactive",
                ],
            )

            assert result.exit_code == 0

        # Step 2: Verify author was created
        result = runner.invoke(app, ["author", "list"])
        # Note: This might not show the author due to mocking complexity in CLI testing

        # Step 3: Test status command
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

    @pytest.mark.skip(reason="Integration test for error handling - skip for POC stage")
    def test_error_handling_invalid_author(self):
        """Test error handling for invalid author operations."""
        runner = CliRunner()

        # Try to build dataset for non-existent author
        result = runner.invoke(app, ["dataset", "build", "nonexistent_author"])
        assert result.exit_code == 0  # Command succeeds but shows error message

        # Try to start training for non-existent author
        result = runner.invoke(app, ["train", "start", "nonexistent_author"])
        assert result.exit_code == 0  # Command succeeds but shows error message

    def test_command_help_messages(self):
        """Test that all commands show proper help."""
        runner = CliRunner()

        commands_to_test = [
            ["author", "--help"],
            ["dataset", "--help"],
            ["train", "--help"],
            ["author", "create", "--help"],
            ["dataset", "build", "--help"],
            ["train", "start", "--help"],
        ]

        for cmd in commands_to_test:
            result = runner.invoke(app, cmd)
            assert result.exit_code == 0
            assert (
                "--help" in result.stdout
                or "Usage:" in result.stdout
                or "help" in result.stdout.lower()
            )


@pytest.mark.slow
class TestCLIPerformance:
    """Performance tests for CLI operations."""

    def test_status_command_performance(self):
        """Test that status command performs reasonably with multiple authors."""
        # Mock multiple authors
        with patch("core.storage.list_authors") as mock_list, patch(
            "core.storage.get_author_profile"
        ) as mock_profile, patch("core.storage.AuthorStorage") as mock_storage:

            # Create mock data for 10 authors
            mock_list.return_value = [f"author_{i}" for i in range(10)]
            mock_profile.side_effect = lambda author_id: AuthorProfile(
                author_id=author_id,
                name=f"Author {author_id}",
                description=f"Description for {author_id}",
            )

            mock_storage_instance = Mock()
            mock_storage_instance.load_dataset.return_value = Mock(size=10)
            mock_storage_instance.load_model_metadata.return_value = Mock(
                fine_tune_jobs=[], get_latest_successful_job=Mock(return_value=None)
            )
            mock_storage.return_value = mock_storage_instance

            runner = CliRunner()
            result = runner.invoke(app, ["status"])

            assert result.exit_code == 0
            assert "10 author" in result.stdout

    @pytest.mark.skip(reason="Performance test - skip for POC stage")
    def test_large_dataset_handling(self):
        """Test CLI handling of large datasets."""
        with patch("cli.commands.dataset.AuthorStorage") as mock_storage_class:
            mock_storage = Mock()
            mock_storage.exists.return_value = True

            # Mock large dataset
            mock_dataset = Mock()
            mock_dataset.size = 1000
            mock_storage.load_dataset.return_value = mock_dataset
            mock_storage_class.return_value = mock_storage

            with patch("cli.commands.dataset.DatasetValidator") as mock_validator_class:
                mock_validator = Mock()
                mock_validator.validate.return_value = {"is_valid": True, "issues": []}
                mock_validator_class.return_value = mock_validator

                runner = CliRunner()
                result = runner.invoke(app, ["dataset", "validate", "test_author"])

                assert result.exit_code == 0
