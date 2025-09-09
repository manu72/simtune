import json
from unittest.mock import Mock, mock_open, patch

import pytest
import responses

from core.adapters.openai_adapter import OpenAIAdapter
from core.models import FineTuneJob, JobStatus, Provider


class TestOpenAIAdapterInit:
    """Test OpenAI adapter initialization."""

    @patch("core.adapters.openai_adapter.settings")
    def test_init_with_api_key(self, mock_settings):
        """Test initialization with API key."""
        mock_settings.has_openai_key.return_value = True
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.openai_org_id = "test-org-id"

        with patch("openai.OpenAI") as mock_openai:
            adapter = OpenAIAdapter()

            mock_openai.assert_called_once_with(
                api_key="test-api-key", organization="test-org-id"
            )
            assert adapter.client == mock_openai.return_value

    @patch("core.adapters.openai_adapter.settings")
    def test_init_without_api_key(self, mock_settings):
        """Test initialization fails without API key."""
        mock_settings.has_openai_key.return_value = False

        with pytest.raises(ValueError) as exc_info:
            OpenAIAdapter()

        assert "OpenAI API key not found" in str(exc_info.value)


class TestOpenAIAdapterFileUpload:
    """Test file upload functionality."""

    def test_upload_training_file(self, mock_openai_client, sample_dataset):
        """Test successful training file upload."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                file_id = adapter.upload_training_file(sample_dataset)

                assert file_id == "file-test123"
                mock_openai_client.files.create.assert_called_once()

                # Verify the file was created with correct purpose
                call_args = mock_openai_client.files.create.call_args
                assert call_args[1]["purpose"] == "fine-tune"

    @pytest.mark.skip(reason="Requires real OpenAI API key - skip for POC stage")
    @patch("tempfile.NamedTemporaryFile")
    @patch("builtins.open", new_callable=mock_open)
    def test_upload_training_file_content(
        self, mock_file_open, mock_tempfile, mock_openai_client, sample_dataset
    ):
        """Test that training file contains correct JSONL content."""
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test_file.jsonl"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()
                adapter.upload_training_file(sample_dataset)

                # Verify JSONL content was written
                write_calls = mock_temp.write.call_args_list
                assert len(write_calls) >= sample_dataset.size

                # Check that JSON was written for each example
                json_lines = 0
                newlines = 0
                for call in write_calls:
                    content = call[0][0]
                    if "\n" == content:
                        newlines += 1
                    else:
                        # Should be valid JSON
                        json.loads(content)
                        json_lines += 1

                assert json_lines == sample_dataset.size
                assert newlines == sample_dataset.size

    def test_upload_training_file_api_error(self, mock_openai_client, sample_dataset):
        """Test upload file with API error."""
        mock_openai_client.files.create.side_effect = Exception("API Error")

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                with pytest.raises(Exception) as exc_info:
                    adapter.upload_training_file(sample_dataset)

                assert "API Error" in str(exc_info.value)

    @pytest.mark.skip(reason="Requires real OpenAI API key - skip for POC stage")
    @patch("core.adapters.openai_adapter.Path")
    def test_upload_training_file_cleanup(
        self, mock_path, mock_openai_client, sample_dataset
    ):
        """Test that temporary file is cleaned up."""
        mock_temp_path = Mock()
        mock_path.return_value = mock_temp_path

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                mock_temp = Mock()
                mock_temp.name = "/tmp/test_file.jsonl"
                mock_tempfile.return_value.__enter__.return_value = mock_temp

                with patch("openai.OpenAI", return_value=mock_openai_client):
                    adapter = OpenAIAdapter()
                    adapter.upload_training_file(sample_dataset)

                # Verify cleanup was called
                mock_temp_path.unlink.assert_called_once_with(missing_ok=True)


class TestOpenAIAdapterFineTuning:
    """Test fine-tuning job functionality."""

    def test_create_fine_tune_job(self, mock_openai_client):
        """Test successful fine-tune job creation."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                job = adapter.create_fine_tune_job(
                    author_id="test_author",
                    training_file_id="file-test123",
                    base_model="gpt-3.5-turbo",
                )

                assert isinstance(job, FineTuneJob)
                assert job.job_id == "ft-job-test123"
                assert job.author_id == "test_author"
                assert job.provider == Provider.OPENAI
                assert job.base_model == "gpt-3.5-turbo"
                assert job.status == JobStatus.PENDING
                assert job.training_file_id == "file-test123"

                # Verify API was called correctly
                mock_openai_client.fine_tuning.jobs.create.assert_called_once()
                call_args = mock_openai_client.fine_tuning.jobs.create.call_args
                assert call_args[1]["training_file"] == "file-test123"
                assert call_args[1]["model"] == "gpt-3.5-turbo"

    def test_create_fine_tune_job_with_hyperparameters(self, mock_openai_client):
        """Test fine-tune job creation with custom hyperparameters."""
        custom_hyperparams = {"n_epochs": 5, "learning_rate_multiplier": 0.5}

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True
            mock_settings.get_default_model.return_value = "gpt-3.5-turbo"

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                job = adapter.create_fine_tune_job(
                    author_id="test_author",
                    training_file_id="file-test123",
                    hyperparameters=custom_hyperparams,
                )

                # Check that custom hyperparameters are merged with defaults
                expected_hyperparams = {"n_epochs": 5, "learning_rate_multiplier": 0.5}

                call_args = mock_openai_client.fine_tuning.jobs.create.call_args
                assert call_args[1]["hyperparameters"] == expected_hyperparams
                assert job.hyperparameters == expected_hyperparams

    def test_create_fine_tune_job_api_error(self, mock_openai_client):
        """Test fine-tune job creation with API error."""
        mock_openai_client.fine_tuning.jobs.create.side_effect = Exception("API Error")

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                with pytest.raises(Exception) as exc_info:
                    adapter.create_fine_tune_job(
                        author_id="test_author", training_file_id="file-test123"
                    )

                assert "API Error" in str(exc_info.value)


class TestOpenAIAdapterJobStatus:
    """Test job status functionality."""

    def test_get_job_status_success(self, mock_openai_client):
        """Test getting job status successfully."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                status_info = adapter.get_job_status("ft-job-test123")

                assert status_info["status"] == JobStatus.SUCCEEDED
                assert status_info["openai_status"] == "succeeded"
                assert (
                    status_info["fine_tuned_model"]
                    == "ft:gpt-3.5-turbo:test:test-model:123"
                )
                assert status_info["error"] is None

    def test_get_job_status_mapping(self):
        """Test OpenAI status mapping to our JobStatus."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            status_mappings = [
                ("validating_files", JobStatus.PENDING),
                ("queued", JobStatus.PENDING),
                ("running", JobStatus.RUNNING),
                ("succeeded", JobStatus.SUCCEEDED),
                ("failed", JobStatus.FAILED),
                ("cancelled", JobStatus.CANCELLED),
                ("unknown_status", JobStatus.PENDING),  # Default fallback
            ]

            for openai_status, expected_status in status_mappings:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.status = openai_status
                mock_response.fine_tuned_model = None
                mock_client.fine_tuning.jobs.retrieve.return_value = mock_response

                with patch("openai.OpenAI", return_value=mock_client):
                    adapter = OpenAIAdapter()
                    status_info = adapter.get_job_status("test-job")

                    assert status_info["status"] == expected_status
                    assert status_info["openai_status"] == openai_status

    def test_get_job_status_api_error(self, mock_openai_client):
        """Test get job status with API error."""
        mock_openai_client.fine_tuning.jobs.retrieve.side_effect = Exception(
            "API Error"
        )

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                status_info = adapter.get_job_status("ft-job-test123")

                assert status_info["status"] == JobStatus.FAILED
                assert "API Error" in status_info["error"]

    def test_update_job_status(self, mock_openai_client, sample_fine_tune_job):
        """Test updating job status."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                updated_job = adapter.update_job_status(sample_fine_tune_job)

                assert updated_job.status == JobStatus.SUCCEEDED
                assert (
                    updated_job.fine_tuned_model
                    == "ft:gpt-3.5-turbo:test:test-model:123"
                )
                assert updated_job.error_message is None


class TestOpenAIAdapterWaitForCompletion:
    """Test wait for completion functionality."""

    @patch("time.sleep")  # Speed up the test
    @patch("core.adapters.openai_adapter.Progress")
    def test_wait_for_completion_success(
        self, mock_progress, mock_sleep, mock_openai_client, sample_fine_tune_job
    ):
        """Test waiting for job completion successfully."""
        # Simulate job progression: pending -> running -> succeeded
        status_responses = [
            Mock(status="running", fine_tuned_model=None, error=None),
            Mock(status="succeeded", fine_tuned_model="ft:model:123", error=None),
        ]
        mock_openai_client.fine_tuning.jobs.retrieve.side_effect = status_responses

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                final_job = adapter.wait_for_completion(
                    sample_fine_tune_job, check_interval=1
                )

                assert final_job.status == JobStatus.SUCCEEDED
                assert final_job.fine_tuned_model == "ft:model:123"

                # Should have called sleep at least once
                assert mock_sleep.call_count >= 1

    @patch("time.sleep")
    @patch("core.adapters.openai_adapter.Progress")
    def test_wait_for_completion_failure(
        self, mock_progress, mock_sleep, mock_openai_client, sample_fine_tune_job
    ):
        """Test waiting for job that fails."""
        mock_response = Mock()
        mock_response.status = "failed"
        mock_response.fine_tuned_model = None
        mock_response.error = "Training failed"
        mock_openai_client.fine_tuning.jobs.retrieve.return_value = mock_response

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                final_job = adapter.wait_for_completion(sample_fine_tune_job)

                assert final_job.status == JobStatus.FAILED

    @patch("time.sleep")
    @patch("core.adapters.openai_adapter.Progress")
    def test_wait_for_completion_cancelled(
        self, mock_progress, mock_sleep, mock_openai_client, sample_fine_tune_job
    ):
        """Test waiting for job that gets cancelled."""
        mock_response = Mock()
        mock_response.status = "cancelled"
        mock_response.fine_tuned_model = None
        mock_response.error = None
        mock_openai_client.fine_tuning.jobs.retrieve.return_value = mock_response

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                final_job = adapter.wait_for_completion(sample_fine_tune_job)

                assert final_job.status == JobStatus.CANCELLED


class TestOpenAIAdapterTextGeneration:
    """Test text generation functionality."""

    def test_generate_text(self, mock_openai_client):
        """Test text generation."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                response = adapter.generate_text(
                    model_id="ft:gpt-3.5-turbo:model:123", prompt="Test prompt"
                )

                assert response == "Test response from fine-tuned model"

                # Verify API call
                mock_openai_client.chat.completions.create.assert_called_once()
                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]["model"] == "ft:gpt-3.5-turbo:model:123"
                assert call_args[1]["messages"][0]["content"] == "Test prompt"
                assert call_args[1]["max_completion_tokens"] == 500
                # Temperature parameter was removed for ChatGPT 5 compatibility TESTING ONLY

    def test_generate_text_custom_params(self, mock_openai_client):
        """Test text generation with custom parameters."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                adapter.generate_text(
                    model_id="ft:model:123",
                    prompt="Custom prompt",
                    max_completion_tokens=1000,
                )

                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]["max_completion_tokens"] == 1000

    def test_generate_text_api_error(self, mock_openai_client):
        """Test text generation with API error."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                with pytest.raises(Exception) as exc_info:
                    adapter.generate_text("ft:model:123", "Test prompt")

                assert "API Error" in str(exc_info.value)


class TestOpenAIAdapterModelTesting:
    """Test model testing functionality."""

    @patch("core.adapters.openai_adapter.console")
    def test_test_fine_tuned_model_default_prompts(
        self, mock_console, mock_openai_client
    ):
        """Test testing fine-tuned model with default prompts."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                results = adapter.test_fine_tuned_model("ft:model:123")

                assert len(results) == 3  # Default test prompts
                assert "Write a brief introduction about yourself." in results
                assert "What's your writing style?" in results
                assert "Tell me about your expertise." in results

                # All should have the same response
                for response in results.values():
                    assert response == "Test response from fine-tuned model"

    @patch("core.adapters.openai_adapter.console")
    def test_test_fine_tuned_model_custom_prompts(
        self, mock_console, mock_openai_client
    ):
        """Test testing fine-tuned model with custom prompts."""
        custom_prompts = ["Custom prompt 1", "Custom prompt 2"]

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                results = adapter.test_fine_tuned_model("ft:model:123", custom_prompts)

                assert len(results) == 2
                assert "Custom prompt 1" in results
                assert "Custom prompt 2" in results

    @patch("core.adapters.openai_adapter.console")
    def test_test_fine_tuned_model_with_error(self, mock_console, mock_openai_client):
        """Test testing fine-tuned model when API error occurs."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                results = adapter.test_fine_tuned_model("ft:model:123", ["Test prompt"])

                assert "Test prompt" in results
                assert "Error: API Error" in results["Test prompt"]


class TestOpenAIAdapterModelManagement:
    """Test model management functionality."""

    @pytest.mark.skip(reason="Requires real OpenAI API key - skip for POC stage")
    def test_list_fine_tuned_models(self, mock_openai_client):
        """Test listing fine-tuned models."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                models = adapter.list_fine_tuned_models()

                assert len(models) == 1
                assert "ft:" in models[0].id
                assert models[0].owned_by == "user"

    def test_list_fine_tuned_models_api_error(self, mock_openai_client):
        """Test listing models with API error."""
        mock_openai_client.models.list.side_effect = Exception("API Error")

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                models = adapter.list_fine_tuned_models()

                assert models == []

    @patch("core.adapters.openai_adapter.console")
    def test_delete_fine_tuned_model_success(self, mock_console, mock_openai_client):
        """Test successful model deletion."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                result = adapter.delete_fine_tuned_model("ft:model:123")

                assert result is True
                mock_openai_client.models.delete.assert_called_once_with("ft:model:123")

    @patch("core.adapters.openai_adapter.console")
    def test_delete_fine_tuned_model_error(self, mock_console, mock_openai_client):
        """Test model deletion with API error."""
        mock_openai_client.models.delete.side_effect = Exception("API Error")

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                result = adapter.delete_fine_tuned_model("ft:model:123")

                assert result is False


@pytest.mark.integration
class TestOpenAIAdapterIntegration:
    """Integration tests using responses mock."""

    @pytest.mark.skip(reason="Integration test with API - skip for POC stage")
    @responses.activate
    def test_full_workflow_mock_responses(self, sample_dataset):
        """Test full workflow with mocked HTTP responses."""
        # Setup responses
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/files",
            json={"id": "file-test123", "purpose": "fine-tune"},
            status=200,
        )

        responses.add(
            responses.POST,
            "https://api.openai.com/v1/fine_tuning/jobs",
            json={
                "id": "ft-job-test123",
                "status": "validating_files",
                "model": "gpt-3.5-turbo",
                "training_file": "file-test123",
            },
            status=200,
        )

        responses.add(
            responses.GET,
            "https://api.openai.com/v1/fine_tuning/jobs/ft-job-test123",
            json={
                "id": "ft-job-test123",
                "status": "succeeded",
                "fine_tuned_model": "ft:gpt-3.5-turbo:test:model:123",
            },
            status=200,
        )

        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_org_id = None

            adapter = OpenAIAdapter()

            # Test file upload
            file_id = adapter.upload_training_file(sample_dataset)
            assert file_id == "file-test123"

            # Test job creation
            job = adapter.create_fine_tune_job("test_author", file_id)
            assert job.job_id == "ft-job-test123"

            # Test status update
            updated_job = adapter.update_job_status(job)
            assert updated_job.status == JobStatus.SUCCEEDED
            assert updated_job.fine_tuned_model == "ft:gpt-3.5-turbo:test:model:123"


class TestOpenAIAdapterChat:
    """Test chat-related functionality of OpenAI adapter."""

    def test_generate_chat_response_success(self, mock_openai_client):
        """Test successful chat response generation."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            # Mock successful chat completion response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = (
                "Hello! How can I help you today?"
            )
            mock_openai_client.chat.completions.create.return_value = mock_response

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                messages = [
                    {"role": "user", "content": "Hello there!"},
                    {"role": "assistant", "content": "Hi! How can I help?"},
                    {"role": "user", "content": "Tell me a joke"},
                ]

                response = adapter.generate_chat_response(
                    "ft:gpt-3.5-turbo:model:123", messages
                )

                assert response == "Hello! How can I help you today?"
                mock_openai_client.chat.completions.create.assert_called_once_with(
                    model="ft:gpt-3.5-turbo:model:123",
                    messages=messages,
                    max_completion_tokens=500,
                    temperature=0.7,
                )

    def test_generate_chat_response_with_custom_tokens(self, mock_openai_client):
        """Test chat response generation with custom token limit."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Custom response"
            mock_openai_client.chat.completions.create.return_value = mock_response

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                messages = [{"role": "user", "content": "Test message"}]

                response = adapter.generate_chat_response(
                    "ft:gpt-3.5-turbo:model:123", messages, max_completion_tokens=1000
                )

                assert response == "Custom response"
                mock_openai_client.chat.completions.create.assert_called_once_with(
                    model="ft:gpt-3.5-turbo:model:123",
                    messages=messages,
                    max_completion_tokens=1000,
                    temperature=0.7,
                )

    def test_generate_chat_response_error(self, mock_openai_client):
        """Test chat response generation with API error."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            mock_openai_client.chat.completions.create.side_effect = Exception(
                "API Error"
            )

            with patch("openai.OpenAI", return_value=mock_openai_client):
                adapter = OpenAIAdapter()

                messages = [{"role": "user", "content": "Test message"}]

                with pytest.raises(Exception) as exc_info:
                    adapter.generate_chat_response(
                        "ft:gpt-3.5-turbo:model:123", messages
                    )

                assert "API Error" in str(exc_info.value)

    def test_truncate_messages_under_limit(self):
        """Test message truncation when under character limit."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI"):
                adapter = OpenAIAdapter()

                messages = [
                    {"role": "user", "content": "Short message"},
                    {"role": "assistant", "content": "Short response"},
                ]

                truncated = adapter._truncate_messages(messages, 500)
                assert truncated == messages

    def test_truncate_messages_over_limit(self):
        """Test message truncation when over character limit."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI"):
                adapter = OpenAIAdapter()

                # Create messages that exceed limits
                long_content = "x" * 5000  # Very long message
                messages = [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": long_content},
                    {"role": "user", "content": "Last message"},
                ]

                truncated = adapter._truncate_messages(messages, 500)

                # Should keep most recent messages that fit
                assert len(truncated) <= len(messages)
                assert truncated[-1]["content"] == "Last message"  # Most recent kept

    def test_truncate_messages_empty_list(self):
        """Test message truncation with empty messages list."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI"):
                adapter = OpenAIAdapter()

                truncated = adapter._truncate_messages([], 500)
                assert truncated == []

    def test_truncate_messages_keeps_at_least_last_message(self):
        """Test that truncation keeps at least the last message."""
        with patch("core.adapters.openai_adapter.settings") as mock_settings:
            mock_settings.has_openai_key.return_value = True

            with patch("openai.OpenAI"):
                adapter = OpenAIAdapter()

                # Create message that's too long but should still be kept
                very_long_content = "x" * 50000
                messages = [{"role": "user", "content": very_long_content}]

                truncated = adapter._truncate_messages(messages, 500)

                # Should keep at least one message
                assert len(truncated) == 1
                assert truncated[0]["content"] == very_long_content
