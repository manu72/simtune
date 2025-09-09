import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import responses
from freezegun import freeze_time

from core.config import Settings
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
from core.storage import AuthorStorage


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    authors_dir = data_dir / "authors"
    authors_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_settings(temp_data_dir):
    """Mock settings with temporary directory."""
    settings = Settings(
        data_dir=temp_data_dir,
        authors_dir=temp_data_dir / "authors",
        openai_api_key="test-api-key",
        openai_org_id="test-org-id",
        llm_provider="OPENAI",
        openai_model="gpt-3.5-turbo",
        gemini_model="gemini-1.5-flash",
        _skip_directory_creation=True,
    )
    return settings


@pytest.fixture
def sample_style_guide():
    """Sample style guide for testing."""
    return StyleGuide(
        tone="casual",
        voice="first_person",
        formality="moderate",
        length_preference="medium",
        topics=["technology", "writing", "productivity"],
        avoid_topics=["politics", "controversial topics"],
        writing_style_notes="Direct and engaging, with occasional humor",
    )


@pytest.fixture
def sample_author_profile(sample_style_guide):
    """Sample author profile for testing."""
    return AuthorProfile(
        author_id="test_author",
        name="Test Author",
        description="A test author profile for unit testing",
        style_guide=sample_style_guide,
        created_at=datetime(2023, 1, 1, 12, 0, 0),
        updated_at=datetime(2023, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_training_examples():
    """Sample training examples for testing."""
    return [
        TrainingExample(
            messages=[
                {"role": "system", "content": "You are a helpful writing assistant."},
                {
                    "role": "user",
                    "content": "Write a brief introduction about yourself.",
                },
                {
                    "role": "assistant",
                    "content": "Hello! I'm an AI writing assistant focused on helping you create engaging content in your unique style.",
                },
            ]
        ),
        TrainingExample(
            messages=[
                {"role": "system", "content": "You are a helpful writing assistant."},
                {
                    "role": "user",
                    "content": "Explain the importance of documentation in software development.",
                },
                {
                    "role": "assistant",
                    "content": "Documentation is crucial in software development because it serves as a roadmap for future developers and your future self. Good docs save time, reduce bugs, and make collaboration smoother.",
                },
            ]
        ),
        TrainingExample(
            messages=[
                {"role": "system", "content": "You are a helpful writing assistant."},
                {"role": "user", "content": "Write tips for productive remote work."},
                {
                    "role": "assistant",
                    "content": "Here are my top remote work tips: 1) Create a dedicated workspace, 2) Establish clear boundaries between work and personal time, 3) Use video calls to maintain human connection, 4) Take regular breaks to avoid burnout.",
                },
            ]
        ),
    ]


@pytest.fixture
def sample_dataset(sample_training_examples):
    """Sample dataset for testing."""
    dataset = Dataset(author_id="test_author")
    for example in sample_training_examples:
        dataset.add_example(example)
    return dataset


@pytest.fixture
def sample_fine_tune_job():
    """Sample fine-tuning job for testing."""
    return FineTuneJob(
        job_id="ft-job-test-123",
        author_id="test_author",
        provider=Provider.OPENAI,
        base_model="gpt-3.5-turbo",
        status=JobStatus.PENDING,
        created_at=datetime(2023, 1, 1, 12, 0, 0),
        training_file_id="file-abc123",
        hyperparameters={"n_epochs": 3, "learning_rate_multiplier": 0.1},
    )


@pytest.fixture
def sample_model_metadata(sample_fine_tune_job):
    """Sample model metadata for testing."""
    metadata = ModelMetadata()
    metadata.add_job(sample_fine_tune_job)
    return metadata


@pytest.fixture
def author_storage(mock_settings):
    """Author storage instance with temporary directory."""
    with patch("core.storage.settings", mock_settings):
        return AuthorStorage("test_author")


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = Mock()

    # Mock file creation
    file_response = Mock()
    file_response.id = "file-test123"
    client.files.create.return_value = file_response

    # Mock fine-tuning job creation
    job_response = Mock()
    job_response.id = "ft-job-test123"
    job_response.status = "validating_files"
    client.fine_tuning.jobs.create.return_value = job_response

    # Mock job status retrieval
    status_response = Mock()
    status_response.id = "ft-job-test123"
    status_response.status = "succeeded"
    status_response.fine_tuned_model = "ft:gpt-3.5-turbo:test:test-model:123"
    status_response.error = None
    status_response.estimated_finish = None
    status_response.result_files = []
    client.fine_tuning.jobs.retrieve.return_value = status_response

    # Mock chat completions
    completion_response = Mock()
    completion_response.choices = [Mock()]
    completion_response.choices[0].message.content = (
        "Test response from fine-tuned model"
    )
    client.chat.completions.create.return_value = completion_response

    # Mock model listing
    models_response = Mock()
    models_response.data = [
        Mock(id="ft:gpt-3.5-turbo:test:test-model:123", owned_by="user")
    ]
    client.models.list.return_value = models_response

    return client


@pytest.fixture
def mock_console():
    """Mock Rich console for testing CLI output."""
    return Mock()


@pytest.fixture
def mock_prompt():
    """Mock Typer prompt for testing user input."""
    return Mock()


@pytest.fixture
def openai_responses():
    """Setup responses mock for OpenAI API calls."""
    with responses.RequestsMock() as rsps:
        # Mock file upload
        rsps.add(
            responses.POST,
            "https://api.openai.com/v1/files",
            json={"id": "file-test123", "purpose": "fine-tune"},
            status=200,
        )

        # Mock fine-tuning job creation
        rsps.add(
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

        # Mock job status retrieval
        rsps.add(
            responses.GET,
            "https://api.openai.com/v1/fine_tuning/jobs/ft-job-test123",
            json={
                "id": "ft-job-test123",
                "status": "succeeded",
                "model": "gpt-3.5-turbo",
                "fine_tuned_model": "ft:gpt-3.5-turbo:test:test-model:123",
                "training_file": "file-test123",
            },
            status=200,
        )

        yield rsps


# Test configuration
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_data_dir):
    """Setup test environment for each test."""
    # Create a test settings instance
    test_settings = Settings(
        data_dir=temp_data_dir,
        authors_dir=temp_data_dir / "authors",
        openai_api_key="test-api-key",
        openai_org_id="test-org-id",
        llm_provider="OPENAI",
        openai_model="gpt-3.5-turbo",
        gemini_model="gemini-1.5-flash",
        _skip_directory_creation=True,
    )

    # Replace the global settings with test settings
    monkeypatch.setattr("core.config.settings", test_settings)
    monkeypatch.setattr("core.storage.settings", test_settings)
    if hasattr(monkeypatch, "core.dataset.builder.settings"):
        monkeypatch.setattr("core.dataset.builder.settings", test_settings)


@pytest.fixture
def freeze_datetime():
    """Freeze time for consistent datetime testing."""
    with freeze_time("2023-01-01 12:00:00"):
        yield


# Sample file content fixtures
@pytest.fixture
def sample_text_content():
    """Sample text content for file import testing."""
    return """
This is the first paragraph of sample content. It contains enough text to be meaningful for testing purposes.

This is a second paragraph that talks about different topics. It should be long enough to test the text splitting functionality properly.

A third paragraph with more content about various subjects. This helps test how the system handles multiple sections of text when importing from files.

Short paragraph.

This final paragraph concludes our sample text with additional meaningful content that can be used for comprehensive testing of the import functionality.
""".strip()


@pytest.fixture
def sample_jsonl_content(sample_training_examples):
    """Sample JSONL content for dataset testing."""
    lines = []
    for example in sample_training_examples:
        lines.append(json.dumps(example.model_dump()))
    return "\n".join(lines)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "openai: mark test as requiring OpenAI API")
