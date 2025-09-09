import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from core.markdown_utils import (
    create_content_markdown,
    create_markdown_content,
    ensure_content_directory,
    ensure_examples_directory,
    generate_content_filename,
    generate_markdown_filename,
    sanitize_subject,
    save_content_as_markdown,
    save_example_as_markdown,
)


class TestMarkdownUtils:
    """Test markdown utilities."""

    def test_sanitize_subject_normal_text(self):
        """Test sanitizing normal text."""
        result = sanitize_subject("Write about technology")
        assert result == "write_about_technology"

    def test_sanitize_subject_with_special_chars(self):
        """Test sanitizing text with special characters."""
        result = sanitize_subject("What's your favorite topic?!@#")
        assert result == "whats_your_favorite_topic"

    def test_sanitize_subject_max_length(self):
        """Test sanitizing text with length limit."""
        long_text = "This is a very long text that should be truncated"
        result = sanitize_subject(long_text, max_length=20)
        assert len(result) <= 20
        assert result == "this_is_a_very_long"

    def test_sanitize_subject_empty_fallback(self):
        """Test sanitizing empty or special-only text."""
        result = sanitize_subject("!@#$%^&*()")
        assert result == "example"

    def test_generate_markdown_filename(self):
        """Test generating markdown filename."""
        timestamp = datetime(2023, 1, 15, 14, 30, 45)
        result = generate_markdown_filename("user", "Write about technology", timestamp)
        assert result == "user_write_about_technology_20230115_143045.md"

    def test_generate_markdown_filename_current_time(self):
        """Test generating filename with current time."""
        with patch("core.markdown_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 15, 14, 30, 45)
            result = generate_markdown_filename("llm", "Test prompt")
            assert result == "llm_test_prompt_20230115_143045.md"

    def test_create_markdown_content(self):
        """Test creating markdown content."""
        timestamp = datetime(2023, 1, 15, 14, 30, 45)
        result = create_markdown_content(
            prompt="What is AI?",
            response="AI is artificial intelligence.",
            example_type="user",
            timestamp=timestamp,
        )

        expected = """# What is AI?

**Type:** user  
**Created:** 2023-01-15 14:30:45  
**Prompt:** What is AI?

## Content

AI is artificial intelligence.
"""
        assert result == expected

    def test_create_markdown_content_long_title(self):
        """Test creating markdown content with long prompt."""
        long_prompt = "This is a very long prompt that should be truncated in the title but kept full in the prompt field"
        result = create_markdown_content(
            prompt=long_prompt,
            response="Response",
            example_type="llm",
            timestamp=datetime(2023, 1, 15, 14, 30, 45),
        )

        # Title should be truncated to 50 characters + "..."
        assert result.startswith(
            "# This is a very long prompt that should be truncate..."
        )
        # But full prompt should be in the prompt field
        assert f"**Prompt:** {long_prompt}" in result

    def test_ensure_examples_directory(self):
        """Test ensuring examples directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            author_dir = Path(temp_dir) / "test_author"
            author_dir.mkdir()

            examples_dir = ensure_examples_directory(author_dir)

            assert examples_dir.exists()
            assert examples_dir.is_dir()
            assert examples_dir == author_dir / "examples"

    def test_ensure_examples_directory_already_exists(self):
        """Test ensuring examples directory when it already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            author_dir = Path(temp_dir) / "test_author"
            author_dir.mkdir()

            # Create examples directory first
            examples_dir = author_dir / "examples"
            examples_dir.mkdir()

            result = ensure_examples_directory(author_dir)

            assert result == examples_dir
            assert examples_dir.exists()

    def test_save_example_as_markdown(self):
        """Test saving example as markdown file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            author_dir = Path(temp_dir) / "test_author"
            author_dir.mkdir()

            timestamp = datetime(2023, 1, 15, 14, 30, 45)

            result_path = save_example_as_markdown(
                author_dir=author_dir,
                prompt="Test prompt",
                response="Test response",
                example_type="user",
                timestamp=timestamp,
            )

            # Check file was created
            assert result_path.exists()
            assert result_path.name == "user_test_prompt_20230115_143045.md"

            # Check file content
            content = result_path.read_text(encoding="utf-8")
            assert "# Test prompt" in content
            assert "**Type:** user" in content
            assert "**Prompt:** Test prompt" in content
            assert "Test response" in content

    def test_save_example_as_markdown_creates_directory(self):
        """Test that saving markdown creates examples directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            author_dir = Path(temp_dir) / "test_author"
            author_dir.mkdir()

            # Examples directory doesn't exist yet
            examples_dir = author_dir / "examples"
            assert not examples_dir.exists()

            save_example_as_markdown(
                author_dir=author_dir,
                prompt="Test prompt",
                response="Test response",
                example_type="llm",
            )

            # Should have been created
            assert examples_dir.exists()
            assert examples_dir.is_dir()


class TestContentMarkdownUtils:
    """Test content-specific markdown utilities."""

    def test_generate_content_filename(self):
        """Test generating content filename."""
        timestamp = datetime(2023, 1, 15, 14, 30, 45)
        result = generate_content_filename(
            "my_author", "Write about productivity tips", timestamp
        )
        assert result == "my_author_20230115_143045_write_abou.md"

    def test_generate_content_filename_current_time(self):
        """Test generating content filename with current time."""
        with patch("core.markdown_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 15, 14, 30, 45)
            result = generate_content_filename("author123", "Test prompt")
            assert result == "author123_20230115_143045_test_promp.md"

    def test_create_content_markdown(self):
        """Test creating content markdown."""
        timestamp = datetime(2023, 1, 15, 14, 30, 45)
        result = create_content_markdown(
            prompt="What is productivity?",
            response="Productivity is the effectiveness of productive effort.",
            author_name="John Doe",
            model_id="ft:gpt-3.5-turbo:example:123",
            timestamp=timestamp,
        )

        expected = """# Generated Content

**Author:** John Doe  
**Model:** ft:gpt-3.5-turbo:example:123  
**Created:** 2023-01-15 14:30:45  
**Prompt:** What is productivity?

## Content

Productivity is the effectiveness of productive effort.
"""
        assert result == expected

    def test_create_content_markdown_current_time(self):
        """Test creating content markdown with current time."""
        with patch("core.markdown_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 15, 14, 30, 45)
            result = create_content_markdown(
                prompt="Test prompt",
                response="Test response",
                author_name="Test Author",
                model_id="test-model",
            )

            assert "**Created:** 2023-01-15 14:30:45" in result
            assert "**Author:** Test Author" in result
            assert "**Model:** test-model" in result

    def test_ensure_content_directory(self):
        """Test ensuring content directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            author_dir = Path(temp_dir) / "test_author"
            author_dir.mkdir()

            content_dir = ensure_content_directory(author_dir)

            assert content_dir.exists()
            assert content_dir.is_dir()
            assert content_dir == author_dir / "content"

    def test_ensure_content_directory_already_exists(self):
        """Test ensuring content directory when it already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            author_dir = Path(temp_dir) / "test_author"
            author_dir.mkdir()

            # Create content directory first
            content_dir = author_dir / "content"
            content_dir.mkdir()

            result = ensure_content_directory(author_dir)

            assert result == content_dir
            assert content_dir.exists()

    def test_save_content_as_markdown(self):
        """Test saving content as markdown file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            author_dir = Path(temp_dir) / "test_author"
            author_dir.mkdir()

            timestamp = datetime(2023, 1, 15, 14, 30, 45)

            result_path = save_content_as_markdown(
                author_dir=author_dir,
                author_id="my_author",
                author_name="John Doe",
                prompt="Write about AI",
                response="AI is transforming technology.",
                model_id="ft:gpt-3.5-turbo:model:123",
                timestamp=timestamp,
            )

            # Check file was created
            assert result_path.exists()
            assert result_path.name == "my_author_20230115_143045_write_abou.md"

            # Check file content
            content = result_path.read_text(encoding="utf-8")
            assert "# Generated Content" in content
            assert "**Author:** John Doe" in content
            assert "**Model:** ft:gpt-3.5-turbo:model:123" in content
            assert "**Prompt:** Write about AI" in content
            assert "AI is transforming technology." in content

    def test_save_content_as_markdown_creates_directory(self):
        """Test that saving content creates content directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            author_dir = Path(temp_dir) / "test_author"
            author_dir.mkdir()

            # Content directory doesn't exist yet
            content_dir = author_dir / "content"
            assert not content_dir.exists()

            save_content_as_markdown(
                author_dir=author_dir,
                author_id="test_author",
                author_name="Test Author",
                prompt="Test prompt",
                response="Test response",
                model_id="test-model",
            )

            # Should have been created
            assert content_dir.exists()
            assert content_dir.is_dir()
