from datetime import datetime
from unittest.mock import patch

import pytest

from core.models import ChatMessage, ChatSession
from core.storage import AuthorStorage


class TestAuthorStorageContent:
    """Test content-related functionality of AuthorStorage."""

    def test_content_directory_created_on_init(self, temp_data_dir, mock_settings):
        """Test that content directory is created when AuthorStorage is initialized."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Content directory should exist
            content_dir = storage.content_dir
            assert content_dir.exists()
            assert content_dir.is_dir()
            assert content_dir.parent == storage.author_dir

    def test_content_dir_property(self, temp_data_dir, mock_settings):
        """Test content_dir property returns correct path."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            expected_path = storage.author_dir / "content"
            assert storage.content_dir == expected_path

    def test_save_generated_content(self, temp_data_dir, mock_settings):
        """Test saving generated content."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            timestamp = datetime(2023, 1, 15, 14, 30, 45)

            content_path = storage.save_generated_content(
                prompt="Write about productivity",
                response="Productivity is key to success.",
                author_name="Test Author",
                model_id="ft:gpt-3.5-turbo:model:123",
                timestamp=timestamp,
            )

            # Check file was created
            assert content_path.exists()
            assert content_path.name == "test_author_20230115_143045_write_abou.md"

            # Check file content
            content = content_path.read_text(encoding="utf-8")
            assert "# Generated Content" in content
            assert "**Author:** Test Author" in content
            assert "**Model:** ft:gpt-3.5-turbo:model:123" in content
            assert "**Prompt:** Write about productivity" in content
            assert "Productivity is key to success." in content

    def test_save_generated_content_current_time(self, temp_data_dir, mock_settings):
        """Test saving generated content with current time."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            content_path = storage.save_generated_content(
                prompt="Test prompt",
                response="Test response",
                author_name="Test Author",
                model_id="test-model",
            )

            # Check file was created (filename will have current timestamp)
            assert content_path.exists()
            assert content_path.name.startswith("test_author_")
            assert content_path.name.endswith("_test_promp.md")

    def test_save_generated_content_error_handling(self, temp_data_dir, mock_settings):
        """Test that content saving errors are handled gracefully."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Mock the save_content_as_markdown to raise an exception
            with patch(
                "core.storage.save_content_as_markdown",
                side_effect=Exception("Test error"),
            ):
                with pytest.raises(Exception, match="Test error"):
                    storage.save_generated_content(
                        prompt="Test prompt",
                        response="Test response",
                        author_name="Test Author",
                        model_id="test-model",
                    )


class TestAuthorStorageChat:
    """Test chat-related functionality of AuthorStorage."""

    def test_chats_directory_created_on_init(self, temp_data_dir, mock_settings):
        """Test that chats directory is created when AuthorStorage is initialized."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Chats directory should exist
            chats_dir = storage.chats_dir
            assert chats_dir.exists()
            assert chats_dir.is_dir()
            assert chats_dir.parent == storage.author_dir

    def test_chats_dir_property(self, temp_data_dir, mock_settings):
        """Test chats_dir property returns correct path."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            expected_path = storage.author_dir / "chats"
            assert storage.chats_dir == expected_path

    def test_save_chat_session(self, temp_data_dir, mock_settings):
        """Test saving a chat session."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            session = ChatSession(author_id="test_author")
            session.add_message("user", "Hello!")
            session.add_message("assistant", "Hi there!")

            saved_path = storage.save_chat_session(session)

            # Check file was created
            assert saved_path.exists()
            assert saved_path.name == f"{session.session_id}.json"
            assert saved_path.parent == storage.chats_dir

    def test_load_chat_session(self, temp_data_dir, mock_settings):
        """Test loading a chat session."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create and save a session
            original_session = ChatSession(author_id="test_author")
            original_session.add_message("user", "Hello!")
            original_session.add_message("assistant", "Hi there!")
            storage.save_chat_session(original_session)

            # Load the session
            loaded_session = storage.load_chat_session(original_session.session_id)

            assert loaded_session is not None
            assert loaded_session.session_id == original_session.session_id
            assert loaded_session.author_id == original_session.author_id
            assert loaded_session.message_count == 2
            assert loaded_session.messages[0].content == "Hello!"
            assert loaded_session.messages[1].content == "Hi there!"

    def test_load_nonexistent_chat_session(self, temp_data_dir, mock_settings):
        """Test loading a nonexistent chat session."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            loaded_session = storage.load_chat_session("nonexistent-id")
            assert loaded_session is None

    def test_list_chat_sessions_empty(self, temp_data_dir, mock_settings):
        """Test listing chat sessions when directory is empty."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            sessions = storage.list_chat_sessions()
            assert sessions == []

    def test_list_chat_sessions(self, temp_data_dir, mock_settings):
        """Test listing chat sessions."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create and save multiple sessions
            session1 = ChatSession(author_id="test_author")
            session2 = ChatSession(author_id="test_author")
            session3 = ChatSession(author_id="test_author")

            storage.save_chat_session(session1)
            storage.save_chat_session(session2)
            storage.save_chat_session(session3)

            sessions = storage.list_chat_sessions()

            # Should return all session IDs
            assert len(sessions) == 3
            assert session1.session_id in sessions
            assert session2.session_id in sessions
            assert session3.session_id in sessions

    def test_delete_chat_session(self, temp_data_dir, mock_settings):
        """Test deleting a chat session."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create and save a session
            session = ChatSession(author_id="test_author")
            storage.save_chat_session(session)

            # Verify it exists
            assert storage.load_chat_session(session.session_id) is not None

            # Delete it
            result = storage.delete_chat_session(session.session_id)
            assert result is True

            # Verify it's gone
            assert storage.load_chat_session(session.session_id) is None

    def test_delete_nonexistent_chat_session(self, temp_data_dir, mock_settings):
        """Test deleting a nonexistent chat session."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            result = storage.delete_chat_session("nonexistent-id")
            assert result is False

    def test_export_chat_session_as_markdown(self, temp_data_dir, mock_settings):
        """Test exporting a chat session as markdown."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create a session with messages
            session = ChatSession(author_id="test_author")
            session.add_message("user", "Hello there!")
            session.add_message("assistant", "Hi! How can I help you?")
            session.add_message("user", "Tell me a joke")
            session.add_message(
                "assistant",
                "Why don't scientists trust atoms? Because they make up everything!",
            )

            # Export as markdown
            export_path = storage.export_chat_session_as_markdown(
                session, "Test Author"
            )

            # Check file was created
            assert export_path.exists()
            assert export_path.suffix == ".md"
            assert export_path.parent == storage.chats_dir

            # Check file content
            content = export_path.read_text(encoding="utf-8")
            assert "# Chat Session - Test Author" in content
            assert session.session_id in content
            assert "🧑 **You**" in content
            assert "🤖 **Test Author**" in content
            assert "Hello there!" in content
            assert "Hi! How can I help you?" in content
            assert "Tell me a joke" in content
            assert "Why don't scientists trust atoms?" in content

    def test_export_empty_chat_session(self, temp_data_dir, mock_settings):
        """Test exporting an empty chat session."""
        with patch("core.storage.settings", mock_settings):
            storage = AuthorStorage("test_author")

            # Create empty session
            session = ChatSession(author_id="test_author")

            # Export as markdown
            export_path = storage.export_chat_session_as_markdown(
                session, "Test Author"
            )

            # Check file was created
            assert export_path.exists()

            # Check file content
            content = export_path.read_text(encoding="utf-8")
            assert "# Chat Session - Test Author" in content
            assert "**Messages:** 0" in content
