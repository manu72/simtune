from datetime import datetime
from unittest.mock import Mock, call, patch
from uuid import uuid4

import pytest
import typer
from rich.console import Console

from cli.commands.generate import (
    _display_assistant_message,
    _display_available_sessions,
    _display_chat_header,
    _display_chat_help,
    _display_conversation_history,
    _display_session_info,
    _display_user_message,
    _format_timestamp,
    _handle_chat_command,
    chat_session,
)
from core.models import AuthorProfile, ChatMessage, ChatSession, StyleGuide


class TestChatSessionCommand:
    """Test the main chat_session command."""

    @patch("cli.commands.generate.get_author_profile")
    def test_chat_session_author_not_found(self, mock_get_profile):
        """Test chat session with non-existent author."""
        mock_get_profile.return_value = None

        with pytest.raises(typer.Exit) as exc_info:
            chat_session("nonexistent_author")

        assert exc_info.value.exit_code == 1

    @patch("cli.commands.generate.console.print")
    @patch("cli.commands.generate.get_author_profile")
    @patch("cli.commands.generate.AuthorStorage")
    def test_chat_session_no_fine_tuned_model(
        self, mock_storage_class, mock_get_profile, mock_print
    ):
        """Test chat session when no fine-tuned model exists."""
        # Setup mocks
        mock_profile = Mock(spec=AuthorProfile)
        mock_profile.name = "Test Author"
        mock_get_profile.return_value = mock_profile

        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        mock_metadata = Mock()
        mock_metadata.get_latest_successful_job.return_value = None
        mock_storage.load_model_metadata.return_value = mock_metadata

        with pytest.raises(typer.Exit) as exc_info:
            chat_session("test_author")

        assert exc_info.value.exit_code == 1
        mock_print.assert_any_call(
            "[red]No fine-tuned model found for 'test_author'.[/red]"
        )

    @patch("cli.commands.generate.Prompt.ask")
    @patch("cli.commands.generate.OpenAIAdapter")
    @patch("cli.commands.generate._display_conversation_history")
    @patch("cli.commands.generate._display_chat_header")
    @patch("cli.commands.generate._display_user_message")
    @patch("cli.commands.generate._display_assistant_message")
    @patch("cli.commands.generate.console.print")
    @patch("cli.commands.generate.get_author_profile")
    @patch("cli.commands.generate.AuthorStorage")
    def test_chat_session_basic_flow(
        self,
        mock_storage_class,
        mock_get_profile,
        mock_print,
        mock_display_assistant,
        mock_display_user,
        mock_display_header,
        mock_display_history,
        mock_adapter_class,
        mock_prompt_ask,
    ):
        """Test basic chat session flow."""
        # Setup profile
        mock_profile = Mock(spec=AuthorProfile)
        mock_profile.name = "Test Author"
        mock_get_profile.return_value = mock_profile

        # Setup storage and model
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        mock_job = Mock()
        mock_job.fine_tuned_model = "ft:gpt-3.5-turbo:model:123"

        mock_metadata = Mock()
        mock_metadata.get_latest_successful_job.return_value = mock_job
        mock_storage.load_model_metadata.return_value = mock_metadata

        # Setup adapter
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter.generate_chat_response.return_value = "Hello! How can I help?"

        # Setup prompt sequence: user input, then quit command
        mock_prompt_ask.side_effect = ["Hello there!", "/quit"]

        # Run the command
        chat_session("test_author", session_id=None, save=True)

        # Verify session was created and saved
        mock_storage.save_chat_session.assert_called()

        # Verify OpenAI adapter was used
        mock_adapter.generate_chat_response.assert_called_once()

    @patch("cli.commands.generate.console.print")
    @patch("cli.commands.generate.get_author_profile")
    @patch("cli.commands.generate.AuthorStorage")
    def test_chat_session_resume_nonexistent_session(
        self, mock_storage_class, mock_get_profile, mock_print
    ):
        """Test resuming a non-existent chat session."""
        # Setup profile
        mock_profile = Mock(spec=AuthorProfile)
        mock_get_profile.return_value = mock_profile

        # Setup storage
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage
        mock_storage.load_chat_session.return_value = None

        with pytest.raises(typer.Exit) as exc_info:
            chat_session("test_author", session_id="nonexistent-session")

        assert exc_info.value.exit_code == 1
        mock_print.assert_any_call(
            "[red]Chat session 'nonexistent-session' not found.[/red]"
        )


class TestChatDisplayFunctions:
    """Test chat display helper functions."""

    def test_display_chat_header(self):
        """Test chat header display."""
        session = ChatSession(author_id="test_author")
        session.add_message("user", "Test message")

        with patch("cli.commands.generate.console.print") as mock_print:
            _display_chat_header(
                "Test Author", "ft:gpt-3.5-turbo:model:123", session, True
            )

            # Verify header was printed
            mock_print.assert_called_once()

    def test_display_conversation_history(self):
        """Test conversation history display."""
        session = ChatSession(author_id="test_author")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")

        with patch(
            "cli.commands.generate._display_user_message"
        ) as mock_display_user, patch(
            "cli.commands.generate._display_assistant_message"
        ) as mock_display_assistant, patch(
            "cli.commands.generate.console.print"
        ) as mock_print:

            _display_conversation_history(session, "Test Author")

            # Verify messages were displayed
            assert mock_display_user.call_count == 1
            assert mock_display_assistant.call_count == 1

    def test_display_user_message(self):
        """Test user message display."""
        with patch("cli.commands.generate.console.print") as mock_print:
            _display_user_message("Hello there!", show_timestamp=False)

            # Verify user message was printed
            mock_print.assert_any_call("\n[bold cyan]🧑 You:[/bold cyan]")
            mock_print.assert_any_call("[white]Hello there![/white]")

    def test_display_assistant_message(self):
        """Test assistant message display."""
        with patch("cli.commands.generate.console.print") as mock_print:
            _display_assistant_message(
                "Hello there!", "Test Author", show_timestamp=False
            )

            # Verify assistant message header was printed
            mock_print.assert_called()

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Just test that it returns a string in the correct format
        result = _format_timestamp()
        assert isinstance(result, str)
        assert len(result) == 8  # HH:MM:SS format
        assert result.count(":") == 2

    def test_display_session_info(self):
        """Test session info display."""
        session = ChatSession(author_id="test_author")
        session.add_message("user", "Test message")

        with patch("cli.commands.generate.console.print") as mock_print:
            _display_session_info(session)

            # Verify table was printed
            mock_print.assert_called_once()

    def test_display_available_sessions_empty(self):
        """Test displaying available sessions when none exist."""
        mock_storage = Mock()
        mock_storage.list_chat_sessions.return_value = []

        with patch("cli.commands.generate.console.print") as mock_print:
            _display_available_sessions(mock_storage)

            # Verify empty message was printed
            mock_print.assert_any_call("[yellow]No saved chat sessions found.[/yellow]")

    def test_display_available_sessions_with_data(self):
        """Test displaying available sessions with data."""
        mock_storage = Mock()

        # Create mock sessions
        session1 = ChatSession(author_id="test_author")
        session1.add_message("user", "Test 1")
        session2 = ChatSession(author_id="test_author")
        session2.add_message("user", "Test 2")

        mock_storage.list_chat_sessions.return_value = [
            session1.session_id,
            session2.session_id,
        ]
        mock_storage.load_chat_session.side_effect = [session1, session2]

        with patch("cli.commands.generate.console.print") as mock_print:
            _display_available_sessions(mock_storage)

            # Verify table was printed
            mock_print.assert_called()

    def test_display_chat_help(self):
        """Test chat help display."""
        with patch("cli.commands.generate.console.print") as mock_print:
            _display_chat_help()

            # Verify help panel was printed
            mock_print.assert_called_once()


class TestChatCommandHandlers:
    """Test chat command handling functions."""

    def test_handle_quit_commands(self):
        """Test quit command variants."""
        session = ChatSession(author_id="test_author")
        storage = Mock()

        with patch("cli.commands.generate.console.print") as mock_print:
            for quit_cmd in ["/quit", "/exit", "/q"]:
                result = _handle_chat_command(quit_cmd, session, storage, "Test Author")
                assert result is False

            # Verify exit messages were printed
            assert mock_print.call_count >= 3

    def test_handle_clear_command(self):
        """Test clear command."""
        session = ChatSession(author_id="test_author")
        session.add_message("user", "Test message")
        storage = Mock()

        with patch("cli.commands.generate.console.print") as mock_print:
            result = _handle_chat_command("/clear", session, storage, "Test Author")

            assert result is True
            assert session.message_count == 0
            storage.save_chat_session.assert_called_once_with(session)
            mock_print.assert_any_call("[green]Conversation cleared.[/green]")

    def test_handle_history_command_with_messages(self):
        """Test history command with existing messages."""
        session = ChatSession(author_id="test_author")
        session.add_message("user", "Test message")
        storage = Mock()

        with patch(
            "cli.commands.generate._display_conversation_history"
        ) as mock_display:
            result = _handle_chat_command("/history", session, storage, "Test Author")

            assert result is True
            mock_display.assert_called_once_with(session, "Test Author")

    def test_handle_history_command_empty(self):
        """Test history command with no messages."""
        session = ChatSession(author_id="test_author")
        storage = Mock()

        with patch("cli.commands.generate.console.print") as mock_print:
            result = _handle_chat_command("/history", session, storage, "Test Author")

            assert result is True
            mock_print.assert_any_call("[yellow]No conversation history yet.[/yellow]")

    def test_handle_save_command(self):
        """Test save command."""
        session = ChatSession(author_id="test_author")
        storage = Mock()

        with patch("cli.commands.generate.console.print") as mock_print:
            result = _handle_chat_command("/save", session, storage, "Test Author")

            assert result is True
            storage.save_chat_session.assert_called_once_with(session)
            mock_print.assert_any_call(
                f"[green]Session saved: {session.session_id}[/green]"
            )

    def test_handle_export_command(self):
        """Test export command variants."""
        session = ChatSession(author_id="test_author")
        storage = Mock()
        mock_path = Mock()
        mock_path.name = "chat_export.md"
        storage.export_chat_session_as_markdown.return_value = mock_path

        with patch("cli.commands.generate.console.print") as mock_print:
            for export_cmd in ["/export", "/export-md"]:
                result = _handle_chat_command(
                    export_cmd, session, storage, "Test Author"
                )
                assert result is True

            storage.export_chat_session_as_markdown.assert_called_with(
                session, "Test Author"
            )
            mock_print.assert_any_call(
                "[green]Chat exported to: chat_export.md[/green]"
            )

    def test_handle_info_command(self):
        """Test info command."""
        session = ChatSession(author_id="test_author")
        storage = Mock()

        with patch("cli.commands.generate._display_session_info") as mock_display:
            result = _handle_chat_command("/info", session, storage, "Test Author")

            assert result is True
            mock_display.assert_called_once_with(session)

    def test_handle_sessions_command(self):
        """Test sessions command."""
        session = ChatSession(author_id="test_author")
        storage = Mock()

        with patch("cli.commands.generate._display_available_sessions") as mock_display:
            result = _handle_chat_command("/sessions", session, storage, "Test Author")

            assert result is True
            mock_display.assert_called_once_with(storage)

    def test_handle_help_commands(self):
        """Test help command variants."""
        session = ChatSession(author_id="test_author")
        storage = Mock()

        with patch("cli.commands.generate._display_chat_help") as mock_display:
            for help_cmd in ["/help", "/?"]:
                result = _handle_chat_command(help_cmd, session, storage, "Test Author")
                assert result is True

            assert mock_display.call_count == 2

    def test_handle_unknown_command(self):
        """Test handling of unknown commands."""
        session = ChatSession(author_id="test_author")
        storage = Mock()

        with patch("cli.commands.generate.console.print") as mock_print:
            result = _handle_chat_command("/unknown", session, storage, "Test Author")

            assert result is True
            mock_print.assert_any_call("[yellow]Unknown command: /unknown[/yellow]")
            mock_print.assert_any_call("Use /help to see available commands.")


class TestChatSessionIntegration:
    """Test integration scenarios for chat functionality."""

    @patch("cli.commands.generate.Prompt.ask")
    @patch("cli.commands.generate.OpenAIAdapter")
    @patch("cli.commands.generate._display_conversation_history")
    @patch("cli.commands.generate._display_chat_header")
    @patch("cli.commands.generate._display_user_message")
    @patch("cli.commands.generate._display_assistant_message")
    @patch("cli.commands.generate.get_author_profile")
    @patch("cli.commands.generate.AuthorStorage")
    def test_full_chat_conversation_flow(
        self,
        mock_storage_class,
        mock_get_profile,
        mock_display_assistant,
        mock_display_user,
        mock_display_header,
        mock_display_history,
        mock_adapter_class,
        mock_prompt_ask,
    ):
        """Test a complete chat conversation flow."""
        # Setup profile
        mock_profile = Mock(spec=AuthorProfile)
        mock_profile.name = "Test Author"
        mock_get_profile.return_value = mock_profile

        # Setup storage
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        mock_job = Mock()
        mock_job.fine_tuned_model = "ft:gpt-3.5-turbo:model:123"
        mock_metadata = Mock()
        mock_metadata.get_latest_successful_job.return_value = mock_job
        mock_storage.load_model_metadata.return_value = mock_metadata

        # Setup adapter with realistic responses
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter.generate_chat_response.side_effect = [
            "Hello! How can I help you?",
            "That's a great question about productivity!",
        ]

        # Setup conversation flow: greeting, question, quit
        mock_prompt_ask.side_effect = [
            "Hello there!",
            "Can you give me productivity tips?",
            "/quit",
        ]

        # Run the chat session
        chat_session("test_author", session_id=None, save=True)

        # Verify the conversation flow
        assert mock_adapter.generate_chat_response.call_count == 2

        # Verify session was saved
        assert mock_storage.save_chat_session.call_count >= 2

    @patch("cli.commands.generate.Prompt.ask")
    @patch("cli.commands.generate.OpenAIAdapter")
    @patch("cli.commands.generate._display_conversation_history")
    @patch("cli.commands.generate._display_chat_header")
    @patch("cli.commands.generate._display_user_message")
    @patch("cli.commands.generate._display_assistant_message")
    @patch("cli.commands.generate.get_author_profile")
    @patch("cli.commands.generate.AuthorStorage")
    def test_chat_with_error_handling(
        self,
        mock_storage_class,
        mock_get_profile,
        mock_display_assistant,
        mock_display_user,
        mock_display_header,
        mock_display_history,
        mock_adapter_class,
        mock_prompt_ask,
    ):
        """Test chat session with API error handling."""
        # Setup profile and storage
        mock_profile = Mock(spec=AuthorProfile)
        mock_profile.name = "Test Author"
        mock_get_profile.return_value = mock_profile

        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        mock_job = Mock()
        mock_job.fine_tuned_model = "ft:gpt-3.5-turbo:model:123"
        mock_metadata = Mock()
        mock_metadata.get_latest_successful_job.return_value = mock_job
        mock_storage.load_model_metadata.return_value = mock_metadata

        # Setup adapter to raise an error on first call, succeed on second
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter.generate_chat_response.side_effect = [
            Exception("API Error"),
            "This works fine!",
        ]

        # Setup prompt sequence: message causing error, then successful message, then quit
        mock_prompt_ask.side_effect = ["First message", "Second message", "/quit"]

        with patch("cli.commands.generate.console.print"):
            # Run the command with explicit parameters - should handle the error gracefully
            chat_session("test_author", session_id=None, save=True)

        # Verify the first call failed but the session continued
        assert mock_adapter.generate_chat_response.call_count == 2
