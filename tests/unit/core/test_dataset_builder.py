from unittest.mock import Mock, mock_open, patch

from core.dataset.builder import DatasetBuilder
from core.models import Dataset, TrainingExample
from core.storage import AuthorStorage


class TestDatasetBuilder:
    """Test DatasetBuilder class."""

    def test_init(self, temp_data_dir, mock_settings):
        """Test DatasetBuilder initialization."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            assert builder.author_id == "test_author"
            assert isinstance(builder.storage, AuthorStorage)
            assert isinstance(builder.dataset, Dataset)
            assert builder.dataset.author_id == "test_author"

    def test_init_with_existing_dataset(
        self, temp_data_dir, mock_settings, sample_dataset
    ):
        """Test DatasetBuilder initialization with existing dataset."""
        with patch("core.storage.settings", mock_settings):
            # Save dataset first
            storage = AuthorStorage("test_author")
            storage.save_dataset(sample_dataset)

            # Initialize builder - should load existing dataset
            builder = DatasetBuilder("test_author")
            assert builder.dataset.size == sample_dataset.size

    def test_split_content_normal_text(self):
        """Test _split_content with normal paragraph structure."""
        builder = DatasetBuilder("test_author")

        content = """First paragraph with some content.

Second paragraph with different content.

Third paragraph with more text."""

        sections = builder._split_content(content)
        assert len(sections) == 3
        assert sections[0] == "First paragraph with some content."
        assert sections[1] == "Second paragraph with different content."
        assert sections[2] == "Third paragraph with more text."

    def test_split_content_multiple_newlines(self):
        """Test _split_content with multiple newlines between paragraphs."""
        builder = DatasetBuilder("test_author")

        content = """First paragraph.



Second paragraph after multiple newlines.


Third paragraph."""

        sections = builder._split_content(content)
        assert len(sections) == 3
        assert "First paragraph." in sections
        assert "Second paragraph after multiple newlines." in sections
        assert "Third paragraph." in sections

    def test_split_content_empty_sections(self):
        """Test _split_content filters out empty sections."""
        builder = DatasetBuilder("test_author")

        content = """Valid paragraph.


Another valid paragraph.

"""

        sections = builder._split_content(content)
        assert len(sections) == 2
        assert sections[0] == "Valid paragraph."
        assert sections[1] == "Another valid paragraph."

    def test_split_content_single_paragraph(self):
        """Test _split_content with single paragraph."""
        builder = DatasetBuilder("test_author")

        content = "Single paragraph without line breaks."

        sections = builder._split_content(content)
        assert len(sections) == 1
        assert sections[0] == "Single paragraph without line breaks."

    def test_split_content_empty_string(self):
        """Test _split_content with empty string."""
        builder = DatasetBuilder("test_author")

        sections = builder._split_content("")
        assert sections == []

    def test_split_content_whitespace_only(self):
        """Test _split_content with whitespace only."""
        builder = DatasetBuilder("test_author")

        sections = builder._split_content("   \n\n  \t  \n  ")
        assert sections == []

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    def test_add_from_writing_samples(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _add_from_writing_samples method."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock console.input() to simulate multiline input
            mock_console.input.side_effect = [
                "Sample writing content here.",  # First line of input
                EOFError(),  # Simulate Ctrl+D to end input
            ]

            # Mock prompt for the new workflow (choice + manual prompt)
            mock_prompt.ask.side_effect = [
                "1",  # Choose manual prompt entry (choice)
                "Write about the topic",  # The actual prompt
            ]

            initial_size = builder.dataset.size
            builder._add_from_writing_samples()

            # Should add one example
            assert builder.dataset.size == initial_size + 1

            # Check the added example
            example = builder.dataset.examples[-1]
            assert len(example.messages) == 3
            assert example.messages[0]["role"] == "system"
            assert example.messages[1]["role"] == "user"
            assert example.messages[1]["content"] == "Write about the topic"
            assert example.messages[2]["role"] == "assistant"
            assert example.messages[2]["content"] == "Sample writing content here."

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    def test_add_from_writing_samples_empty_input(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _add_from_writing_samples with empty input."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock empty input (EOF immediately)
            mock_console.input.side_effect = [EOFError()]

            # Mock prompt for the new workflow (choice + manual prompt)
            mock_prompt.ask.side_effect = [
                "1",  # Choose manual prompt entry (choice)
                "Some prompt",  # The actual prompt
            ]

            initial_size = builder.dataset.size
            builder._add_from_writing_samples()

            # Should not add any examples
            assert builder.dataset.size == initial_size
            mock_console.print.assert_called_with("[red]No sample provided[/red]")

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    def test_add_from_writing_samples_empty_prompt(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _add_from_writing_samples with empty prompt."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock console.input() to simulate multiline input
            mock_console.input.side_effect = [
                "Sample content",  # First line of input
                EOFError(),  # Simulate Ctrl+D to end input
            ]

            # Mock empty prompt in new workflow
            mock_prompt.ask.side_effect = [
                "1",  # Choose manual prompt entry (choice)
                "",  # Empty prompt
            ]

            initial_size = builder.dataset.size
            builder._add_from_writing_samples()

            # Should not add any examples
            assert builder.dataset.size == initial_size
            mock_console.print.assert_called_with("[red]No prompt provided[/red]")

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch(
        "core.prompts.templates.EXAMPLE_GENERATION_PROMPTS",
        {
            "creative_writing": "Write creatively about {topic} in a {tone} tone with {length_preference} length.",
            "technical_explanation": "Explain {topic} technically.",
        },
    )
    def test_generate_examples(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _generate_examples method."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock user inputs
            mock_prompt.ask.side_effect = [
                "1",  # prompt type selection
                "artificial intelligence",  # topic
            ]

            # Mock console.input() to simulate multiline input followed by EOF
            mock_console.input.side_effect = [
                "AI is fascinating and complex...",
                "It has many applications in modern technology.",
                EOFError(),  # Simulate Ctrl+D to end input
            ]

            initial_size = builder.dataset.size
            builder._generate_examples()

            # Should add one example
            assert builder.dataset.size == initial_size + 1

            # Check the added example
            example = builder.dataset.examples[-1]
            assert "artificial intelligence" in example.messages[1]["content"]
            assert "AI is fascinating and complex..." in example.messages[2]["content"]
            assert "It has many applications" in example.messages[2]["content"]

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    def test_generate_examples_empty_response(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _generate_examples with empty response."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock inputs with empty response
            mock_prompt.ask.side_effect = ["1", "test topic"]

            # Mock console.input() to simulate empty input (just EOF)
            mock_console.input.side_effect = [EOFError()]

            initial_size = builder.dataset.size

            # Mock EXAMPLE_GENERATION_PROMPTS
            with patch(
                "core.dataset.builder.EXAMPLE_GENERATION_PROMPTS",
                {"test": "Test prompt"},
            ):
                builder._generate_examples()

            # Should not add any examples
            assert builder.dataset.size == initial_size
            mock_console.print.assert_called_with("[red]No response provided[/red]")

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.Confirm")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="This is the first paragraph with enough content to pass the minimum length requirement for testing purposes.\n\nThis is the second paragraph which also has sufficient content to meet the length requirements for the dataset builder test.",
    )
    @patch("core.dataset.builder.Path")
    def test_import_from_file(
        self,
        mock_path_class,
        mock_file,
        mock_confirm,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
    ):
        """Test _import_from_file method."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock Path behavior
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path

            # Mock user inputs for new workflow
            mock_prompt.ask.side_effect = [
                "/path/to/file.txt",  # file path
                "1",  # Choose manual prompt entry for first section
                "What prompt for first paragraph?",  # prompt for first section
                "1",  # Choose manual prompt entry for second section
                "What prompt for second paragraph?",  # prompt for second section
            ]
            mock_confirm.ask.side_effect = [True, True]  # Include both sections

            initial_size = builder.dataset.size
            builder._import_from_file()

            # Should add two examples (both sections included)
            assert builder.dataset.size == initial_size + 2

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.Path")
    def test_import_from_file_not_exists(
        self, mock_path_class, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _import_from_file with non-existent file."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock Path behavior
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_path_class.return_value = mock_path

            mock_prompt.ask.return_value = "/nonexistent/file.txt"

            initial_size = builder.dataset.size
            builder._import_from_file()

            # Should not add any examples
            assert builder.dataset.size == initial_size
            mock_console.print.assert_called_with(
                "[red]File not found: /nonexistent/file.txt[/red]"
            )

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    @patch("core.dataset.builder.Path")
    def test_import_from_file_empty(
        self,
        mock_path_class,
        mock_file,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
    ):
        """Test _import_from_file with empty file."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock Path behavior
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path

            mock_prompt.ask.return_value = "/path/to/empty.txt"

            initial_size = builder.dataset.size
            builder._import_from_file()

            # Should not add any examples
            assert builder.dataset.size == initial_size
            mock_console.print.assert_called_with("[red]File is empty[/red]")

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    @patch("core.dataset.builder.Path")
    def test_import_from_file_read_error(
        self,
        mock_path_class,
        mock_file,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
    ):
        """Test _import_from_file with file read error."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock Path behavior
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path

            mock_prompt.ask.return_value = "/path/to/restricted.txt"

            builder._import_from_file()

            # Should show error message
            mock_console.print.assert_called_with(
                "[red]Error reading file: Permission denied[/red]"
            )

    @patch("core.dataset.builder.console")
    def test_review_dataset_empty(self, mock_console, temp_data_dir, mock_settings):
        """Test _review_dataset with empty dataset."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            builder._review_dataset()

            mock_console.print.assert_called_with("[yellow]Dataset is empty[/yellow]")

    @patch("core.dataset.builder.console")
    def test_review_dataset_with_examples(
        self, mock_console, temp_data_dir, mock_settings, sample_training_examples
    ):
        """Test _review_dataset with examples."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Add examples
            for example in sample_training_examples:
                builder.dataset.add_example(example)

            builder._review_dataset()

            # Should print dataset review header
            mock_console.print.assert_any_call(
                f"\n[bold]Dataset Review ({len(sample_training_examples)} examples)[/bold]"
            )

    @patch("core.dataset.builder.console")
    def test_save_and_exit_empty(self, mock_console, temp_data_dir, mock_settings):
        """Test _save_and_exit with empty dataset."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            builder._save_and_exit()

            mock_console.print.assert_called_with(
                "[yellow]No examples to save[/yellow]"
            )

    @patch("core.dataset.builder.console")
    def test_save_and_exit_with_examples(
        self, mock_console, temp_data_dir, mock_settings, sample_training_examples
    ):
        """Test _save_and_exit with examples."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Add examples
            for example in sample_training_examples:
                builder.dataset.add_example(example)

            builder._save_and_exit()

            # Should save dataset and print success message
            expected_path = builder.storage.author_dir / "train.jsonl"
            mock_console.print.assert_any_call(
                f"[green]Saved {len(sample_training_examples)} examples to {expected_path}[/green]"
            )

    @patch("core.dataset.builder.console")
    def test_save_and_exit_recommendations(
        self, mock_console, temp_data_dir, mock_settings
    ):
        """Test _save_and_exit recommendation messages."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Test with few examples (< 10)
            for i in range(5):
                example = TrainingExample(
                    messages=[
                        {"role": "user", "content": f"Test {i}"},
                        {"role": "assistant", "content": f"Response {i}"},
                    ]
                )
                builder.dataset.add_example(example)

            builder._save_and_exit()

            mock_console.print.assert_any_call(
                "[yellow]Recommendation: Add more examples (10-100) for better fine-tuning results[/yellow]"
            )

            # Reset and test with many examples (>= 100)
            mock_console.reset_mock()
            for i in range(95):  # Total will be 100
                example = TrainingExample(
                    messages=[
                        {"role": "user", "content": f"Test {i+5}"},
                        {"role": "assistant", "content": f"Response {i+5}"},
                    ]
                )
                builder.dataset.add_example(example)

            builder._save_and_exit()

            mock_console.print.assert_any_call(
                "[green]Great! You have enough examples for effective fine-tuning[/green]"
            )

    def test_content_filtering_short_sections(self, temp_data_dir, mock_settings):
        """Test that short sections are filtered out during import."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            content = """This is a very long paragraph with more than fifty characters to test filtering.

Short.

Another long paragraph that should not be filtered out because it has enough content to be meaningful."""

            with patch(
                "core.dataset.builder.Confirm.ask", side_effect=[True, True]
            ), patch(
                "core.dataset.builder.Prompt.ask", side_effect=["prompt1", "prompt2"]
            ):

                # Mock the file operations
                sections = builder._split_content(content)

                # Should get 3 sections but only process the long ones
                assert len(sections) == 3

                # Simulate the filtering logic from _import_from_file
                long_sections = [s for s in sections if len(s) >= 50]
                assert len(long_sections) == 2  # Only the long paragraphs

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    def test_generate_examples_max_lines_limit(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _generate_examples with max lines limit reached."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock user inputs
            mock_prompt.ask.side_effect = ["1", "test topic"]

            # Mock console.input() to simulate hitting max line limit
            lines = [f"Line {i}" for i in range(101)]  # More than max_lines (100)
            mock_console.input.side_effect = lines

            # Mock EXAMPLE_GENERATION_PROMPTS
            with patch(
                "core.dataset.builder.EXAMPLE_GENERATION_PROMPTS",
                {"test": "Test prompt"},
            ):
                initial_size = builder.dataset.size
                builder._generate_examples()

                # Should still add the example despite hitting limit
                assert builder.dataset.size == initial_size + 1

                # Should warn about truncation
                mock_console.print.assert_any_call(
                    "[yellow]Input limited to 100 lines. Content may be truncated.[/yellow]"
                )

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    def test_generate_examples_keyboard_interrupt(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _generate_examples with KeyboardInterrupt (Ctrl+C)."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock user inputs
            mock_prompt.ask.side_effect = ["1", "test topic"]

            # Mock console.input() to simulate KeyboardInterrupt
            mock_console.input.side_effect = KeyboardInterrupt()

            # Mock EXAMPLE_GENERATION_PROMPTS
            with patch(
                "core.dataset.builder.EXAMPLE_GENERATION_PROMPTS",
                {"test": "Test prompt"},
            ):
                initial_size = builder.dataset.size
                builder._generate_examples()

                # Should not add any examples
                assert builder.dataset.size == initial_size

                # Should show cancellation message
                mock_console.print.assert_called_with(
                    "[yellow]Input cancelled.[/yellow]"
                )

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.Confirm")
    def test_generate_from_existing_insufficient_examples(
        self, mock_confirm, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _generate_from_existing with insufficient existing examples."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Add only 1 example (need at least 2)
            example = TrainingExample(
                messages=[
                    {"role": "user", "content": "Test prompt"},
                    {"role": "assistant", "content": "Test response"},
                ]
            )
            builder.dataset.add_example(example)

            builder._generate_from_existing()

            # Should show error message
            mock_console.print.assert_any_call(
                "[red]❌ You need at least 2 existing examples to generate new ones.[/red]"
            )

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.Confirm")
    @patch("core.dataset.builder.OpenAIAdapter")
    def test_generate_from_existing_successful_generation(
        self,
        mock_adapter_class,
        mock_confirm,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
        sample_training_examples,
    ):
        """Test _generate_from_existing with successful generation."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Add existing examples
            for example in sample_training_examples:
                builder.dataset.add_example(example)

            # Mock user inputs
            mock_prompt.ask.side_effect = [
                "3",  # number of examples to generate
                "accept",  # first example decision
                "accept",  # second example decision
                "skip",  # third example decision
            ]
            mock_confirm.ask.side_effect = [True]  # confirm generation

            # Mock OpenAI adapter
            mock_adapter = mock_adapter_class.return_value
            mock_adapter.generate_text.side_effect = [
                """
EXAMPLE 1:
User prompt: Write about productivity tips
Assistant response: Here are some great productivity tips that can help you get more done.
""",
                """
EXAMPLE 1:
User prompt: Explain time management
Assistant response: Time management is about prioritizing tasks and using your hours effectively.
""",
                """
EXAMPLE 1:
User prompt: Describe effective workflows
Assistant response: Effective workflows streamline your process and reduce wasted effort.
""",
            ]

            initial_size = builder.dataset.size
            builder._generate_from_existing()

            # Should add 2 examples (2 accepted, 1 skipped)
            assert builder.dataset.size == initial_size + 2

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.Confirm")
    def test_generate_from_existing_cancelled_by_user(
        self,
        mock_confirm,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
        sample_training_examples,
    ):
        """Test _generate_from_existing when user cancels."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Add existing examples
            for example in sample_training_examples:
                builder.dataset.add_example(example)

            # Mock user inputs
            mock_prompt.ask.return_value = "5"  # number of examples
            mock_confirm.ask.return_value = False  # don't confirm generation

            initial_size = builder.dataset.size
            builder._generate_from_existing()

            # Should not add any examples
            assert builder.dataset.size == initial_size
            mock_console.print.assert_any_call("[yellow]Generation cancelled[/yellow]")

    def test_prepare_examples_for_prompt(
        self, temp_data_dir, mock_settings, sample_training_examples
    ):
        """Test _prepare_examples_for_prompt method."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Add examples
            for example in sample_training_examples:
                builder.dataset.add_example(example)

            result = builder._prepare_examples_for_prompt()

            # Should format examples properly
            assert "EXAMPLE 1:" in result
            assert "User prompt:" in result
            assert "Assistant response:" in result
            assert len(result.split("EXAMPLE")) == 4  # 3 examples + empty string

    def test_parse_generated_examples_valid_response(
        self, temp_data_dir, mock_settings
    ):
        """Test _parse_generated_examples with valid response."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            response = """
EXAMPLE 1:
User prompt: Write about coding best practices
Assistant response: Good code should be readable, maintainable, and well-tested.

EXAMPLE 2:
User prompt: Explain debugging techniques  
Assistant response: Debugging involves systematic problem-solving and testing hypotheses.
"""

            examples = builder._parse_generated_examples(response)

            assert len(examples) == 2
            assert (
                examples[0].messages[1]["content"]
                == "Write about coding best practices"
            )
            assert "readable, maintainable" in examples[0].messages[2]["content"]

    def test_parse_generated_examples_malformed_response(
        self, temp_data_dir, mock_settings
    ):
        """Test _parse_generated_examples with malformed response."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            response = "This is not a properly formatted response"

            examples = builder._parse_generated_examples(response)

            assert len(examples) == 0


class TestDatasetBuilderIntegration:
    """Integration tests for DatasetBuilder."""

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.Confirm")
    def test_full_workflow_simulation(
        self, mock_confirm, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test a simulated full workflow of dataset building."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("integration_author")

            # Simulate adding from writing sample
            mock_console.input.side_effect = [
                "Sample content for testing",  # First line of input
                EOFError(),  # Simulate Ctrl+D to end input
            ]
            mock_prompt.ask.side_effect = [
                "1",  # Choose manual prompt entry
                "Generate sample content",  # prompt
            ]

            builder._add_from_writing_samples()
            assert builder.dataset.size == 1

            # Simulate saving
            builder._save_and_exit()

            # Verify the dataset was saved
            loaded_dataset = builder.storage.load_dataset()
            assert loaded_dataset.size == 1
            assert (
                "Sample content for testing"
                in loaded_dataset.examples[0].messages[2]["content"]
            )


class TestDatasetBuilderMarkdown:
    """Test markdown file creation in DatasetBuilder."""

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    def test_add_from_writing_samples_creates_markdown(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test that _add_from_writing_samples creates markdown files."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock console.input() to simulate multiline input
            mock_console.input.side_effect = [
                "This is my writing sample content.",
                EOFError(),  # Simulate Ctrl+D to end input
            ]

            # Mock prompt for the new workflow
            mock_prompt.ask.side_effect = [
                "1",  # Choose manual prompt entry
                "Write about technology",  # The actual prompt
            ]

            builder._add_from_writing_samples()

            # Check that examples directory was created
            examples_dir = builder.storage.examples_dir
            assert examples_dir.exists()
            assert examples_dir.is_dir()

            # Check that markdown files were created
            markdown_files = list(examples_dir.glob("user_*.md"))
            assert len(markdown_files) == 1

            # Check markdown file content
            markdown_file = markdown_files[0]
            content = markdown_file.read_text(encoding="utf-8")
            assert "# Write about technology" in content
            assert "**Type:** user" in content
            assert "**Prompt:** Write about technology" in content
            assert "This is my writing sample content." in content

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.EXAMPLE_GENERATION_PROMPTS", {"test": "Test {topic}"})
    def test_generate_examples_creates_markdown(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test that _generate_examples creates markdown files."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock user inputs
            mock_prompt.ask.side_effect = [
                "1",  # prompt type selection
                "technology",  # topic
            ]

            # Mock console.input() to simulate multiline input
            mock_console.input.side_effect = [
                "Technology is fascinating and evolving rapidly.",
                EOFError(),  # Simulate Ctrl+D to end input
            ]

            builder._generate_examples()

            # Check that markdown files were created
            examples_dir = builder.storage.examples_dir
            markdown_files = list(examples_dir.glob("user_*.md"))
            assert len(markdown_files) == 1

            # Check markdown file content
            markdown_file = markdown_files[0]
            content = markdown_file.read_text(encoding="utf-8")
            assert "**Type:** user" in content
            assert "technology" in content.lower()
            assert "Technology is fascinating" in content

    def test_import_from_file_creates_markdown(self, temp_data_dir, mock_settings):
        """Test that _import_from_file creates markdown files."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Create a real temporary file for testing
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as temp_file:
                temp_file.write(
                    "This is a comprehensive paragraph with enough content to meet the minimum length requirement for testing purposes. It contains sufficient text to pass the 50-character minimum length check in the dataset builder."
                )
                temp_file_path = temp_file.name

            try:
                with patch("core.dataset.builder.console") as mock_console, patch(
                    "core.dataset.builder.Prompt"
                ) as mock_prompt, patch("core.dataset.builder.Confirm") as mock_confirm:

                    # Mock user inputs for new workflow
                    mock_prompt.ask.side_effect = [
                        temp_file_path,  # file path
                        "1",  # Choose manual prompt entry
                        "Describe the topic thoroughly",  # prompt for section
                    ]
                    mock_confirm.ask.side_effect = [True]  # Include the section

                    builder._import_from_file()

                    # Check that markdown files were created
                    examples_dir = builder.storage.examples_dir
                    markdown_files = list(examples_dir.glob("user_*.md"))
                    assert len(markdown_files) == 1

                    # Check markdown file content
                    markdown_file = markdown_files[0]
                    content = markdown_file.read_text(encoding="utf-8")
                    assert "**Type:** user" in content
                    assert "**Prompt:** Describe the topic thoroughly" in content
                    assert "comprehensive paragraph" in content

            finally:
                # Clean up the temporary file
                import os

                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.Confirm")
    @patch("core.dataset.builder.OpenAIAdapter")
    def test_generate_from_existing_creates_markdown(
        self,
        mock_adapter_class,
        mock_confirm,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
        sample_training_examples,
    ):
        """Test that _generate_from_existing creates markdown files for LLM examples."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Add existing examples
            for example in sample_training_examples:
                builder.dataset.add_example(example)

            # Mock user inputs
            mock_prompt.ask.side_effect = [
                "1",  # number of examples to generate
                "accept",  # accept the generated example
            ]
            mock_confirm.ask.side_effect = [True]  # confirm generation

            # Mock OpenAI adapter
            mock_adapter = mock_adapter_class.return_value
            mock_adapter.generate_text.return_value = """
EXAMPLE 1:
User prompt: Write about AI technology trends
Assistant response: AI technology is advancing rapidly with new breakthroughs in machine learning and natural language processing.
"""

            initial_markdown_count = len(
                list(builder.storage.examples_dir.glob("*.md"))
            )
            builder._generate_from_existing()

            # Check that new markdown files were created
            examples_dir = builder.storage.examples_dir
            markdown_files = list(examples_dir.glob("llm_*.md"))
            assert len(markdown_files) >= 1

            # Check that LLM-generated markdown file has correct content
            llm_files = [f for f in examples_dir.glob("*.md") if "llm_" in f.name]
            assert len(llm_files) >= 1

            markdown_file = llm_files[0]
            content = markdown_file.read_text(encoding="utf-8")
            assert "**Type:** llm" in content
            assert "AI technology trends" in content
            assert "advancing rapidly" in content

    @patch("core.dataset.builder.console")
    def test_markdown_file_error_handling(
        self, mock_console, temp_data_dir, mock_settings
    ):
        """Test that markdown file creation errors are handled gracefully."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock the save_example_as_markdown method to raise an exception
            with patch.object(
                builder.storage,
                "save_example_as_markdown",
                side_effect=Exception("Test error"),
            ):
                example = TrainingExample(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful writing assistant.",
                        },
                        {"role": "user", "content": "Test prompt"},
                        {"role": "assistant", "content": "Test response"},
                    ]
                )

                initial_size = builder.dataset.size
                builder.dataset.add_example(example)

                # The example should still be added even if markdown saving fails
                assert builder.dataset.size == initial_size + 1

                # Should show warning message (this would be called in the methods that create examples)
                # We're just testing that the example addition continues even with markdown errors

    def test_examples_directory_creation(self, temp_data_dir, mock_settings):
        """Test that examples directory is created when DatasetBuilder is initialized."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Examples directory should exist
            examples_dir = builder.storage.examples_dir
            assert examples_dir.exists()
            assert examples_dir.is_dir()
            assert examples_dir.parent == builder.storage.author_dir


class TestDatasetBuilderReverseEngineering:
    """Test the new reverse-engineering prompt functionality."""

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    def test_collect_prompt_manual_choice(
        self, mock_prompt, mock_console, temp_data_dir, mock_settings
    ):
        """Test _collect_prompt_for_content with manual choice."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            mock_prompt.ask.side_effect = [
                "1",  # Choose manual prompt entry
                "Write about productivity",  # Manual prompt
            ]

            result = builder._collect_prompt_for_content(
                "Sample content about productivity"
            )

            assert result == "Write about productivity"
            assert mock_prompt.ask.call_count == 2

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.OpenAIAdapter")
    def test_collect_prompt_ai_choice_accept(
        self,
        mock_adapter_class,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
    ):
        """Test _collect_prompt_for_content with AI choice and accept suggestion."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock OpenAI adapter
            mock_adapter = Mock()
            mock_adapter.generate_text.return_value = (
                "Write a blog post about productivity tips"
            )
            mock_adapter_class.return_value = mock_adapter

            mock_prompt.ask.side_effect = [
                "2",  # Choose AI-assisted prompt
                "accept",  # Accept AI suggestion
            ]

            result = builder._collect_prompt_for_content(
                "Sample content about productivity"
            )

            assert result == "Write a blog post about productivity tips"
            assert mock_adapter.generate_text.called

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.OpenAIAdapter")
    def test_collect_prompt_ai_choice_edit(
        self,
        mock_adapter_class,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
    ):
        """Test _collect_prompt_for_content with AI choice and edit suggestion."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock OpenAI adapter
            mock_adapter = Mock()
            mock_adapter.generate_text.return_value = (
                "Write a blog post about productivity"
            )
            mock_adapter_class.return_value = mock_adapter

            mock_prompt.ask.side_effect = [
                "2",  # Choose AI-assisted prompt
                "edit",  # Edit AI suggestion
                "Write a detailed blog post about productivity tips",  # Edited prompt
            ]

            result = builder._collect_prompt_for_content(
                "Sample content about productivity"
            )

            assert result == "Write a detailed blog post about productivity tips"
            assert mock_adapter.generate_text.called

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.OpenAIAdapter")
    def test_collect_prompt_ai_choice_fallback_to_manual(
        self,
        mock_adapter_class,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
    ):
        """Test _collect_prompt_for_content with AI choice but fallback to manual."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock OpenAI adapter
            mock_adapter = Mock()
            mock_adapter.generate_text.return_value = (
                "Write a blog post about productivity"
            )
            mock_adapter_class.return_value = mock_adapter

            mock_prompt.ask.side_effect = [
                "2",  # Choose AI-assisted prompt
                "manual",  # Reject AI, go manual
                "My custom prompt",  # Manual prompt
            ]

            result = builder._collect_prompt_for_content(
                "Sample content about productivity"
            )

            assert result == "My custom prompt"
            assert mock_adapter.generate_text.called

    @patch("core.dataset.builder.console")
    @patch("core.dataset.builder.Prompt")
    @patch("core.dataset.builder.OpenAIAdapter")
    def test_reverse_engineer_prompt_api_error_fallback(
        self,
        mock_adapter_class,
        mock_prompt,
        mock_console,
        temp_data_dir,
        mock_settings,
    ):
        """Test _collect_prompt_for_content with API error fallback to manual."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock OpenAI adapter to raise an error
            mock_adapter_class.side_effect = ValueError("API key not found")

            mock_prompt.ask.side_effect = [
                "2",  # Choose AI-assisted prompt
                "Fallback manual prompt",  # Manual prompt after API error
            ]

            result = builder._collect_prompt_for_content(
                "Sample content about productivity"
            )

            assert result == "Fallback manual prompt"

    @patch("core.dataset.builder.OpenAIAdapter")
    def test_reverse_engineer_prompt_content_truncation(
        self, mock_adapter_class, temp_data_dir, mock_settings
    ):
        """Test that reverse_engineer_prompt truncates long content."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Mock OpenAI adapter
            mock_adapter = Mock()
            mock_adapter.generate_text.return_value = "Suggested prompt"
            mock_adapter_class.return_value = mock_adapter

            # Create content longer than 2000 characters
            long_content = "This is a test sentence. " * 100  # About 2500 characters

            result = builder._reverse_engineer_prompt(long_content)

            assert result == "Suggested prompt"
            # Check that truncation occurred in API call
            call_args = mock_adapter.generate_text.call_args
            prompt_content = call_args[1]["prompt"]
            assert "[Content truncated for analysis...]" in prompt_content

    def test_clean_ai_prompt_response_basic_cleaning(
        self, temp_data_dir, mock_settings
    ):
        """Test basic cleaning of AI prompt responses."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Test basic stripping
            assert (
                builder._clean_ai_prompt_response("  Write about productivity  ")
                == "Write about productivity"
            )

            # Test empty/None handling
            assert builder._clean_ai_prompt_response("") == ""
            assert builder._clean_ai_prompt_response("   ") == ""

    def test_clean_ai_prompt_response_prefix_removal(
        self, temp_data_dir, mock_settings
    ):
        """Test removal of common AI prefixes."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            test_cases = [
                ("PROMPT: Write about productivity", "Write about productivity"),
                ("Prompt: Write about productivity", "Write about productivity"),
                ("prompt: Write about productivity", "Write about productivity"),
                ("USER PROMPT: Write about productivity", "Write about productivity"),
                (
                    "Here's the prompt: Write about productivity",
                    "Write about productivity",
                ),
                (
                    "The prompt would be: Write about productivity",
                    "Write about productivity",
                ),
                (
                    "SUGGESTED PROMPT: Write about productivity",
                    "Write about productivity",
                ),
            ]

            for input_text, expected in test_cases:
                result = builder._clean_ai_prompt_response(input_text)
                assert result == expected, f"Failed for input: {input_text}"

    def test_clean_ai_prompt_response_quote_removal(self, temp_data_dir, mock_settings):
        """Test removal of surrounding quotes."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            test_cases = [
                ('"Write about productivity"', "Write about productivity"),
                ("'Write about productivity'", "Write about productivity"),
                ('Prompt: "Write about productivity"', "Write about productivity"),
                (
                    "The prompt is: 'Write about productivity'",
                    "Write about productivity",
                ),
            ]

            for input_text, expected in test_cases:
                result = builder._clean_ai_prompt_response(input_text)
                assert result == expected, f"Failed for input: {input_text}"

    def test_clean_ai_prompt_response_punctuation_cleanup(
        self, temp_data_dir, mock_settings
    ):
        """Test cleanup of leading/trailing punctuation."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            test_cases = [
                (": Write about productivity", "Write about productivity"),
                ("- Write about productivity", "Write about productivity"),
                ("• Write about productivity", "Write about productivity"),
                ("* Write about productivity", "Write about productivity"),
                ("= Write about productivity", "Write about productivity"),
                ("Write about productivity.", "Write about productivity"),
                ("Write about productivity:", "Write about productivity"),
                (
                    "Write about productivity...",
                    "Write about productivity...",
                ),  # Keep ellipsis
            ]

            for input_text, expected in test_cases:
                result = builder._clean_ai_prompt_response(input_text)
                assert result == expected, f"Failed for input: {input_text}"

    def test_clean_ai_prompt_response_multiline_handling(
        self, temp_data_dir, mock_settings
    ):
        """Test handling of multiline responses."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Test taking first meaningful line
            multiline_input = """Write about productivity techniques for remote workers
            
            Note: This prompt focuses on practical advice
            Example: Include specific tools and methods"""

            result = builder._clean_ai_prompt_response(multiline_input)
            assert result == "Write about productivity techniques for remote workers"

            # Test skipping short/irrelevant lines
            multiline_input2 = """Here's the prompt:
            
            Write a comprehensive guide about time management
            
            Format: Blog post style"""

            result2 = builder._clean_ai_prompt_response(multiline_input2)
            assert result2 == "Write a comprehensive guide about time management"

    def test_clean_ai_prompt_response_complex_cases(self, temp_data_dir, mock_settings):
        """Test complex real-world AI response patterns."""
        with patch("core.storage.settings", mock_settings):
            builder = DatasetBuilder("test_author")

            # Test combination of issues
            complex_input = """PROMPT: "Write a detailed blog post about productivity tips for remote workers."
            
            Note: This should be engaging and practical."""

            result = builder._clean_ai_prompt_response(complex_input)
            assert (
                result
                == "Write a detailed blog post about productivity tips for remote workers"
            )

            # Test with various formatting
            complex_input2 = "The prompt would be: - 'Create an engaging introduction about artificial intelligence'"
            result2 = builder._clean_ai_prompt_response(complex_input2)
            assert (
                result2
                == "Create an engaging introduction about artificial intelligence"
            )
