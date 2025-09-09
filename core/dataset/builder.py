import re
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from core.adapters.openai_adapter import OpenAIAdapter
from core.config import settings
from core.models import Dataset, TrainingExample
from core.prompts.templates import (
    DATASET_BUILDING_PROMPTS,
    EXAMPLE_GENERATION_FROM_EXISTING_TEMPLATE,
    EXAMPLE_GENERATION_PROMPTS,
    REVERSE_ENGINEER_PROMPT_TEMPLATE,
)
from core.storage import AuthorStorage

console = Console()


class DatasetBuilder:
    def __init__(self, author_id: str) -> None:
        self.author_id = author_id
        self.storage = AuthorStorage(author_id)
        self.dataset = self.storage.load_dataset() or Dataset(author_id=author_id)

    def interactive_build(self) -> None:
        console.print("\n[bold blue]Dataset Builder[/bold blue]")
        console.print("Let's build a training dataset for your writing style!")

        while True:
            console.print(
                f"\n[green]Current dataset size: {self.dataset.size} examples[/green]"
            )

            choices = [
                "1. Add examples from writing samples",
                "2. Generate examples from prompts",
                "3. Import from text file",
                "4. Review current dataset",
                "5. Generate more examples from existing examples",
                "6. Save and exit",
            ]

            for choice in choices:
                console.print(choice)

            action = Prompt.ask(
                "\nWhat would you like to do?", choices=["1", "2", "3", "4", "5", "6"]
            )

            if action == "1":
                self._add_from_writing_samples()
            elif action == "2":
                self._generate_examples()
            elif action == "3":
                self._import_from_file()
            elif action == "4":
                self._review_dataset()
            elif action == "5":
                self._generate_from_existing()
            elif action == "6":
                self._save_and_exit()
                break

    def _add_from_writing_samples(self) -> None:
        console.print(
            Panel(DATASET_BUILDING_PROMPTS["writing_sample"], title="Writing Sample")
        )

        # Collect multiline input using console.input() in a loop
        console.print("[dim]Press Ctrl+D (or Ctrl+Z on Windows) when finished[/dim]")
        sample_lines = []
        max_lines = 100  # Safety limit to prevent infinite loops
        line_count = 0

        try:
            while line_count < max_lines:
                line = console.input()
                sample_lines.append(line)
                line_count += 1
        except EOFError:
            # User pressed Ctrl+D (or Ctrl+Z on Windows) to finish input
            pass
        except KeyboardInterrupt:
            # User pressed Ctrl+C to cancel
            console.print("[yellow]Input cancelled.[/yellow]")
            return

        sample = "\n".join(sample_lines)

        # Warn if max lines reached
        if line_count >= max_lines:
            console.print(
                f"[yellow]Input limited to {max_lines} lines. Content may be truncated.[/yellow]"
            )
        if not sample.strip():
            console.print("[red]No sample provided[/red]")
            return

        prompt = self._collect_prompt_for_content(sample)
        if not prompt or not prompt.strip():
            console.print("[red]No prompt provided[/red]")
            return

        example = TrainingExample(
            messages=[
                {"role": "system", "content": "You are a helpful writing assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": sample},
            ]
        )

        # Save as markdown file
        try:
            markdown_path = self.storage.save_example_as_markdown(
                prompt=prompt, response=sample, example_type="user"
            )
            console.print(f"[dim]Saved markdown: {markdown_path.name}[/dim]")
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not save markdown file: {e}[/yellow]"
            )

        self.dataset.add_example(example)
        console.print(
            f"[green]Added example! Dataset now has {self.dataset.size} examples[/green]"
        )

    def _generate_examples(self) -> None:
        console.print("\n[bold]Generate Training Examples[/bold]")
        console.print("Choose a prompt type:")

        prompt_types = list(EXAMPLE_GENERATION_PROMPTS.keys())
        for i, ptype in enumerate(prompt_types, 1):
            console.print(f"{i}. {ptype.replace('_', ' ').title()}")

        choice = Prompt.ask(
            "Select prompt type",
            choices=[str(i) for i in range(1, len(prompt_types) + 1)],
        )
        prompt_type = prompt_types[int(choice) - 1]

        topic = Prompt.ask("What topic should this example be about?")

        prompt_template = EXAMPLE_GENERATION_PROMPTS[prompt_type]
        user_prompt = prompt_template.format(
            topic=topic, tone="professional", length_preference="medium"
        )

        console.print(f"\n[yellow]Prompt: {user_prompt}[/yellow]")
        console.print("[blue]Now write your response in your style:[/blue]")
        console.print("[dim]Press Ctrl+D (or Ctrl+Z on Windows) when finished[/dim]")

        # Collect multiline input using console.input() in a loop
        response_lines = []
        max_lines = 100  # Safety limit to prevent infinite loops
        line_count = 0

        try:
            while line_count < max_lines:
                line = console.input()
                response_lines.append(line)
                line_count += 1
        except EOFError:
            # User pressed Ctrl+D (or Ctrl+Z on Windows) to finish input
            pass
        except KeyboardInterrupt:
            # User pressed Ctrl+C to cancel
            console.print("[yellow]Input cancelled.[/yellow]")
            return

        response = "\n".join(response_lines)
        if not response.strip():
            console.print("[red]No response provided[/red]")
            return

        # Warn if max lines reached
        if line_count >= max_lines:
            console.print(
                f"[yellow]Input limited to {max_lines} lines. Content may be truncated.[/yellow]"
            )

        example = TrainingExample(
            messages=[
                {"role": "system", "content": "You are a helpful writing assistant."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
        )

        # Save as markdown file
        try:
            markdown_path = self.storage.save_example_as_markdown(
                prompt=user_prompt, response=response, example_type="user"
            )
            console.print(f"[dim]Saved markdown: {markdown_path.name}[/dim]")
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not save markdown file: {e}[/yellow]"
            )

        self.dataset.add_example(example)
        console.print(
            f"[green]Added example! Dataset now has {self.dataset.size} examples[/green]"
        )

    def _generate_from_existing(self) -> None:
        """Generate new training examples based on existing examples using OpenAI API."""
        console.print(
            Panel(
                "[bold blue]Generate Examples from Existing[/bold blue]\n\n"
                "This will use AI to create new examples based on your existing ones, "
                "maintaining the same writing style and tone.",
                title="🤖 AI Generation",
            )
        )

        # Check if we have enough existing examples
        if self.dataset.size < 2:
            console.print(
                "[red]❌ You need at least 2 existing examples to generate new ones.[/red]"
            )
            console.print("Please add some examples first using options 1, 2, or 3.")
            return

        # Show current dataset size
        console.print(f"[green]Current dataset: {self.dataset.size} examples[/green]")

        # Get number of examples to generate
        while True:
            try:
                count = int(
                    Prompt.ask(
                        "How many new examples would you like to generate?", default="5"
                    )
                )
                if 1 <= count <= 20:
                    break
                else:
                    console.print(
                        "[yellow]Please enter a number between 1 and 20[/yellow]"
                    )
            except ValueError:
                console.print("[yellow]Please enter a valid number[/yellow]")

        # Estimate costs and confirm
        estimated_cost = count * 0.002  # Rough estimate: $0.002 per generation
        console.print(
            f"[yellow]💰 Estimated cost: ${estimated_cost:.3f} (OpenAI API)[/yellow]"
        )

        if not Confirm.ask("Continue with generation?"):
            console.print("[yellow]Generation cancelled[/yellow]")
            return

        # Initialize OpenAI adapter
        try:
            adapter = OpenAIAdapter()
        except ValueError as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            return

        # Prepare existing examples for the prompt
        sample_examples = self._prepare_examples_for_prompt()

        console.print(f"\n[blue]🔄 Generating {count} new examples...[/blue]")

        generated_examples = []
        for i in range(count):
            try:
                console.print(f"[dim]Generating example {i+1}/{count}...[/dim]")

                # Create the generation prompt
                generation_prompt = EXAMPLE_GENERATION_FROM_EXISTING_TEMPLATE.format(
                    existing_examples=sample_examples,
                    count=1,  # Generate one at a time for better control
                )

                console.print(
                    f"[dim]Generation prompt length: {len(generation_prompt)}[/dim]"
                )
                console.print(
                    f"[dim]Sample examples length: {len(sample_examples)}[/dim]"
                )

                # Generate new example using OpenAI
                try:
                    response = adapter.generate_text(
                        model_id=settings.get_default_model("openai"),
                        prompt=generation_prompt,
                        max_completion_tokens=800,
                    )
                    console.print(
                        f"[dim]Generated response length: {len(response) if response else 0}[/dim]"
                    )
                except Exception as api_error:
                    console.print(
                        f"[red]❌ API Error generating example {i+1}: {str(api_error)}[/red]"
                    )
                    if not Confirm.ask("Continue with remaining examples?"):
                        break
                    continue

                # Parse the generated response
                parsed_examples = self._parse_generated_examples(response)

                if parsed_examples:
                    generated_examples.extend(parsed_examples)
                else:
                    console.print(
                        f"[yellow]⚠️  Failed to parse example {i+1}, skipping...[/yellow]"
                    )

            except Exception as e:
                console.print(f"[red]❌ Error generating example {i+1}: {str(e)}[/red]")
                if not Confirm.ask("Continue with remaining examples?"):
                    break

        if not generated_examples:
            console.print("[red]❌ No examples were successfully generated[/red]")
            return

        # Review and approve generated examples
        console.print(
            f"\n[green]✅ Generated {len(generated_examples)} examples[/green]"
        )
        console.print("[blue]Please review each example:[/blue]")

        approved_examples = []
        for i, example in enumerate(generated_examples):
            console.print(f"\n[cyan]--- Example {i+1} ---[/cyan]")

            # Extract user prompt and assistant response
            user_msg = next(
                (msg["content"] for msg in example.messages if msg["role"] == "user"),
                "No user message found",
            )
            assistant_msg = next(
                (
                    msg["content"]
                    for msg in example.messages
                    if msg["role"] == "assistant"
                ),
                "No assistant message found",
            )

            console.print(f"[yellow]Prompt:[/yellow] {user_msg}")
            console.print(
                f"[green]Response:[/green] {assistant_msg[:200]}{'...' if len(assistant_msg) > 200 else ''}"
            )

            # Get user decision
            action = Prompt.ask(
                "What would you like to do with this example?",
                choices=["accept", "edit", "skip"],
                default="accept",
            )

            if action == "accept":
                approved_examples.append(example)
                console.print("[green]✅ Example accepted[/green]")

            elif action == "edit":
                # Allow user to edit the response
                console.print("[blue]Edit the assistant response:[/blue]")
                console.print(f"[dim]Current: {assistant_msg}[/dim]")

                new_response = Prompt.ask(
                    "New response (press Enter to keep current)", default=assistant_msg
                )

                if new_response.strip():
                    # Create edited example
                    edited_example = TrainingExample(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful writing assistant.",
                            },
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": new_response},
                        ]
                    )
                    approved_examples.append(edited_example)
                    console.print("[green]✅ Edited example accepted[/green]")
                else:
                    console.print(
                        "[yellow]⚠️  No changes made, skipping example[/yellow]"
                    )

            else:  # skip
                console.print("[yellow]⚠️  Example skipped[/yellow]")

        # Add approved examples to dataset
        if approved_examples:
            for example in approved_examples:
                # Extract prompt and response for markdown saving
                user_msg = next(
                    (
                        msg["content"]
                        for msg in example.messages
                        if msg["role"] == "user"
                    ),
                    "No user message found",
                )
                assistant_msg = next(
                    (
                        msg["content"]
                        for msg in example.messages
                        if msg["role"] == "assistant"
                    ),
                    "No assistant message found",
                )

                # Save as markdown file
                try:
                    markdown_path = self.storage.save_example_as_markdown(
                        prompt=user_msg, response=assistant_msg, example_type="llm"
                    )
                    console.print(f"[dim]Saved markdown: {markdown_path.name}[/dim]")
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not save markdown file: {e}[/yellow]"
                    )

                self.dataset.add_example(example)

            console.print(
                f"\n[green]🎉 Added {len(approved_examples)} new examples! "
                f"Dataset now has {self.dataset.size} examples[/green]"
            )
        else:
            console.print("[yellow]No examples were approved[/yellow]")

    def _reverse_engineer_prompt(self, content: str) -> Optional[str]:
        """Use AI to suggest a prompt that would generate the given content."""
        try:
            adapter = OpenAIAdapter()
            # Limit content length to prevent excessive API costs
            truncated_content = content[:2000]
            if len(content) > 2000:
                truncated_content += "\n\n[Content truncated for analysis...]"

            prompt = REVERSE_ENGINEER_PROMPT_TEMPLATE.format(content=truncated_content)

            response = adapter.generate_text(
                model_id=settings.get_default_model("openai"),
                prompt=prompt,
                max_completion_tokens=150,  # Prompts should be concise
            )

            return self._clean_ai_prompt_response(response) if response else None

        except Exception as e:
            console.print(
                f"[yellow]⚠️  Could not generate prompt suggestion: {e}[/yellow]"
            )
            return None

    def _clean_ai_prompt_response(self, response: str) -> str:
        """Clean up AI-generated prompt response to extract just the prompt text."""
        if not response:
            return ""

        # Start with basic cleaning
        cleaned = response.strip()

        # Remove common prefixes that AI might include
        prefixes_to_remove = [
            "PROMPT:",
            "Prompt:",
            "prompt:",
            "USER PROMPT:",
            "User prompt:",
            "user prompt:",
            "SUGGESTED PROMPT:",
            "Suggested prompt:",
            "suggested prompt:",
            "Here's the prompt:",
            "Here is the prompt:",
            "The prompt is:",
            "The prompt would be:",
        ]

        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()
                break

        # Remove any remaining leading/trailing punctuation that doesn't belong
        while cleaned.startswith((":", "-", "•", "*", "=")) and len(cleaned) > 1:
            cleaned = cleaned[1:].strip()

        # Handle cases where AI includes multiple lines - take first meaningful line FIRST
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if lines:
            # Take the first substantial line (>5 characters)
            for line in lines:
                if len(line) > 5 and not line.lower().startswith(
                    ("note:", "example:", "format:")
                ):
                    cleaned = line
                    break

        # Remove surrounding quotes (single or double) - after getting the main line
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()

        # Final cleanup
        cleaned = cleaned.strip()

        # Ensure it doesn't end with unnecessary punctuation
        if cleaned.endswith((".", ":")) and not cleaned.endswith("..."):
            cleaned = cleaned[:-1].strip()

        return cleaned

    def _collect_prompt_for_content(self, content: str) -> Optional[str]:
        """Collect a prompt for given content, offering manual or AI-assisted options."""

        console.print("\n[yellow]💡 Prompt Options:[/yellow]")
        console.print("1. Write your own prompt (free)")
        console.print("2. Get AI-suggested prompt (~$0.001)")

        choice = Prompt.ask(
            "How would you like to create the prompt?", choices=["1", "2"], default="1"
        )

        if choice == "2":
            # AI-assisted prompt
            console.print("[blue]🤖 Generating prompt suggestion...[/blue]")
            suggested_prompt = self._reverse_engineer_prompt(content)

            if suggested_prompt:
                console.print(f"\n[green]💡 AI Suggested Prompt:[/green]")
                console.print(f"[cyan]{suggested_prompt}[/cyan]")

                action = Prompt.ask(
                    "What would you like to do?",
                    choices=["accept", "edit", "manual"],
                    default="accept",
                )

                if action == "accept":
                    return suggested_prompt
                elif action == "edit":
                    return Prompt.ask("Edit the prompt", default=suggested_prompt)
                # else: fall through to manual
            else:
                console.print(
                    "[red]Could not generate prompt suggestion. Using manual entry.[/red]"
                )

        # Manual prompt entry (fallback or chosen)
        console.print(
            Panel(DATASET_BUILDING_PROMPTS["prompt_for_sample"], title="Manual Prompt")
        )
        return Prompt.ask("Your prompt")

    def _prepare_examples_for_prompt(self) -> str:
        """Prepare a sample of existing examples for the generation prompt."""
        # Select up to 3 representative examples
        sample_size = min(3, self.dataset.size)
        sample_examples = self.dataset.examples[:sample_size]

        formatted_examples = []
        for i, example in enumerate(sample_examples, 1):
            user_msg = next(
                (msg["content"] for msg in example.messages if msg["role"] == "user"),
                "",
            )
            assistant_msg = next(
                (
                    msg["content"]
                    for msg in example.messages
                    if msg["role"] == "assistant"
                ),
                "",
            )

            formatted_examples.append(
                f"EXAMPLE {i}:\n"
                f"User prompt: {user_msg}\n"
                f"Assistant response: {assistant_msg}\n"
            )

        return "\n".join(formatted_examples)

    def _parse_generated_examples(self, response: str) -> List[TrainingExample]:
        """Parse the AI-generated response into TrainingExample objects."""
        examples = []

        try:
            # Debug: Print the raw response to understand the format
            console.print(f"[dim]Raw AI response (length: {len(response)}):[/dim]")
            console.print(f"[dim]{response}[/dim]")

            # Look for the pattern: User prompt: ... Assistant response: ...
            # Try multiple patterns to handle different AI response formats
            patterns = [
                # Original pattern
                r"User prompt:\s*(.*?)\s*(?:\n|^)Assistant response:\s*(.*?)(?=(?:\n(?:User prompt:|EXAMPLE|\Z))|$)",
                # More flexible pattern
                r"User prompt:\s*(.*?)\s*Assistant response:\s*(.*?)(?=\n\n|\nUser prompt:|\nEXAMPLE|\Z)",
                # Even more flexible - just look for the two sections
                r"User prompt:\s*(.*?)\s*Assistant response:\s*(.*?)(?=\n|$)",
            ]

            matches = []
            for pattern in patterns:
                matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
                if matches:
                    console.print(f"[dim]Using pattern: {pattern[:50]}...[/dim]")
                    break

            console.print(f"[dim]Found {len(matches)} matches with regex[/dim]")

            for user_prompt, assistant_response in matches:
                user_prompt = user_prompt.strip()
                assistant_response = assistant_response.strip()

                if user_prompt and assistant_response:
                    example = TrainingExample(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful writing assistant.",
                            },
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_response},
                        ]
                    )
                    examples.append(example)
                else:
                    console.print("[dim]Skipping empty prompt or response[/dim]")

            # If no matches found, try a fallback approach
            if not matches and response.strip():
                console.print(
                    "[dim]No regex matches found, trying fallback parsing...[/dim]"
                )
                # Look for any text that might be a prompt and response
                lines = response.split("\n")
                current_prompt = None
                current_response: list[str] = []

                for line in lines:
                    line = line.strip()
                    if line.startswith("User prompt:") or line.startswith("Prompt:"):
                        if current_prompt and current_response:
                            # Save previous example
                            example = TrainingExample(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a helpful writing assistant.",
                                    },
                                    {"role": "user", "content": current_prompt},
                                    {
                                        "role": "assistant",
                                        "content": "\n".join(current_response),
                                    },
                                ]
                            )
                            examples.append(example)
                        current_prompt = line.split(":", 1)[1].strip()
                        current_response = []
                    elif line.startswith("Assistant response:") or line.startswith(
                        "Response:"
                    ):
                        current_response = [line.split(":", 1)[1].strip()]
                    elif current_response is not None and line:
                        current_response.append(line)

                # Don't forget the last example
                if current_prompt and current_response:
                    example = TrainingExample(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful writing assistant.",
                            },
                            {"role": "user", "content": current_prompt},
                            {
                                "role": "assistant",
                                "content": "\n".join(current_response),
                            },
                        ]
                    )
                    examples.append(example)

                console.print(
                    f"[dim]Fallback parsing found {len(examples)} examples[/dim]"
                )

        except Exception as e:
            console.print(f"[yellow]⚠️  Error parsing generated examples: {e}[/yellow]")

        return examples

    def _import_from_file(self) -> None:
        file_path = Prompt.ask("Enter the path to your text file")
        path = Path(file_path)

        if not path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                console.print("[red]File is empty[/red]")
                return

            sections = self._split_content(content)
            console.print(f"[green]Found {len(sections)} sections in the file[/green]")

            for i, section in enumerate(sections):
                if len(section) < 50:  # Skip very short sections
                    continue

                console.print(f"\n[yellow]Section {i+1}:[/yellow]")
                console.print(section[:200] + "..." if len(section) > 200 else section)

                if Confirm.ask("Include this section?"):
                    prompt = self._collect_prompt_for_content(section)
                    if not prompt or not prompt.strip():
                        console.print(
                            "[yellow]Skipping section - no prompt provided[/yellow]"
                        )
                        continue

                    example = TrainingExample(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful writing assistant.",
                            },
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": section},
                        ]
                    )

                    # Save as markdown file
                    try:
                        markdown_path = self.storage.save_example_as_markdown(
                            prompt=prompt, response=section, example_type="user"
                        )
                        console.print(
                            f"[dim]Saved markdown: {markdown_path.name}[/dim]"
                        )
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Could not save markdown file: {e}[/yellow]"
                        )

                    self.dataset.add_example(example)
                    console.print("[green]Added![/green]")

            console.print(
                f"[green]Import complete! Dataset now has {self.dataset.size} examples[/green]"
            )

        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")

    def _split_content(self, content: str) -> List[str]:
        # Split on common paragraph separators
        sections = re.split(r"\n\s*\n", content)
        return [section.strip() for section in sections if section.strip()]

    def _review_dataset(self) -> None:
        if self.dataset.size == 0:
            console.print("[yellow]Dataset is empty[/yellow]")
            return

        console.print(f"\n[bold]Dataset Review ({self.dataset.size} examples)[/bold]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", width=3)
        table.add_column("Prompt", max_width=40)
        table.add_column("Response Preview", max_width=50)

        for i, example in enumerate(self.dataset.examples[:10], 1):  # Show first 10
            user_msg = next(
                (msg["content"] for msg in example.messages if msg["role"] == "user"),
                "",
            )
            assistant_msg = next(
                (
                    msg["content"]
                    for msg in example.messages
                    if msg["role"] == "assistant"
                ),
                "",
            )

            prompt_preview = user_msg[:37] + "..." if len(user_msg) > 40 else user_msg
            response_preview = (
                assistant_msg[:47] + "..." if len(assistant_msg) > 50 else assistant_msg
            )

            table.add_row(str(i), prompt_preview, response_preview)

        console.print(table)

        if self.dataset.size > 10:
            console.print(f"[dim]... and {self.dataset.size - 10} more examples[/dim]")

    def _save_and_exit(self) -> None:
        if self.dataset.size == 0:
            console.print("[yellow]No examples to save[/yellow]")
            return

        self.storage.save_dataset(self.dataset)
        console.print(
            f"[green]Saved {self.dataset.size} examples to {self.storage.author_dir / 'train.jsonl'}[/green]"
        )

        if self.dataset.size < 10:
            console.print(
                "[yellow]Recommendation: Add more examples (10-100) for better fine-tuning results[/yellow]"
            )
        elif self.dataset.size >= 100:
            console.print(
                "[green]Great! You have enough examples for effective fine-tuning[/green]"
            )
