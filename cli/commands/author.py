from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from core.models import AuthorProfile, StyleGuide
from core.storage import AuthorStorage, get_author_profile, list_authors

console = Console()
author_app = typer.Typer()


@author_app.command("create")
def create_author(
    author_id: str = typer.Argument(..., help="Unique identifier for the author"),
    name: str = typer.Option(None, "--name", "-n", help="Display name for the author"),
    description: str = typer.Option(
        "", "--description", "-d", help="Brief description"
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", "-i/-ni", help="Interactive setup"
    ),
) -> None:
    """📝 Create a new author profile."""

    if interactive:
        create_author_interactive(author_id, name, description)
    else:
        create_author_simple(author_id, name, description)


@author_app.command("list")
def list_authors_cmd() -> None:
    """📋 List all author profiles."""

    authors = list_authors()

    if not authors:
        console.print(
            "[yellow]No authors found. Run 'simtune author create' to create one.[/yellow]"
        )
        return

    console.print(f"\n[bold blue]Authors ({len(authors)})[/bold blue]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Description", max_width=50)
    table.add_column("Style", max_width=30)
    table.add_column("Created", style="dim")

    for author_id in authors:
        profile = get_author_profile(author_id)
        if not profile:
            continue

        style_summary = f"{profile.style_guide.tone}, {profile.style_guide.formality}"
        created_str = (
            profile.created_at.strftime("%Y-%m-%d")
            if hasattr(profile, "created_at")
            else "Unknown"
        )

        table.add_row(
            author_id,
            profile.name,
            (
                profile.description[:47] + "..."
                if len(profile.description) > 50
                else profile.description
            ),
            style_summary,
            created_str,
        )

    console.print(table)


@author_app.command("show")
def show_author(
    author_id: str = typer.Argument(..., help="Author ID to display")
) -> None:
    """👤 Show detailed information about an author."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    # Profile info
    console.print(f"\n[bold blue]Author: {profile.name}[/bold blue]")
    console.print(f"[dim]ID: {profile.author_id}[/dim]")

    if profile.description:
        console.print(f"\n{profile.description}")

    # Style guide
    style = profile.style_guide
    console.print(
        Panel(
            f"[bold]Tone:[/bold] {style.tone}\n"
            f"[bold]Voice:[/bold] {style.voice}\n"
            f"[bold]Formality:[/bold] {style.formality}\n"
            f"[bold]Length Preference:[/bold] {style.length_preference}\n"
            f"[bold]Topics:[/bold] {', '.join(style.topics) if style.topics else 'Any'}\n"
            f"[bold]Avoid Topics:[/bold] {', '.join(style.avoid_topics) if style.avoid_topics else 'None'}\n"
            f"[bold]Style Notes:[/bold] {style.writing_style_notes or 'None'}",
            title="📝 Writing Style",
            border_style="blue",
        )
    )

    # Dataset info
    storage = AuthorStorage(author_id)
    dataset = storage.load_dataset()
    metadata = storage.load_model_metadata()

    if dataset and dataset.size > 0:
        console.print(f"\n[green]📊 Dataset: {dataset.size} training examples[/green]")
    else:
        console.print("\n[yellow]📊 No dataset created yet[/yellow]")

    # Model info
    if metadata and metadata.fine_tune_jobs:
        console.print(
            f"\n[green]🤖 Fine-tuning jobs: {len(metadata.fine_tune_jobs)}[/green]"
        )

        latest_job = metadata.get_latest_successful_job()
        if latest_job and latest_job.fine_tuned_model:
            console.print(
                f"[green]✅ Active model: {latest_job.fine_tuned_model}[/green]"
            )
    else:
        console.print("\n[yellow]🤖 No fine-tuned models yet[/yellow]")


@author_app.command("edit")
def edit_author(author_id: str = typer.Argument(..., help="Author ID to edit")) -> None:
    """✏️  Edit an existing author profile."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Editing Author: {profile.name}[/bold blue]")

    # Edit basic info
    new_name = Prompt.ask("Name", default=profile.name)
    new_description = Prompt.ask("Description", default=profile.description)

    profile.name = new_name
    profile.description = new_description

    # Edit style guide
    style = profile.style_guide

    console.print("\n[bold yellow]Style Guide[/bold yellow]")

    style.tone = Prompt.ask(
        "Tone",
        default=style.tone,
        choices=[
            "casual",
            "professional",
            "friendly",
            "authoritative",
            "witty",
            "formal",
        ],
    )

    style.voice = Prompt.ask(
        "Voice",
        default=style.voice,
        choices=["first_person", "second_person", "third_person"],
    )

    style.formality = Prompt.ask(
        "Formality",
        default=style.formality,
        choices=["very_casual", "casual", "moderate", "formal", "academic"],
    )

    style.length_preference = Prompt.ask(
        "Length Preference",
        default=style.length_preference,
        choices=["short", "medium", "long", "variable"],
    )

    # Topics
    topics_str = ", ".join(style.topics) if style.topics else ""
    new_topics = Prompt.ask("Preferred topics (comma-separated)", default=topics_str)
    style.topics = [t.strip() for t in new_topics.split(",") if t.strip()]

    avoid_str = ", ".join(style.avoid_topics) if style.avoid_topics else ""
    new_avoid = Prompt.ask("Topics to avoid (comma-separated)", default=avoid_str)
    style.avoid_topics = [t.strip() for t in new_avoid.split(",") if t.strip()]

    style.writing_style_notes = Prompt.ask(
        "Additional style notes", default=style.writing_style_notes
    )

    # Save changes
    profile.updated_at = datetime.now()
    storage = AuthorStorage(author_id)
    storage.save_profile(profile)

    console.print(f"[green]✅ Author '{author_id}' updated successfully![/green]")


def create_author_interactive(
    author_id: str = None, name: str = None, description: str = None
) -> None:
    """Interactive author creation flow."""

    console.print(
        Panel(
            "[bold blue]Create New Author Profile[/bold blue]\n\n"
            "Let's set up your writing profile so we can fine-tune an AI to match your style.",
            title="📝 Author Setup",
        )
    )

    # Get basic info
    if not author_id:
        author_id = Prompt.ask(
            "Enter a unique ID for this author (e.g., 'john_doe', 'blog_writer')"
        )

    # Check if author already exists
    if get_author_profile(author_id):
        console.print(f"[red]Author '{author_id}' already exists![/red]")
        if not Confirm.ask("Do you want to edit the existing author instead?"):
            return
        edit_author(author_id)
        return

    if not name:
        name = Prompt.ask(
            "What should we call this author?",
            default=author_id.replace("_", " ").title(),
        )

    if not description:
        description = Prompt.ask(
            "Brief description of this author (optional)", default=""
        )

    # Style guide setup
    console.print("\n[bold yellow]Writing Style Setup[/bold yellow]")
    console.print("Help us understand your writing style:")

    tone = Prompt.ask(
        "What's your typical tone?",
        choices=[
            "casual",
            "professional",
            "friendly",
            "authoritative",
            "witty",
            "formal",
        ],
        default="professional",
    )

    voice = Prompt.ask(
        "What narrative voice do you use?",
        choices=["first_person", "second_person", "third_person"],
        default="first_person",
    )

    formality = Prompt.ask(
        "How formal is your writing?",
        choices=["very_casual", "casual", "moderate", "formal", "academic"],
        default="moderate",
    )

    length_preference = Prompt.ask(
        "Do you prefer writing that is...",
        choices=["short", "medium", "long", "variable"],
        default="medium",
    )

    topics_str = Prompt.ask(
        "What topics do you typically write about? (comma-separated, optional)",
        default="",
    )
    topics = [t.strip() for t in topics_str.split(",") if t.strip()]

    avoid_str = Prompt.ask(
        "Any topics you want to avoid? (comma-separated, optional)", default=""
    )
    avoid_topics = [t.strip() for t in avoid_str.split(",") if t.strip()]

    style_notes = Prompt.ask(
        "Any additional notes about your writing style? (optional)", default=""
    )

    # Create profile
    style_guide = StyleGuide(
        tone=tone,
        voice=voice,
        formality=formality,
        length_preference=length_preference,
        topics=topics,
        avoid_topics=avoid_topics,
        writing_style_notes=style_notes,
    )

    profile = AuthorProfile(
        author_id=author_id, name=name, description=description, style_guide=style_guide
    )

    # Save profile
    storage = AuthorStorage(author_id)
    storage.save_profile(profile)

    console.print(f"\n[green]✅ Author '{name}' created successfully![/green]")

    # Next steps
    console.print(
        Panel(
            f"[bold green]What's Next?[/bold green]\n\n"
            f"1. Build a training dataset:\n"
            f"   [cyan]simtune dataset build {author_id}[/cyan]\n\n"
            f"2. Start fine-tuning:\n"
            f"   [cyan]simtune train start {author_id}[/cyan]\n\n"
            f"3. View your author:\n"
            f"   [cyan]simtune author show {author_id}[/cyan]",
            title="🚀 Ready to go!",
        )
    )


def create_author_simple(
    author_id: str, name: str = None, description: str = ""
) -> None:
    """Simple non-interactive author creation."""

    if get_author_profile(author_id):
        console.print(f"[red]Author '{author_id}' already exists![/red]")
        raise typer.Exit(1)

    if not name:
        name = author_id.replace("_", " ").title()

    profile = AuthorProfile(
        author_id=author_id,
        name=name,
        description=description,
        style_guide=StyleGuide(),  # Use defaults
    )

    storage = AuthorStorage(author_id)
    storage.save_profile(profile)

    console.print(f"[green]✅ Author '{name}' created with default settings.[/green]")
    console.print(
        f"[dim]Use 'simtune author edit {author_id}' to customize the style.[/dim]"
    )
