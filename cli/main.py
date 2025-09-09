import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.commands.author import author_app
from cli.commands.dataset import dataset_app
from cli.commands.generate import generate_app
from cli.commands.train import train_app

console = Console()

app = typer.Typer(
    name="simtune",
    help="Personal AI Writing Assistant - Fine-tune LLMs with your writing style",
    rich_markup_mode="rich",
)

# Add subcommands
app.add_typer(author_app, name="author", help="Manage author profiles")
app.add_typer(dataset_app, name="dataset", help="Build and manage training datasets")
app.add_typer(train_app, name="train", help="Fine-tune and manage models")
app.add_typer(
    generate_app, name="generate", help="Generate content with fine-tuned models"
)


@app.command()
def init() -> None:
    """Initialize Simtune and create your first author profile."""
    console.print(
        Panel.fit(
            "[bold blue]Welcome to Simtune![/bold blue]\n\n"
            "Simtune helps you create a personal AI that writes in your style.\n"
            "Let's get started by creating your first author profile.",
            title="Simtune Setup",
        )
    )

    # Import here to avoid circular imports
    from cli.commands.author import create_author_interactive

    create_author_interactive()


@app.command()
def status() -> None:
    """Show overview of all authors and their status."""
    from core.storage import AuthorStorage, get_author_profile, list_authors

    authors = list_authors()

    if not authors:
        console.print(
            "[yellow]No authors found. Run 'simtune init' to create your first author.[/yellow]"
        )
        return

    console.print(
        f"\n[bold blue]Simtune Status[/bold blue] - {len(authors)} author(s)"
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Author", style="cyan")
    table.add_column("Description", max_width=40)
    table.add_column("Dataset Size", justify="right")
    table.add_column("Models", justify="right")
    table.add_column("Status", style="green")

    for author_id in authors:
        profile = get_author_profile(author_id)
        if not profile:
            continue

        storage = AuthorStorage(author_id)
        dataset = storage.load_dataset()
        metadata = storage.load_model_metadata()

        dataset_size = dataset.size if dataset else 0
        model_count = len(metadata.fine_tune_jobs) if metadata else 0

        # Determine status
        if model_count > 0:
            latest_job = metadata.get_latest_successful_job()
            if latest_job:
                status = "✅ Ready"
            else:
                status = "🔄 Training"
        elif dataset_size > 0:
            status = "📊 Dataset Ready"
        else:
            status = "📝 Setup Needed"

        table.add_row(
            profile.name,
            (
                profile.description[:37] + "..."
                if len(profile.description) > 40
                else profile.description
            ),
            str(dataset_size),
            str(model_count),
            status,
        )

    console.print(table)

    console.print("\n[dim]Commands:[/dim]")
    console.print("• [cyan]simtune author list[/cyan] - Manage authors")
    console.print(
        "• [cyan]simtune dataset build <author>[/cyan] - Build training dataset"
    )
    console.print("• [cyan]simtune train start <author>[/cyan] - Start fine-tuning")
    console.print(
        "• [cyan]simtune generate text <author>[/cyan] - Generate content"
    )


@app.command()
def version() -> None:
    """Show version information."""
    console.print("[bold blue]Simtune[/bold blue] v0.1.0 (Stage 1 POC)")
    console.print("Personal AI Writing Assistant")


if __name__ == "__main__":
    app()
