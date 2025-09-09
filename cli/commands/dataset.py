import typer
from rich.console import Console
from rich.prompt import Confirm

from core.dataset.builder import DatasetBuilder
from core.dataset.validator import DatasetValidator
from core.storage import AuthorStorage, get_author_profile

console = Console()
dataset_app = typer.Typer()


@dataset_app.command("build")
def build_dataset(
    author_id: str = typer.Argument(..., help="Author ID to build dataset for")
):
    """📊 Interactively build a training dataset for an author."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        console.print("Use 'simtune author create' to create an author first.")
        raise typer.Exit(1)

    console.print(f"[bold blue]Building dataset for: {profile.name}[/bold blue]")

    builder = DatasetBuilder(author_id)
    builder.interactive_build()


@dataset_app.command("validate")
def validate_dataset(
    author_id: str = typer.Argument(..., help="Author ID to validate dataset for")
):
    """✅ Validate the training dataset for an author."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    storage = AuthorStorage(author_id)
    dataset = storage.load_dataset()

    if not dataset or dataset.size == 0:
        console.print(f"[red]No dataset found for '{author_id}'.[/red]")
        console.print(f"Use 'simtune dataset build {author_id}' to create one.")
        raise typer.Exit(1)

    console.print(f"[bold blue]Validating dataset for: {profile.name}[/bold blue]")

    validator = DatasetValidator(dataset)
    validator.validate()

    # Show summary
    summary = validator.get_validation_summary()
    if summary == "ready":
        console.print("\n[bold green]🎉 Dataset is ready for fine-tuning![/bold green]")
    elif summary == "acceptable":
        console.print(
            "\n[bold yellow]⚠️  Dataset has some issues but may work for fine-tuning.[/bold yellow]"
        )
    else:
        console.print(
            "\n[bold red]❌ Dataset needs attention before fine-tuning.[/bold red]"
        )


@dataset_app.command("show")
def show_dataset(
    author_id: str = typer.Argument(..., help="Author ID to show dataset for")
):
    """📋 Show dataset information and statistics."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    storage = AuthorStorage(author_id)
    dataset = storage.load_dataset()

    if not dataset or dataset.size == 0:
        console.print(f"[red]No dataset found for '{author_id}'.[/red]")
        console.print(f"Use 'simtune dataset build {author_id}' to create one.")
        raise typer.Exit(1)

    console.print(f"[bold blue]Dataset for: {profile.name}[/bold blue]")
    console.print(f"[green]Size: {dataset.size} examples[/green]")
    console.print(
        f"[dim]Created: {dataset.created_at.strftime('%Y-%m-%d %H:%M')}[/dim]"
    )
    console.print(
        f"[dim]Updated: {dataset.updated_at.strftime('%Y-%m-%d %H:%M')}[/dim]"
    )

    if dataset.size > 0:
        # Show first few examples
        console.print("\n[bold]Sample Examples:[/bold]")

        for i, example in enumerate(dataset.examples[:3], 1):
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

            console.print(f"\n[yellow]Example {i}:[/yellow]")
            console.print(
                f"[dim]Prompt:[/dim] {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}"
            )
            console.print(
                f"[dim]Response:[/dim] {assistant_msg[:200]}{'...' if len(assistant_msg) > 200 else ''}"
            )

        if dataset.size > 3:
            console.print(f"\n[dim]... and {dataset.size - 3} more examples[/dim]")


@dataset_app.command("clear")
def clear_dataset(
    author_id: str = typer.Argument(..., help="Author ID to clear dataset for")
):
    """🗑️  Clear all training examples for an author."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    storage = AuthorStorage(author_id)
    dataset = storage.load_dataset()

    if not dataset or dataset.size == 0:
        console.print(f"[yellow]No dataset found for '{author_id}'.[/yellow]")
        return

    console.print(
        f"[red]⚠️  This will delete all {dataset.size} training examples for '{profile.name}'[/red]"
    )

    if Confirm.ask("Are you sure you want to continue?"):
        # Create empty dataset
        from core.models import Dataset

        empty_dataset = Dataset(author_id=author_id)
        storage.save_dataset(empty_dataset)

        console.print(f"[green]✅ Dataset cleared for '{profile.name}'[/green]")
    else:
        console.print("Cancelled.")


@dataset_app.command("export")
def export_dataset(
    author_id: str = typer.Argument(..., help="Author ID to export dataset for"),
    output_file: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: <author_id>_dataset.jsonl)",
    ),
):
    """📤 Export dataset to a JSONL file."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    storage = AuthorStorage(author_id)
    dataset = storage.load_dataset()

    if not dataset or dataset.size == 0:
        console.print(f"[red]No dataset found for '{author_id}'.[/red]")
        raise typer.Exit(1)

    if not output_file:
        output_file = f"{author_id}_dataset.jsonl"

    try:
        import jsonlines

        with jsonlines.open(output_file, "w") as writer:
            for example in dataset.examples:
                writer.write(example.model_dump())

        console.print(
            f"[green]✅ Dataset exported to '{output_file}' ({dataset.size} examples)[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error exporting dataset: {str(e)}[/red]")
        raise typer.Exit(1)
