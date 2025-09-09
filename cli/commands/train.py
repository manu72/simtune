import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from core.adapters.openai_adapter import OpenAIAdapter
from core.config import settings
from core.dataset.validator import DatasetValidator
from core.models import JobStatus
from core.storage import AuthorStorage, get_author_profile

console = Console()
train_app = typer.Typer()


@train_app.command("start")
def start_training(
    author_id: str = typer.Argument(..., help="Author ID to train model for"),
    base_model: str = typer.Option(
        None, "--model", "-m", help="Base model to fine-tune"
    ),
    wait: bool = typer.Option(
        False, "--wait", "-w", help="Wait for training to complete"
    ),
):
    """🚀 Start fine-tuning a model for an author."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    # Choose model: explicit flag > training-specific default > general default
    if base_model is None:
        base_model = settings.get_training_model("openai")

    storage = AuthorStorage(author_id)
    dataset = storage.load_dataset()

    # Check dataset exists
    if not dataset or dataset.size == 0:
        console.print(f"[red]No dataset found for '{author_id}'.[/red]")
        console.print(f"Use 'simtune dataset build {author_id}' to create one first.")
        raise typer.Exit(1)

    console.print(f"[bold blue]Starting fine-tuning for: {profile.name}[/bold blue]")
    console.print(f"Dataset size: {dataset.size} examples")
    console.print(f"Base model: {base_model}")

    # Validate dataset first
    console.print("\n📋 Validating dataset...")
    validator = DatasetValidator(dataset)
    validator.validate()

    validation_status = validator.get_validation_summary()
    if validation_status == "needs_attention":
        console.print("[red]❌ Dataset validation failed.[/red]")
        if not Confirm.ask("Continue anyway? (not recommended)"):
            raise typer.Exit(1)
    elif validation_status == "acceptable":
        console.print("[yellow]⚠️  Dataset has some issues but should work.[/yellow]")
        if not Confirm.ask("Continue with training?"):
            raise typer.Exit(1)
    else:
        console.print("[green]✅ Dataset validation passed.[/green]")

    # Initialize adapter
    try:
        adapter = OpenAIAdapter()
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)

    try:
        # Upload training file
        training_file_id = adapter.upload_training_file(dataset)

        # Create fine-tuning job
        job = adapter.create_fine_tune_job(
            author_id=author_id,
            training_file_id=training_file_id,
            base_model=base_model,
        )

        # Save job metadata
        metadata = storage.load_model_metadata()
        metadata.add_job(job)
        storage.save_model_metadata(metadata)

        console.print(f"\n[green]✅ Fine-tuning job started: {job.job_id}[/green]")
        console.print(f"Status: {job.status.value}")

        console.print("\n[bold yellow]Next steps:[/bold yellow]")
        console.print(f"• Check status: [cyan]simtune train status {author_id}[/cyan]")
        console.print(f"• List all jobs: [cyan]simtune train list {author_id}[/cyan]")

        # Wait for completion if requested
        if wait:
            console.print("\n⏳ Waiting for training to complete...")
            job = adapter.wait_for_completion(job)

            # Update metadata
            metadata = storage.load_model_metadata()
            stored_job = metadata.get_job(job.job_id)
            if stored_job:
                stored_job.update_status(
                    job.status,
                    fine_tuned_model=job.fine_tuned_model,
                    error_message=job.error_message,
                )
                storage.save_model_metadata(metadata)

            if job.status == JobStatus.SUCCEEDED:
                console.print("\n[green]🎉 Training completed successfully![/green]")
                console.print(f"Fine-tuned model: {job.fine_tuned_model}")

                # Test the model
                if Confirm.ask("Would you like to test the fine-tuned model?"):
                    test_model(author_id, job.fine_tuned_model)
            else:
                console.print(f"\n[red]❌ Training failed: {job.error_message}[/red]")

    except Exception as e:
        console.print(f"[red]Error starting training: {str(e)}[/red]")
        raise typer.Exit(1)


@train_app.command("status")
def check_status(
    author_id: str = typer.Argument(..., help="Author ID to check training status for")
):
    """📊 Check the status of fine-tuning jobs for an author."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    storage = AuthorStorage(author_id)
    metadata = storage.load_model_metadata()

    if not metadata.fine_tune_jobs:
        console.print(f"[yellow]No fine-tuning jobs found for '{author_id}'.[/yellow]")
        console.print(f"Use 'simtune train start {author_id}' to start training.")
        return

    console.print(f"[bold blue]Training Status for: {profile.name}[/bold blue]")

    try:
        adapter = OpenAIAdapter()

        # Update job statuses
        for job in metadata.fine_tune_jobs:
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                updated_job = adapter.update_job_status(job)
                job.update_status(
                    updated_job.status,
                    fine_tuned_model=updated_job.fine_tuned_model,
                    error_message=updated_job.error_message,
                )

        # Save updated metadata
        storage.save_model_metadata(metadata)

        # Display jobs table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Job ID", style="cyan", max_width=20)
        table.add_column("Status", style="white")
        table.add_column("Model", style="white", max_width=30)
        table.add_column("Fine-tuned Model", style="green", max_width=30)
        table.add_column("Created", style="dim")

        for job in sorted(
            metadata.fine_tune_jobs, key=lambda x: x.created_at, reverse=True
        ):
            status_color = {
                JobStatus.SUCCEEDED: "green",
                JobStatus.FAILED: "red",
                JobStatus.RUNNING: "yellow",
                JobStatus.PENDING: "blue",
                JobStatus.CANCELLED: "dim",
            }.get(job.status, "white")

            status_display = (
                f"[{status_color}]{job.status.value.upper()}[/{status_color}]"
            )

            table.add_row(
                job.job_id[:17] + "..." if len(job.job_id) > 20 else job.job_id,
                status_display,
                job.base_model,
                (
                    job.fine_tuned_model[:27] + "..."
                    if job.fine_tuned_model and len(job.fine_tuned_model) > 30
                    else (job.fine_tuned_model or "-")
                ),
                job.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

        # Show active model
        latest_successful = metadata.get_latest_successful_job()
        if latest_successful and latest_successful.fine_tuned_model:
            console.print(
                f"\n[green]🤖 Active Model: {latest_successful.fine_tuned_model}[/green]"
            )

    except Exception as e:
        console.print(f"[red]Error checking status: {str(e)}[/red]")


@train_app.command("list")
def list_jobs(author_id: str = typer.Argument(..., help="Author ID to list jobs for")):
    """📋 List all fine-tuning jobs for an author."""
    check_status(author_id)  # Reuse the status command


@train_app.command("test")
def test_model(
    author_id: str = typer.Argument(..., help="Author ID to test model for"),
    model_id: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Specific model ID to test (default: latest successful)",
    ),
):
    """🧪 Test a fine-tuned model with sample prompts."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    storage = AuthorStorage(author_id)
    metadata = storage.load_model_metadata()

    # Find model to test
    if model_id:
        job = next(
            (
                job
                for job in metadata.fine_tune_jobs
                if job.fine_tuned_model == model_id
            ),
            None,
        )
        if not job:
            console.print(f"[red]Model '{model_id}' not found.[/red]")
            raise typer.Exit(1)
    else:
        job = metadata.get_latest_successful_job()
        if not job or not job.fine_tuned_model:
            console.print(f"[red]No fine-tuned model found for '{author_id}'.[/red]")
            console.print(f"Use 'simtune train start {author_id}' to create one.")
            raise typer.Exit(1)
        model_id = job.fine_tuned_model

    console.print(f"[bold blue]Testing model: {model_id}[/bold blue]")

    try:
        adapter = OpenAIAdapter()

        # Test with predefined prompts
        test_prompts = [
            "Write a brief introduction about yourself.",
            "What's your writing style like?",
            "Tell me about your expertise.",
        ]

        # Ask user for custom prompt
        custom_prompt = Prompt.ask(
            "\nEnter a custom prompt to test (or press Enter to skip)", default=""
        )
        if custom_prompt.strip():
            test_prompts.append(custom_prompt.strip())

        adapter.test_fine_tuned_model(model_id, test_prompts)

        console.print("\n[green]✅ Model test completed![/green]")

    except Exception as e:
        console.print(f"[red]Error testing model: {str(e)}[/red]")
        raise typer.Exit(1)


@train_app.command("generate")
def generate_text(
    author_id: str = typer.Argument(..., help="Author ID to generate text for"),
    prompt: str = typer.Option(
        None, "--prompt", "-p", help="Prompt for text generation"
    ),
    model_id: str = typer.Option(
        None, "--model", "-m", help="Specific model ID (default: latest successful)"
    ),
    max_completion_tokens: int = typer.Option(
        500, "--max-completion-tokens", help="Maximum tokens to generate"
    ),
):
    """✍️  Generate text using a fine-tuned model."""

    profile = get_author_profile(author_id)
    if not profile:
        console.print(f"[red]Author '{author_id}' not found.[/red]")
        raise typer.Exit(1)

    storage = AuthorStorage(author_id)
    metadata = storage.load_model_metadata()

    # Find model to use
    if model_id:
        job = next(
            (
                job
                for job in metadata.fine_tune_jobs
                if job.fine_tuned_model == model_id
            ),
            None,
        )
        if not job:
            console.print(f"[red]Model '{model_id}' not found.[/red]")
            raise typer.Exit(1)
    else:
        job = metadata.get_latest_successful_job()
        if not job or not job.fine_tuned_model:
            console.print(f"[red]No fine-tuned model found for '{author_id}'.[/red]")
            raise typer.Exit(1)
        model_id = job.fine_tuned_model

    # Get prompt if not provided
    if not prompt:
        prompt = Prompt.ask("Enter your prompt")

    console.print(f"\n[yellow]Prompt:[/yellow] {prompt}")
    console.print(f"[dim]Using model: {model_id}[/dim]")

    try:
        adapter = OpenAIAdapter()
        response = adapter.generate_text(model_id, prompt, max_completion_tokens)

        console.print(
            Panel(
                response, title=f"✍️  Generated by {profile.name}", border_style="green"
            )
        )

    except Exception as e:
        console.print(f"[red]Error generating text: {str(e)}[/red]")
        raise typer.Exit(1)
