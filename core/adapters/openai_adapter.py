import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import openai
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config import settings
from core.models import Dataset, FineTuneJob, JobStatus, Provider

console = Console()


class OpenAIAdapter:
    def __init__(self) -> None:
        if not settings.has_openai_key():
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.\n"
                "Copy .env.example to .env and add your API key."
            )

        self.client = openai.OpenAI(
            api_key=settings.openai_api_key, organization=settings.openai_org_id
        )

    def upload_training_file(self, dataset: Dataset) -> str:
        console.print("📤 Uploading training data to OpenAI...")

        # Convert dataset to OpenAI JSONL format
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp_file:
            for example in dataset.examples:
                json.dump(example.model_dump(), tmp_file)
                tmp_file.write("\n")

            tmp_file_path = tmp_file.name

        try:
            # Upload file
            with open(tmp_file_path, "rb") as f:
                response = self.client.files.create(file=f, purpose="fine-tune")

            console.print(f"✅ Training file uploaded: {response.id}")
            return response.id

        finally:
            # Clean up temporary file
            Path(tmp_file_path).unlink(missing_ok=True)

    def create_fine_tune_job(
        self,
        author_id: str,
        training_file_id: str,
        base_model: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> FineTuneJob:
        # Use default model if none specified
        if base_model is None:
            base_model = settings.get_default_model("openai")

        console.print(f"🚀 Starting fine-tuning job for {base_model}...")

        # Default hyperparameters
        hyperparams_dict = {"n_epochs": "auto", "learning_rate_multiplier": "auto"}

        if hyperparameters:
            hyperparams_dict.update(hyperparameters)

        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=base_model,
                hyperparameters=hyperparams_dict,  # type: ignore
            )

            job = FineTuneJob(
                job_id=response.id,
                author_id=author_id,
                provider=Provider.OPENAI,
                base_model=base_model,
                status=JobStatus.PENDING,
                training_file_id=training_file_id,
                hyperparameters=hyperparams_dict,
            )

            console.print(f"✅ Fine-tuning job created: {response.id}")
            console.print(f"📊 Status: {response.status}")

            return job

        except Exception as e:
            console.print(f"❌ Error creating fine-tuning job: {str(e)}")
            raise

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)

            # Map OpenAI status to our JobStatus
            status_mapping = {
                "validating_files": JobStatus.PENDING,
                "queued": JobStatus.PENDING,
                "running": JobStatus.RUNNING,
                "succeeded": JobStatus.SUCCEEDED,
                "failed": JobStatus.FAILED,
                "cancelled": JobStatus.CANCELLED,
            }

            our_status = status_mapping.get(response.status, JobStatus.PENDING)

            # Normalise error to a simple string (or None)
            error_obj = getattr(response, "error", None)
            error_message: Optional[str]
            if error_obj is None:
                error_message = None
            else:
                # Try common shapes: object with .message, dict-like with ['message'], else str()
                message_attr = getattr(error_obj, "message", None)
                if message_attr:
                    error_message = str(message_attr)
                else:
                    try:
                        error_message = (
                            error_obj.get("message")
                            if hasattr(error_obj, "get")
                            else None
                        )
                    except Exception:
                        error_message = None
                    if not error_message:
                        error_message = str(error_obj)

            # Normalise result_files to a list of ids/strings
            raw_files_attr = getattr(response, "result_files", None)
            raw_files: List[Any]
            if isinstance(raw_files_attr, list):
                raw_files = raw_files_attr
            else:
                raw_files = []

            normalised_files: List[str] = []
            for f in raw_files:
                try:
                    file_id = getattr(f, "id", None)
                    if file_id:
                        normalised_files.append(str(file_id))
                    else:
                        normalised_files.append(str(f))
                except Exception:
                    normalised_files.append(str(f))

            return {
                "status": our_status,
                "openai_status": response.status,
                "fine_tuned_model": response.fine_tuned_model,
                "error": error_message,
                "estimated_finish": getattr(response, "estimated_finish", None),
                "result_files": normalised_files,
            }

        except Exception as e:
            console.print(f"❌ Error checking job status: {str(e)}")
            return {"status": JobStatus.FAILED, "error": str(e)}

    def update_job_status(self, job: FineTuneJob) -> FineTuneJob:
        status_info = self.get_job_status(job.job_id)

        job.update_status(
            status=status_info["status"],
            fine_tuned_model=status_info.get("fine_tuned_model"),
            error_message=status_info.get("error"),
            result_files=status_info.get("result_files", []),
        )

        return job

    def wait_for_completion(
        self, job: FineTuneJob, check_interval: int = 60
    ) -> FineTuneJob:
        console.print("⏳ Waiting for fine-tuning to complete...")
        console.print("This may take 20+ minutes depending on dataset size.")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                description="Fine-tuning in progress...", total=None
            )

            while job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                time.sleep(check_interval)
                job = self.update_job_status(job)

                if job.status == JobStatus.SUCCEEDED:
                    progress.update(task, description="Fine-tuning completed! ✅")
                    break
                elif job.status == JobStatus.FAILED:
                    progress.update(task, description="Fine-tuning failed ❌")
                    break
                elif job.status == JobStatus.CANCELLED:
                    progress.update(task, description="Fine-tuning cancelled")
                    break

        return job

    def generate_text(
        self, model_id: str, prompt: str, max_completion_tokens: int = 500
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                temperature=0.7,
            )

            return response.choices[0].message.content

        except Exception as e:
            console.print(f"❌ Error generating text: {str(e)}")
            raise

    def generate_chat_response(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_completion_tokens: int = 500,
    ) -> str:
        """Generate response for a chat conversation with full message history."""
        try:
            # Ensure we don't exceed token limits - basic implementation
            # In future could implement smarter truncation/summarization
            truncated_messages = self._truncate_messages(
                messages, max_completion_tokens
            )

            response = self.client.chat.completions.create(
                model=model_id,
                messages=cast(Any, truncated_messages),
                max_completion_tokens=max_completion_tokens,
                temperature=0.7,
            )

            return response.choices[0].message.content

        except Exception as e:
            console.print(f"❌ Error generating chat response: {str(e)}")
            raise

    def _truncate_messages(
        self, messages: List[Dict[str, str]], max_completion_tokens: int
    ) -> List[Dict[str, str]]:
        """Truncate message history to stay within context window limits.

        Basic implementation - keeps most recent messages.
        Could be enhanced with smarter strategies like summarization.
        """
        # Rough estimation: 4 chars per token, keep substantial context
        # Reserve tokens for completion, leave room for conversation
        max_context_tokens = 4096 - max_completion_tokens - 500  # Conservative buffer
        max_chars = max_context_tokens * 4

        # Calculate total character count
        total_chars = sum(len(msg.get("content", "")) for msg in messages)

        # If under limit, return all messages
        if total_chars <= max_chars:
            return messages

        # Otherwise, keep most recent messages that fit
        truncated: List[Dict[str, str]] = []
        char_count = 0

        # Process in reverse to keep most recent
        for msg in reversed(messages):
            msg_chars = len(msg.get("content", ""))
            if char_count + msg_chars <= max_chars:
                truncated.insert(0, msg)
                char_count += msg_chars
            else:
                break

        # Always keep at least the last user message if possible
        if not truncated and messages:
            truncated = [messages[-1]]

        return truncated

    def test_fine_tuned_model(
        self, model_id: str, test_prompts: Optional[List[str]] = None
    ) -> Dict[str, str]:
        if not test_prompts:
            test_prompts = [
                "Write a brief introduction about yourself.",
                "What's your writing style?",
                "Tell me about your expertise.",
            ]

        console.print(Panel("🧪 Testing Fine-tuned Model", style="blue"))

        results = {}
        for prompt in test_prompts:
            console.print(f"\n[yellow]Prompt:[/yellow] {prompt}")

            try:
                response = self.generate_text(model_id, prompt)
                results[prompt] = response
                console.print(f"[green]Response:[/green] {response}")
            except Exception as e:
                results[prompt] = f"Error: {str(e)}"
                console.print(f"[red]Error:[/red] {str(e)}")

        return results

    def list_fine_tuned_models(self) -> List[Any]:
        try:
            response = self.client.models.list()

            # Filter for fine-tuned models (they contain "ft-" in the ID)
            fine_tuned = [
                model
                for model in response.data
                if "ft-" in model.id and model.owned_by != "system"
            ]

            return fine_tuned

        except Exception as e:
            console.print(f"❌ Error listing models: {str(e)}")
            return []

    def delete_fine_tuned_model(self, model_id: str) -> bool:
        try:
            self.client.models.delete(model_id)
            console.print(f"✅ Model {model_id} deleted successfully")
            return True
        except Exception as e:
            console.print(f"❌ Error deleting model {model_id}: {str(e)}")
            return False
