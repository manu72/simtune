from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    """Get current UTC datetime - used for default factory to support time mocking."""
    return datetime.now()


class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingExample(BaseModel):
    messages: List[Dict[str, str]] = Field(
        description="OpenAI chat format training example"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not v or len(v) < 2:
            raise ValueError("Training example must have at least 2 messages")

        valid_roles = {"system", "user", "assistant"}
        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg["role"] not in valid_roles:
                raise ValueError(f"Role must be one of {valid_roles}")

        return v


class StyleGuide(BaseModel):
    tone: str = Field(
        default="professional",
        description="Writing tone (e.g., casual, professional, witty)",
    )
    voice: str = Field(
        default="first_person",
        description="Narrative voice (e.g., first_person, third_person)",
    )
    formality: str = Field(
        default="moderate",
        description="Formality level (e.g., casual, moderate, formal)",
    )
    length_preference: str = Field(
        default="medium",
        description="Preferred content length (e.g., short, medium, long)",
    )
    topics: List[str] = Field(
        default_factory=list, description="Preferred topics or domains"
    )
    avoid_topics: List[str] = Field(default_factory=list, description="Topics to avoid")
    writing_style_notes: str = Field(
        default="", description="Additional style preferences"
    )

    class Config:
        extra = "allow"


class AuthorProfile(BaseModel):
    author_id: str = Field(description="Unique identifier for the author")
    name: str = Field(description="Display name for the author")
    description: str = Field(default="", description="Brief description of the author")
    style_guide: StyleGuide = Field(default_factory=StyleGuide)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def author_dir(self) -> Path:
        return Path("data/authors") / self.author_id

    @property
    def dataset_path(self) -> Path:
        return self.author_dir / "train.jsonl"

    @property
    def style_guide_path(self) -> Path:
        return self.author_dir / "style_guide.yml"

    @property
    def models_path(self) -> Path:
        return self.author_dir / "models.json"


class Dataset(BaseModel):
    author_id: str
    examples: List[TrainingExample] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def size(self) -> int:
        return len(self.examples)

    def add_example(self, example: TrainingExample) -> None:
        self.examples.append(example)
        self.updated_at = datetime.now()


class ChatMessage(BaseModel):
    role: str = Field(description="Message role: 'user' or 'assistant'")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=utc_now)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        valid_roles = {"user", "assistant", "system"}
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return v


class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    author_id: str = Field(description="ID of the author for this chat session")
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation."""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.updated_at = utc_now()

    def get_openai_messages(self) -> List[Dict[str, str]]:
        """Convert messages to OpenAI API format."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def clear_messages(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = utc_now()

    @property
    def message_count(self) -> int:
        """Get the total number of messages in the session."""
        return len(self.messages)

    @property
    def last_message_time(self) -> Optional[datetime]:
        """Get the timestamp of the last message."""
        return self.messages[-1].timestamp if self.messages else None


class FineTuneJob(BaseModel):
    job_id: str = Field(description="Provider's job identifier")
    author_id: str
    provider: Provider
    base_model: str = Field(description="Base model name (e.g., gpt-3.5-turbo)")
    status: JobStatus = Field(default=JobStatus.PENDING)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    training_file_id: Optional[str] = Field(
        default=None, description="Provider's training file ID"
    )
    fine_tuned_model: Optional[str] = Field(
        default=None, description="ID of the fine-tuned model"
    )
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    result_files: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None

    def update_status(self, status: JobStatus, **kwargs: Any) -> None:
        self.status = status
        self.updated_at = datetime.now()

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # --- Normalisers for backward/robust loading of stored metadata ---
    @field_validator("error_message", mode="before")
    @classmethod
    def _normalise_error_message(cls, value: Any) -> Optional[str]:
        if value is None or isinstance(value, str):
            return value
        try:
            # If dict-like with 'message', prefer that
            if isinstance(value, dict):
                msg = value.get("message")
                if msg:
                    return str(msg)
            # If object with .message attribute
            msg_attr = getattr(value, "message", None)
            if msg_attr:
                return str(msg_attr)
        except Exception:
            pass
        # Fallback to string representation
        return str(value)

    @field_validator("result_files", mode="before")
    @classmethod
    def _normalise_result_files(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            normalised: List[str] = []
            for item in value:
                try:
                    if isinstance(item, dict) and "id" in item:
                        normalised.append(str(item["id"]))
                    else:
                        file_id = getattr(item, "id", None)
                        normalised.append(str(file_id) if file_id else str(item))
                except Exception:
                    normalised.append(str(item))
            return normalised
        # If it isn't a list, coerce to one-string list
        return [str(value)]


class ModelMetadata(BaseModel):
    fine_tune_jobs: List[FineTuneJob] = Field(default_factory=list)
    active_model: Optional[str] = Field(
        default=None, description="Currently active fine-tuned model"
    )

    def add_job(self, job: FineTuneJob) -> None:
        self.fine_tune_jobs.append(job)

    def get_job(self, job_id: str) -> Optional[FineTuneJob]:
        return next((job for job in self.fine_tune_jobs if job.job_id == job_id), None)

    def get_latest_successful_job(self) -> Optional[FineTuneJob]:
        successful_jobs = [
            job for job in self.fine_tune_jobs if job.status == JobStatus.SUCCEEDED
        ]
        return (
            max(successful_jobs, key=lambda x: x.updated_at)
            if successful_jobs
            else None
        )
