import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import jsonlines
import yaml

from core.config import settings
from core.markdown_utils import save_content_as_markdown, save_example_as_markdown
from core.models import (
    AuthorProfile,
    ChatSession,
    Dataset,
    ModelMetadata,
    TrainingExample,
)


class AuthorStorage:
    def __init__(self, author_id: str) -> None:
        self.author_id = author_id
        self.author_dir = settings.authors_dir / author_id
        self.author_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_examples_directory()
        self._ensure_content_directory()
        self._ensure_chats_directory()

    def save_profile(self, profile: AuthorProfile) -> None:
        profile_path = self.author_dir / "profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile.model_dump(), f, indent=2, default=str)

        style_guide_path = self.author_dir / "style_guide.yml"
        with open(style_guide_path, "w") as f:
            yaml.dump(profile.style_guide.model_dump(), f, default_flow_style=False)

    def load_profile(self) -> Optional[AuthorProfile]:
        profile_path = self.author_dir / "profile.json"
        if not profile_path.exists():
            return None

        with open(profile_path, "r") as f:
            data = json.load(f)
        return AuthorProfile(**data)

    def save_dataset(self, dataset: Dataset) -> None:
        dataset_path = self.author_dir / "train.jsonl"
        with jsonlines.open(dataset_path, "w") as writer:
            for example in dataset.examples:
                writer.write(example.model_dump())

    def load_dataset(self) -> Optional[Dataset]:
        dataset_path = self.author_dir / "train.jsonl"
        if not dataset_path.exists():
            return Dataset(author_id=self.author_id)

        examples = []
        with jsonlines.open(dataset_path, "r") as reader:
            for obj in reader:
                examples.append(TrainingExample(**obj))

        return Dataset(author_id=self.author_id, examples=examples)

    def save_model_metadata(self, metadata: ModelMetadata) -> None:
        models_path = self.author_dir / "models.json"
        with open(models_path, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)

    def load_model_metadata(self) -> ModelMetadata:
        models_path = self.author_dir / "models.json"
        if not models_path.exists():
            return ModelMetadata()

        with open(models_path, "r") as f:
            data = json.load(f)
        return ModelMetadata(**data)

    def exists(self) -> bool:
        return self.author_dir.exists() and (self.author_dir / "profile.json").exists()

    def _ensure_examples_directory(self) -> Path:
        """Ensure the examples directory exists for this author."""
        examples_dir = self.author_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        return examples_dir

    def save_example_as_markdown(
        self,
        prompt: str,
        response: str,
        example_type: str,
        timestamp: Optional[datetime] = None,
    ) -> Path:
        """Save a training example as a markdown file.

        Args:
            prompt: The user prompt
            response: The assistant response
            example_type: 'user' or 'llm'
            timestamp: Optional timestamp, defaults to current time

        Returns:
            Path to the created markdown file
        """
        return save_example_as_markdown(
            self.author_dir, prompt, response, example_type, timestamp
        )

    def _ensure_content_directory(self) -> Path:
        """Ensure the content directory exists for this author."""
        content_dir = self.author_dir / "content"
        content_dir.mkdir(exist_ok=True)
        return content_dir

    def save_generated_content(
        self,
        prompt: str,
        response: str,
        author_name: str,
        model_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Path:
        """Save generated content as a markdown file.

        Args:
            prompt: The user prompt
            response: The generated response
            author_name: The author's display name
            model_id: The model used for generation
            timestamp: Optional timestamp, defaults to current time

        Returns:
            Path to the created markdown file
        """
        return save_content_as_markdown(
            self.author_dir,
            self.author_id,
            author_name,
            prompt,
            response,
            model_id,
            timestamp,
        )

    @property
    def examples_dir(self) -> Path:
        """Get the examples directory path."""
        return self.author_dir / "examples"

    @property
    def content_dir(self) -> Path:
        """Get the content directory path."""
        return self.author_dir / "content"

    def _ensure_chats_directory(self) -> Path:
        """Ensure the chats directory exists for this author."""
        chats_dir = self.author_dir / "chats"
        chats_dir.mkdir(exist_ok=True)
        return chats_dir

    @property
    def chats_dir(self) -> Path:
        """Get the chats directory path."""
        return self.author_dir / "chats"

    def save_chat_session(self, session: ChatSession) -> Path:
        """Save a chat session to a JSON file.

        Args:
            session: The ChatSession to save

        Returns:
            Path to the saved session file
        """
        session_file = self.chats_dir / f"{session.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session.model_dump(), f, indent=2, default=str)
        return session_file

    def load_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a chat session from file.

        Args:
            session_id: The session ID to load

        Returns:
            ChatSession if found, None otherwise
        """
        session_file = self.chats_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        with open(session_file, "r") as f:
            data = json.load(f)
        return ChatSession(**data)

    def list_chat_sessions(self) -> List[str]:
        """List all chat session IDs for this author.

        Returns:
            List of session IDs sorted by modification time (newest first)
        """
        if not self.chats_dir.exists():
            return []

        sessions = []
        for session_file in self.chats_dir.glob("*.json"):
            sessions.append((session_file.stem, session_file.stat().st_mtime))

        # Sort by modification time, newest first
        sessions.sort(key=lambda x: x[1], reverse=True)
        return [session_id for session_id, _ in sessions]

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session file.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted successfully, False if file didn't exist
        """
        session_file = self.chats_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        return False

    def export_chat_session_as_markdown(
        self, session: ChatSession, author_name: str
    ) -> Path:
        """Export a chat session as markdown.

        Args:
            session: The chat session to export
            author_name: Display name of the author

        Returns:
            Path to the created markdown file
        """
        # Create markdown content
        lines = [
            f"# Chat Session - {author_name}",
            "",
            f"**Session ID:** {session.session_id}",
            f"**Created:** {session.created_at}",
            f"**Messages:** {session.message_count}",
            "",
        ]

        for msg in session.messages:
            role_display = (
                "🧑 **You**" if msg.role == "user" else f"🤖 **{author_name}**"
            )
            lines.extend(
                [
                    f"## {role_display}",
                    f"*{msg.timestamp}*",
                    "",
                    msg.content,
                    "",
                    "---",
                    "",
                ]
            )

        # Remove trailing separator
        if lines and lines[-1] == "" and lines[-2] == "---":
            lines = lines[:-2]

        markdown_content = "\n".join(lines)

        # Save to chats directory
        filename = f"chat_{session.session_id}_{session.created_at.strftime('%Y%m%d_%H%M%S')}.md"
        export_path = self.chats_dir / filename

        with open(export_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return export_path


def list_authors() -> List[str]:
    if not settings.authors_dir.exists():
        return []

    authors = []
    for author_dir in settings.authors_dir.iterdir():
        if author_dir.is_dir() and (author_dir / "profile.json").exists():
            authors.append(author_dir.name)

    return sorted(authors)


def get_author_profile(author_id: str) -> Optional[AuthorProfile]:
    storage = AuthorStorage(author_id)
    return storage.load_profile()
