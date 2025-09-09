import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def sanitize_subject(text: str, max_length: int = 30) -> str:
    """Sanitize text for use in filename, keeping only safe characters."""
    # Remove special characters, keep only alphanumeric and common separators
    sanitized = re.sub(r"[^\w\s-]", "", text.lower())
    # Replace spaces and multiple separators with single underscores
    sanitized = re.sub(r"[\s_-]+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("_")

    return sanitized or "example"


def generate_markdown_filename(
    example_type: str, prompt: str, timestamp: Optional[datetime] = None
) -> str:
    """Generate a filename for markdown example files.

    Format: {type}_{subject}_{datetime}.md

    Args:
        example_type: 'user' or 'llm'
        prompt: The user prompt text to derive subject from
        timestamp: Optional timestamp, defaults to current time

    Returns:
        Generated filename string
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Extract subject from prompt
    subject = sanitize_subject(prompt)

    # Format timestamp
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

    return f"{example_type}_{subject}_{timestamp_str}.md"


def create_markdown_content(
    prompt: str,
    response: str,
    example_type: str,
    timestamp: Optional[datetime] = None,
    title: Optional[str] = None,
) -> str:
    """Create markdown content for a training example.

    Args:
        prompt: The user prompt
        response: The assistant response
        example_type: 'user' or 'llm'
        timestamp: Optional timestamp, defaults to current time
        title: Optional custom title, defaults to prompt (truncated)

    Returns:
        Formatted markdown string
    """
    if timestamp is None:
        timestamp = datetime.now()

    if title is None:
        # Use first 50 characters of prompt as title
        title = prompt[:50].strip()
        if len(prompt) > 50:
            title += "..."

    # Format timestamp for display
    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return f"""# {title}

**Type:** {example_type}  
**Created:** {formatted_time}  
**Prompt:** {prompt}

## Content

{response}
"""


def ensure_examples_directory(author_dir: Path) -> Path:
    """Ensure the examples directory exists for an author.

    Args:
        author_dir: Path to the author's directory

    Returns:
        Path to the examples directory
    """
    examples_dir = author_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    return examples_dir


def save_example_as_markdown(
    author_dir: Path,
    prompt: str,
    response: str,
    example_type: str,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Save a training example as a markdown file.

    Args:
        author_dir: Path to the author's directory
        prompt: The user prompt
        response: The assistant response
        example_type: 'user' or 'llm'
        timestamp: Optional timestamp, defaults to current time

    Returns:
        Path to the created markdown file
    """
    examples_dir = ensure_examples_directory(author_dir)

    if timestamp is None:
        timestamp = datetime.now()

    filename = generate_markdown_filename(example_type, prompt, timestamp)
    filepath = examples_dir / filename

    content = create_markdown_content(prompt, response, example_type, timestamp)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def generate_content_filename(
    author_id: str, prompt: str, timestamp: Optional[datetime] = None
) -> str:
    """Generate a filename for generated content files.

    Format: {author_id}_{datetime}_{prompt_10chars}.md

    Args:
        author_id: The author's identifier
        prompt: The user prompt text to derive subject from
        timestamp: Optional timestamp, defaults to current time

    Returns:
        Generated filename string
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Extract first 10 characters from prompt, sanitized
    prompt_chars = sanitize_subject(prompt, max_length=10)

    # Format timestamp
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

    return f"{author_id}_{timestamp_str}_{prompt_chars}.md"


def create_content_markdown(
    prompt: str,
    response: str,
    author_name: str,
    model_id: str,
    timestamp: Optional[datetime] = None,
) -> str:
    """Create markdown content for generated text.

    Args:
        prompt: The user prompt
        response: The generated response
        author_name: The author's display name
        model_id: The model used for generation
        timestamp: Optional timestamp, defaults to current time

    Returns:
        Formatted markdown string
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Format timestamp for display
    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return f"""# Generated Content

**Author:** {author_name}  
**Model:** {model_id}  
**Created:** {formatted_time}  
**Prompt:** {prompt}

## Content

{response}
"""


def ensure_content_directory(author_dir: Path) -> Path:
    """Ensure the content directory exists for an author.

    Args:
        author_dir: Path to the author's directory

    Returns:
        Path to the content directory
    """
    content_dir = author_dir / "content"
    content_dir.mkdir(exist_ok=True)
    return content_dir


def save_content_as_markdown(
    author_dir: Path,
    author_id: str,
    author_name: str,
    prompt: str,
    response: str,
    model_id: str,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Save generated content as a markdown file.

    Args:
        author_dir: Path to the author's directory
        author_id: The author's identifier
        author_name: The author's display name
        prompt: The user prompt
        response: The generated response
        model_id: The model used for generation
        timestamp: Optional timestamp, defaults to current time

    Returns:
        Path to the created markdown file
    """
    content_dir = ensure_content_directory(author_dir)

    if timestamp is None:
        timestamp = datetime.now()

    filename = generate_content_filename(author_id, prompt, timestamp)
    filepath = content_dir / filename

    content = create_content_markdown(
        prompt, response, author_name, model_id, timestamp
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath
