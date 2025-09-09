# Simtune

Simtune is a tool that empowers non technical users to create their own personal writing AI.
It helps you finetune a language model so it learns your style, tone, and preferences.
Over time, your Simtune becomes a unique voice or author that improves through feedback and edits you provide.

---

## Vision

The goal of Simtune is to democratise finetuning. Anyone, not just developers, should be able to:

- Collect and structure a small dataset of writing samples.
- Fine-tune an existing large language model (LLM) into a personal voice or author.
- Generate content that reflects their voice: blog posts, articles, essays, or even books.
- Provide feedback on drafts and see the author evolve continuously.

---

## Features (Stage 1)

- **Dataset Builder**: Guided prompts help you prepare a small training dataset in the correct format for the LLM.
- **Fine-Tune Runner**: Simple CLI to start a finetuning job with a commercial LLM.
- **Author Runtime**: Generate drafts from your tuned author with a single command.
- **Content Persistence**: All generated content is automatically saved as markdown files with metadata.
- **Feedback Loop**: Rate or edit drafts, turning feedback into new training examples.

---

## Roadmap

### Stage 1: Terminal based POC

- CLI tool for dataset building, validation, and fine-tuning.
- Adaptor for a single commercial provider (OpenAI or Gemini).
- Feedback mechanism for turning edits into new examples.
- Local storage of datasets (.jsonl) and author profiles.

### Stage 2: Basic Browser UI

- Minimal web interface to guide dataset creation and fine-tuning.
- Inline editor for reviewing drafts and logging feedback.
- Backend built on FastAPI or Flask, frontend with React or Svelte or Streamlit.
- users will provide their own API keys

### Stage 3: Multiple model support

- Support multiple LLMs (OpenAI, Gemini).
- Local model support via Ollama (DeepSeek, Mistral Small).
- LoRA or PEFT-based fine-tuning for open models.

### Stage 4: Multiple user accounts

- Secure account creation and login.
- Each user has isolated storage and their own fine-tuned authors.
- Bring-your-own-key support for provider APIs.

### Stage 5: Production ready UI/UX

- Full author dashboard with progress tracking and dataset versioning.
- Job queueing, monitoring, and error handling.
- Export, backup, and delete-my-data options.
- Observability and cost management tools.

---

## Project Structure

```
simtune/
├── cli/                    # Typer-based CLI entrypoints
├── core/
│   ├── adapters/          # openai_adapter.py, gemini_adapter.py
│   ├── dataset/           # builders, validators, splitters
│   ├── feedback/          # edit diff, new examples from feedback
│   ├── eval/              # style metrics, safety checks
│   └── prompts/           # system templates
├── data/
│   └── authors/<author_id>/
│       ├── style_guide.yml
│       ├── train.jsonl
│       ├── examples/      # training examples as markdown
│       ├── content/       # generated content as markdown
│       ├── edits/
│       └── models.json    # fine-tune job metadata
├── .env.example           # API keys and secrets
├── requirements.txt
└── README.md
```

---

## Quickstart (Stage 1 CLI)

1. Clone the repo:

```bash
git clone https://github.com/manu72/simtune.git
cd simtune
```

2. Set up a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Copy `.env.example` → `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY and/or GEMINI_API_KEY
```

4. Run the CLI:

```bash
python -m cli.main init
```

---

## CLI Commands

### Author Management

```bash
# Initialize a new author profile
python -m cli.main init

# List all authors
python -m cli.main list
```

### Dataset Building

```bash
# Build training dataset interactively
python -m cli.main dataset build <author_id>

# Import examples from files
python -m cli.main dataset import <author_id> <file_path>
```

### Fine-tuning

```bash
# Start fine-tuning job
python -m cli.main train start <author_id>

# Check training status
python -m cli.main train status <author_id>

# Wait for completion
python -m cli.main train wait <author_id>
```

### Content Generation

```bash
# Generate single piece of content
python -m cli.main generate text <author_id> --prompt "Write about productivity"

# Disable content saving
python -m cli.main generate text <author_id> --prompt "Quick note" --no-save

# Interactive generation session
python -m cli.main generate interactive <author_id>
```

---

## Status

Currently in Stage 1 (terminal-based proof of concept). Expect rapid iteration.

---

## Testing

Simtune includes a comprehensive test suite with 90%+ coverage. The testing framework uses pytest with extensive mocking for external APIs and file operations.

### Quick Start

```bash
# Install all dependencies (including test tools)
pip install -r requirements.txt

# Run all tests
make test
# or
python -m pytest

# Run with coverage report
make test-coverage
# or
python -m pytest --cov-report=html --cov-report=term
```

### Test Commands

#### Using Make (Recommended)

```bash
make test           # Run all tests with coverage
make test-unit      # Run only unit tests
make test-integration  # Run only integration tests
make test-fast      # Skip slow tests
make test-coverage  # Generate detailed coverage report
make clean          # Clean up generated files

# Code quality
make lint           # Run linting checks
make format         # Auto-format code with black/isort
make format-check   # Check formatting without changes
make type-check     # Run mypy type checking
make security       # Run bandit security scan
make check          # Run all quality checks

# Development setup
make dev-install    # Install with pre-commit hooks
```

#### Using pytest directly

```bash
# Basic test runs
python -m pytest                    # All tests
python -m pytest tests/unit         # Unit tests only
python -m pytest tests/integration  # Integration tests only
python -m pytest -m "not slow"      # Fast tests only

# With coverage
python -m pytest --cov=core --cov=cli --cov-report=html

# Specific tests
python -m pytest tests/unit/test_models.py                    # Single file
python -m pytest tests/unit/test_models.py::TestAuthorProfile # Single class
python -m pytest -k "test_author"                            # Pattern matching

# Debugging
python -m pytest -v               # Verbose output
python -m pytest -s               # Don't capture output
python -m pytest --pdb            # Drop into debugger on failure
python -m pytest -x               # Stop on first failure
```

#### Using the test runner script

```bash
python run_tests.py deps          # Check dependencies
python run_tests.py unit          # Unit tests
python run_tests.py integration   # Integration tests
python run_tests.py all           # All tests with coverage
python run_tests.py fast          # Fast tests only
python run_tests.py coverage      # Detailed coverage report
python run_tests.py specific --test-path tests/unit/test_models.py
```

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── unit/                          # Unit tests (90%+ coverage target)
│   ├── test_models.py            # Pydantic model validation
│   ├── test_config.py            # Settings and environment
│   └── core/
│       ├── test_dataset_builder.py    # Dataset processing logic
│       ├── test_markdown_utils.py     # Markdown content generation
│       ├── test_storage.py           # File I/O and data persistence
│       └── adapters/
│           └── test_openai_adapter.py # API integration (mocked)
├── integration/                   # End-to-end workflow tests
│   └── test_cli_workflows.py     # Complete CLI command testing
└── fixtures/                     # Sample data and API responses
    ├── sample_datasets/
    ├── author_profiles/
    └── api_responses/
```

### Test Coverage

- **Minimum Required**: 80% (enforced by CI/CD)
- **Target**: 90%+ for core business logic
- **Current Coverage**: View with `make test-coverage` and open `htmlcov/index.html`

#### Coverage by Module

- **Core Models**: Pydantic validation, properties, business logic
- **Storage System**: File I/O, JSON/YAML/JSONL handling, content persistence, error cases
- **Markdown Utils**: Content generation, filename sanitization, metadata handling
- **Dataset Builder**: Content processing, imports, user interactions
- **OpenAI Adapter**: Full API mocking, job management, error scenarios
- **CLI Commands**: End-to-end command testing with Typer

### Writing Tests

#### Guidelines

1. **Follow AAA Pattern**: Arrange, Act, Assert
2. **Use Descriptive Names**: `test_author_profile_creation_with_custom_style_guide`
3. **Test Edge Cases**: Include error conditions and boundary cases
4. **Mock External Dependencies**: Don't hit real APIs or file systems
5. **Keep Tests Independent**: Each test should be isolated

#### Example Test

```python
def test_author_profile_creation(sample_style_guide):
    """Test AuthorProfile creation with custom StyleGuide."""
    # Arrange
    profile_data = {
        "author_id": "test_author",
        "name": "Test Author",
        "style_guide": sample_style_guide
    }

    # Act
    profile = AuthorProfile(**profile_data)

    # Assert
    assert profile.author_id == "test_author"
    assert profile.name == "Test Author"
    assert profile.style_guide == sample_style_guide
```

### Continuous Integration

Tests run automatically on:

- **Push/PR to main/develop branches**
- **Multiple Python versions**: 3.8, 3.9, 3.10, 3.11
- **Code quality checks**: linting, formatting, type checking, security

#### GitHub Actions Jobs

- **test**: Run unit and integration tests across Python versions
- **lint**: Code formatting (black, isort), linting (flake8), typing (mypy)
- **security**: Security scanning with bandit

### Troubleshooting Tests

#### Common Issues

```bash
# Import errors - ensure PYTHONPATH is set
export PYTHONPATH=$PWD
python -m pytest

# Permission errors - use temp directories in tests
pytest --basetemp=/tmp/pytest

# Slow tests - run fast subset only
python -m pytest -m "not slow"

# Debug specific test
python -m pytest tests/unit/test_models.py::test_author_profile_creation -v -s --pdb
```

#### Test Markers

```bash
# Available markers
python -m pytest --markers

# Run by marker
python -m pytest -m unit           # Unit tests only
python -m pytest -m integration    # Integration tests only
python -m pytest -m "not slow"     # Exclude slow tests
python -m pytest -m openai         # OpenAI-related tests only
```

### Performance Testing

For performance-sensitive tests:

```bash
# Time test execution
python -m pytest --durations=10    # Show 10 slowest tests
python -m pytest --benchmark-only  # Run benchmark tests only
```

### Test Documentation

- **Comprehensive Guide**: See [TESTING.md](TESTING.md) for detailed documentation
- **API Documentation**: Tests serve as examples of API usage
- **Coverage Reports**: Generated in `htmlcov/` directory

---

## Contributing

This project is in early development. Contributions, feedback, and suggestions are welcome!

### Development Setup

```bash
# Clone and setup
git clone https://github.com/manu72/simtune.git
cd simtune
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup development tools
make dev-install  # Installs pre-commit hooks

# Before submitting PR
make check        # Run all quality checks
make test         # Ensure tests pass
```

### Code Quality Standards

- **Formatting**: Use `black` and `isort` for consistent code formatting
- **Linting**: Pass `flake8` and `mypy` checks without errors
- **Test Coverage**: Maintain 80%+ coverage for new code
- **Type Hints**: Use type annotations for all new functions
- **Documentation**: Update relevant docs for new features
- **Security**: No hardcoded secrets or unsafe patterns

### Pre-commit Checklist

Before submitting a PR:

```bash
# Format code
black .
isort .

# Run quality checks
flake8
mypy .

# Run tests
python -m pytest

# Or use the comprehensive check
make check  # if Makefile available
```

## License

MIT
