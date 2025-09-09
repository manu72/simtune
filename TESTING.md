# Simtune Testing Guide

This document describes the comprehensive testing strategy and setup for the Simtune project.

## Overview

The Simtune project uses a robust testing framework built on pytest with comprehensive coverage of:
- Core business logic (unit tests)
- External API integrations (mocked)
- CLI workflows (integration tests)
- File I/O and storage operations
- Error handling and edge cases

## Quick Start

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
make test
# or
python -m pytest

# Run only unit tests
make test-unit
# or
python -m pytest tests/unit

# Run only integration tests
make test-integration
# or
python -m pytest tests/integration

# Run with coverage report
make test-coverage
# or
python -m pytest --cov-report=html --cov-report=term
```

### Test Runner Script

For convenience, use the test runner script:

```bash
# Check dependencies
python run_tests.py deps

# Run unit tests
python run_tests.py unit

# Run all tests with coverage
python run_tests.py all

# Run fast tests (excluding slow ones)
python run_tests.py fast

# Run specific test
python run_tests.py specific --test-path tests/unit/test_models.py
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                     # Shared fixtures and configuration
├── unit/                          # Unit tests
│   ├── test_models.py            # Core model validation tests
│   ├── test_storage.py           # File I/O and storage tests
│   ├── test_config.py            # Configuration tests
│   └── core/
│       ├── test_dataset_builder.py     # Dataset building logic
│       └── adapters/
│           └── test_openai_adapter.py  # API adapter tests
├── integration/                   # Integration tests
│   ├── test_cli_workflows.py     # End-to-end CLI command testing
│   └── test_data_persistence.py  # Data storage integration tests
└── fixtures/                     # Test data and fixtures
    ├── sample_datasets/
    ├── author_profiles/
    └── api_responses/
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **Models (`test_models.py`)**: Pydantic model validation, properties, and business logic
- **Storage (`test_storage.py`)**: File operations, JSON/YAML/JSONL handling, error cases
- **Config (`test_config.py`)**: Settings and environment variable handling
- **Dataset Builder (`test_dataset_builder.py`)**: Content processing, file imports, example generation
- **OpenAI Adapter (`test_openai_adapter.py`)**: API interactions with comprehensive mocking

### Integration Tests (`tests/integration/`)

Test complete workflows and component interactions:

- **CLI Workflows (`test_cli_workflows.py`)**: End-to-end command testing with Typer
- **Data Persistence**: Cross-component data flow testing

## Test Fixtures

### Shared Fixtures (`conftest.py`)

- `sample_author_profile`: Complete AuthorProfile with StyleGuide
- `sample_dataset`: Dataset with multiple TrainingExamples
- `sample_fine_tune_job`: FineTuneJob for testing workflows
- `mock_openai_client`: Comprehensive OpenAI client mock
- `temp_data_dir`: Isolated temporary directory for file operations

### File Fixtures (`tests/fixtures/`)

- Sample datasets in JSONL format
- Author profile JSON files
- OpenAI API response mocks
- Text files for import testing

## Mocking Strategy

### External APIs

All OpenAI API calls are mocked using:
- `responses` library for HTTP-level mocking
- `unittest.mock` for client-level mocking
- Comprehensive error scenario testing

### File System

- `tmp_path` pytest fixture for isolated file operations
- Mock file I/O for error condition testing
- Temporary directories for integration tests

### CLI Interactions

- Typer CliRunner for command testing
- Mock user inputs with `unittest.mock`
- Rich console output verification

## Coverage Requirements

- **Minimum Coverage**: 80% (enforced by pytest)
- **Target Coverage**: 90%+ for core business logic
- **Reports**: HTML, XML, and terminal formats available

### Viewing Coverage

```bash
# Generate HTML report
make test-coverage
open htmlcov/index.html

# Terminal report
python -m pytest --cov-report=term-missing
```

## Test Markers

Use pytest markers to categorize tests:

```bash
# Run only fast tests
python -m pytest -m "not slow"

# Run only integration tests
python -m pytest -m integration

# Run only OpenAI-related tests
python -m pytest -m openai
```

Available markers:
- `slow`: Long-running tests
- `integration`: Integration tests
- `openai`: Tests involving OpenAI API
- `unit`: Unit tests
- `cli`: CLI-specific tests

## Continuous Integration

### GitHub Actions

The project includes automated testing via GitHub Actions:
- Tests on Python 3.8, 3.9, 3.10, 3.11
- Code quality checks (linting, formatting, type checking)
- Security scanning with bandit
- Coverage reporting to Codecov

### Pre-commit Hooks

Set up development environment with pre-commit hooks:

```bash
make dev-install
```

## Writing New Tests

### Guidelines

1. **Follow AAA Pattern**: Arrange, Act, Assert
2. **Use Descriptive Names**: Test names should explain what they test
3. **Test Edge Cases**: Include error conditions and boundary cases
4. **Mock External Dependencies**: Don't hit real APIs or file systems
5. **Keep Tests Independent**: Each test should be isolated

### Example Test

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

### Testing CLI Commands

```python
def test_author_list_command():
    """Test author list CLI command."""
    with patch('cli.commands.author.list_authors', return_value=['author1']):
        runner = CliRunner()
        result = runner.invoke(app, ["author", "list"])
        
        assert result.exit_code == 0
        assert "author1" in result.stdout
```

### Testing with Fixtures

```python
def test_dataset_operations(sample_dataset, temp_data_dir, mock_settings):
    """Test dataset save/load operations."""
    with patch('core.storage.settings', mock_settings):
        storage = AuthorStorage("test_author")
        storage.save_dataset(sample_dataset)
        
        loaded_dataset = storage.load_dataset()
        assert loaded_dataset.size == sample_dataset.size
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **File Permission Errors**: Use temporary directories for file tests
3. **Mock Issues**: Verify mock objects have expected methods/attributes
4. **Fixture Scope**: Use appropriate fixture scopes for performance

### Debugging Tests

```bash
# Run with verbose output
python -m pytest -v

# Stop on first failure
python -m pytest -x

# Drop into debugger on failure
python -m pytest --pdb

# Show local variables on failure
python -m pytest -l

# Run specific test with debugging
python -m pytest tests/unit/test_models.py::TestAuthorProfile::test_author_profile_creation -v -s
```

## Performance Considerations

- Mark slow tests with `@pytest.mark.slow`
- Use appropriate fixture scopes to avoid recreation
- Mock heavy operations (file I/O, API calls)
- Consider parallel test execution for large test suites

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure tests cover happy path and error cases
3. Update fixtures if new test data is needed
4. Maintain or improve coverage percentage
5. Add integration tests for new CLI commands
6. Update this documentation for significant changes

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-mock plugin](https://pytest-mock.readthedocs.io/)
- [responses library](https://github.com/getsentry/responses)
- [Typer testing](https://typer.tiangolo.com/tutorial/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)