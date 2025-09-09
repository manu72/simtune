# Simtune Testing Makefile

.PHONY: help install test test-unit test-integration test-fast test-coverage clean lint format type-check security

help: ## Show this help message
	@echo "Simtune Test Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies including test dependencies
	pip install -r requirements.txt

test: ## Run all tests with coverage
	python -m pytest

test-unit: ## Run only unit tests
	python -m pytest tests/unit -v

test-integration: ## Run only integration tests
	python -m pytest tests/integration -v

test-fast: ## Run tests excluding slow ones
	python -m pytest -m "not slow"

test-coverage: ## Generate detailed coverage report
	python -m pytest --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

test-specific: ## Run specific test (use TEST=path/to/test)
	python -m pytest $(TEST) -v

clean: ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete -type d

lint: ## Run linting checks
	flake8 core cli tests --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 core cli tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Format code with black and isort
	black core cli tests
	isort core cli tests

format-check: ## Check code formatting without making changes
	black --check core cli tests
	isort --check-only core cli tests

type-check: ## Run type checking with mypy
	mypy core cli --ignore-missing-imports

security: ## Run security scan with bandit
	bandit -r core cli

check: format-check lint type-check security ## Run all code quality checks

ci: install test lint type-check security ## Run full CI pipeline

dev-install: ## Install in development mode with pre-commit hooks
	pip install -r requirements.txt
	pip install pre-commit black isort flake8 mypy bandit
	pre-commit install

# Test shortcuts
ut: test-unit ## Shortcut for unit tests
it: test-integration ## Shortcut for integration tests
cov: test-coverage ## Shortcut for coverage report