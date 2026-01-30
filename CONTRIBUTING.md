# Contributing to Developer Sentinel

Thank you for your interest in contributing to Developer Sentinel! This document provides guidelines and conventions for contributing to the project.

## Code Style

### Logging Convention

**Use lazy evaluation (% formatting) for all log statements instead of f-strings.**

This is enforced by ruff rule G004.

#### Why?

F-strings in logger calls are always evaluated, even when the log level is disabled. This causes unnecessary performance overhead:

```python
# BAD - f-string is always evaluated, even if DEBUG logging is disabled
logger.debug(f"Processing issue {issue_key} with {len(items)} items")

# GOOD - formatting is deferred until the message is actually logged
logger.debug("Processing issue %s with %s items", issue_key, len(items))
```

#### Format Specifiers

Use appropriate format specifiers for different data types:

```python
# Strings and general objects
logger.info("Processing %s", issue_key)

# Multiple arguments
logger.info("Found %s issues in project %s", count, project_name)

# Floating point with precision (use f-string for the argument only)
logger.debug("Operation took %s seconds", f"{elapsed:.2f}")
```

#### Exception Logging

When logging exceptions, include the exception as an argument:

```python
logger.error("Failed to process %s: %s", issue_key, error)
```

### Other Conventions

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep line length to 100 characters (enforced by ruff)

## Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/unit -v

# Run all tests with coverage
pytest tests/unit -v --cov=sentinel --cov-report=xml

# Run linting
ruff check src tests
isort --check-only --diff src tests
mypy src
```

## Pull Request Process

1. Ensure all tests pass locally
2. Update documentation if needed
3. Add appropriate labels to your PR
4. Request review from maintainers
