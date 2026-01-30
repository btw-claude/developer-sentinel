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

### Docstring Convention

**Use Google style docstrings for all public functions, methods, and classes.**

This is enforced by pydocstyle with the Google convention.

#### Why Google Style?

Google style docstrings provide a clean, readable format that works well with:
- IDE tooltips and autocomplete
- Sphinx documentation generation
- Type hint integration

#### Format

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Short one-line summary of the function.

    Longer description if needed. This can span multiple lines and provides
    more context about what the function does.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with its default value.

    Returns:
        Description of what is returned.

    Raises:
        ValueError: Description of when this exception is raised.
        TypeError: Description of when this exception is raised.

    Example:
        ```python
        result = example_function("test", 20)
        ```
    """
    ...
```

#### Key Sections

- **Summary**: A brief one-line description (required)
- **Args**: Document all parameters (required if function has parameters)
- **Returns**: Document return value (required if function returns something)
- **Raises**: Document exceptions (required if function raises exceptions)
- **Example**: Usage examples (optional but encouraged for complex functions)
- **Note**: Additional notes (optional)
- **Attributes**: For class docstrings, document instance attributes

#### Class Docstrings

```python
class ExampleClass:
    """Short one-line summary of the class.

    Longer description of the class purpose and behavior.

    Attributes:
        name: Description of the name attribute.
        value: Description of the value attribute.

    Example:
        ```python
        obj = ExampleClass("example")
        ```
    """

    def __init__(self, name: str) -> None:
        """Initialize the ExampleClass.

        Args:
            name: The name for this instance.
        """
        self.name = name
        self.value = 0
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
pydocstyle src
mypy src
```

## Pull Request Process

1. Ensure all tests pass locally
2. Update documentation if needed
3. Add appropriate labels to your PR
4. Request review from maintainers
