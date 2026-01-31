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

# Floating point with precision
logger.debug("Operation took %.2f seconds", elapsed)
```

#### Exception Logging

Use the appropriate logging method based on whether you need the traceback:

```python
# For exceptions where you want the full traceback:
# NOTE: logger.exception() automatically captures exception info from sys.exc_info()
# and should ONLY be called from within an exception handler (try/except block)
try:
    process_issue(issue_key)
except ProcessingError:
    logger.exception("Failed to process %s", issue_key)

# For errors without traceback:
logger.error("Failed to process %s: %s", issue_key, error)
```

**Important:** `logger.exception()` automatically includes the current exception's traceback by
calling `sys.exc_info()` internally. If called outside an exception handler, it will log
`NoneType: None` instead of useful exception information.

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

#### Docstring Inheritance

Python supports docstring inheritance through the `__doc__` attribute, but pydocstyle does not
recognize inherited docstrings when checking for missing documentation. This means you may see
warnings for methods that intentionally rely on parent class documentation.

**When to add explicit docstrings:**

```python
class BaseProcessor:
    """Base class for all processors."""

    def process(self, data: str) -> str:
        """Process the input data.

        Args:
            data: The input data to process.

        Returns:
            The processed data.
        """
        raise NotImplementedError


class ConcreteProcessor(BaseProcessor):
    """Concrete implementation of the processor."""

    def process(self, data: str) -> str:
        """Process the input data by converting to uppercase.

        Args:
            data: The input data to process.

        Returns:
            The processed data in uppercase.
        """
        # Explicit docstring needed because pydocstyle checks each class independently
        return data.upper()
```

**When inheritance applies (but pydocstyle still warns):**

```python
from abc import ABC, abstractmethod


class Repository(ABC):
    """Abstract repository interface."""

    @abstractmethod
    def save(self, entity: dict) -> None:
        """Save an entity to the repository.

        Args:
            entity: The entity to save.
        """
        ...


class InMemoryRepository(Repository):
    """In-memory implementation of Repository."""

    def save(self, entity: dict) -> None:
        # pydocstyle may warn here, but the docstring is inherited at runtime
        # Add an explicit docstring if implementation details differ significantly
        self._storage.append(entity)
```

**Best practices:**

1. **Always add docstrings for methods with different behavior** - If your implementation has
   specific behavior, side effects, or exceptions not documented in the parent, add an explicit
   docstring.

2. **Consider adding brief docstrings for clarity** - Even when inheriting, a short docstring
   like `"""See base class."""` can satisfy pydocstyle and signal intentional inheritance.

3. **Use `# noqa: D102` sparingly** - If a method truly just implements the parent interface
   with no additional behavior, you can suppress the warning:
   ```python
   def save(self, entity: dict) -> None:  # noqa: D102
       self._storage.append(entity)
   ```

**Project configuration:**

Our pydocstyle configuration in `pyproject.toml` ignores certain rules to reduce noise:

- `D102`: Missing docstring in public method - allows simple implementations to omit docstrings
- `D105`: Missing docstring in magic method - magic methods are self-documenting
- `D107`: Missing docstring in `__init__` - class docstring is often sufficient

These relaxed rules help reduce warnings for inherited and obvious methods while still enforcing
documentation for public APIs.

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
