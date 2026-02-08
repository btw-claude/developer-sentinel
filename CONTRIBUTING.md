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
- `D214`: Section is over-indented - flexible indentation in examples is acceptable
- `D301`: Use r""" if any backslashes - raw strings not needed for well-escaped content
- `D402`: First line should not be function's signature - some property docstrings are fine
- `D403`: First word should be capitalized - false positives for proper nouns (e.g., "GitHub")

These relaxed rules help reduce warnings for inherited and obvious methods while still enforcing
documentation for public APIs.

### Test Docstring Cross-References

**Use cross-referencing docstrings to link related tests that cover overlapping behavior.**

When two or more tests exercise similar or complementary aspects of the same feature, their
docstrings should explicitly reference each other and explain each test's distinct focus. This
helps developers understand why seemingly similar tests both exist without reading the full test
body.

#### Reference Patterns

Use one of two reference patterns depending on the relationship between tests:

- **"See also"** (forward reference): Use when pointing to a test that covers a complementary
  aspect of the same behavior. This is a neutral pointer that says "there's more to explore."
- **"Unlike"** (contrast reference): Use when highlighting how the current test differs from
  a related one. This draws attention to a specific distinction.

Both patterns can be combined in the same docstring when appropriate.

#### Two-Way vs. Three-Way (or More) Cross-References

- **Two-way references** are the default. Use them when exactly two tests form a natural pair
  covering complementary aspects of the same behavior (e.g., success vs. failure, enabled vs.
  disabled, sanitization vs. truncation).
- **Three-way (or more) references** are appropriate when three or more tests cover distinct
  variants of the same behavior and the set is small enough to enumerate without clutter
  (e.g., omitted vs. empty-string vs. whitespace-only input, or basic vs. with-reset vs.
  restore-on-exit). Use parenthetical annotations to keep each variant distinguishable at a
  glance.
- **Avoid exhaustive cross-references** when the set of related tests exceeds roughly four.
  In such cases, reference only the most closely related tests or reference a shared grouping
  (e.g., "See the other `test_parse_*_field` tests in this class for analogous validation").

#### Format and Placement

Cross-references belong in the extended description of the docstring, after the summary line
and any behavioral explanation. Place them at the end of the description paragraph, before
any Args/Returns/Raises sections.

#### Examples

**Two-way "See also" reference:**

```python
def test_transitions_to_open_after_failure_threshold(self) -> None:
    """Test circuit opens after reaching failure threshold.

    Verifies the transition condition: consecutive failures equal to the
    threshold move the circuit from CLOSED to OPEN. See also
    test_rejects_requests_when_open, which tests the resulting behavior
    once the circuit is in the OPEN state.
    """
    ...

def test_rejects_requests_when_open(self) -> None:
    """Test that requests are rejected when circuit is open.

    Unlike test_transitions_to_open_after_failure_threshold (which verifies
    the transition condition from CLOSED to OPEN), this test focuses on the
    observable behavior once the circuit is already open: requests are
    rejected with CircuitOpenError.
    """
    ...
```

**Three-way reference with parenthetical annotations:**

```python
def test_create_branch_true_without_branch_pattern_raises_error(self) -> None:
    """Test that create_branch=True without a branch pattern raises error.

    Validates the case where the branch field is omitted entirely. See also
    test_create_branch_true_with_empty_branch_raises_error (explicit empty
    string) and test_create_branch_true_with_whitespace_only_branch_raises_error
    (whitespace-only string) for the same validation under different empty
    branch conditions.
    """
    ...
```

**Shared-grouping reference (many related tests):**

When more than roughly four tests cover analogous behavior (e.g., field-parsing
validation across many fields), avoid enumerating every sibling. Instead,
reference the shared naming pattern and the containing class:

```python
def test_parse_priority_field_missing(self) -> None:
    """Test that a missing priority field falls back to the default.

    Validates that omitting the priority key from the YAML mapping
    produces the documented default value without raising an error.
    See the other test_parse_*_field tests in this class for
    analogous validation of remaining orchestration fields.
    """
    ...
```

This keeps docstrings concise while still directing developers to the full
family of related tests. Use it whenever the set of siblings is large enough
that listing each one individually would add more noise than signal.

**Preserving existing context alongside cross-references:**

```python
def test_run_agent_context_sanitizes_newlines(self) -> None:
    """Test that newline characters in context are replaced with spaces.

    Verifies that newline and carriage return characters in context
    keys and values are replaced with spaces to prevent prompt
    injection from untrusted sources (DS-666). See also
    test_run_agent_context_truncates_long_values, which tests the
    complementary length-based truncation of context keys and values.
    """
    ...
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
