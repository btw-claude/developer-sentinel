# Testing Guide

This document describes the testing strategy and structure for Developer Sentinel.

## Test Organization

Tests are organized into two categories based on their dependencies and execution requirements:

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures and test configuration
├── helpers.py           # Test helper functions
├── mocks.py             # Mock implementations
├── unit/                # Unit tests (fast, no external dependencies)
│   ├── __init__.py
│   └── test_*.py
└── integration/         # Integration tests (may require external services)
    ├── __init__.py
    └── test_*.py
```

### Unit Tests (`tests/unit/`)

Unit tests are **fast, isolated tests** that:
- Do not require external services (APIs, databases, etc.)
- Use mocks for all external dependencies
- Can run in any environment without special setup
- Should complete quickly (typically <1 second per test)

**Examples of unit tests:**
- Testing configuration parsing
- Testing data validation logic
- Testing pure functions
- Testing class methods with mocked dependencies

### Integration Tests (`tests/integration/`)

Integration tests are tests that:
- Require external services or real dependencies
- May need specific environment setup (CLI tools, network access, etc.)
- Are marked with `@pytest.mark.integration`
- May take longer to execute

**Examples of integration tests:**
- Tests that call real CLI tools (e.g., `claude` CLI)
- Tests that make actual network requests
- Tests that require real file system operations beyond temp files

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Only Unit Tests

```bash
pytest tests/unit
```

This is the fastest way to run tests during development.

### Run Only Integration Tests

```bash
pytest tests/integration -m integration
```

### Run All Tests Except Integration Tests

```bash
pytest -m "not integration"
```

This is useful for CI pipelines where integration tests should run separately.

### Run Tests with Coverage

```bash
pytest --cov=sentinel --cov-report=html
```

## Writing Tests

### Unit Test Guidelines

1. **Use mocks from `tests/mocks.py`:**
   ```python
   from tests.mocks import MockJiraClient, MockAgentClient

   def test_example():
       jira_client = MockJiraClient(issues=[...])
       # ... test code
   ```

2. **Use helpers from `tests/helpers.py`:**
   ```python
   from tests.helpers import make_config, make_orchestration

   def test_with_config():
       config = make_config(poll_interval=30)
       # ... test code
   ```

3. **Keep tests focused and fast:**
   - One assertion per test when possible
   - No network calls or subprocess execution
   - No sleeps or time-dependent operations

### Integration Test Guidelines

1. **Mark tests with `@pytest.mark.integration`:**
   ```python
   import pytest

   @pytest.mark.integration
   def test_cli_execution():
       # Test that requires external CLI
       ...
   ```

2. **Handle missing dependencies gracefully:**
   ```python
   @pytest.mark.integration
   def test_with_optional_dependency(self):
       if not self._is_dependency_available():
           pytest.skip("Required dependency not available")
       # ... test code
   ```

3. **Document requirements in test docstrings:**
   ```python
   @pytest.mark.integration
   def test_with_claude_cli():
       """Test CLI integration.

       Requires: claude CLI installed and configured.
       """
       ...
   ```

## CI Configuration

The GitHub Actions workflow (`/.github/workflows/tests.yml`) is configured to:

1. **Run unit tests** on every push and PR (fast, required to pass)
2. **Run integration tests** separately (may be skipped if dependencies unavailable)
3. **Run linting** (ruff, mypy) for code quality

Unit tests run across multiple Python versions (3.11, 3.12, 3.13) to ensure compatibility.
