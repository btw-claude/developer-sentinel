# Test Suite Documentation

This directory contains unit and integration tests for Developer Sentinel.

## Directory Structure

```
tests/
├── unit/           # Unit tests (fast, no external dependencies)
├── integration/    # Integration tests (require external services)
├── conftest.py     # Shared pytest fixtures
├── helpers.py      # Shared test helper functions
└── mocks.py        # Mock implementations for testing
```

## Running Tests

### Unit Tests
```bash
pytest tests/unit -v
```

### Integration Tests
```bash
pytest tests/integration -v -m integration
```

## Mock Usage Guidelines

The `tests/mocks.py` module provides reusable mock implementations of core interfaces.
When writing tests, follow these guidelines to ensure maintainable and reliable tests.

### Core Principle: Simulate Behavior, Don't Inspect Internals

**Mocks should simulate real behavior by maintaining their own state**, rather than
inspecting the internal implementation details of other mocks.

**Bad Pattern (avoid):**
```python
# DON'T: Access internal state of other mocks
def search_issues(self, jql, max_results):
    if self.tag_client:
        removed = self.tag_client.remove_calls  # Accessing internal state!
        for issue in self.issues:
            if any(key == issue.key for key, _ in removed):
                continue
```

**Good Pattern (use this):**
```python
# DO: Maintain your own state, updated via observer pattern
def search_issues(self, jql, max_results):
    # Filter based on own state, which is updated by observer callbacks
    return [i for i in self.issues if i.key not in self.processed_issue_keys]
```

### Direct Instantiation

**Prefer direct instantiation** over pytest fixtures for most tests:

```python
from tests.mocks import MockJiraClient, MockAgentClient, MockTagClient

def test_example():
    tag_client = MockTagClient()
    jira_client = MockJiraClient(issues=[...], tag_client=tag_client)
    agent_client = MockAgentClient(responses=["SUCCESS"])
    # ... use in test ...
```

**Rationale:**
- Explicit setup makes test behavior immediately clear
- No hidden dependencies through fixture injection
- Parameters are visible at the call site

### Available Mock Classes

#### MockTagClient

Simulates Jira label operations. Supports an observer pattern for notifying
other mocks when labels are removed.

```python
tag_client = MockTagClient()

# Add/remove labels
tag_client.add_label("PROJ-123", "needs-review")
tag_client.remove_label("PROJ-123", "needs-review")

# Check recorded calls
assert ("PROJ-123", "needs-review") in tag_client.add_calls
assert ("PROJ-123", "needs-review") in tag_client.remove_calls
```

##### Observer Pattern

The `MockTagClient` implements a complete observer pattern for label removal
notifications. Observers can be registered to receive callbacks when labels
are removed, and can also be unregistered when no longer needed.

Observers are stored using weak references, so bound methods on objects that
are garbage collected will be automatically cleaned up. This prevents memory
leaks when mocks are reused across multiple tests.

```python
tag_client = MockTagClient()

# Define an observer callback
def on_label_removed(issue_key: str, label: str) -> None:
    print(f"Label {label} removed from {issue_key}")

# Register the observer
tag_client.register_observer(on_label_removed)

# The callback will be invoked when labels are removed
tag_client.remove_label("PROJ-123", "needs-review")

# Explicit unregistration (optional - observers are also auto-cleaned via weak refs)
tag_client.unregister_observer(on_label_removed)
```

For bound methods, weak references work automatically:

```python
class MyHandler:
    def __init__(self, tag_client: MockTagClient) -> None:
        tag_client.register_observer(self.on_label_removed)

    def on_label_removed(self, issue_key: str, label: str) -> None:
        print(f"Label {label} removed from {issue_key}")

tag_client = MockTagClient()
handler = MyHandler(tag_client)
# When handler is garbage collected, the observer is automatically cleaned up
```

#### MockJiraClient

Simulates Jira search operations. Automatically filters out processed issues
when linked to a MockTagClient.

```python
tag_client = MockTagClient()
jira_client = MockJiraClient(
    issues=[{"key": "PROJ-123", "fields": {...}}],
    tag_client=tag_client  # Links for automatic filtering
)

# Issues are filtered when their labels are removed
tag_client.remove_label("PROJ-123", "trigger-tag")
results = jira_client.search_issues("labels = trigger-tag")
assert len(results) == 0  # PROJ-123 is now filtered out
```

#### MockJiraPoller

Simulates the JiraPoller for polling trigger-tagged issues.

```python
tag_client = MockTagClient()
poller = MockJiraPoller(
    issues=[JiraIssue(...)],
    tag_client=tag_client
)

# Poll returns issues
results = poller.poll(trigger_config)

# After processing (label removed), issue is filtered
tag_client.remove_label("PROJ-123", "needs-agent")
results = poller.poll(trigger_config)
# PROJ-123 is no longer returned
```

#### MockAgentClient

Simulates agent execution without running actual agents.

```python
agent_client = MockAgentClient(
    responses=["SUCCESS: Task completed", "FAILURE: Something went wrong"],
    workdir=Path("/tmp/work")
)

# Responses cycle through the list
result1 = await agent_client.run_agent(prompt, tools)  # Returns first response
result2 = await agent_client.run_agent(prompt, tools)  # Returns second response
```

### Error Simulation

MockAgentClient supports error simulation for testing error handling:

```python
agent_client = MockAgentClient(responses=["SUCCESS"])
agent_client.should_error = True
agent_client.max_errors = 2

# First two calls raise AgentClientError
# Third call succeeds
```

### Testing Async Code

Use `pytest-asyncio` for async tests:

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    agent_client = MockAgentClient(responses=["SUCCESS"])
    result = await agent_client.run_agent("prompt", ["tool"])
    assert "SUCCESS" in result.response
```

## Writing New Tests

1. **Use mocks for external dependencies** - Don't make real API calls in unit tests
2. **Test behavior, not implementation** - Focus on what the code does, not how
3. **Keep tests isolated** - Each test should be independent
4. **Use descriptive test names** - `test_search_issues_filters_processed_items`
5. **Follow the AAA pattern** - Arrange, Act, Assert
