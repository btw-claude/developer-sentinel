# Dependency Injection Guide

This guide explains the dependency injection (DI) patterns used in Developer Sentinel and how contributors can leverage them for testing and extending the application.

## Overview

Developer Sentinel uses the [dependency-injector](https://python-dependency-injector.ets-labs.org/) library to manage dependencies. This approach provides:

- **Testability**: Easy swapping of implementations for unit and integration testing
- **Loose coupling**: Components depend on abstractions, not concrete implementations
- **Configuration management**: Centralized configuration through providers
- **Lifecycle management**: Singleton instances managed by the container

## Quick Start

### Production Usage

```python
from sentinel.container import create_container

# Create container with auto-configured providers
container = create_container()

# Get a fully configured Sentinel instance
sentinel = container.sentinel()

# Run the application
sentinel.run()
```

### Testing Usage

```python
from unittest.mock import Mock
from dependency_injector import providers
from sentinel.container import create_test_container

# Create test container
container = create_test_container()

# Override with mocks
mock_jira_client = Mock()
mock_jira_tag_client = Mock()
container.clients.jira_client.override(providers.Object(mock_jira_client))
container.clients.jira_tag_client.override(providers.Object(mock_jira_tag_client))

# Create Sentinel with mocked dependencies
sentinel = container.sentinel()
```

## Container Architecture

The DI container follows a hierarchical structure:

```
SentinelContainer (root)
├── config (Config)
├── orchestrations (list[Orchestration])
├── clients (ClientsContainer)
│   ├── jira_client (JiraClient)
│   ├── jira_tag_client (JiraTagClient)
│   ├── github_client (GitHubClient | None)
│   └── github_tag_client (GitHubTagClient | None)
└── services (ServicesContainer)
    ├── agent_factory (AgentClientFactory)
    ├── agent_logger (AgentLogger)
    └── tag_manager (TagManager)
```

### ClientsContainer

Groups all API client dependencies (Jira, GitHub). This makes it easy to swap entire client sets for different environments.

### ServicesContainer

Groups service-layer dependencies that coordinate between clients and business logic.

## Registered Services

### Jira Clients

| Provider | Type | Description |
|----------|------|-------------|
| `clients.jira_client` | `JiraClient` | Polls Jira for issues |
| `clients.jira_tag_client` | `JiraTagClient` | Manages Jira labels |

The container automatically selects between REST and SDK implementations based on configuration:
- **REST clients**: Used when `JIRA_BASE_URL`, `JIRA_EMAIL`, and `JIRA_API_TOKEN` are set
- **SDK clients**: Used as fallback (slower, uses Claude for API calls)

### GitHub Clients

| Provider | Type | Description |
|----------|------|-------------|
| `clients.github_client` | `GitHubClient \| None` | Polls GitHub for issues/PRs |
| `clients.github_tag_client` | `GitHubTagClient \| None` | Manages GitHub labels |

GitHub clients are `None` when `GITHUB_TOKEN` is not configured.

### Services

| Provider | Type | Description |
|----------|------|-------------|
| `services.agent_factory` | `AgentClientFactory` | Creates agent clients (Claude, Cursor) |
| `services.agent_logger` | `AgentLogger` | Logs agent execution output |
| `services.tag_manager` | `TagManager` | Manages issue tags across platforms |

## Common Patterns

### Pattern 1: Override a Single Dependency

```python
from dependency_injector import providers
from sentinel.container import create_container

container = create_container()

# Override just the Jira client
container.clients.jira_client.override(
    providers.Object(my_custom_jira_client)
)
```

### Pattern 2: Test with Complete Mocks

```python
from unittest.mock import Mock, MagicMock
from dependency_injector import providers
from sentinel.container import create_test_container
from sentinel.config import Config

# Create test config
config = Config(
    poll_interval=1,
    max_concurrent_executions=1,
)

container = create_test_container(config=config)

# Create mocks
mock_jira = Mock()
mock_jira.search_issues.return_value = []

mock_jira_tag = Mock()
mock_agent_factory = MagicMock()
mock_agent_logger = Mock()

# Wire up all mocks
container.clients.jira_client.override(providers.Object(mock_jira))
container.clients.jira_tag_client.override(providers.Object(mock_jira_tag))
container.services.agent_factory.override(providers.Object(mock_agent_factory))
container.services.agent_logger.override(providers.Object(mock_agent_logger))

# Now create Sentinel - it will use all mocks
sentinel = container.sentinel()
```

### Pattern 3: Use Factory Functions Directly

For simpler cases, you can use the factory functions without the container:

```python
from sentinel.container import (
    create_jira_rest_client,
    create_agent_factory,
    create_tag_manager,
)
from sentinel.config import load_config

config = load_config()

jira_client = create_jira_rest_client(config)
agent_factory = create_agent_factory(config)
```

### Pattern 4: Context Manager for Test Overrides

```python
from contextlib import contextmanager
from dependency_injector import providers

@contextmanager
def override_jira_client(container, mock_client):
    """Temporarily override Jira client for a test."""
    container.clients.jira_client.override(providers.Object(mock_client))
    try:
        yield
    finally:
        container.clients.jira_client.reset_override()

# Usage in test
def test_something():
    container = create_container()
    mock_jira = Mock()

    with override_jira_client(container, mock_jira):
        sentinel = container.sentinel()
        # Test with mock...

    # After context, original provider is restored
```

### Pattern 5: Integration Test with Real Clients

```python
from sentinel.container import create_container
from sentinel.config import Config

# Create config pointing to test Jira instance
config = Config(
    jira_base_url="https://test-jira.example.com",
    jira_email="test@example.com",
    jira_api_token="test-token",
)

# Container will use REST clients with test config
container = create_container(config=config)
sentinel = container.sentinel()
```

## Extending the Container

### Adding a New Service

1. Create your service class:

```python
# sentinel/my_service.py
class MyService:
    def __init__(self, config: Config, jira_client: JiraClient):
        self.config = config
        self.jira_client = jira_client

    def do_something(self):
        ...
```

2. Add a factory function in `container.py`:

```python
def create_my_service(config: Config, jira_client: JiraClient) -> MyService:
    from sentinel.my_service import MyService
    return MyService(config, jira_client)
```

3. Add a provider to ServicesContainer:

```python
class ServicesContainer(containers.DeclarativeContainer):
    ...
    my_service = providers.Singleton(
        providers.Callable(lambda: None)
    )
```

4. Configure the provider in `create_container()`:

```python
container.services.my_service.override(
    providers.Singleton(
        create_my_service,
        container.config,
        container.clients.jira_client,
    )
)
```

### Adding a New Client Type

Follow the same pattern as Jira/GitHub clients:

1. Create factory functions for the client
2. Add providers to `ClientsContainer`
3. Configure in `create_container()` with conditional logic if needed

## Migration Guide

### From Manual Injection

If you have code that manually creates Sentinel:

```python
# Before (manual injection)
config = load_config()
jira_client = JiraRestClient(...)
tag_client = JiraRestTagClient(...)
agent_factory = create_default_factory(config)
agent_logger = AgentLogger(...)

sentinel = Sentinel(
    config=config,
    orchestrations=orchestrations,
    jira_client=jira_client,
    tag_client=tag_client,
    agent_factory=agent_factory,
    agent_logger=agent_logger,
)
```

```python
# After (using DI container)
container = create_container()
sentinel = container.sentinel()
```

### Gradual Migration

You can migrate incrementally by using factory functions:

```python
from sentinel.container import (
    create_jira_rest_client,
    create_jira_rest_tag_client,
)

# Use DI factory functions, but still manual composition
jira_client = create_jira_rest_client(config)
tag_client = create_jira_rest_tag_client(config)

# Rest of manual setup...
```

## Best Practices

1. **Prefer `create_container()` in production**: It handles all the wiring automatically.

2. **Use `create_test_container()` for tests**: It's pre-configured for easy mocking.

3. **Override at the right level**: Override specific providers, not entire sub-containers.

4. **Use `providers.Object()` for mocks**: It wraps your mock as a provider.

5. **Reset overrides in tests**: Use `container.reset_override()` or context managers.

6. **Keep factory functions pure**: They should only create instances, not have side effects.

## Troubleshooting

### "No builder registered for agent type"

Ensure `agent_factory` is properly configured:

```python
container.services.agent_factory.override(
    providers.Singleton(create_agent_factory, container.config)
)
```

### "NoneType object has no attribute..."

Check if you're accessing a client that might be `None`:

```python
# GitHub clients can be None if not configured
github_client = container.clients.github_client()
if github_client is not None:
    github_client.poll(...)
```

### Tests affecting each other

Reset overrides between tests:

```python
def teardown_method(self):
    self.container.reset_override()
```

## References

- [dependency-injector documentation](https://python-dependency-injector.ets-labs.org/)
- [Dependency Injection principles](https://en.wikipedia.org/wiki/Dependency_injection)
- [sentinel/container.py](../src/sentinel/container.py) - Container implementation
