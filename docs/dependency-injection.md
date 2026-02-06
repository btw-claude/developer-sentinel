# Dependency Injection Guide

This guide explains the dependency injection patterns used in Developer Sentinel.

## Overview

Developer Sentinel uses **constructor injection** via the bootstrap module (`bootstrap.py`) to manage dependencies. The `BootstrapContext` class serves as the composition root, wiring together all dependencies before the application starts running.

> **Note:** The `dependency-injector` library and `container.py` were removed in DS-581 as vestigial code. The application has always used the bootstrap module for dependency wiring. See the [Bootstrap Architecture](#bootstrap-architecture) section for current patterns.

## Bootstrap Architecture

The bootstrap module (`src/sentinel/bootstrap.py`) acts as the composition root:

```
bootstrap()
├── load_config()
├── setup_logging()
├── load_orchestrations()
├── create_jira_clients()
├── create_github_clients()
├── create_default_factory()
└── BootstrapContext (holds all dependencies)
      └── create_sentinel_from_context()
            └── Sentinel (fully wired)
```

### Production Usage

```python
from sentinel.app import main

# The main() function handles bootstrap and running
exit_code = main()
```

Or for more control:

```python
from sentinel.bootstrap import bootstrap, create_sentinel_from_context
from sentinel.cli import parse_args

parsed = parse_args()
context = bootstrap(parsed)
sentinel = create_sentinel_from_context(context)
sentinel.run()
```

### Testing Usage

For unit tests, inject dependencies directly via constructor:

```python
from unittest.mock import Mock, MagicMock
from sentinel.main import Sentinel

config = make_config()
sentinel = Sentinel(
    config=config,
    orchestrations=[],
    tag_client=Mock(),
    agent_factory=MagicMock(),
    jira_poller=Mock(),
)
```

## Circuit Breaker Dependency Injection

The circuit breaker pattern is implemented using dependency injection to avoid global mutable state and enable proper test isolation.

### Thread Safety

The `CircuitBreakerRegistry` is thread-safe and can be safely accessed from multiple threads in an async context. Internally, it uses a `threading.Lock()` to protect the registry of circuit breakers. Individual `CircuitBreaker` instances are also thread-safe, using `threading.RLock()` to protect their state transitions and metrics updates.

This means you can safely:
- Call `registry.get("service")` from multiple threads concurrently
- Share the same circuit breaker instance across async tasks
- Access circuit breaker state and metrics while other threads are recording successes/failures

### Design Principles

1. **No Global Registry**: Circuit breakers are not stored in a global registry. Instead, a `CircuitBreakerRegistry` is created during bootstrap and passed to components that need it.

2. **Constructor Injection**: Components receive their circuit breaker via constructor injection, making dependencies explicit and testable.

3. **Test Isolation**: Each test can create its own `CircuitBreakerRegistry`, ensuring tests don't interfere with each other.

### How It Works

The bootstrap process creates a `CircuitBreakerRegistry` and passes it to components:

```python
from sentinel.circuit_breaker import CircuitBreakerRegistry

# During bootstrap
circuit_breaker_registry = CircuitBreakerRegistry()

# Inject into Jira clients
jira_client = JiraRestClient(
    base_url=config.jira.base_url,
    email=config.jira.email,
    api_token=config.jira.api_token,
    circuit_breaker=circuit_breaker_registry.get("jira"),
)

# Inject into GitHub clients
github_client = GitHubRestClient(
    token=config.github.token,
    circuit_breaker=circuit_breaker_registry.get("github"),
)

# Inject into agent factory
agent_factory = create_default_factory(config, circuit_breaker_registry)
```

### Testing with Circuit Breakers

For unit tests, create isolated registries to avoid test pollution:

```python
class TestMyComponent:
    def test_circuit_breaker_behavior(self) -> None:
        # Create isolated registry for this test
        registry = CircuitBreakerRegistry()

        # Create component with injected circuit breaker
        client = JiraRestClient(
            base_url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            circuit_breaker=registry.get("jira"),
        )

        # Test that circuit breaker is in initial state
        assert client.circuit_breaker.state == CircuitState.CLOSED
```

For integration tests that need to share circuit breaker state:

```python
class TestIntegration:
    def setUp(self) -> None:
        # Shared registry for integration test
        self.registry = CircuitBreakerRegistry()

    def test_shared_circuit_breaker_state(self) -> None:
        # Both clients share the same Jira circuit breaker
        client1 = JiraRestClient(
            base_url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            circuit_breaker=self.registry.get("jira"),
        )
        client2 = JiraRestTagClient(
            base_url="https://test.atlassian.net",
            email="test@example.com",
            api_token="test-token",
            circuit_breaker=self.registry.get("jira"),
        )

        # Same circuit breaker instance
        assert client1.circuit_breaker is client2.circuit_breaker
```

### Default Behavior

When no circuit breaker is provided, components create their own default circuit breaker:

```python
# Without explicit injection, creates default circuit breaker
client = JiraRestClient(
    base_url="https://test.atlassian.net",
    email="test@example.com",
    api_token="test-token",
    # circuit_breaker defaults to CircuitBreaker("jira", CircuitBreakerConfig.from_env("jira"))
)
```

This ensures backward compatibility while encouraging explicit injection for better testability.

## References

- [Dependency Injection principles](https://en.wikipedia.org/wiki/Dependency_injection)
- [sentinel/bootstrap.py](../src/sentinel/bootstrap.py) - Bootstrap and dependency wiring
- [sentinel/circuit_breaker.py](../src/sentinel/circuit_breaker.py) - Circuit breaker implementation
