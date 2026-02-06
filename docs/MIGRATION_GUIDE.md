# Migration Guide

This guide helps you migrate from deprecated APIs to current best practices.

## Table of Contents

- [Sentinel Constructor Parameters (Removed in v1.0)](#sentinel-constructor-parameters-removed-in-v10)
- [Config Backward Compatibility Properties (Deprecated)](#config-backward-compatibility-properties-deprecated)

---

# Sentinel Constructor Parameters (Removed in v1.0)

This section helps you migrate from the deprecated `Sentinel.__init__` parameters to the current API.

## Deprecation History

The following table documents the full deprecation timeline for the removed parameters:

| Parameter | Introduced | Deprecated | Removed | Replacement |
|-----------|------------|------------|---------|-------------|
| `jira_client` | v0.1.0 (DS-4) | v0.1.0 (DS-296) | v1.0 (DS-503) | `jira_poller` |
| `github_client` | v0.1.0 (DS-51) | v0.1.0 (DS-296) | v1.0 (DS-503) | `github_poller` |
| `agent_client` | v0.1.0 (DS-8) | v0.1.0 (DS-296) | v1.0 (DS-503) | `agent_factory` |

### Timeline

- **v0.1.0 (Initial Release)**: The original `jira_client` (DS-4), `github_client` (DS-51), and `agent_client` (DS-8) parameters were introduced as part of the initial Sentinel implementation.
- **v0.1.0 (DS-296)**: The `AgentClientFactory` pattern was introduced to support per-orchestration agent configuration. Backward compatibility was maintained with deprecation warnings for the legacy parameters.
- **v1.0 (DS-503)**: The deprecated parameters were removed to reduce maintenance burden and cognitive load. Users must now use the new poller and factory patterns.

## Overview

The following parameters have been removed from `Sentinel.__init__`:

| Removed Parameter | Replacement | Description |
|------------------|-------------|-------------|
| `jira_client` | `jira_poller` | Wrap your `JiraClient` in a `JiraPoller` |
| `github_client` | `github_poller` | Wrap your `GitHubClient` in a `GitHubPoller` |
| `agent_client` | `agent_factory` | Use `AgentClientFactory` for per-orchestration agent management |

## Why These Changes?

The deprecated parameters created maintenance burden:

1. **Parallel code paths**: The `Sentinel.__init__` had to handle both old and new parameter styles
2. **Deprecation warnings in logs**: Users saw warnings but the old API continued to work
3. **Cognitive load**: New contributors had to understand both old and new patterns

The new API provides:

- **Single responsibility**: Each component (poller, factory) handles one concern
- **Per-orchestration agents**: `AgentClientFactory` allows different agent configurations per orchestration
- **Cleaner separation**: Polling logic is in pollers, not in Sentinel initialization

## Migration Steps

### Step 1: Update Jira Client Usage

**Before:**
```python
from sentinel.poller import JiraClient
from sentinel.main import Sentinel

jira_client = JiraClient(base_url, email, api_token)

sentinel = Sentinel(
    config=config,
    orchestrations=orchestrations,
    tag_client=tag_client,
    jira_client=jira_client,  # Deprecated!
)
```

**After:**
```python
from sentinel.poller import JiraClient, JiraPoller
from sentinel.main import Sentinel

jira_client = JiraClient(base_url, email, api_token)
jira_poller = JiraPoller(
    jira_client,
    epic_link_field=config.jira_epic_link_field,
)

sentinel = Sentinel(
    config=config,
    orchestrations=orchestrations,
    tag_client=tag_client,
    jira_poller=jira_poller,
)
```

### Step 2: Update GitHub Client Usage

**Before:**
```python
from sentinel.github_poller import GitHubClient

github_client = GitHubClient(token)

sentinel = Sentinel(
    # ...
    github_client=github_client,  # Deprecated!
)
```

**After:**
```python
from sentinel.github_poller import GitHubClient, GitHubPoller

github_client = GitHubClient(token)
github_poller = GitHubPoller(github_client)

sentinel = Sentinel(
    # ...
    github_poller=github_poller,
)
```

### Step 3: Update Agent Client Usage

**Before:**
```python
from sentinel.executor import AgentClient

agent_client = MyAgentClient()

sentinel = Sentinel(
    # ...
    agent_client=agent_client,  # Deprecated!
)
```

**After:**
```python
from sentinel.agent_clients.factory import AgentClientFactory

agent_factory = AgentClientFactory()

sentinel = Sentinel(
    # ...
    agent_factory=agent_factory,
)
```

### Benefits of AgentClientFactory

The `AgentClientFactory` provides per-orchestration agent management:

```python
from sentinel.agent_clients.factory import AgentClientFactory

# The factory creates appropriate clients based on orchestration configuration
factory = AgentClientFactory()

# Factory automatically creates clients for different agent types
# based on orchestration.agent.agent_type
```

## Complete Example

**Before (deprecated):**
```python
from sentinel.main import Sentinel
from sentinel.config import Config
from sentinel.poller import JiraClient
from sentinel.github_poller import GitHubClient
from sentinel.executor import AgentClient

config = Config.from_env()
jira_client = JiraClient(...)
github_client = GitHubClient(...)
agent_client = MyAgentClient()

sentinel = Sentinel(
    config=config,
    orchestrations=orchestrations,
    tag_client=tag_client,
    jira_client=jira_client,
    github_client=github_client,
    agent_client=agent_client,
)
```

**After (current):**
```python
from sentinel.main import Sentinel
from sentinel.config import Config
from sentinel.poller import JiraClient, JiraPoller
from sentinel.github_poller import GitHubClient, GitHubPoller
from sentinel.agent_clients.factory import AgentClientFactory
from sentinel.tag_manager import JiraTagClient

config = Config.from_env()

# Create pollers from clients
jira_client = JiraClient(...)
jira_poller = JiraPoller(jira_client, epic_link_field=config.jira_epic_link_field)

github_client = GitHubClient(...)
github_poller = GitHubPoller(github_client)

# Use the factory
agent_factory = AgentClientFactory()

sentinel = Sentinel(
    config=config,
    orchestrations=orchestrations,
    tag_client=tag_client,
    jira_poller=jira_poller,
    github_poller=github_poller,
    agent_factory=agent_factory,
)
```

## Using the Bootstrap Module (Recommended)

For new projects, we recommend using the bootstrap module which handles all dependency wiring automatically:

```python
from sentinel.app import main

# main() handles bootstrap and running
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

See [dependency-injection.md](dependency-injection.md) for more details on dependency injection patterns.

## Troubleshooting

### TypeError: Sentinel.__init__() got an unexpected keyword argument 'jira_client'

You're using a removed parameter. Update your code to use `jira_poller` instead:

```python
# Instead of jira_client=..., use:
jira_poller = JiraPoller(jira_client, epic_link_field=config.jira_epic_link_field)
sentinel = Sentinel(..., jira_poller=jira_poller)
```

### TypeError: Sentinel.__init__() got an unexpected keyword argument 'github_client'

Update to use `github_poller`:

```python
# Instead of github_client=..., use:
github_poller = GitHubPoller(github_client)
sentinel = Sentinel(..., github_poller=github_poller)
```

### TypeError: Sentinel.__init__() got an unexpected keyword argument 'agent_client'

Update to use `agent_factory`:

```python
# Instead of agent_client=..., use:
agent_factory = AgentClientFactory()
sentinel = Sentinel(..., agent_factory=agent_factory)
```

## Questions?

If you encounter issues during migration, please:

1. Check this guide for the correct replacement parameter
2. Review the [dependency-injection.md](dependency-injection.md) for the recommended DI approach
3. Open an issue on GitHub if you need further assistance

For a complete history of changes and removal timelines, see the [CHANGELOG](../CHANGELOG.md).

---

# Config Backward Compatibility Properties (Deprecated)

The `Config` class was refactored to use focused sub-configs for each subsystem (Jira, GitHub, Dashboard, etc.). The old flat property access pattern is deprecated and will be removed in a future major version.

## Why These Changes?

The backward compatibility properties created maintenance burden:

1. **Code duplication**: ~270 lines of properties that simply delegate to sub-configs
2. **Harder maintenance**: Changes to sub-configs require updating corresponding properties
3. **Cognitive load**: New contributors must understand both access patterns

The new sub-config pattern provides:

- **Grouped configuration**: Related settings are organized together
- **Immutable sub-configs**: Each sub-config is a frozen dataclass
- **Clear namespacing**: `config.jira.base_url` is clearer than `config.jira_base_url`

## Deprecation Timeline

| Version | Status |
|---------|--------|
| Current | Deprecated with warnings (DS-559) |
| Next Major | Removal planned |

## Migration Reference

### Polling Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.poll_interval` | `config.polling.interval` |
| `config.max_issues_per_poll` | `config.polling.max_issues_per_poll` |

### Logging Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.log_level` | `config.logging_config.level` |
| `config.log_json` | `config.logging_config.json` |

### Execution Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.max_concurrent_executions` | `config.execution.max_concurrent_executions` |
| `config.orchestrations_dir` | `config.execution.orchestrations_dir` |
| `config.agent_workdir` | `config.execution.agent_workdir` |
| `config.agent_logs_dir` | `config.execution.agent_logs_dir` |
| `config.orchestration_logs_dir` | `config.execution.orchestration_logs_dir` |
| `config.cleanup_workdir_on_success` | `config.execution.cleanup_workdir_on_success` |
| `config.disable_streaming_logs` | `config.execution.disable_streaming_logs` |
| `config.subprocess_timeout` | `config.execution.subprocess_timeout` |
| `config.default_base_branch` | `config.execution.default_base_branch` |
| `config.attempt_counts_ttl` | `config.execution.attempt_counts_ttl` |
| `config.max_queue_size` | `config.execution.max_queue_size` |
| `config.inter_message_times_threshold` | `config.execution.inter_message_times_threshold` |
| `config.shutdown_timeout_seconds` | `config.execution.shutdown_timeout_seconds` |

### Jira Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.jira_base_url` | `config.jira.base_url` |
| `config.jira_email` | `config.jira.email` |
| `config.jira_api_token` | `config.jira.api_token` |
| `config.jira_epic_link_field` | `config.jira.epic_link_field` |
| `config.jira_configured` | `config.jira.configured` |

### GitHub Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.github_token` | `config.github.token` |
| `config.github_api_url` | `config.github.api_url` |
| `config.github_configured` | `config.github.configured` |

### Dashboard Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.dashboard_enabled` | `config.dashboard.enabled` |
| `config.dashboard_port` | `config.dashboard.port` |
| `config.dashboard_host` | `config.dashboard.host` |
| `config.toggle_cooldown_seconds` | `config.dashboard.toggle_cooldown_seconds` |
| `config.rate_limit_cache_ttl` | `config.dashboard.rate_limit_cache_ttl` |
| `config.rate_limit_cache_maxsize` | `config.dashboard.rate_limit_cache_maxsize` |
| `config.max_recent_executions` | `config.dashboard.max_recent_executions` |

### Rate Limiting Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.claude_rate_limit_enabled` | `config.rate_limit.enabled` |
| `config.claude_rate_limit_per_minute` | `config.rate_limit.per_minute` |
| `config.claude_rate_limit_per_hour` | `config.rate_limit.per_hour` |
| `config.claude_rate_limit_strategy` | `config.rate_limit.strategy` |
| `config.claude_rate_limit_warning_threshold` | `config.rate_limit.warning_threshold` |
| `config.claude_rate_limit_max_queued` | `config.rate_limit.max_queued` |
| `config.claude_rate_limit_queue_full_strategy` | `config.rate_limit.queue_full_strategy` |

### Circuit Breaker Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.circuit_breaker_enabled` | `config.circuit_breaker.enabled` |
| `config.circuit_breaker_failure_threshold` | `config.circuit_breaker.failure_threshold` |
| `config.circuit_breaker_recovery_timeout` | `config.circuit_breaker.recovery_timeout` |
| `config.circuit_breaker_half_open_max_calls` | `config.circuit_breaker.half_open_max_calls` |

### Health Check Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.health_check_enabled` | `config.health_check.enabled` |
| `config.health_check_timeout` | `config.health_check.timeout` |

### Cursor Configuration

| Deprecated Property | New Property |
|---------------------|--------------|
| `config.default_agent_type` | `config.cursor.default_agent_type` |
| `config.cursor_path` | `config.cursor.path` |
| `config.cursor_default_model` | `config.cursor.default_model` |
| `config.cursor_default_mode` | `config.cursor.default_mode` |

## Migration Example

**Before (deprecated):**
```python
from sentinel.config import load_config

config = load_config()

# Deprecated flat access pattern
jira_url = config.jira_base_url
poll_seconds = config.poll_interval
max_workers = config.max_concurrent_executions
```

**After (recommended):**
```python
from sentinel.config import load_config

config = load_config()

# New sub-config access pattern
jira_url = config.jira.base_url
poll_seconds = config.polling.interval
max_workers = config.execution.max_concurrent_executions
```

## Suppressing Deprecation Warnings

If you need time to migrate and want to suppress the warnings temporarily, you can use Python's warnings filter:

```python
import warnings

# Suppress all deprecation warnings from config module
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="sentinel.config"
)
```

**Note:** This is not recommended as a long-term solution. Plan to migrate to the new API before the next major version.

For a complete history of changes and removal timelines, see the [CHANGELOG](../CHANGELOG.md).
