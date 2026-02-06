# Migration Guide

This guide helps you migrate from deprecated APIs to current best practices.

## Table of Contents

- [Sentinel Constructor Parameters (Removed in v1.0)](#sentinel-constructor-parameters-removed-in-v10)
- [Config Sub-Config Migration (Completed)](#config-sub-config-migration-completed)

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
    jira_client=jira_client,  # Removed!
)
```

**After:**
```python
from sentinel.poller import JiraClient, JiraPoller
from sentinel.main import Sentinel

jira_client = JiraClient(base_url, email, api_token)
jira_poller = JiraPoller(
    jira_client,
    epic_link_field=config.jira.epic_link_field,
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
    github_client=github_client,  # Removed!
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

**Before (removed — this import path no longer works):**
```python
# OLD: from sentinel.executor import AgentClient  # removed in DS-586
from sentinel.agent_clients.base import AgentClient  # canonical import path

agent_client = MyAgentClient()

sentinel = Sentinel(
    # ...
    agent_client=agent_client,  # Removed!
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

**Before (removed — these parameters and import path no longer work):**
```python
from sentinel.main import Sentinel
from sentinel.config import Config
from sentinel.poller import JiraClient
from sentinel.github_poller import GitHubClient
# OLD: from sentinel.executor import AgentClient  # removed in DS-586
from sentinel.agent_clients.base import AgentClient  # canonical import path

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
jira_poller = JiraPoller(jira_client, epic_link_field=config.jira.epic_link_field)

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
jira_poller = JiraPoller(jira_client, epic_link_field=config.jira.epic_link_field)
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

# Config Sub-Config Migration (Completed)

The `Config` class was refactored to use focused sub-configs for each subsystem (Jira, GitHub, Dashboard, etc.). The old flat property access pattern (e.g., `config.jira_base_url`) and the backward-compatibility `@property` shim layer have been **removed** as of DS-573.

## Migration History

| Ticket | Change |
|--------|--------|
| DS-559 | Deprecated flat properties with warnings, introduced sub-config access |
| DS-572 | Migrated all internal callers to sub-config paths |
| DS-573 | Removed all deprecated `@property` shims from the `Config` class |
| DS-584 | Documentation cleanup (this update) |

## Current API

All configuration access uses the sub-config pattern. Each sub-config is a frozen dataclass grouping related settings:

```python
from sentinel.config import load_config

config = load_config()

# Sub-config access (the only supported pattern)
jira_url = config.jira.base_url
poll_seconds = config.polling.interval
max_workers = config.execution.max_concurrent_executions
```

### Available Sub-Configs

| Sub-Config | Description | Example Access |
|------------|-------------|----------------|
| `config.jira` | Jira REST API settings | `config.jira.base_url`, `config.jira.epic_link_field` |
| `config.github` | GitHub REST API settings | `config.github.token`, `config.github.api_url` |
| `config.dashboard` | Dashboard server settings | `config.dashboard.port`, `config.dashboard.enabled` |
| `config.rate_limit` | Claude API rate limiting | `config.rate_limit.per_minute`, `config.rate_limit.strategy` |
| `config.circuit_breaker` | Circuit breaker settings | `config.circuit_breaker.enabled`, `config.circuit_breaker.failure_threshold` |
| `config.health_check` | Health check settings | `config.health_check.timeout`, `config.health_check.enabled` |
| `config.execution` | Agent execution settings | `config.execution.max_concurrent_executions`, `config.execution.agent_workdir` |
| `config.cursor` | Cursor CLI settings | `config.cursor.default_agent_type`, `config.cursor.default_mode` |
| `config.logging_config` | Logging settings | `config.logging_config.level`, `config.logging_config.json` |
| `config.polling` | Polling settings | `config.polling.interval`, `config.polling.max_issues_per_poll` |

## Troubleshooting

### AttributeError when accessing flat config properties

If you encounter an `AttributeError` such as:

```
AttributeError: 'Config' object has no attribute 'jira_base_url'
```

This means your code is using the old flat property access pattern, which has been removed. Update to the sub-config path:

```python
# Old (removed):
# jira_url = config.jira_base_url

# New:
jira_url = config.jira.base_url
```

For a complete history of changes and removal timelines, see the [CHANGELOG](../CHANGELOG.md).
