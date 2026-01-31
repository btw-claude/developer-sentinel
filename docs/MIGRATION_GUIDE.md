# Migration Guide: Removing Deprecated Sentinel Parameters

This guide helps you migrate from the deprecated `Sentinel.__init__` parameters to the current API.

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

## Using the DI Container (Recommended)

For new projects, we recommend using the dependency injection container which handles all wiring automatically:

```python
from sentinel.container import create_container

# Container handles all dependency wiring
container = create_container()
sentinel = container.sentinel()

# Run the sentinel
sentinel.run()
```

See [dependency-injection.md](dependency-injection.md) for more details on using the DI container.

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
