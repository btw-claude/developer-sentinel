# Changelog

All notable changes to Developer Sentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **BREAKING**: Removed deprecated backward compatibility parameters from `Sentinel.__init__`:
  - `jira_client` - use `jira_poller` parameter instead
  - `github_client` - use `github_poller` parameter instead
  - `agent_client` - use `agent_factory` parameter instead

  These parameters were deprecated since v0.x and have been removed to reduce
  maintenance burden and cognitive load. See the [Migration Guide](docs/MIGRATION_GUIDE.md)
  for upgrade instructions.

### Changed

- Cleaned up `__all__` exports in `sentinel.main` module to remove re-exports that
  are better imported from their respective modules directly.

## Migration from Pre-1.0 Versions

If you were using the deprecated parameters, update your code as follows:

### Before (deprecated)

```python
from sentinel.poller import JiraClient
from sentinel.github_poller import GitHubClient
from sentinel.executor import AgentClient

sentinel = Sentinel(
    config=config,
    orchestrations=orchestrations,
    tag_client=tag_client,
    jira_client=jira_client,      # Deprecated
    github_client=github_client,  # Deprecated
    agent_client=agent_client,    # Deprecated
)
```

### After (current)

```python
from sentinel.poller import JiraPoller
from sentinel.github_poller import GitHubPoller
from sentinel.agent_clients.factory import AgentClientFactory

# Create pollers from clients
jira_poller = JiraPoller(jira_client, epic_link_field=config.jira_epic_link_field)
github_poller = GitHubPoller(github_client)

# Use the factory for agent clients
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

See [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for more details.
