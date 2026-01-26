# Developer Sentinel

Python orchestration app that polls Jira for tagged issues and routes them to Claude-based agents for processing.

## Overview

Developer Sentinel monitors Jira for issues with specific tags (e.g., `@code-review`, `@docs-update`) and automatically routes them to specialized Claude agents for processing. Agents have access to Jira, Confluence, and GitHub tooling.

## Features

- **Tag-based routing**: Configure which agent handles issues based on Jira labels
- **GitHub Project polling**: Poll GitHub Projects (v2) with JQL-like filter expressions
- **YAML orchestration configs**: Define agent behavior, tools, and prompts in YAML files
- **Multiple tool integrations**: Jira, Confluence, and GitHub tools available to agents
- **Multi-repository support**: GitHub Projects can contain issues/PRs from multiple repos
- **Configurable polling**: Adjustable poll intervals and issue limits
- **Structured logging**: JSON-formatted logs with context for debugging
- **Hot-reload orchestrations**: Automatically load, reload, and unload orchestration files without restart
- **Eager polling**: Polls immediately when work is submitted to maximize throughput

## Installation

```bash
# Clone the repository
git clone https://github.com/btw-claude/developer-sentinel.git
cd developer-sentinel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file with your credentials:

```bash
# Jira REST API Configuration (recommended for faster polling)
JIRA_BASE_URL=https://your-instance.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token

# Sentinel Configuration
SENTINEL_POLL_INTERVAL=60
SENTINEL_MAX_ISSUES=50
SENTINEL_LOG_LEVEL=INFO
SENTINEL_LOG_JSON=false
SENTINEL_ORCHESTRATIONS_DIR=./orchestrations
SENTINEL_AGENT_WORKDIR=./workdir
SENTINEL_AGENT_LOGS_DIR=./logs
```

### Environment Variables Reference

#### Jira REST API (Recommended)

Configure these for direct REST API access, which is faster than SDK-based polling:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JIRA_BASE_URL` | No* | - | Jira instance URL (e.g., `https://company.atlassian.net`) |
| `JIRA_EMAIL` | No* | - | Email address for Jira authentication |
| `JIRA_API_TOKEN` | No* | - | Jira API token ([create one here](https://id.atlassian.com/manage-profile/security/api-tokens)) |

*All three Jira variables must be set together. If not configured, falls back to SDK-based Jira access via Claude Agent SDK.

**Rate Limiting**: The Jira REST client automatically handles rate limiting with exponential backoff and jitter per [Atlassian's recommendations](https://developer.atlassian.com/cloud/jira/platform/rate-limiting/). When a 429 (Too Many Requests) response is received, the client retries up to 4 times with increasing delays (max 30 seconds).

#### Sentinel Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SENTINEL_POLL_INTERVAL` | No | `60` | Seconds between Jira polls |
| `SENTINEL_MAX_ISSUES` | No | `50` | Maximum issues to process per poll |
| `SENTINEL_LOG_LEVEL` | No | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `SENTINEL_LOG_JSON` | No | `false` | Enable JSON log output (`true`, `false`) |
| `SENTINEL_ORCHESTRATIONS_DIR` | No | `./orchestrations` | Path to orchestration YAML files |
| `SENTINEL_AGENT_WORKDIR` | No | `./workdir` | Base directory for agent working directories |
| `SENTINEL_AGENT_LOGS_DIR` | No | `./logs` | Base directory for agent execution logs |

#### Cursor CLI Configuration

Configure these to enable Cursor as an alternative agent type:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SENTINEL_DEFAULT_AGENT_TYPE` | No | `claude` | Default agent type (`claude`, `cursor`) |
| `SENTINEL_CURSOR_PATH` | Yes* | - | Path to Cursor CLI executable. *Required when using `cursor` agent type. |
| `SENTINEL_CURSOR_DEFAULT_MODEL` | No | - | Default model for Cursor agent |
| `SENTINEL_CURSOR_DEFAULT_MODE` | No | `agent` | Default Cursor mode (`agent`, `plan`, `ask`) |

**Note:** When `SENTINEL_DEFAULT_AGENT_TYPE` is set to `cursor` (or when an orchestration specifies `agent_type: cursor`), the `SENTINEL_CURSOR_PATH` environment variable must be configured with the path to the Cursor CLI executable.

## Usage

```bash
# Run the sentinel
sentinel

# With custom options
sentinel --interval 30 --log-level DEBUG --config-dir ./my-orchestrations
```

## Orchestration Configuration

Define orchestrations in YAML files under the `orchestrations/` directory:

```yaml
orchestrations:
  - name: "code-review"
    trigger:
      source: jira
      project: "DEV"
      tags:
        - "needs-review"
    agent:
      prompt: |
        You are a code review assistant. Review issue {jira_issue_key}.

        Summary: {jira_summary}
        Description: {jira_description}

        Respond with SUCCESS when complete.
      tools:
        - jira
        - github
      github:
        host: "github.com"
        org: "your-org"
        repo: "your-repo"
    on_complete:
      add_tag: "reviewed"
```

### Agent Type Selection

You can select which AI agent to use for each orchestration:

```yaml
orchestrations:
  - name: "cursor-code-review"
    trigger:
      source: jira
      project: "DEV"
      tags:
        - "cursor-review"
    agent:
      # Use Cursor instead of Claude (default: config.default_agent_type)
      agent_type: cursor

      # Cursor-specific mode (only valid when agent_type is "cursor")
      # Values: "agent" (default), "plan", "ask"
      cursor_mode: agent

      prompt: |
        Review the code changes for {jira_issue_key}.
      tools:
        - jira
        - github
```

**Agent Types:**
- `claude` - Use Claude AI agent (default)
- `cursor` - Use Cursor AI agent

**Cursor Modes (only valid when `agent_type: cursor`):**
- `agent` - Full autonomous agent mode (default)
- `plan` - Planning mode - creates plans without executing
- `ask` - Ask mode - waits for user confirmation before actions

See `orchestrations/README.md` for full configuration reference.

### GitHub Project Triggers

Sentinel also supports GitHub Projects (v2) as a trigger source:

```yaml
orchestrations:
  - name: "github-code-review"
    trigger:
      source: github
      project_number: 42          # Project number from URL
      project_owner: "your-org"   # Organization or username
      project_scope: "org"        # "org" or "user"
      project_filter: 'Status = "Ready for Review"'
    agent:
      prompt: |
        Review the code changes for this item.
        End with APPROVED or CHANGES REQUESTED.
      tools:
        - github
      github:
        host: "github.com"
        org: "your-org"
        repo: "your-repo"
    on_complete:
      add_tag: "reviewed"
```

See [docs/GITHUB_TRIGGER_MIGRATION.md](docs/GITHUB_TRIGGER_MIGRATION.md) for the migration guide from deprecated GitHub trigger fields.

## Dynamic Orchestration Management

### Hot-Reload

Sentinel automatically detects changes to orchestration files at the start of each poll cycle:

- **New files**: Automatically loaded and routed to matching issues
- **Modified files**: Reloaded with version protection - running executions continue with their original configuration
- **Removed files**: Unloaded after running executions complete

This allows updating orchestration configurations without restarting the service.

### Eager Polling

When work is submitted for execution, Sentinel polls immediately for more work instead of waiting for the next poll interval. This maximizes throughput when there's a backlog of issues to process while conserving resources during idle periods.

## Development

```bash
# Run tests
pytest

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

## License

MIT
