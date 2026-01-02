# Developer Sentinel

Python orchestration app that polls Jira for tagged issues and routes them to Claude-based agents for processing.

## Overview

Developer Sentinel monitors Jira for issues with specific tags (e.g., `@code-review`, `@docs-update`) and automatically routes them to specialized Claude agents for processing. Agents have access to Jira, Confluence, and GitHub tooling.

## Features

- **Tag-based routing**: Configure which agent handles issues based on Jira labels
- **YAML orchestration configs**: Define agent behavior, tools, and prompts in YAML files
- **Multiple tool integrations**: Jira, Confluence, and GitHub tools available to agents
- **Configurable polling**: Adjustable poll intervals and issue limits
- **Structured logging**: JSON-formatted logs with context for debugging

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

Configure these for direct REST API access, which is faster than MCP-based polling:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JIRA_BASE_URL` | No* | - | Jira instance URL (e.g., `https://company.atlassian.net`) |
| `JIRA_EMAIL` | No* | - | Email address for Jira authentication |
| `JIRA_API_TOKEN` | No* | - | Jira API token ([create one here](https://id.atlassian.com/manage-profile/security/api-tokens)) |

*All three Jira variables must be set together. If not configured, falls back to MCP-based Jira access via Claude Code CLI.

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

See `orchestrations/README.md` for full configuration reference.

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
