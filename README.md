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
# Required - Claude API
ANTHROPIC_API_KEY=your-anthropic-api-key

# Required - Jira Configuration
JIRA_URL=https://your-instance.atlassian.net
JIRA_USER=your-email@example.com
JIRA_API_TOKEN=your-api-token

# Optional - Sentinel Configuration
SENTINEL_POLL_INTERVAL=60
SENTINEL_MAX_ISSUES=50
SENTINEL_LOG_LEVEL=INFO
SENTINEL_LOG_JSON=false
SENTINEL_ORCHESTRATIONS_DIR=./orchestrations
```

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Your Anthropic API key for Claude |
| `JIRA_URL` | Yes | - | Jira instance URL (e.g., `https://company.atlassian.net`) |
| `JIRA_USER` | Yes | - | Jira username (email) |
| `JIRA_API_TOKEN` | Yes | - | Jira API token ([create one here](https://id.atlassian.com/manage-profile/security/api-tokens)) |
| `SENTINEL_POLL_INTERVAL` | No | `60` | Seconds between Jira polls |
| `SENTINEL_MAX_ISSUES` | No | `50` | Maximum issues to process per poll |
| `SENTINEL_LOG_LEVEL` | No | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `SENTINEL_LOG_JSON` | No | `false` | Enable JSON log output (`true`, `false`) |
| `SENTINEL_ORCHESTRATIONS_DIR` | No | `./orchestrations` | Path to orchestration YAML files |

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
name: code-review
description: Automated code review agent
version: "1.0"

trigger:
  tag: "@code-review"
  project_filter: "project = DEV"

agent:
  model: claude-sonnet-4-20250514
  system_prompt: |
    You are a code review assistant...
  tools:
    - jira
    - github
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
