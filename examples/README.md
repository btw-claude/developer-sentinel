# Developer Sentinel Examples

This directory contains working examples to help you get started with Developer Sentinel quickly.

## Deployment Examples

| Example | Description | Best For |
|---------|-------------|----------|
| [basic-setup](./basic-setup/) | Minimal working configuration | First-time users, quick start |
| [multi-orchestration](./multi-orchestration/) | Multiple triggers and workflows | Complex automation scenarios |
| [docker](./docker/) | Docker and docker-compose deployment | Production deployments |

## Orchestration Reference Examples

The [`orchestrations/`](./orchestrations/) directory contains focused examples for each orchestration feature. Each file demonstrates one capability and is safe to drop into a live setup (`enabled: false` by default).

| Example | Feature | Key Fields |
|---------|---------|------------|
| [jira-polling.yaml](./orchestrations/jira-polling.yaml) | Jira project polling | `source`, `project`, `tags`, `jql_filter`, lifecycle hooks |
| [github-project-polling.yaml](./orchestrations/github-project-polling.yaml) | GitHub Project (v2) polling | `project_number`, `project_owner`, `project_scope`, `project_filter`, `labels` |
| [branch-patterns.yaml](./orchestrations/branch-patterns.yaml) | Feature branches and base branches | `github.branch`, `create_branch`, `base_branch`, fork workflows |
| [jira-template-variables.yaml](./orchestrations/jira-template-variables.yaml) | All Jira template variables | All 11 `{jira_*}` variables in a prompt |
| [github-template-variables.yaml](./orchestrations/github-template-variables.yaml) | All GitHub template variables | All 14 `{github_*}` variables + common variables |
| [outcomes.yaml](./orchestrations/outcomes.yaml) | Outcome-based tagging | `outcomes`, `patterns`, `regex:` prefix, `default_outcome`, `default_status` |
| [chained-pipeline.yaml](./orchestrations/chained-pipeline.yaml) | Multi-step tag-driven pipeline | Outcome tags triggering next steps, review loops |
| [agent-types.yaml](./orchestrations/agent-types.yaml) | Agent type options | `agent_type`, `cursor_mode`, `agent_teams`, `model` |
| [advanced-options.yaml](./orchestrations/advanced-options.yaml) | Miscellaneous options | `max_concurrent`, `strict_template_variables`, `enabled`, `default_status`, file-level trigger inheritance |

## Quick Start

1. **Choose an example** that matches your use case
2. **Copy the configuration files** to your project
3. **Update credentials** in the `.env` file
4. **Customize orchestrations** for your needs
5. **Run the sentinel**

## Prerequisites

All examples assume you have:
- Python 3.11 or later installed
- A Jira Cloud instance with API access
- (Optional) GitHub repository access for code-related workflows

## Getting API Tokens

### Jira API Token
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Give it a descriptive name (e.g., "Developer Sentinel")
4. Copy the token and save it securely

### GitHub Token (Optional)
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo`, `read:org`, `read:project`
4. Copy the token and save it securely

## Example Usage

```bash
# Navigate to an example
cd examples/basic-setup

# Copy environment template
cp .env.example .env

# Edit with your credentials
vim .env

# Install dependencies (from project root)
cd ../..
pip install -e ".[dev]"

# Run the sentinel
sentinel --config-dir examples/basic-setup/orchestrations
```

## Customizing Examples

Each example includes detailed comments explaining configuration options. Key files to customize:

- `.env` - Credentials and runtime settings
- `orchestrations/*.yaml` - Workflow definitions and triggers

See the main [README.md](../README.md) for complete configuration reference.
