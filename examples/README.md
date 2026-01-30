# Developer Sentinel Examples

This directory contains working examples to help you get started with Developer Sentinel quickly.

## Examples Overview

| Example | Description | Best For |
|---------|-------------|----------|
| [basic-setup](./basic-setup/) | Minimal working configuration | First-time users, quick start |
| [multi-orchestration](./multi-orchestration/) | Multiple triggers and workflows | Complex automation scenarios |
| [docker](./docker/) | Docker and docker-compose deployment | Production deployments |

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
