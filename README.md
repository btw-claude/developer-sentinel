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

## Requirements

- **Python 3.11 or later** is required. The codebase uses modern Python features including `typing.TypeAlias` and other 3.11+ syntax.
- **Note:** pip will automatically enforce the Python version requirement during installation (via `requires-python` in pyproject.toml). If you attempt to install with an older Python version, pip will fail gracefully with a clear error message.

## Installation

```bash
# Clone the repository
git clone https://github.com/btw-claude/developer-sentinel.git
cd developer-sentinel

# Create virtual environment (ensure Python 3.11+)
python3.11 -m venv .venv  # or python3.12, python3.13
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
| `JIRA_EPIC_LINK_FIELD` | No | `customfield_10014` | Custom field ID for epic links (see below) |

*All three Jira authentication variables must be set together. If not configured, falls back to SDK-based Jira access via Claude Agent SDK.

**Finding Your Epic Link Field ID:**

The epic link custom field ID varies between Jira instances. To find yours:

1. Navigate to any issue that has an epic link
2. Use the Jira REST API to inspect the issue: `GET /rest/api/3/issue/{issueKey}`
3. Look in the `fields` object for a field containing your epic key (e.g., `"customfield_10014": "EPIC-123"`)
4. The field name (e.g., `customfield_10014`) is what you should set for `JIRA_EPIC_LINK_FIELD`

Alternatively, you can find custom field IDs in Jira Admin under **Settings > Issues > Custom Fields**.

**Note on Empty Values:** If `JIRA_EPIC_LINK_FIELD` is explicitly set to an empty string, epic link detection via the classic custom field will be disabled (epic_key will always be `None` for issues without a next-gen parent epic). This may be intentional for Jira instances that don't use classic epic links.

**Troubleshooting:** If the configured epic link field is not found on any polled issues, a warning will be logged suggesting verification of the field ID. This helps identify misconfigured field names.

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

### Branch Pattern Support

Orchestrations can specify a branch pattern to automatically checkout or create a branch before the agent runs. This is useful for feature development workflows where each Jira issue or GitHub issue gets its own branch.

```yaml
orchestrations:
  - name: "feature-development"
    trigger:
      source: jira
      project: DS
      tags: ["ready-for-dev"]
    agent:
      prompt: "Implement the feature described in {jira_summary}..."
      github:
        org: my-org
        repo: my-project
        branch: "feature/{jira_issue_key}"  # Creates feature/DS-290
        create_branch: true
        base_branch: main
```

#### Branch Pattern Template Variables

The following template variables can be used in branch patterns:

| Variable | Description | Example |
|----------|-------------|---------|
| `{jira_issue_key}` | Jira issue key | `DS-290` |
| `{github_issue_number}` | GitHub issue/PR number | `123` |
| `{jira_summary}` | Issue summary (use with caution) | `Add login feature` |
| `{github_issue_title}` | Issue/PR title (use with caution) | `Fix bug in auth` |

**Note:** When using `{jira_summary}` or `{github_issue_title}` in branch names, be aware that these may contain characters that are invalid for Git branch names. Consider using `{jira_issue_key}` or `{github_issue_number}` for safer branch names.

**Example of invalid branch name failure:**

If your Jira issue summary is "Fix login/authentication bug" and you use this configuration:

```yaml
github:
  branch: "feature/{jira_summary}"
  create_branch: true
```

The branch creation will fail because the resulting branch name `feature/Fix login/authentication bug` contains invalid characters (spaces and slashes within the summary portion). Git will return an error like:

```
fatal: 'feature/Fix login/authentication bug' is not a valid branch name
```

**Recommended approach:** Use `{jira_issue_key}` instead, which always produces valid branch names:

```yaml
github:
  branch: "feature/{jira_issue_key}"  # Results in: feature/DS-290
  create_branch: true
```

**Future Enhancement:** Automatic branch name sanitization (replacing spaces with dashes, removing slashes, etc.) could be implemented to make `{jira_summary}` and `{github_issue_title}` safer to use in branch patterns.

#### Branch Behavior

| Setting | Default | Description |
|---------|---------|-------------|
| `branch` | (none) | Branch pattern to checkout. If not specified, uses the repository's default branch. |
| `create_branch` | `false` | If `true`, creates the branch from `base_branch` if it doesn't exist. If `false`, fails if the branch doesn't exist. |
| `base_branch` | `main` | The base branch to create new branches from when `create_branch` is `true`. |

**Execution flow:**
1. Branch checkout happens **before** the agent runs
2. If `create_branch: true` and the branch doesn't exist, it's created from `base_branch`
3. If `create_branch: false` (default) and the branch doesn't exist, the execution fails
4. If no branch is specified, the repository's default branch is used

#### Fork Workflow Example

When working with forks, configure the `org` to point to your fork:

```yaml
orchestrations:
  - name: "fork-feature-development"
    trigger:
      source: jira
      project: DS
      tags: ["ready-for-dev"]
    agent:
      prompt: "Implement the feature described in {jira_summary}..."
      tools:
        - jira
        - github
      github:
        org: my-username     # Your fork
        repo: project-name
        branch: "feature/{jira_issue_key}"
        create_branch: true
        base_branch: main
```

This allows agents to push changes to a fork branch, which can then be used to create a pull request to the upstream repository.

**Creating a PR to upstream:** After the agent pushes changes to your fork branch, you can create a pull request to the upstream repository using the GitHub CLI or the GitHub web interface:

```bash
# Using GitHub CLI
gh pr create --repo upstream-org/project-name --head my-username:feature/DS-290 --base main

# Or navigate to GitHub UI:
# https://github.com/upstream-org/project-name/compare/main...my-username:feature/DS-290
```

**Note:** Before using the GitHub CLI (`gh`) commands, ensure you have the [GitHub CLI installed](https://cli.github.com/) and have authenticated with GitHub by running `gh auth login`. This is required for the CLI to access your GitHub account and perform operations like creating pull requests.

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

To enable Claude Code's experimental Agent Teams feature, use `agent_teams: true` in the `agent` block:

```yaml
    agent:
      agent_type: claude

      # Enable Claude Code's experimental Agent Teams feature
      # (only valid when agent_type is "claude"; default: false)
      agent_teams: true
```

**Agent Types:**
- `claude` - Use Claude AI agent (default)
- `cursor` - Use Cursor AI agent

**Default Behavior:** When `agent_type` is omitted from an orchestration config, it defaults to the value of the `SENTINEL_DEFAULT_AGENT_TYPE` environment variable. If that environment variable is also not set, it defaults to `claude`.

**Cursor Modes (only valid when `agent_type: cursor`):**
- `agent` - Full autonomous agent mode (default)
- `plan` - Planning mode - creates plans without executing
- `ask` - Ask mode - waits for user confirmation before actions

**Agent Teams (only valid when `agent_type: claude`):**
- `agent_teams` - Boolean flag to enable Claude Code's experimental Agent Teams feature (default: `false`)
- When set to `true`, enables multi-agent collaboration via Claude Code's Agent Teams capability
- Only supported when `agent_type` is `claude` (or unset, since the default agent type is `claude`)

See `orchestrations/README.md` for full configuration reference.

### Failure Pattern Configuration

Sentinel uses pattern matching to determine agent execution outcomes. Proper pattern selection is critical for reliable success/failure detection.

**Recommended failure patterns:**

| Pattern | Description |
|---------|-------------|
| `FAILURE` | All-caps keyword indicating explicit failure |
| `TASK_FAILED` | Specific compound keyword for task failures |
| `ERROR:` | Error prefix with colon (avoids matching "error" in prose) |
| `COULD_NOT_COMPLETE:` | Specific prefix for incomplete tasks |

**Avoid generic patterns** like `error`, `failed`, or `could not` as they may cause false positives (e.g., "No errors found" would incorrectly match `error`).

Example configuration:

```yaml
retry:
  max_attempts: 3
  failure_patterns:
    - "FAILURE"
    - "TASK_FAILED"
    - "ERROR:"
    - "COULD_NOT_COMPLETE:"
```

See [docs/FAILURE_PATTERNS.md](docs/FAILURE_PATTERNS.md) for detailed guidance on pattern selection, examples of good vs. bad patterns, and migration recommendations.

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
      labels:                     # Filter by GitHub labels (optional)
        - "bug"                   # Issues must have ALL labels (AND logic)
        - "needs-triage"          # Case-insensitive matching
      project_filter: 'Status = "Ready for Review"'  # Can combine with labels
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

**GitHub Labels Field:**
- `labels`: List of GitHub labels to filter by
- Issues must have **ALL** specified labels (AND logic)
- Label matching is **case-insensitive** ("Bug" matches "bug", "BUG", etc.)
- Can be combined with `project_filter` for more precise filtering
- Similar to Jira's `tags` field for users familiar with Jira triggers

## Dynamic Orchestration Management

### Hot-Reload

Sentinel automatically detects changes to orchestration files at the start of each poll cycle:

- **New files**: Automatically loaded and routed to matching issues
- **Modified files**: Reloaded with version protection - running executions continue with their original configuration
- **Removed files**: Unloaded after running executions complete

This allows updating orchestration configurations without restarting the service.

### Eager Polling

When work is submitted for execution, Sentinel polls immediately for more work instead of waiting for the next poll interval. This maximizes throughput when there's a backlog of issues to process while conserving resources during idle periods.

## Troubleshooting

### Common Issues

#### Sentinel Not Picking Up Issues

**Symptom:** Issues with the correct tags aren't being processed.

**Solutions:**
1. Verify the project key matches your Jira project exactly (case-sensitive)
2. Check that tags in the orchestration match Jira labels exactly
3. Ensure the issue doesn't already have the `on_start` tag (e.g., `sentinel-processing`)
4. Enable debug logging: `SENTINEL_LOG_LEVEL=DEBUG`

```bash
# Check which orchestrations are loaded
sentinel --config-dir ./orchestrations 2>&1 | grep "Loaded orchestration"
```

#### API Authentication Failures

**Symptom:** Errors like "401 Unauthorized" or "Authentication failed".

**Solutions:**
1. Verify your API tokens are correct and not expired
2. For Jira: Ensure you're using the email associated with the API token
3. For GitHub: Verify token has required scopes (`repo`, `read:org`, `read:project`)

```bash
# Test Jira connectivity
curl -u "your-email@example.com:your-api-token" \
  "https://your-instance.atlassian.net/rest/api/3/myself"
```

#### Agent Execution Fails

**Symptom:** Agent starts but fails with errors.

**Solutions:**
1. Check agent logs in `SENTINEL_AGENT_LOGS_DIR` (default: `./logs`)
2. Verify Claude API key is valid: `ANTHROPIC_API_KEY`
3. Check for rate limiting (see Rate Limiting section in `.env.example`)
4. Review the agent prompt for issues with template variables

#### Out of Memory Errors

**Symptom:** Process killed or "MemoryError" exceptions.

**Solutions:**
1. Reduce `SENTINEL_MAX_CONCURRENT_EXECUTIONS`
2. Increase system memory or container limits
3. Check for runaway processes in workdir

#### Hot-Reload Not Working

**Symptom:** Changes to orchestration files aren't picked up.

**Solutions:**
1. Verify file has valid YAML syntax: `python -c "import yaml; yaml.safe_load(open('file.yaml'))"`
2. Check file permissions (must be readable by sentinel process)
3. Look for load errors in logs at start of poll cycle

### Debug Mode

Enable comprehensive debug logging:

```bash
SENTINEL_LOG_LEVEL=DEBUG SENTINEL_LOG_JSON=false sentinel
```

### Getting Help

1. Check the [examples](./examples/) for working configurations
2. Review [docs/FAILURE_PATTERNS.md](docs/FAILURE_PATTERNS.md) for outcome detection
3. See [docs/TESTING.md](docs/TESTING.md) for running tests
4. Open an issue on GitHub with:
   - Sentinel version
   - Relevant log output (sanitize credentials!)
   - Orchestration configuration (sanitize sensitive data)

## Development

```bash
# Run all tests
pytest

# Run only unit tests (fast, no external dependencies)
pytest tests/unit

# Run only integration tests
pytest tests/integration -m integration

# Run tests excluding integration tests
pytest -m "not integration"

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

For detailed testing guidelines, see [docs/TESTING.md](docs/TESTING.md).

### Contribution Guidelines

When contributing to this project, please follow these guidelines:

#### Code Comments

- **Do not include ticket references in code comments** (e.g., "DS-123:", "PROJ-456:"). These make the code read like a changelog rather than maintainable source code.
- Ticket references belong in commit messages and pull request descriptions where they provide valuable context for git blame and code review.
- Comments should explain *why* code exists, not *when* it was added or which ticket requested it.

**Bad:**
```python
# DS-123: Add rate limiting to prevent abuse
rate_limiter = RateLimiter()
```

**Good:**
```python
# Rate limiting prevents abuse from excessive API calls
rate_limiter = RateLimiter()
```

#### Commit Messages

- Reference tickets in commit messages where appropriate (e.g., "Add rate limiting [DS-123]")
- This preserves the connection between code and tickets via git blame without cluttering the source code

## License

MIT
