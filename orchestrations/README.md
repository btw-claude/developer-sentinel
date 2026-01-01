# Orchestrations Directory

This directory contains YAML configuration files that define how Developer Sentinel processes Jira issues. Each orchestration connects a Jira trigger to a Claude agent workflow.

## Quick Start

1. Copy `example.yaml` to create your own orchestration
2. Configure the trigger to match your Jira project and tags
3. Write a prompt that instructs the agent what to do
4. Configure tag management for the workflow lifecycle

## Creating a New Orchestration

Create a new `.yaml` or `.yml` file in this directory. Each file can contain multiple orchestrations:

```yaml
orchestrations:
  - name: "my-orchestration"
    trigger:
      # ... trigger configuration
    agent:
      # ... agent configuration
    retry:
      # ... retry configuration (optional)
    on_complete:
      # ... success actions (optional)
    on_failure:
      # ... failure actions (optional)
```

## Configuration Reference

### Orchestration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the orchestration |
| `trigger` | object | Yes | Defines which Jira issues activate this orchestration |
| `agent` | object | Yes | Configures the Claude agent behavior |
| `retry` | object | No | Retry logic and pattern matching |
| `on_complete` | object | No | Actions after successful processing |
| `on_failure` | object | No | Actions when all retries are exhausted |

### Trigger Configuration

```yaml
trigger:
  source: jira                    # Only "jira" is supported
  project: "PROJ"                 # Jira project key
  jql_filter: "status != Closed"  # Additional JQL filter (optional)
  tags:                           # Labels that trigger this orchestration
    - "needs-review"
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source` | string | No | `"jira"` | Issue source (only "jira" supported) |
| `project` | string | No | `""` | Jira project key to watch |
| `jql_filter` | string | No | `""` | Additional JQL conditions |
| `tags` | list | No | `[]` | Labels that must be present on the issue |

**Note:** Issues matching the trigger must have ALL specified tags.

### Agent Configuration

```yaml
agent:
  prompt: |
    You are an assistant. Process this issue and respond with SUCCESS or FAILURE.
  tools:
    - jira
    - github
  github:
    host: "github.com"
    org: "your-org"
    repo: "your-repo"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Instructions for the Claude agent |
| `tools` | list | No | Tools available to the agent |
| `github` | object | No | GitHub repository context |

#### Available Tools

| Tool | Description |
|------|-------------|
| `jira` | Read/write Jira issues, comments, and labels |
| `confluence` | Read/write Confluence pages |
| `github` | Access GitHub repositories, PRs, and code |

#### GitHub Context

Required when using the `github` tool:

```yaml
github:
  host: "github.com"      # GitHub host (default: github.com)
  org: "your-org"         # Organization or user name
  repo: "your-repo"       # Repository name
```

### Retry Configuration

```yaml
retry:
  max_attempts: 3
  success_patterns:
    - "SUCCESS"
    - "completed successfully"
  failure_patterns:
    - "FAILURE"
    - "failed"
    - "error"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_attempts` | int | `3` | Maximum execution attempts |
| `success_patterns` | list | `["SUCCESS", "completed successfully"]` | Patterns indicating success |
| `failure_patterns` | list | `["FAILURE", "failed", "error"]` | Patterns indicating failure |

#### Pattern Matching

Patterns are matched case-insensitively against the agent's response:

- **Simple strings**: Matched as substrings (e.g., `"SUCCESS"` matches `"Task SUCCESS"`)
- **Regex patterns**: Patterns starting with `^`, ending with `$`, or containing `*` are treated as regex

If the response matches a success pattern, execution is considered successful. If it matches a failure pattern (and no success pattern), the execution is retried. If neither pattern matches, execution defaults to success.

### On Complete Configuration

Actions taken after successful execution:

```yaml
on_complete:
  remove_tag: "needs-review"    # Remove this tag from the issue
  add_tag: "reviewed"           # Add this tag to the issue
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `remove_tag` | string | `""` | Tag to remove after success |
| `add_tag` | string | `""` | Tag to add after success |

### On Failure Configuration

Actions taken when all retry attempts are exhausted:

```yaml
on_failure:
  add_tag: "review-failed"      # Add this tag for investigation
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `add_tag` | string | `""` | Tag to add after failure |

**Note:** Trigger tags are NOT removed on failure, allowing for manual investigation and retry.

## Tag-Based Workflow

Sentinel uses a stateless, tag-based workflow to track issue processing:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tag Workflow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Issue created with trigger tag (e.g., "needs-review")      │
│                            │                                     │
│                            ▼                                     │
│   2. Sentinel polls and picks up the issue                      │
│                            │                                     │
│                            ▼                                     │
│   3. Agent processes the issue                                  │
│                            │                                     │
│              ┌─────────────┴─────────────┐                      │
│              │                           │                       │
│              ▼                           ▼                       │
│   4a. SUCCESS                  4b. FAILURE (after retries)      │
│   - Remove trigger tag         - Keep trigger tag                │
│   - Add completion tag         - Add failure tag                 │
│   (e.g., "reviewed")          (e.g., "review-failed")           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Approach?

1. **Idempotent Processing**: Issues without trigger tags won't be reprocessed
2. **Visible State**: Tag state clearly shows processing status
3. **Easy Retry**: Keep trigger tags on failure for manual investigation and retry
4. **Audit Trail**: Completion/failure tags provide history

### Example Tag Flow

| State | Tags on Issue |
|-------|---------------|
| Initial | `needs-review` |
| Processing | `needs-review` (unchanged) |
| Success | `reviewed` |
| Failure | `needs-review`, `review-failed` |

## Best Practices

### Writing Effective Prompts

1. **Be specific**: Clearly state what the agent should do
2. **Include context**: Mention what tools are available
3. **Define success/failure**: Tell the agent how to indicate completion
4. **Provide examples**: Give examples of expected output when helpful

```yaml
prompt: |
  You are a code review assistant with access to Jira and GitHub.

  Your task:
  1. Read the linked GitHub PR from the Jira issue
  2. Review the code for bugs, security issues, and style
  3. Post your review as a GitHub PR review
  4. Add a summary comment to the Jira issue

  When complete, respond with:
  - SUCCESS: if the review was posted successfully
  - FAILURE: if you could not complete the review (explain why)
```

### Choosing Retry Settings

- **`max_attempts: 1`**: For operations that shouldn't be retried (e.g., sending notifications)
- **`max_attempts: 2-3`**: For most operations (default is 3)
- **`max_attempts: 5+`**: For operations with transient failures

### Tag Naming Conventions

Use consistent, descriptive tag names:

| Purpose | Pattern | Example |
|---------|---------|---------|
| Trigger | `needs-<action>` | `needs-review`, `needs-docs` |
| Completion | `<action>-complete` or `<past-tense>` | `reviewed`, `docs-updated` |
| Failure | `<action>-failed` | `review-failed`, `docs-update-failed` |

## Troubleshooting

### Issue Not Being Picked Up

1. Check that the issue has ALL required trigger tags
2. Verify the project key matches
3. Check the JQL filter isn't excluding the issue
4. Ensure the issue status isn't excluded (default excludes Done/Closed/Resolved)

### Agent Keeps Retrying

1. Ensure success patterns match the agent's actual response
2. Check for typos in pattern strings
3. Use broader patterns if needed (e.g., `"complete"` instead of `"completed successfully"`)

### Tags Not Updating

1. Verify the Jira credentials have permission to modify labels
2. Check the tag names don't contain special characters
3. Look for errors in the Sentinel logs
