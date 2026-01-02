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
| `on_start` | object | No | Actions when issue is picked up (prevents duplicate processing) |
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
    You are an assistant. Review issue {jira_issue_key}: {jira_summary}

    Description:
    {jira_description}

    Repository: {github_org}/{github_repo}

    Respond with SUCCESS or FAILURE when complete.
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
| `prompt` | string | Yes | Instructions for the Claude agent (supports template variables) |
| `model` | string | No | Model to use (e.g., `claude-opus-4-5-20251101`). Uses CLI default if not specified. |
| `tools` | list | No | Tools available to the agent |
| `github` | object | No | GitHub repository context |
| `timeout_seconds` | int | No | Optional timeout in seconds for agent execution |

#### Template Variables

The prompt supports template variables that are substituted with actual values from the Jira issue and GitHub context. Use `{variable_name}` syntax in your prompt:

**Jira Variables:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `{jira_issue_key}` | Issue key | `"DS-123"` |
| `{jira_summary}` | Issue title/summary | `"Fix login bug"` |
| `{jira_description}` | Full issue description | `"Users cannot log in..."` |
| `{jira_status}` | Current status | `"In Progress"` |
| `{jira_assignee}` | Assignee display name | `"John Smith"` |
| `{jira_labels}` | Comma-separated labels | `"bug, high-priority"` |
| `{jira_comments}` | Recent comments (last 3, truncated to 500 chars each) | `"1. Comment text..."` |
| `{jira_links}` | Comma-separated linked issue keys | `"DS-100, DS-101"` |

**GitHub Variables:**

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `{github_host}` | GitHub host | `"github.com"` |
| `{github_org}` | Organization name | `"your-org"` |
| `{github_repo}` | Repository name | `"your-repo"` |

**Example prompt with template variables:**

```yaml
prompt: |
  You are a code review assistant. Review issue {jira_issue_key}.

  ## Issue Details
  **Summary:** {jira_summary}
  **Status:** {jira_status}
  **Assignee:** {jira_assignee}
  **Labels:** {jira_labels}

  **Description:**
  {jira_description}

  **Recent Comments:**
  {jira_comments}

  **Related Issues:** {jira_links}

  **Repository:** {github_org}/{github_repo}

  ## Your Task
  Review the code changes and provide feedback. Post your findings as a
  comment on the Jira issue.

  End your response with SUCCESS or FAILURE.
```

**Notes:**
- Unknown variables (not in the list above) are preserved as-is, allowing literal braces in prompts
- Empty values are substituted with empty strings
- GitHub variables require the `github` section to be configured

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

#### Model Selection

Optionally specify which Claude model to use for this orchestration:

```yaml
agent:
  model: "claude-opus-4-5-20251101"
  prompt: |
    ...
```

| Model | Identifier |
|-------|------------|
| Opus 4.5 | `claude-opus-4-5-20251101` |
| Sonnet 4 | `claude-sonnet-4-20250514` |
| Haiku 3.5 | `claude-haiku-3-5-20241022` |

If not specified, the agent uses the Claude CLI's default model (typically your configured default).

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
| `default_status` | string | `"success"` | Status when no patterns match (`"success"` or `"failure"`) |

#### Pattern Matching

Patterns are matched case-insensitively against the agent's response:

- **Simple strings**: Matched as substrings (e.g., `"SUCCESS"` matches `"Task SUCCESS"`)
- **Regex patterns**: Patterns starting with `^`, ending with `$`, or containing `*` are treated as regex

If the response matches a success pattern, execution is considered successful. If it matches a failure pattern (and no success pattern), the execution is retried. If neither pattern matches, the `default_status` setting determines the outcome (defaults to `"success"`).

### On Start Configuration

Actions taken immediately when an issue is picked up for processing. Use this to prevent duplicate processing if your poll interval is shorter than the typical processing time.

```yaml
on_start:
  add_tag: "sentinel-processing"  # Mark issue as being processed
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `add_tag` | string | `""` | Tag to add when processing starts (automatically removed after processing) |

**Important:**
- Configure your trigger to exclude issues with this tag, so the poller will skip issues already being processed
- The `add_tag` is **automatically removed** after processing completes (on both success and failure)

### On Complete Configuration

Actions taken after successful execution:

```yaml
on_complete:
  add_tag: "reviewed"           # Add completion tag to the issue
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `add_tag` | string | `""` | Tag to add after success |

**Note:** Trigger tags are already removed when processing starts, so `remove_tag` is typically not needed.

### On Failure Configuration

Actions taken when all retry attempts are exhausted:

```yaml
on_failure:
  add_tag: "review-failed"      # Add this tag for investigation
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `add_tag` | string | `""` | Tag to add after failure |

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
│      - Removes trigger tag                                       │
│      - Adds in-progress tag (e.g., "sentinel-processing")       │
│                            │                                     │
│                            ▼                                     │
│   3. Agent processes the issue                                  │
│                            │                                     │
│              ┌─────────────┴─────────────┐                      │
│              │                           │                       │
│              ▼                           ▼                       │
│   4a. SUCCESS                  4b. FAILURE (after retries)      │
│   - Remove in-progress tag     - Remove in-progress tag         │
│   - Add completion tag         - Add failure tag                 │
│   (e.g., "reviewed")          (e.g., "review-failed")           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Approach?

1. **Idempotent Processing**: Trigger tags are removed immediately, preventing duplicate processing
2. **Visible State**: In-progress tag clearly shows which issues are being processed
3. **No Race Conditions**: Trigger tag removal + in-progress tag addition happens atomically at pickup
4. **Audit Trail**: Completion/failure tags provide history

### Example Tag Flow

| State | Tags on Issue |
|-------|---------------|
| Initial | `needs-review` |
| Processing | `sentinel-processing` |
| Success | `reviewed` |
| Failure | `review-failed` |

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
