# GitHub Trigger Migration Guide

This guide explains how to migrate from the deprecated repository-based GitHub triggers to the new project-based triggers introduced in Developer Sentinel.

## Overview of Changes

Developer Sentinel now uses **GitHub Projects (v2)** as the source for GitHub-triggered orchestrations. This provides several advantages:

1. **Multi-repository support**: A single project can contain issues/PRs from multiple repositories
2. **Rich filtering**: Filter by project fields (Status, Priority, Sprint, etc.) using JQL-like expressions
3. **Better deduplication**: Avoids polling the same project multiple times
4. **Unified workflow**: Manage work across repos in a single project board

## Breaking Changes

The following trigger fields are **deprecated** for GitHub triggers:

| Deprecated Field | Status | Migration |
|-----------------|--------|-----------|
| `repo` | Deprecated | Use `project_owner` + project containing the repo |
| `tags` | Deprecated | Use `project_filter` with label conditions |
| `query_filter` | Deprecated | Use `project_filter` |

**Note**: These fields will continue to work but will emit deprecation warnings. They will be removed in a future version.

## Migration Examples

### Before: Repository-based trigger (deprecated)

```yaml
orchestrations:
  - name: "code-review"
    trigger:
      source: github
      repo: "my-org/my-repo"           # DEPRECATED
      tags:                             # DEPRECATED
        - "needs-review"
      query_filter: "is:pr is:open"    # DEPRECATED
    agent:
      # ...
```

### After: Project-based trigger (recommended)

```yaml
orchestrations:
  - name: "code-review"
    trigger:
      source: github

      # New required fields
      project_number: 42              # From your project URL
      project_owner: "my-org"         # Organization or user
      project_scope: "org"            # "org" or "user"

      # New filter field (replaces tags and query_filter)
      project_filter: 'Status = "Ready for Review"'
    agent:
      # ...
```

## New Configuration Fields

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `project_number` | integer | The project number from the GitHub project URL |
| `project_owner` | string | The organization name or username that owns the project |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project_scope` | string | `"org"` | Either `"org"` (organization project) or `"user"` (personal project) |
| `project_filter` | string | `""` | JQL-like filter expression to select items |

### Finding Your Project Number

Your project number is in the GitHub project URL:

```
https://github.com/orgs/YOUR-ORG/projects/42
                                          ^^
                                    project_number: 42
```

For personal projects:
```
https://github.com/users/YOUR-USERNAME/projects/5
                                                ^
                                    project_number: 5
```

## Filter Expression Syntax

The `project_filter` field supports a JQL-like syntax for filtering project items.

### Basic Comparisons

```yaml
# Equality
project_filter: 'Status = "Ready"'

# Inequality
project_filter: 'Status != "Done"'

# Field names are case-sensitive and must match your project's custom fields
```

### Logical Operators

```yaml
# AND - both conditions must be true
project_filter: 'Status = "Ready" AND Priority = "High"'

# OR - either condition can be true
project_filter: 'Status = "Ready" OR Status = "In Progress"'

# Parentheses for grouping
project_filter: '(Status = "Ready" OR Status = "In Progress") AND Priority = "High"'
```

### Common Filter Patterns

```yaml
# Items ready for review
project_filter: 'Status = "Ready for Review"'

# High priority items not done
project_filter: 'Priority = "High" AND Status != "Done"'

# Items in current sprint
project_filter: 'Sprint = "Sprint 23"'

# Multiple statuses
project_filter: 'Status = "Ready" OR Status = "Needs Triage"'
```

## Multi-Repository Support

GitHub Projects (v2) can contain issues and PRs from multiple repositories. Developer Sentinel automatically extracts the repository context from each item's URL.

### How It Works

1. Sentinel polls the configured GitHub Project
2. For each item, it extracts the repository from the item's URL (e.g., `https://github.com/org/repo-a/issues/123`)
3. GitHub operations (labels, comments, etc.) are automatically directed to the correct repository

### Example Configuration

```yaml
orchestrations:
  - name: "cross-repo-triage"
    trigger:
      source: github
      project_number: 10
      project_owner: "my-org"
      project_scope: "org"
      # This will match items from ANY repo in the project
      project_filter: 'Status = "Needs Triage"'

    agent:
      prompt: |
        Triage the issue. The system will automatically handle
        the correct repository for each item.
      tools:
        - github
      github:
        host: "github.com"
        org: "my-org"
        repo: "default-repo"  # Default for cloning; actual repo from item URL
```

## Deduplication Changes

### Previous Behavior (deprecated)

Deduplication was based on `repo` + `tags`:
```
github:my-org/my-repo:needs-review,high-priority
```

### New Behavior

Deduplication is now based on `project_owner` + `project_number`:
```
github:my-org/42
```

This means:
- Multiple orchestrations can watch the same project with different filters
- The project is only polled once per cycle
- Each orchestration's filter is applied to the cached results

## Validation Requirements

GitHub triggers now require:

1. `project_number` must be a positive integer
2. `project_owner` must be non-empty
3. `project_scope` must be either `"org"` or `"user"`

Invalid configurations will raise an `OrchestrationError` at load time.

## Deprecation Timeline

| Version | Status |
|---------|--------|
| Current | Deprecated fields emit warnings at load time |
| Future | Deprecated fields will be removed |

## Troubleshooting

### "GitHub trigger requires project_number to be set"

Your configuration is missing the required `project_number` field. Add it:

```yaml
trigger:
  source: github
  project_number: 42  # Add this
  project_owner: "your-org"
```

### "GitHub trigger requires project_owner to be set"

Your configuration is missing the required `project_owner` field. Add it:

```yaml
trigger:
  source: github
  project_number: 42
  project_owner: "your-org"  # Add this
```

### "Invalid project_filter expression"

Check your filter syntax:
- Field names are case-sensitive
- String values must be quoted with double quotes
- Operators: `=`, `!=`, `AND`, `OR`

### Issues not appearing in poll results

1. Verify the project number and owner are correct
2. Check that items in the project have the expected field values
3. Test your filter expression by simplifying it (remove conditions)
4. Ensure items are not DraftIssues (these are skipped)

## Additional Resources

- [GitHub Projects (v2) documentation](https://docs.github.com/en/issues/planning-and-tracking-with-projects)
- [Example orchestration configuration](../orchestrations/example.yaml.example)
- [Integration tests](../tests/test_github_project_integration.py)
