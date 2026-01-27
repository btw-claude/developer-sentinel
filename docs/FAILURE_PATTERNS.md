# Failure Pattern Guide

This document describes best practices for configuring `failure_patterns` and `success_patterns` in orchestration configurations. Proper pattern selection ensures reliable detection of agent execution outcomes while minimizing false positives.

## Overview

Sentinel uses pattern matching to determine whether an agent execution succeeded or failed. Patterns are matched against the agent's response text using case-sensitive substring matching (or regex with the `regex:` prefix).

**Key Principle:** Use specific, unambiguous patterns that are unlikely to appear in normal prose or error descriptions.

## Recommended Failure Patterns

The following patterns are recommended for detecting agent failures:

| Pattern | Description | Recommended |
|---------|-------------|-------------|
| `FAILURE` | All-caps keyword indicating explicit failure | Yes |
| `TASK_FAILED` | Specific compound keyword for task failures | Yes |
| `ERROR:` | Error prefix with colon (avoids matching "error" in prose) | Yes |
| `COULD_NOT_COMPLETE:` | Specific prefix for incomplete tasks | Yes |
| `COMPLETION_FAILED` | Specific compound keyword | Yes |

### Example Configuration

```yaml
retry:
  max_attempts: 3
  success_patterns:
    - "SUCCESS"
    - "TASK_COMPLETED"
  failure_patterns:
    - "FAILURE"
    - "TASK_FAILED"
    - "ERROR:"
    - "COULD_NOT_COMPLETE:"
```

## Patterns to Avoid

Some patterns are too generic and may cause false positives:

| Pattern | Problem | Better Alternative |
|---------|---------|-------------------|
| `error` | Matches "No errors found", "error-free", etc. | `ERROR:` |
| `failed` | Matches "tests failed to find issues" (good thing) | `TASK_FAILED` |
| `could not complete` | Too generic, could match prose descriptions | `COULD_NOT_COMPLETE:` |
| `problem` | Common word in descriptions | `PROBLEM_DETECTED:` |

### Why Specificity Matters

Consider an agent response like:

> "Code review complete. No errors found in the implementation. The error handling looks solid."

A generic `error` pattern would incorrectly match this successful response. Using `ERROR:` (with the colon) ensures we only match explicit error declarations like:

> "ERROR: Could not access the repository"

## Good vs. Bad Pattern Examples

### Good Patterns

```yaml
# Good: Specific, unambiguous patterns
failure_patterns:
  - "FAILURE"           # All-caps, unlikely in normal prose
  - "TASK_FAILED"       # Compound keyword, very specific
  - "ERROR:"            # Colon prevents matching "error" in prose
  - "COULD_NOT_COMPLETE:" # Specific prefix format
  - "API_ERROR"         # Technical, specific
  - "BLOCKED:"          # Clear status indicator
```

### Bad Patterns

```yaml
# Bad: Generic patterns prone to false positives
failure_patterns:
  - "error"             # Matches "No errors found"
  - "failed"            # Matches "failed to find issues" (success!)
  - "could not"         # Too generic
  - "problem"           # Common word in descriptions
  - "issue"             # Matches "resolved the issue"
  - "bug"               # Matches "no bugs detected"
```

## Default Patterns

When no `failure_patterns` are specified, Sentinel uses these defaults:

```python
failure_patterns = ["FAILURE", "failed", "error"]
```

**Note:** The defaults include `failed` and `error` for backwards compatibility. For new orchestrations, we recommend using the more specific patterns documented above.

## Pattern Matching Details

### Substring Matching (Default)

By default, patterns are matched as case-sensitive substrings:

```yaml
failure_patterns:
  - "FAILURE"  # Matches "FAILURE", "FAILURE: reason", "Task FAILURE"
```

### Regex Matching

Use the `regex:` prefix for more complex matching:

```yaml
failure_patterns:
  - "regex:FAIL(ED|URE)"           # Matches FAILED or FAILURE
  - "regex:ERROR\\s*:"             # ERROR with optional whitespace before colon
  - "regex:(?i)task.?failed"       # Case-insensitive with optional character
```

## Agent Prompt Guidelines

When writing agent prompts, instruct agents to use specific outcome keywords:

```yaml
agent:
  prompt: |
    Review the code changes for {jira_issue_key}.

    After completing your review, end your response with one of:
    - "SUCCESS" if the review was completed
    - "FAILURE: <reason>" if you couldn't complete the review
    - "ERROR: <description>" if you encountered a technical error
```

This ensures agents produce output that matches your configured patterns reliably.

## Troubleshooting

### False Positives (Success Detected as Failure)

If successful executions are being marked as failures:
1. Check if your failure patterns are too generic
2. Look for the pattern in successful agent responses
3. Make the pattern more specific (add colon, use compound words, etc.)

### False Negatives (Failure Detected as Success)

If failed executions are being marked as successes:
1. Ensure agents are using the expected failure keywords in their responses
2. Add the actual failure text to your failure patterns
3. Consider using regex patterns for more flexible matching

## Migration Guide

If you're using the older `could not complete` pattern, consider migrating to the more specific `COULD_NOT_COMPLETE:` format:

**Before:**
```yaml
failure_patterns:
  - "FAILURE"
  - "could not complete"  # Generic, potential false positives
```

**After:**
```yaml
failure_patterns:
  - "FAILURE"
  - "COULD_NOT_COMPLETE:"  # Specific prefix format
```

Update your agent prompts to use the new format:

**Before:**
> "I could not complete the task because..."

**After:**
> "COULD_NOT_COMPLETE: I was unable to finish because..."
