# Multi-Orchestration Example

An advanced example demonstrating multiple triggers and workflows working together.

## What This Example Does

This example sets up three different workflows that can run concurrently:

1. **Code Review** - Reviews PRs linked to Jira issues
2. **Documentation Update** - Updates Confluence docs based on Jira issues
3. **GitHub Project Triage** - Triages items in a GitHub Project board

## Files

```
multi-orchestration/
├── README.md                    # This file
├── .env.example                 # Environment template
└── orchestrations/
    ├── code-review.yaml         # PR review workflow
    ├── docs-update.yaml         # Documentation workflow
    └── github-triage.yaml       # GitHub Project workflow
```

## Key Features Demonstrated

### Multiple Trigger Types
- **Jira triggers** - Poll Jira projects for tagged issues
- **GitHub triggers** - Poll GitHub Projects for matching items

### Tool Combinations
- Jira + GitHub - For code review workflows
- Jira + Confluence - For documentation workflows
- GitHub only - For GitHub-native workflows

### Branch Pattern Support
- Automatic branch creation for feature work
- Template variables like `{jira_issue_key}`

### Outcome-Based Tagging
- Different tags for different review outcomes
- Approved vs. Changes Requested workflows

## Setup Instructions

### 1. Install Developer Sentinel

```bash
# From the repository root
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.example .env
vim .env
```

Required settings:
- Jira credentials (for Jira-triggered workflows)
- GitHub token (for GitHub-triggered workflows)

### 3. Customize Orchestrations

Each YAML file includes detailed comments. Key customizations:
- Project keys and repository names
- Trigger tags/labels
- Agent prompts

### 4. Run the Sentinel

```bash
sentinel --config-dir examples/multi-orchestration/orchestrations
```

## Workflow Details

### Code Review (code-review.yaml)

```
Trigger: Jira issue with "needs-code-review" label
Action:  Review linked GitHub PR
Output:  "code-reviewed" or "changes-requested" label
```

### Documentation Update (docs-update.yaml)

```
Trigger: Jira issue with "needs-docs" label
Action:  Update Confluence documentation
Output:  "docs-updated" label
```

### GitHub Triage (github-triage.yaml)

```
Trigger: GitHub Project item with Status = "Needs Triage"
Action:  Categorize and prioritize the issue
Output:  "triaged" label
```

## Running Multiple Workflows

The sentinel automatically:
- Loads all YAML files from the orchestrations directory
- Polls for matching issues/items on each cycle
- Routes items to the appropriate workflow
- Handles concurrent processing (configurable)

```bash
# Increase concurrent executions for faster processing
SENTINEL_MAX_CONCURRENT_EXECUTIONS=3 sentinel --config-dir examples/multi-orchestration/orchestrations
```

## Debugging Tips

1. **Check loaded orchestrations:**
   ```
   INFO - Loaded orchestration: code-review
   INFO - Loaded orchestration: docs-update
   INFO - Loaded orchestration: github-triage
   ```

2. **Enable debug logging:**
   ```bash
   SENTINEL_LOG_LEVEL=DEBUG sentinel ...
   ```

3. **Test one workflow at a time:**
   Move other YAML files out of the directory temporarily.

## Next Steps

- See the [docker](../docker/) example for production deployment
- Read about [hot-reload](../../README.md#hot-reload) for live config updates
- Check [failure patterns](../../docs/FAILURE_PATTERNS.md) for reliable detection
