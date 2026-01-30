# Basic Setup Example

A minimal working configuration to get Developer Sentinel running quickly.

## What This Example Does

This example sets up a simple code review workflow:
1. Monitors a Jira project for issues tagged with `needs-code-review`
2. Sends matching issues to a Claude agent for review
3. Posts review feedback as a Jira comment
4. Adds `code-reviewed` tag on completion

## Files

```
basic-setup/
├── README.md              # This file
├── .env.example           # Environment template
└── orchestrations/
    └── code-review.yaml   # Single orchestration config
```

## Setup Instructions

### 1. Install Developer Sentinel

```bash
# From the repository root
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit with your actual values
vim .env
```

Required settings:
- `JIRA_BASE_URL` - Your Jira instance URL
- `JIRA_EMAIL` - Your Jira account email
- `JIRA_API_TOKEN` - Your Jira API token

### 3. Customize the Orchestration

Edit `orchestrations/code-review.yaml`:
- Change `project: "DEMO"` to your Jira project key
- Adjust the trigger tags if needed
- Customize the agent prompt for your workflow

### 4. Run the Sentinel

```bash
# From the repository root
sentinel --config-dir examples/basic-setup/orchestrations
```

## Testing Your Setup

1. Create a test issue in your Jira project
2. Add the label `needs-code-review` to the issue
3. Watch the sentinel logs for processing
4. Check the issue for a new comment from the agent

## Expected Output

```
INFO - Loaded orchestration: code-review
INFO - Starting poll cycle...
INFO - Found 1 issue matching orchestration 'code-review'
INFO - Processing issue DEMO-123
INFO - Agent completed with SUCCESS
INFO - Added label 'code-reviewed' to DEMO-123
```

## Next Steps

- Try the [multi-orchestration](../multi-orchestration/) example for more complex workflows
- See the [docker](../docker/) example for production deployment
- Read the main [README](../../README.md) for all configuration options
