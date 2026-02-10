# UTC Timestamp Convention

All timestamps throughout the Sentinel codebase use **timezone-aware UTC** (`datetime.now(tz=UTC)`). This convention was established in DS-879 to prevent subtle bugs when comparing naive and timezone-aware datetime objects.

## Why UTC?

- **Consistency:** A single timezone across all components eliminates ambiguity when correlating events across log files, working directories, and database records.
- **Server-side correctness:** UTC is the standard for server-side applications, avoiding issues with daylight saving time transitions and varying server locations.
- **Timezone safety:** Using `datetime.now(tz=UTC)` produces timezone-aware objects, preventing `TypeError` when comparing against other timezone-aware datetimes (e.g., from Jira or GitHub APIs).

## What This Means for Operators

Timestamps visible in the following locations are **UTC, not local time**:

| Location | Format | Example |
|----------|--------|---------|
| Agent working directory names | `{ISSUE_KEY}_{YYYYMMDD_HHMMSS}` | `DS-123_20260210_030000` |
| Streaming log line prefixes | `[HH:MM:SS.mmm]` | `[03:00:00.123]` |
| Log file header/footer timestamps | ISO 8601 | `2026-02-10T03:00:00+00:00` |
| Log filenames | `YYYYMMDD_HHMMSS.log` | `20260210_030000.log` |
| Dashboard state timestamps | ISO 8601 | `2026-02-10T03:00:00+00:00` |

If you are reviewing logs or directory names and the times appear offset from your local clock, this is expected. To convert to local time, apply your timezone offset (e.g., UTC-8 for PST, UTC-5 for EST).

## Developer Guidelines

### Always use timezone-aware UTC

```python
from datetime import UTC, datetime

# Correct
now = datetime.now(tz=UTC)
ts = datetime.fromtimestamp(epoch_value, tz=UTC)

# Incorrect - produces naive datetime
now = datetime.now()
ts = datetime.fromtimestamp(epoch_value)
```

### Import style

Use the established import pattern:

```python
from datetime import UTC, datetime
```

### Affected modules

The following source modules use `datetime.now(tz=UTC)`:

- `src/sentinel/agent_clients/base.py` -- working directory timestamp
- `src/sentinel/agent_clients/claude_sdk.py` -- execution timing
- `src/sentinel/agent_clients/codex.py` -- execution timing
- `src/sentinel/agent_logger.py` -- log line timestamps and execution timing
- `src/sentinel/dashboard/state.py` -- uptime calculations
- `src/sentinel/executor.py` -- execution start/end times
- `src/sentinel/main.py` -- poll timestamps and completion times
- `src/sentinel/orchestration.py` -- orchestration version load times
- `src/sentinel/state_tracker.py` -- tracking start, queue, and run times

## History

- **DS-879:** Audited and replaced all naive `datetime.now()` calls with `datetime.now(tz=UTC)` across 7 source modules and 2 test files.
- **DS-880:** Documented the UTC convention (this file) and added clarifying code comments to `base.py` and `agent_logger.py`.
