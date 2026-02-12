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
| Agent working directory names | `{ISSUE_KEY}_{YYYYMMDD-HHMMSS}_a{N}` | `DS-123_20260210-030000_a1` |
| Streaming log line prefixes | `[HH:MM:SS.mmm]` | `[03:00:00.123]` |
| Log file header/footer timestamps | ISO 8601 | `2026-02-10T03:00:00+00:00` |
| Log filenames | `{ISSUE_KEY}_{YYYYMMDD-HHMMSS}_a{N}.log` | `DS-123_20260210-030000_a1.log` |
| Dashboard state timestamps | ISO 8601 | `2026-02-10T03:00:00+00:00` |

### Naming format details (DS-960)

The naming format `{issue_key}_{YYYYMMDD-HHMMSS}_a{attempt}` uses:
- `_` to separate issue key from timestamp (issue keys contain `-` but not `_`)
- `-` to separate date from time within timestamp (disambiguates from `_` delimiter)
- `_a{N}` suffix to guarantee uniqueness across retry attempts

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

> **Note:** This list is validated automatically by the CI test
> `tests/unit/test_utc_modules_list.py` (added in DS-886). If a module is added
> or removed, the test will fail until this section is updated. To find the
> current set of modules manually, run:
>
> ```bash
> grep -r --include="*.py" "datetime.now(tz=UTC)" src/
> ```

## History

- **DS-879:** Audited and replaced all naive `datetime.now()` calls with `datetime.now(tz=UTC)` across 7 source modules and 2 test files.
- **DS-880:** Documented the UTC convention (this file) and added clarifying code comments to `base.py` and `agent_logger.py`.
- **DS-886:** Added CI validation test (`tests/unit/test_utc_modules_list.py`) to automatically detect when the affected modules list drifts from the actual source tree.
- **DS-960:** Added issue key and attempt number to log filenames and workdir names. Changed naming from `YYYYMMDD_HHMMSS` to `{issue_key}_{YYYYMMDD-HHMMSS}_a{attempt}` to fix collision bugs and add issue context.
