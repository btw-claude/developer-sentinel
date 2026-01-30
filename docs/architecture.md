# Sentinel Architecture

This document describes the internal architecture of the Sentinel orchestrator, focusing on component boundaries and responsibilities.

## Overview

Sentinel follows a **composition over inheritance** design pattern, where the main `Sentinel` class acts as a thin orchestrator that delegates to focused, single-responsibility components. This architecture emerged from refactoring the original "God Object" into composable units (DS-384).

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                      Sentinel                           │
                    │         (Main orchestrator and coordinator)             │
                    └─────────────────────────────────────────────────────────┘
                                              │
           ┌──────────────────┬───────────────┼───────────────┬──────────────────┐
           │                  │               │               │                  │
           ▼                  ▼               ▼               ▼                  ▼
    ┌─────────────┐   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  ┌─────────────┐
    │ StateTracker│   │ Execution   │ │Orchestration│ │    Poll     │  │   Router    │
    │             │   │  Manager    │ │  Registry   │ │ Coordinator │  │             │
    └─────────────┘   └─────────────┘ └─────────────┘ └─────────────┘  └─────────────┘
```

## Core Components

### StateTracker

**Module:** `sentinel/state_tracker.py`

**Purpose:** Manages execution state, metrics, and queues for the Sentinel orchestrator.

**Key Responsibilities:**
- Tracking attempt counts per (issue_key, orchestration_name) pair with TTL-based cleanup
- Managing running step metadata for dashboard display
- Managing the issue queue for issues waiting for execution slots
- Tracking per-orchestration active execution counts for concurrency limits

**Thread Safety:** All public methods are thread-safe and use internal locks with a documented lock ordering discipline to prevent deadlocks.

**Key Data Structures:**
- `_attempt_counts`: Dictionary tracking how many times each issue has been processed
- `_running_steps`: Maps future IDs to `RunningStepInfo` for active executions
- `_issue_queue`: Deque with FIFO eviction when capacity is reached
- `_per_orch_active_counts`: Tracks concurrent executions per orchestration

**Sentinel Integration:** The `Sentinel` class delegates state management through:
- `get_running_steps()` - Returns currently running step metadata
- `get_issue_queue()` - Returns queued issues waiting for slots
- `get_per_orch_count()` / `get_all_per_orch_counts()` - Returns per-orchestration execution counts
- `_get_available_slots_for_orchestration()` - Checks slot availability considering both global and per-orchestration limits

---

### ExecutionManager

**Module:** `sentinel/execution_manager.py`

**Purpose:** Manages thread pool lifecycle, future tracking, and concurrent execution.

**Key Responsibilities:**
- Creating and managing the `ThreadPoolExecutor`
- Tracking active futures with TTL-based cleanup
- Providing execution slot availability information
- Collecting completed results with error handling
- Monitoring and logging long-running futures

**Thread Safety:** All public methods that modify shared state use internal locks.

**Memory Safety:** Futures are tracked with timestamps and cleaned up based on TTL. A maximum list size prevents unbounded memory growth.

**Key Configuration:**
- `max_concurrent_executions`: Maximum number of parallel executions
- `future_ttl_seconds`: TTL for futures before they are considered stale (default: 300s)
- `max_futures`: Maximum number of futures to track (default: 1000)

**Sentinel Integration:** The `Sentinel` class delegates execution management through:
- `_execution_manager.start()` / `_execution_manager.shutdown()` - Lifecycle management
- `_execution_manager.submit()` - Submitting tasks to the thread pool
- `_execution_manager.get_available_slots()` - Checking slot availability
- `_execution_manager.collect_completed_results()` - Collecting execution results

---

### OrchestrationRegistry

**Module:** `sentinel/orchestration_registry.py`

**Purpose:** Manages orchestration loading, hot-reload, and version tracking.

**Key Responsibilities:**
- Tracking known orchestration files and their modification times
- Detecting new, modified, and deleted orchestration files for hot-reload
- Loading and unloading orchestrations with version tracking
- Managing pending removal versions until their executions complete
- Providing observability metrics for hot-reload operations

**Thread Safety:** All public methods that modify shared state use internal locks.

**Hot-Reload Mechanism:**
1. Each polling cycle, the registry scans for file changes (new, modified, deleted)
2. New files are loaded and added to active orchestrations
3. Modified files trigger version replacement with pending removal tracking
4. Deleted files move orchestrations to pending removal until executions complete
5. Pending removal versions are cleaned up once their active executions finish

**Key Data Structures:**
- `_orchestrations`: List of currently active orchestrations
- `_known_orchestration_files`: Maps file paths to last known modification times
- `_active_versions`: List of `OrchestrationVersion` objects currently in use
- `_pending_removal_versions`: Versions being phased out but with active executions

**Sentinel Integration:** The `Sentinel` class delegates orchestration management through:
- `detect_and_load_orchestration_changes()` - Hot-reload new/modified files
- `detect_and_unload_removed_files()` - Handle deleted files
- `cleanup_pending_removal_versions()` - Clean up completed old versions
- `get_version_for_orchestration()` - Get version info for execution tracking

---

### PollCoordinator

**Module:** `sentinel/poll_coordinator.py`

**Purpose:** Coordinates polling cycles for Jira and GitHub issues.

**Key Responsibilities:**
- Polling Jira for issues matching orchestration triggers
- Polling GitHub for issues/PRs matching orchestration triggers
- Managing deduplication of submissions within a polling cycle
- Constructing issue URLs for dashboard display
- Grouping orchestrations by their trigger source (Jira vs GitHub)

**Thread Safety:** Designed for single-threaded polling loop execution. The deduplication manager handles thread safety internally.

**Key Features:**
- Unique trigger collection to avoid duplicate API calls
- Repository context extraction from GitHub URLs
- Issue URL construction for dashboard linking

**Sentinel Integration:** The `Sentinel` class delegates polling coordination through:
- `poll_jira_triggers()` / `poll_github_triggers()` - Execute polling for each source
- `group_orchestrations_by_source()` - Separate orchestrations by trigger type
- `create_cycle_dedup_set()` / `check_and_mark_submitted()` - Deduplication
- `construct_issue_url()` - Build URLs for dashboard display

---

## Supporting Components

### Router

**Module:** `sentinel/router.py`

**Purpose:** Routes issues to matching orchestrations based on trigger configuration.

**Key Method:**
- `route_matched_only(issues)` - Returns routing results for issues that match any orchestration

### TagManager

**Module:** `sentinel/tag_manager.py`

**Purpose:** Manages Jira and GitHub label/tag operations for tracking execution state.

### AgentExecutor

**Module:** `sentinel/executor.py`

**Purpose:** Executes agent tasks (Claude, Cursor) for matched issues.

### AgentClientFactory

**Module:** `sentinel/agent_clients/factory.py`

**Purpose:** Creates agent clients per-orchestration based on configuration.

---

## Data Flow

### Polling Cycle

```
1. Sentinel.run_once()
   │
   ├── OrchestrationRegistry.detect_and_load_orchestration_changes()
   │   └── Hot-reload any new/modified orchestration files
   │
   ├── StateTracker.clear_issue_queue()
   │   └── Reset queue for new cycle
   │
   ├── ExecutionManager.get_available_slots()
   │   └── Check if we have capacity for new work
   │
   ├── PollCoordinator.group_orchestrations_by_source()
   │   └── Separate Jira vs GitHub orchestrations
   │
   ├── PollCoordinator.poll_jira_triggers()
   │   └── Fetch matching Jira issues
   │
   ├── PollCoordinator.poll_github_triggers()
   │   └── Fetch matching GitHub issues/PRs
   │
   └── Sentinel._submit_execution_tasks()
       ├── StateTracker.get_available_slots_for_orchestration()
       ├── ExecutionManager.submit()
       └── StateTracker.add_running_step()
```

### Execution Flow

```
1. ExecutionManager.submit(task)
   │
   ├── Task executes in thread pool
   │   └── Sentinel._execute_orchestration_task()
   │       ├── AgentExecutor.execute()
   │       └── TagManager.update_tags()
   │
   └── On completion:
       ├── ExecutionManager.collect_completed_results()
       ├── StateTracker.remove_running_step()
       └── StateTracker.decrement_per_orch_count()
```

---

## Thread Safety Design

### Lock Ordering

The `StateTracker` uses multiple locks with a strict ordering discipline:

1. `_attempt_counts_lock` (highest priority)
2. `_running_steps_lock`
3. `_queue_lock`
4. `_per_orch_counts_lock` (lowest priority)

When acquiring multiple locks, they must always be acquired in this order to prevent deadlocks. Currently, all methods acquire only a single lock at a time.

### Single-Threaded vs Multi-Threaded

- **Single-threaded:** Main polling loop, `PollCoordinator`
- **Multi-threaded:** Execution tasks in `ExecutionManager` thread pool

---

## Configuration Integration

The components receive configuration through the `Config` object:

| Component | Key Config Parameters |
|-----------|----------------------|
| StateTracker | `max_queue_size`, `attempt_counts_ttl` |
| ExecutionManager | `max_concurrent_executions` |
| OrchestrationRegistry | `orchestrations_dir` |
| PollCoordinator | `max_issues_per_poll`, `jira_base_url` |

---

## Error Handling and Recovery Patterns

This section documents how each component handles failures, how errors propagate between components, and the recovery patterns and retry strategies used throughout the system.

### Circuit Breaker Pattern

**Module:** `sentinel/circuit_breaker.py`

The circuit breaker pattern protects against cascading failures when external services (Jira, GitHub, Claude APIs) are experiencing issues.

**Circuit States:**
- **CLOSED:** Normal operation, requests pass through
- **OPEN:** Service is failing, requests fail immediately without calling the service
- **HALF_OPEN:** Testing recovery, limited requests pass through to test service health

**State Transitions:**
```
    ┌─────────────┐
    │   CLOSED    │◄────────────────────────────────────┐
    │  (Normal)   │                                     │
    └──────┬──────┘                                     │
           │                                            │
           │ Failures exceed threshold                  │ Success count reaches
           │ (default: 5 failures)                      │ half_open_max_calls
           ▼                                            │
    ┌─────────────┐     recovery_timeout      ┌────────┴────────┐
    │    OPEN     │ ─────(default: 30s)─────► │   HALF_OPEN     │
    │ (Fail-fast) │                           │ (Testing)       │
    └─────────────┘                           └────────┬────────┘
           ▲                                           │
           │                                           │
           └─────────Any failure in half-open──────────┘
```

**Configuration (Environment Variables):**
```
SENTINEL_CIRCUIT_BREAKER_ENABLED=true
SENTINEL_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
SENTINEL_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30
SENTINEL_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS=3
```

**Per-Service Overrides:**
```
SENTINEL_JIRA_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
SENTINEL_GITHUB_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
SENTINEL_CLAUDE_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
```

**Usage:**
```python
from sentinel.circuit_breaker import get_circuit_breaker

# Get or create a circuit breaker for a service
cb = get_circuit_breaker("jira")

# Check before making a call
if cb.allow_request():
    try:
        result = call_jira_api()
        cb.record_success()
    except Exception as e:
        cb.record_failure(e)
        raise

# Or use as a decorator
@get_circuit_breaker("github")
def call_github_api():
    ...
```

---

### Retry Strategies

#### REST Client Retry (Rate Limiting)

**Module:** `sentinel/rest_clients.py`

Implements exponential backoff with jitter for handling rate limits from Jira and GitHub APIs.

**Algorithm:**
1. On HTTP 429 (rate limit), calculate delay: `min(initial_delay * 2^attempt, max_delay) * jitter`
2. If `Retry-After` header is present, use that value as the base delay
3. Apply jitter (`0.7` to `1.3` multiplier) to prevent thundering herd
4. Retry up to `max_retries` times (default: 4)

**Configuration:**
```python
@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 4
    initial_delay: float = 1.0
    max_delay: float = 30.0
    jitter_min: float = 0.7
    jitter_max: float = 1.3
```

#### Agent Execution Retry

**Module:** `sentinel/executor.py`

The `AgentExecutor` retries failed agent executions based on orchestration configuration.

**Retry Logic:**
1. Execute agent with prompt
2. Match response against `success_patterns` and `failure_patterns`
3. If failure pattern matches and attempts < `max_attempts`, retry
4. On `AgentTimeoutError` or `AgentClientError`, retry if attempts remain

**Retry Configuration (per orchestration):**
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
  default_status: "failure"  # When no patterns match
```

---

### Error Propagation Between Components

#### Polling Errors

```
JiraRestClient/GitHubRestClient
         │
         │ JiraClientError / GitHubClientError
         ▼
   PollCoordinator
         │
         │ Logs error, continues with other triggers
         ▼
      Sentinel
         │
         │ Logs summary, proceeds to next polling cycle
         ▼
   (Next poll cycle)
```

**Behavior:** Polling errors are logged but don't stop the orchestrator. The circuit breaker may open if errors persist, causing subsequent polls to fail fast.

#### Execution Errors

```
AgentClient.run_agent()
         │
         │ AgentClientError / AgentTimeoutError
         ▼
   AgentExecutor.execute()
         │
         │ Retries if attempts remain
         │ Returns ExecutionResult with ERROR status
         ▼
   Sentinel._execute_orchestration_task()
         │
         │ Cleans up resources (StateTracker, etc.)
         ▼
   ExecutionManager.collect_completed_results()
         │
         │ Logs errors, removes completed futures
         ▼
   (Dashboard shows execution status)
```

**Behavior:** Agent execution errors trigger retries. After all retries are exhausted, the execution is marked as ERROR and the orchestrator continues processing other issues.

#### Thread Pool Errors

```
ExecutionManager.submit()
         │
         │ Task runs in thread pool
         ▼
   Future completes (success or exception)
         │
         │ collect_completed_results() catches exceptions
         ▼
   ExecutionResult or logged error
```

**Exception Handling in `collect_completed_results()`:**
- `OSError`, `TimeoutError`: Logged as I/O or timeout errors
- `RuntimeError`: Logged as runtime errors
- `KeyError`, `TypeError`, `ValueError`: Logged as data errors

---

### Component-Specific Error Handling

#### StateTracker

**Lock Ordering Discipline:** Prevents deadlocks when multiple locks are needed.
```
1. _attempt_counts_lock (highest priority)
2. _running_steps_lock
3. _queue_lock
4. _per_orch_counts_lock (lowest priority)
```

**Memory Safety:**
- `cleanup_stale_attempt_counts()`: Removes entries older than TTL (default: 24 hours)
- Issue queue uses `deque(maxlen=N)` for automatic FIFO eviction

#### ExecutionManager

**Future Cleanup:**
- Tracks futures with creation timestamps
- Stale futures (exceeding TTL, default: 5 minutes) are logged and cancelled
- Maximum futures limit prevents unbounded memory growth
- When limit is reached, removes completed futures first, then oldest stale futures

**Long-Running Future Monitoring:**
- Logs warnings for futures running longer than 60 seconds
- Warning deduplication prevents log flooding (5-minute interval between repeated warnings)

#### OrchestrationRegistry

**Hot-Reload Error Handling:**
- Invalid orchestration files are logged and skipped (don't crash the system)
- Modified files trigger version replacement with pending removal tracking
- Old versions are kept until their active executions complete

---

### Graceful Shutdown

**Module:** `sentinel/shutdown.py`

**Signal Handling:**
- `SIGINT` (Ctrl+C): Triggers graceful shutdown
- `SIGTERM`: Triggers graceful shutdown

**Shutdown Sequence:**
1. Signal handler sets `shutdown_requested` flag
2. Main polling loop exits on next iteration check
3. `ExecutionManager.shutdown()` waits for running tasks (or cancels them)
4. Claude processes receive termination signal via `request_claude_shutdown()`

**Usage:**
```python
handler = create_shutdown_handler(on_shutdown=sentinel.request_shutdown)
# Signal handlers now installed for SIGINT and SIGTERM
```

---

### Recovery Patterns Summary

| Pattern | Use Case | Recovery Action |
|---------|----------|-----------------|
| Circuit Breaker | External service failures | Fail fast, auto-recover after timeout |
| Exponential Backoff | Rate limiting | Retry with increasing delays |
| Agent Retry | Agent execution failures | Retry up to max_attempts |
| TTL-based Cleanup | Memory leaks from stale data | Automatic removal after TTL |
| FIFO Queue Eviction | Queue overflow | Evict oldest items automatically |
| Graceful Shutdown | Process termination | Complete running work, then exit |
| Lock Ordering | Deadlock prevention | Consistent lock acquisition order |

---

## Related Documentation

- [Dependency Injection](dependency-injection.md) - Container and DI patterns
- [Testing](TESTING.md) - Testing patterns and mocking strategies
- [Failure Patterns](FAILURE_PATTERNS.md) - Error handling conventions
