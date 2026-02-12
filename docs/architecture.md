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

The components receive configuration through the `Config` object and its sub-configs:

| Component              | Config Section                           | Key Config Parameters                                                            |
|------------------------|------------------------------------------|----------------------------------------------------------------------------------|
| StateTracker           | `ExecutionConfig`†                       | `max_queue_size`, `attempt_counts_ttl`, `max_recent_executions`                  |
| ExecutionManager       | `ExecutionConfig`†                       | `max_concurrent_executions`                                                      |
| OrchestrationRegistry  | `ExecutionConfig`†                       | `orchestrations_dir`                                                             |
| PollCoordinator        | `PollingConfig`†, `JiraConfig`†          | `max_issues_per_poll`, `base_url`                                                |
| Sentinel               | `ExecutionConfig`†, `PollingConfig`†     | `polling.interval`, delegates sub-configs to components                          |
| DashboardServer        | `DashboardConfig`†                       | `host`, `port`, `enabled`                                                        |
| DashboardRoutes        | `DashboardConfig`†                       | `toggle_cooldown_seconds`, `rate_limit_cache_ttl`, `rate_limit_cache_maxsize`    |
| JiraRestClient         | `JiraConfig`†, `CircuitBreakerConfig`†   | `base_url`, `email`, `api_token`, `epic_link_field`                              |
| GitHubRestClient       | `GitHubConfig`†, `CircuitBreakerConfig`† | `token`, `api_url`                                                               |
| CircuitBreaker         | `CircuitBreakerConfig`†                  | `enabled`, `failure_threshold`, `recovery_timeout`, `half_open_max_calls`        |
| ClaudeRateLimiter      | `RateLimitConfig`†                       | `enabled`, `per_minute`, `per_hour`, `strategy`, `warning_threshold`             |
| ResilienceWrapper      | `RateLimitConfig`†                       | Coordinates `CircuitBreaker` + `ClaudeRateLimiter`                               |
| HealthChecker          | `HealthCheckConfig`†                     | `enabled`, `timeout`                                                             |
| AgentClientFactory     | `CursorConfig`†                          | `default_agent_type`                                                             |
| CursorAgentClient      | `CursorConfig`†                          | `path`, `default_model`, `default_mode`                                          |
| ClaudeSdkAgentClient   | `ExecutionConfig`†                       | `inter_message_times_threshold`                                                  |
| setup_logging          | `LoggingConfig`†                         | `level`, `json`                                                                  |

> **†** All config sections are sub-configs embedded as fields in the main `Config` dataclass via `field(default_factory=...)`. Each sub-config class (`ExecutionConfig`, `PollingConfig`, `JiraConfig`, `DashboardConfig`, `CircuitBreakerConfig`, `GitHubConfig`, `RateLimitConfig`, `HealthCheckConfig`, `CursorConfig`, `LoggingConfig`) is instantiated through this pattern, providing default values when not explicitly configured. See `sentinel/config.py` for the full composition hierarchy.

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

### Resilience Wrapper: Circuit Breaker and Rate Limiter Coordination

**Module:** `sentinel/resilience.py`

The `ResilienceWrapper` coordinates the `CircuitBreaker` and `ClaudeRateLimiter` to avoid wasting rate limit tokens when the circuit breaker is open.

**Problem Solved:**

Without coordination, the following wasteful scenario can occur:
1. Request comes in
2. Rate limiter consumes a token
3. Circuit breaker rejects the request immediately
4. Token is wasted

**Solution:**

The `ResilienceWrapper` checks the circuit breaker state BEFORE acquiring a rate limit token:

```
    ┌─────────────────────┐
    │  Incoming Request   │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐     NO      ┌─────────────────────┐
    │  Circuit Breaker    │────────────►│  Reject Request     │
    │  allows_request()?  │             │  (NO token consumed)│
    └──────────┬──────────┘             └─────────────────────┘
               │ YES
               ▼
    ┌─────────────────────┐     NO      ┌─────────────────────┐
    │  Rate Limiter       │────────────►│  Reject Request     │
    │  acquire()?         │             │  (token consumed)   │
    └──────────┬──────────┘             └─────────────────────┘
               │ YES
               ▼
    ┌─────────────────────┐
    │  Execute Request    │
    └─────────────────────┘
```

**Usage:**

```python
from sentinel.circuit_breaker import get_circuit_breaker
from sentinel.rate_limiter import ClaudeRateLimiter
from sentinel.resilience import ResilienceWrapper

# Create wrapper with existing components
wrapper = ResilienceWrapper(
    circuit_breaker=get_circuit_breaker("claude"),
    rate_limiter=ClaudeRateLimiter.from_config(config),
)

# Acquire permission before making API call
if wrapper.acquire(timeout=30.0):
    try:
        result = await call_claude_api()
        wrapper.record_success()
    except Exception as e:
        wrapper.record_failure(e)
        raise

# Or use as a context manager
with wrapper:
    result = call_claude_api()
```

**Metrics:**

The wrapper tracks coordination-specific metrics:
- `circuit_breaker_rejections`: Requests rejected by circuit breaker (tokens saved)
- `rate_limit_acquired`: Successful rate limit token acquisitions
- `rate_limit_rejections`: Requests rejected by rate limiter
- `tokens_saved`: Number of tokens that would have been wasted (same as `circuit_breaker_rejections`)

```python
metrics = wrapper.get_metrics()
print(f"Tokens saved: {metrics['wrapper_metrics']['tokens_saved']}")
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

### Metrics and Observability

This section describes how to monitor the health and performance of error handling mechanisms in production.

#### Circuit Breaker Metrics

**Monitoring Circuit Breaker States:**

Each circuit breaker instance exposes the following observable properties:

| Metric | Description | How to Access |
|--------|-------------|---------------|
| `state` | Current state (CLOSED, OPEN, HALF_OPEN) | `cb.state` |
| `failure_count` | Number of consecutive failures | `cb.failure_count` |
| `success_count` | Successes in half-open state | `cb.success_count` |
| `last_failure_time` | Timestamp of most recent failure | `cb.last_failure_time` |
| `last_state_change` | Timestamp of last state transition | `cb.last_state_change` |

**Logging Integration:**

Circuit breaker state changes are logged at INFO level:
```
INFO  circuit_breaker.jira: State changed from CLOSED to OPEN (failures: 5)
INFO  circuit_breaker.jira: State changed from OPEN to HALF_OPEN (recovery timeout elapsed)
INFO  circuit_breaker.jira: State changed from HALF_OPEN to CLOSED (recovery successful)
```

**Prometheus-Style Metrics (Optional):**

For production deployments, expose metrics via a `/metrics` endpoint. For more information on setting up Prometheus and AlertManager, see:
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [AlertManager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
```python
# Example metrics to track
sentinel_circuit_breaker_state{service="jira"} 0  # 0=CLOSED, 1=OPEN, 2=HALF_OPEN
sentinel_circuit_breaker_failures_total{service="jira"} 15
sentinel_circuit_breaker_state_transitions_total{service="jira", from="closed", to="open"} 3
```

#### Retry Metrics

**REST Client Retry Metrics:**

| Metric | Description | Log Pattern |
|--------|-------------|-------------|
| Retry attempts | Number of retries per request | `WARNING rest_client: Retry attempt 2/4 for GET /rest/api/3/issue/...` |
| Rate limit hits | 429 responses received | `WARNING rest_client: Rate limited, waiting 5.2s before retry` |
| Backoff delays | Actual delay times applied | Included in retry log messages |

**Agent Execution Retry Metrics:**

| Metric | Description | Log Pattern |
|--------|-------------|-------------|
| Execution start | Per-attempt start with orchestration context | `INFO executor: Executing agent for PROJ-123 with orchestration 'code-review' (attempt 2/3)` |
| Execution result | Per-attempt completion with status and response summary (truncated to 200 chars) | `INFO executor: Agent execution SUCCESS for PROJ-123 (attempt 2/3): <response_summary>` |
| Retry on failure | Failure pattern matched, retrying next attempt | `WARNING executor: Agent execution failed for PROJ-123 on attempt 2, retrying...` |
| Timeout retry | Agent timed out, retrying next attempt. Structured log includes `timeout_seconds` value | `WARNING executor: Agent timed out for PROJ-123 on attempt 2, retrying...` |
| Agent log file footer | Per-attempt log file written by AgentLogger | `Attempt:        2` (singular, inside `EXECUTION SUMMARY` / `AGENT EXECUTION LOG` blocks) |

#### Failure Rate Tracking

**Calculating Failure Rates:**

Track failure rates over time windows for trending and alerting:

```python
# Example: Track failures per 5-minute window
from collections import deque
from threading import Lock
from time import time

class FailureRateTracker:
    """
    Tracks failure rates over a sliding time window.

    Thread Safety Note: This implementation is NOT thread-safe as written.
    While `deque` operations like `append()` and `popleft()` are individually
    atomic in CPython due to the GIL, compound operations (read-then-write)
    are not. For production multi-threaded usage, add a threading.Lock:

        def __init__(self, window_seconds: int = 300):
            self._lock = Lock()
            # ... rest of init

        def record_failure(self) -> None:
            with self._lock:
                self._cleanup()
                self.failures.append(time())

    Alternatively, consider using `queue.Queue` for thread-safe operations
    or restrict usage to a single thread.
    """
    def __init__(self, window_seconds: int = 300):
        self.window = window_seconds
        self.failures: deque[float] = deque()
        self.successes: deque[float] = deque()

    def record_failure(self) -> None:
        self._cleanup()
        self.failures.append(time())

    def record_success(self) -> None:
        self._cleanup()
        self.successes.append(time())

    def get_failure_rate(self) -> float:
        self._cleanup()
        total = len(self.failures) + len(self.successes)
        return len(self.failures) / total if total > 0 else 0.0

    def _cleanup(self) -> None:
        cutoff = time() - self.window
        while self.failures and self.failures[0] < cutoff:
            self.failures.popleft()
        while self.successes and self.successes[0] < cutoff:
            self.successes.popleft()
```

**Key Failure Rate Metrics:**

| Metric | Scope | Calculation |
|--------|-------|-------------|
| Service failure rate | Per external service (Jira, GitHub, Claude) | failures / (failures + successes) per window |
| Orchestration failure rate | Per orchestration | failed executions / total executions |
| Polling failure rate | Per polling cycle | failed polls / total poll attempts |

#### Dashboard Integration

The Sentinel dashboard displays real-time execution status. Key observability data points:

| Dashboard Element | Data Source | Update Frequency |
|-------------------|-------------|------------------|
| Running executions | `StateTracker.get_running_steps()` | Real-time |
| Queued issues | `StateTracker.get_issue_queue()` | Per poll cycle |
| Per-orchestration counts | `StateTracker.get_all_per_orch_counts()` | Real-time |
| Hot-reload events | `OrchestrationRegistry` metrics | On file change |

---

### Testing Strategies

This section describes how to test error handling scenarios in unit and integration tests.

#### Unit Testing Circuit Breakers

**Testing State Transitions:**

```python
import pytest
from unittest.mock import patch
from sentinel.circuit_breaker import CircuitBreaker, CircuitState

class TestCircuitBreaker:
    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        for _ in range(3):
            cb.record_failure(Exception("test"))

        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
        cb.record_failure(Exception("test"))

        assert cb.state == CircuitState.OPEN

        # Fast-forward time
        with patch('time.time', return_value=time.time() + 2.0):
            assert cb.allow_request()
            assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successful_half_open_calls(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0,
            half_open_max_calls=2
        )
        cb.record_failure(Exception("test"))

        # Transition to half-open
        cb.allow_request()

        # Record successes
        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED
```

**Mocking External Services:**

```python
from unittest.mock import Mock, patch

def test_circuit_breaker_protects_jira_calls():
    mock_client = Mock()
    mock_client.get_issue.side_effect = ConnectionError("Service unavailable")

    with patch('sentinel.rest_clients.JiraRestClient', return_value=mock_client):
        # After failures exceed threshold, circuit opens
        # Subsequent calls should fail fast without calling the service
        ...
```

#### Unit Testing Retry Logic

**Testing Exponential Backoff:**

```python
from sentinel.rest_clients import calculate_backoff_delay

def test_exponential_backoff_calculation():
    config = RetryConfig(initial_delay=1.0, max_delay=30.0)

    # Attempt 0: 1.0s base
    # Attempt 1: 2.0s base
    # Attempt 2: 4.0s base
    # Attempt 3: 8.0s base (capped at max_delay)

    delay_0 = calculate_backoff_delay(attempt=0, config=config)
    delay_1 = calculate_backoff_delay(attempt=1, config=config)

    assert 0.7 <= delay_0 <= 1.3  # With jitter
    assert 1.4 <= delay_1 <= 2.6  # With jitter

def test_respects_retry_after_header():
    response = Mock()
    response.headers = {'Retry-After': '10'}

    delay = calculate_backoff_delay(attempt=0, config=config, response=response)

    assert 7.0 <= delay <= 13.0  # 10s base with jitter
```

**Testing Agent Retry Behavior:**

```python
from sentinel.executor import AgentExecutor

def test_retries_on_failure_pattern_match():
    executor = AgentExecutor(
        retry_config={
            'max_attempts': 3,
            'failure_patterns': ['ERROR:', 'FAILED']
        }
    )

    mock_client = Mock()
    mock_client.run_agent.side_effect = [
        "ERROR: Connection timeout",
        "ERROR: Rate limited",
        "SUCCESS: Task completed"
    ]

    result = executor.execute(mock_client, "test prompt")

    assert mock_client.run_agent.call_count == 3
    assert result.status == ExecutionStatus.SUCCESS
```

#### Integration Testing Error Scenarios

**Testing with Simulated Failures:**

```python
import pytest
from unittest.mock import patch

@pytest.mark.integration
class TestErrorRecovery:
    def test_recovers_from_temporary_jira_outage(self, sentinel_instance):
        """Verify Sentinel continues after Jira becomes unavailable then recovers."""

        call_count = 0
        def flaky_jira_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Jira unavailable")
            return {"issues": []}

        with patch.object(
            sentinel_instance.jira_client,
            'search_issues',
            side_effect=flaky_jira_call
        ):
            # Run multiple polling cycles
            for _ in range(5):
                sentinel_instance.run_once()

            # Verify circuit breaker opened then closed
            cb = get_circuit_breaker("jira")
            assert cb.state == CircuitState.CLOSED

    def test_graceful_degradation_with_one_service_down(self, sentinel_instance):
        """Verify GitHub polling continues when Jira is down."""

        with patch.object(
            sentinel_instance.jira_client,
            'search_issues',
            side_effect=ConnectionError("Jira down")
        ):
            # GitHub polling should still work
            result = sentinel_instance.run_once()

            assert result.github_polls_completed > 0
            assert result.jira_polls_completed == 0
```

**Testing Thread Pool Error Handling:**

```python
def test_handles_executor_thread_failure():
    """Verify ExecutionManager handles thread exceptions gracefully."""

    manager = ExecutionManager(max_concurrent=2)
    manager.start()

    def failing_task():
        raise RuntimeError("Simulated thread failure")

    future = manager.submit(failing_task)
    results = manager.collect_completed_results()

    # Error should be captured, not propagate
    assert len(results) == 1
    assert results[0].status == ExecutionStatus.ERROR

    manager.shutdown()
```

#### Test Fixtures and Utilities

**Circuit Breaker Test Helper:**

```python
@pytest.fixture
def reset_circuit_breakers():
    """Reset all circuit breakers before each test."""
    from sentinel.circuit_breaker import _circuit_breakers
    _circuit_breakers.clear()
    yield
    _circuit_breakers.clear()

@pytest.fixture
def fast_circuit_breaker():
    """Circuit breaker with short timeouts for testing."""
    return CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=0.1,
        half_open_max_calls=1
    )
```

**Mock Service Responses:**

```python
@pytest.fixture
def mock_jira_rate_limit():
    """Simulate Jira rate limiting response."""
    response = Mock()
    response.status_code = 429
    response.headers = {'Retry-After': '1'}
    return response

@pytest.fixture
def mock_github_server_error():
    """Simulate GitHub 500 error."""
    response = Mock()
    response.status_code = 500
    response.json.return_value = {'message': 'Internal Server Error'}
    return response
```

---

### Alerting Recommendations

This section provides recommended thresholds and alerting strategies for error handling patterns.

#### Circuit Breaker Alerts

| Alert | Condition | Severity | Response |
|-------|-----------|----------|----------|
| Circuit Open | Any circuit breaker enters OPEN state | Warning | Investigate external service health |
| Prolonged Open | Circuit remains OPEN > 5 minutes | Critical | Check service status, consider manual intervention |
| Flapping | Circuit transitions > 3 times in 10 minutes | Warning | Service instability, review failure patterns |
| All Circuits Open | All external service circuits are OPEN | Critical | Major incident, likely network or infrastructure issue |

**Alerting Rules (Prometheus/AlertManager Style):**

For detailed configuration syntax and options, see [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/) and [AlertManager Configuration](https://prometheus.io/docs/alerting/latest/configuration/).

```yaml
groups:
  - name: sentinel_circuit_breaker
    rules:
      - alert: CircuitBreakerOpen
        expr: sentinel_circuit_breaker_state == 1
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "Circuit breaker open for {{ $labels.service }}"
          description: "The circuit breaker for {{ $labels.service }} has been open for more than 30 seconds."

      - alert: CircuitBreakerProlongedOpen
        expr: sentinel_circuit_breaker_state == 1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker stuck open for {{ $labels.service }}"
          description: "The circuit breaker for {{ $labels.service }} has been open for more than 5 minutes. Manual investigation required."
```

#### Retry Threshold Alerts

| Alert | Condition | Severity | Response |
|-------|-----------|----------|----------|
| High Retry Rate | > 20% of requests require retries | Warning | Check for rate limiting or service degradation |
| Max Retries Exhausted | > 5 requests exhaust all retries in 5 min | Warning | Service likely degraded, check circuit breaker |
| Consistent Rate Limiting | > 10 rate limit (429) responses in 5 min | Warning | Review request patterns, consider backoff tuning |

**Recommended Thresholds:**

```yaml
# Retry-related alerts
- alert: HighRetryRate
  expr: |
    rate(sentinel_retry_attempts_total[5m]) /
    rate(sentinel_requests_total[5m]) > 0.2
  for: 5m
  labels:
    severity: warning

- alert: RetriesExhausted
  expr: increase(sentinel_retries_exhausted_total[5m]) > 5
  for: 0m
  labels:
    severity: warning
```

#### Failure Rate Alerts

| Alert | Condition | Severity | Response |
|-------|-----------|----------|----------|
| Elevated Failure Rate | Service failure rate > 10% over 5 min | Warning | Monitor closely, may self-recover |
| High Failure Rate | Service failure rate > 25% over 5 min | Critical | Active incident, investigate immediately |
| Execution Failure Spike | > 3 orchestration failures in 10 min | Warning | Check agent health, review failure patterns |
| Polling Failures | > 50% of polling cycles fail | Critical | External services likely unavailable |

**Recommended Thresholds:**

```yaml
# Failure rate alerts
- alert: ElevatedServiceFailureRate
  expr: sentinel_service_failure_rate > 0.10
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Elevated failure rate for {{ $labels.service }}"

- alert: HighServiceFailureRate
  expr: sentinel_service_failure_rate > 0.25
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High failure rate for {{ $labels.service }} - immediate attention required"

- alert: OrchestrationFailureSpike
  expr: increase(sentinel_orchestration_failures_total[10m]) > 3
  for: 0m
  labels:
    severity: warning
```

#### Thread Pool and Resource Alerts

| Alert | Condition | Severity | Response |
|-------|-----------|----------|----------|
| Thread Pool Saturation | Active futures > 80% of max_futures | Warning | Executions backing up, may need to scale |
| Stale Futures | > 5 futures exceed TTL in 10 min | Warning | Executions hanging, check agent health |
| Queue Overflow | Issue queue at max capacity | Warning | Processing falling behind, check concurrency |
| Memory Growth | StateTracker entries > 10000 | Warning | Check TTL cleanup, possible memory leak |

**Recommended Thresholds:**

```yaml
# Resource alerts
- alert: ThreadPoolSaturation
  expr: sentinel_active_futures / sentinel_max_futures > 0.8
  for: 5m
  labels:
    severity: warning

- alert: StaleFutures
  expr: increase(sentinel_stale_futures_total[10m]) > 5
  for: 0m
  labels:
    severity: warning

- alert: IssueQueueOverflow
  expr: sentinel_issue_queue_size >= sentinel_max_queue_size
  for: 1m
  labels:
    severity: warning
```

#### Alert Response Playbook

**Circuit Breaker Open:**
1. Check external service status (Jira, GitHub, Claude API status pages)
2. Review recent error logs for specific failure messages
3. Verify network connectivity from Sentinel host
4. If service is healthy, check for configuration issues (credentials, URLs)
5. Consider temporary increase of `failure_threshold` if experiencing transient issues

**High Retry/Failure Rate:**
1. Check for rate limiting (look for 429 responses in logs)
2. Review request patterns for potential optimizations
3. Verify external service SLAs and current performance
4. Consider reducing polling frequency or batch sizes temporarily
5. Check for recent changes that may have introduced bugs

**Resource Alerts:**
1. Check `max_concurrent_executions` vs current workload
2. Review orchestration execution times for anomalies
3. Look for hung or zombie processes
4. Consider scaling horizontally if sustained high load
5. Verify TTL-based cleanup is running (check cleanup logs)

---

## Related Documentation

- [Dependency Injection](dependency-injection.md) - Container and DI patterns
- [Testing](TESTING.md) - Testing patterns and mocking strategies
- [Failure Patterns](FAILURE_PATTERNS.md) - Error handling conventions
- [UTC Timestamps](UTC_TIMESTAMPS.md) - Timezone-aware UTC convention for all timestamps
