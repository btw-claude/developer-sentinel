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

## Related Documentation

- [Dependency Injection](dependency-injection.md) - Container and DI patterns
- [Testing](TESTING.md) - Testing patterns and mocking strategies
- [Failure Patterns](FAILURE_PATTERNS.md) - Error handling conventions
