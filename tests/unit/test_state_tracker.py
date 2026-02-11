"""Tests for StateTracker class in isolation.

This module tests the StateTracker class directly, without going through Sentinel.
It provides comprehensive coverage of:
- Queue operations (add, get, clear, maxlen eviction)
- Attempt count management (increment, TTL-based cleanup)
- Completed execution tracking (ring buffer with appendleft ordering)
- Running steps (add, remove, get with Future filtering)
- Per-orchestration counts (increment, decrement with zero-floor, get)
- Thread safety (concurrent operations)
- Poll times (get/set for Jira/GitHub, start_time)
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

from sentinel.state_tracker import AttemptCountEntry, CompletedExecutionInfo, StateTracker
from tests.helpers import make_orchestration


class TestQueueOperations:
    def test_add_to_issue_queue_adds_entry(self) -> None:
        tracker = StateTracker()
        tracker.add_to_issue_queue("TEST-1", "my-orch")

        queue = tracker.get_issue_queue()
        assert len(queue) == 1
        assert queue[0].issue_key == "TEST-1"
        assert queue[0].orchestration_name == "my-orch"
        assert isinstance(queue[0].queued_at, datetime)

    def test_get_issue_queue_returns_list(self) -> None:
        tracker = StateTracker()
        tracker.add_to_issue_queue("TEST-1", "orch-a")
        tracker.add_to_issue_queue("TEST-2", "orch-b")

        queue = tracker.get_issue_queue()
        assert len(queue) == 2
        assert queue[0].issue_key == "TEST-1"
        assert queue[1].issue_key == "TEST-2"

    def test_clear_issue_queue_empties_queue(self) -> None:
        tracker = StateTracker()
        tracker.add_to_issue_queue("TEST-1", "orch-a")
        tracker.add_to_issue_queue("TEST-2", "orch-b")

        tracker.clear_issue_queue()

        queue = tracker.get_issue_queue()
        assert len(queue) == 0

    def test_queue_maxlen_eviction_when_full(self) -> None:
        tracker = StateTracker(max_queue_size=3)
        tracker.add_to_issue_queue("TEST-1", "orch-a")
        tracker.add_to_issue_queue("TEST-2", "orch-b")
        tracker.add_to_issue_queue("TEST-3", "orch-c")

        queue = tracker.get_issue_queue()
        assert len(queue) == 3

        tracker.add_to_issue_queue("TEST-4", "orch-d")

        queue = tracker.get_issue_queue()
        assert len(queue) == 3
        assert queue[0].issue_key == "TEST-2"
        assert queue[1].issue_key == "TEST-3"
        assert queue[2].issue_key == "TEST-4"

    def test_queue_eviction_order_fifo(self) -> None:
        tracker = StateTracker(max_queue_size=2)
        tracker.add_to_issue_queue("FIRST", "orch")
        tracker.add_to_issue_queue("SECOND", "orch")
        tracker.add_to_issue_queue("THIRD", "orch")

        queue = tracker.get_issue_queue()
        assert len(queue) == 2
        assert queue[0].issue_key == "SECOND"
        assert queue[1].issue_key == "THIRD"


class TestAttemptCountManagement:
    def test_get_and_increment_first_call_returns_one(self) -> None:
        tracker = StateTracker()
        count = tracker.get_and_increment_attempt_count("TEST-1", "my-orch")
        assert count == 1

    def test_get_and_increment_subsequent_calls_increment(self) -> None:
        tracker = StateTracker()
        count1 = tracker.get_and_increment_attempt_count("TEST-1", "my-orch")
        count2 = tracker.get_and_increment_attempt_count("TEST-1", "my-orch")
        count3 = tracker.get_and_increment_attempt_count("TEST-1", "my-orch")

        assert count1 == 1
        assert count2 == 2
        assert count3 == 3

    def test_get_and_increment_separate_by_issue_key(self) -> None:
        tracker = StateTracker()
        tracker.get_and_increment_attempt_count("TEST-1", "my-orch")
        tracker.get_and_increment_attempt_count("TEST-1", "my-orch")

        count = tracker.get_and_increment_attempt_count("TEST-2", "my-orch")
        assert count == 1

    def test_get_and_increment_separate_by_orchestration(self) -> None:
        tracker = StateTracker()
        tracker.get_and_increment_attempt_count("TEST-1", "orch-a")
        tracker.get_and_increment_attempt_count("TEST-1", "orch-a")

        count = tracker.get_and_increment_attempt_count("TEST-1", "orch-b")
        assert count == 1

    def test_get_and_increment_updates_last_access(self) -> None:
        tracker = StateTracker()
        time_before = time.monotonic()
        tracker.get_and_increment_attempt_count("TEST-1", "my-orch")
        time_after = time.monotonic()

        key = ("TEST-1", "my-orch")
        with tracker._attempt_counts_lock:
            entry = tracker._attempt_counts[key]

        assert entry.count == 1
        assert entry.last_access >= time_before
        assert entry.last_access <= time_after

    def test_cleanup_stale_attempt_counts_removes_old_entries(self) -> None:
        tracker = StateTracker(attempt_counts_ttl=1.0)

        current_time = time.monotonic()
        with tracker._attempt_counts_lock:
            tracker._attempt_counts[("OLD-1", "orch")] = AttemptCountEntry(
                count=5, last_access=current_time - 100
            )
            tracker._attempt_counts[("RECENT-1", "orch")] = AttemptCountEntry(
                count=3, last_access=current_time - 0.5
            )

        assert len(tracker._attempt_counts) == 2

        cleaned = tracker.cleanup_stale_attempt_counts()

        assert cleaned == 1
        assert len(tracker._attempt_counts) == 1
        assert ("OLD-1", "orch") not in tracker._attempt_counts
        assert ("RECENT-1", "orch") in tracker._attempt_counts

    def test_cleanup_stale_attempt_counts_returns_zero_when_none_stale(self) -> None:
        tracker = StateTracker(attempt_counts_ttl=3600.0)
        tracker.get_and_increment_attempt_count("TEST-1", "my-orch")

        cleaned = tracker.cleanup_stale_attempt_counts()
        assert cleaned == 0
        assert len(tracker._attempt_counts) == 1

    def test_cleanup_stale_attempt_counts_with_empty_dict(self) -> None:
        tracker = StateTracker()
        cleaned = tracker.cleanup_stale_attempt_counts()
        assert cleaned == 0


class TestCompletedExecutionTracking:
    def test_add_completed_execution_adds_entry(self) -> None:
        tracker = StateTracker()
        info = CompletedExecutionInfo(
            issue_key="TEST-1",
            orchestration_name="my-orch",
            attempt_number=1,
            started_at=datetime.now(tz=UTC),
            completed_at=datetime.now(tz=UTC),
            status="success",
            input_tokens=100,
            output_tokens=50,
            total_cost_usd=0.01,
            issue_url="https://example.com/TEST-1",
        )

        tracker.add_completed_execution(info)
        executions = tracker.get_completed_executions()

        assert len(executions) == 1
        assert executions[0] == info

    def test_add_completed_execution_uses_appendleft_ordering(self) -> None:
        tracker = StateTracker()
        info1 = CompletedExecutionInfo(
            issue_key="TEST-1",
            orchestration_name="orch",
            attempt_number=1,
            started_at=datetime.now(tz=UTC),
            completed_at=datetime.now(tz=UTC),
            status="success",
            input_tokens=100,
            output_tokens=50,
            total_cost_usd=0.01,
            issue_url="https://example.com/TEST-1",
        )
        info2 = CompletedExecutionInfo(
            issue_key="TEST-2",
            orchestration_name="orch",
            attempt_number=1,
            started_at=datetime.now(tz=UTC),
            completed_at=datetime.now(tz=UTC),
            status="failure",
            input_tokens=200,
            output_tokens=100,
            total_cost_usd=0.02,
            issue_url="https://example.com/TEST-2",
        )

        tracker.add_completed_execution(info1)
        tracker.add_completed_execution(info2)

        executions = tracker.get_completed_executions()
        assert len(executions) == 2
        assert executions[0] == info2
        assert executions[1] == info1

    def test_completed_execution_maxlen_eviction(self) -> None:
        tracker = StateTracker(max_completed_executions=3)

        for i in range(1, 5):
            info = CompletedExecutionInfo(
                issue_key=f"TEST-{i}",
                orchestration_name="orch",
                attempt_number=1,
                started_at=datetime.now(tz=UTC),
                completed_at=datetime.now(tz=UTC),
                status="success",
                input_tokens=100,
                output_tokens=50,
                total_cost_usd=0.01,
                issue_url=f"https://example.com/TEST-{i}",
            )
            tracker.add_completed_execution(info)

        executions = tracker.get_completed_executions()
        assert len(executions) == 3
        assert executions[0].issue_key == "TEST-4"
        assert executions[1].issue_key == "TEST-3"
        assert executions[2].issue_key == "TEST-2"

    def test_get_completed_executions_returns_list(self) -> None:
        tracker = StateTracker()
        executions = tracker.get_completed_executions()
        assert isinstance(executions, list)
        assert len(executions) == 0


class TestRunningSteps:
    def test_add_running_step_stores_info(self) -> None:
        tracker = StateTracker()
        future_id = 12345

        tracker.add_running_step(
            future_id=future_id,
            issue_key="TEST-1",
            orchestration_name="my-orch",
            attempt_number=2,
            issue_url="https://example.com/TEST-1",
        )

        with tracker._running_steps_lock:
            assert future_id in tracker._running_steps
            info = tracker._running_steps[future_id]
            assert info.issue_key == "TEST-1"
            assert info.orchestration_name == "my-orch"
            assert info.attempt_number == 2
            assert info.issue_url == "https://example.com/TEST-1"
            assert isinstance(info.started_at, datetime)

    def test_remove_running_step_returns_info(self) -> None:
        tracker = StateTracker()
        future_id = 12345

        tracker.add_running_step(
            future_id=future_id,
            issue_key="TEST-1",
            orchestration_name="my-orch",
            attempt_number=1,
            issue_url="https://example.com/TEST-1",
        )

        info = tracker.remove_running_step(future_id)
        assert info is not None
        assert info.issue_key == "TEST-1"

        with tracker._running_steps_lock:
            assert future_id not in tracker._running_steps

    def test_remove_running_step_returns_none_when_not_found(self) -> None:
        tracker = StateTracker()
        info = tracker.remove_running_step(99999)
        assert info is None

    def test_get_running_steps_filters_by_active_futures(self) -> None:
        tracker = StateTracker()

        future1 = MagicMock()
        future1.done.return_value = False
        future_id1 = id(future1)

        future2 = MagicMock()
        future2.done.return_value = True
        future_id2 = id(future2)

        future3 = MagicMock()
        future3.done.return_value = False
        future_id3 = id(future3)

        tracker.add_running_step(
            future_id=future_id1,
            issue_key="TEST-1",
            orchestration_name="orch",
            attempt_number=1,
            issue_url="https://example.com/TEST-1",
        )
        tracker.add_running_step(
            future_id=future_id2,
            issue_key="TEST-2",
            orchestration_name="orch",
            attempt_number=1,
            issue_url="https://example.com/TEST-2",
        )
        tracker.add_running_step(
            future_id=future_id3,
            issue_key="TEST-3",
            orchestration_name="orch",
            attempt_number=1,
            issue_url="https://example.com/TEST-3",
        )

        running = tracker.get_running_steps([future1, future2, future3])

        assert len(running) == 2
        issue_keys = {info.issue_key for info in running}
        assert issue_keys == {"TEST-1", "TEST-3"}

    def test_get_running_steps_with_empty_futures_list(self) -> None:
        tracker = StateTracker()
        running = tracker.get_running_steps([])
        assert len(running) == 0

    def test_get_running_steps_ignores_unknown_futures(self) -> None:
        tracker = StateTracker()

        future1 = MagicMock()
        future1.done.return_value = False

        running = tracker.get_running_steps([future1])
        assert len(running) == 0


class TestPerOrchestrationCounts:
    def test_increment_per_orch_count_starts_at_one(self) -> None:
        tracker = StateTracker()
        count = tracker.increment_per_orch_count("my-orch")
        assert count == 1

    def test_increment_per_orch_count_increments(self) -> None:
        tracker = StateTracker()
        count1 = tracker.increment_per_orch_count("my-orch")
        count2 = tracker.increment_per_orch_count("my-orch")
        count3 = tracker.increment_per_orch_count("my-orch")

        assert count1 == 1
        assert count2 == 2
        assert count3 == 3

    def test_decrement_per_orch_count_decrements(self) -> None:
        tracker = StateTracker()
        tracker.increment_per_orch_count("my-orch")
        tracker.increment_per_orch_count("my-orch")
        tracker.increment_per_orch_count("my-orch")

        count = tracker.decrement_per_orch_count("my-orch")
        assert count == 2

    def test_decrement_per_orch_count_zero_floor(self) -> None:
        tracker = StateTracker()
        count = tracker.decrement_per_orch_count("my-orch")
        assert count == 0

    def test_decrement_per_orch_count_removes_entry_at_zero(self) -> None:
        tracker = StateTracker()
        tracker.increment_per_orch_count("my-orch")

        tracker.decrement_per_orch_count("my-orch")

        with tracker._per_orch_counts_lock:
            assert "my-orch" not in tracker._per_orch_active_counts

    def test_get_per_orch_count_returns_current_count(self) -> None:
        tracker = StateTracker()
        tracker.increment_per_orch_count("my-orch")
        tracker.increment_per_orch_count("my-orch")

        count = tracker.get_per_orch_count("my-orch")
        assert count == 2

    def test_get_per_orch_count_returns_zero_for_unknown(self) -> None:
        tracker = StateTracker()
        count = tracker.get_per_orch_count("unknown-orch")
        assert count == 0

    def test_get_all_per_orch_counts_returns_dict(self) -> None:
        tracker = StateTracker()
        tracker.increment_per_orch_count("orch-a")
        tracker.increment_per_orch_count("orch-a")
        tracker.increment_per_orch_count("orch-b")

        counts = tracker.get_all_per_orch_counts()
        assert counts == {"orch-a": 2, "orch-b": 1}

    def test_get_all_per_orch_counts_returns_empty_dict_when_none(self) -> None:
        tracker = StateTracker()
        counts = tracker.get_all_per_orch_counts()
        assert counts == {}

    def test_get_available_slots_no_per_orch_limit(self) -> None:
        tracker = StateTracker()
        orch = make_orchestration(max_concurrent=None)

        available = tracker.get_available_slots_for_orchestration(orch, global_available=5)
        assert available == 5

    def test_get_available_slots_with_per_orch_limit(self) -> None:
        tracker = StateTracker()
        orch = make_orchestration(max_concurrent=3)

        tracker.increment_per_orch_count(orch.name)

        available = tracker.get_available_slots_for_orchestration(orch, global_available=5)
        assert available == 2

    def test_get_available_slots_limited_by_global(self) -> None:
        tracker = StateTracker()
        orch = make_orchestration(max_concurrent=10)

        available = tracker.get_available_slots_for_orchestration(orch, global_available=3)
        assert available == 3

    def test_get_available_slots_limited_by_per_orch(self) -> None:
        tracker = StateTracker()
        orch = make_orchestration(max_concurrent=2)

        tracker.increment_per_orch_count(orch.name)

        available = tracker.get_available_slots_for_orchestration(orch, global_available=5)
        assert available == 1

    def test_get_available_slots_returns_zero_when_per_orch_full(self) -> None:
        tracker = StateTracker()
        orch = make_orchestration(max_concurrent=2)

        tracker.increment_per_orch_count(orch.name)
        tracker.increment_per_orch_count(orch.name)

        available = tracker.get_available_slots_for_orchestration(orch, global_available=5)
        assert available == 0

    def test_get_available_slots_returns_zero_when_negative(self) -> None:
        tracker = StateTracker()
        orch = make_orchestration(max_concurrent=1)

        tracker.increment_per_orch_count(orch.name)
        tracker.increment_per_orch_count(orch.name)

        available = tracker.get_available_slots_for_orchestration(orch, global_available=5)
        assert available == 0


class TestPollTimes:
    def test_start_time_is_set_on_init(self) -> None:
        before = datetime.now(tz=UTC)
        tracker = StateTracker()
        after = datetime.now(tz=UTC)

        start_time = tracker.start_time
        assert before <= start_time <= after

    def test_last_jira_poll_defaults_to_none(self) -> None:
        tracker = StateTracker()
        assert tracker.last_jira_poll is None

    def test_last_jira_poll_can_be_set(self) -> None:
        tracker = StateTracker()
        now = datetime.now(tz=UTC)
        tracker.last_jira_poll = now

        assert tracker.last_jira_poll == now

    def test_last_github_poll_defaults_to_none(self) -> None:
        tracker = StateTracker()
        assert tracker.last_github_poll is None

    def test_last_github_poll_can_be_set(self) -> None:
        tracker = StateTracker()
        now = datetime.now(tz=UTC)
        tracker.last_github_poll = now

        assert tracker.last_github_poll == now

    def test_poll_times_are_independent(self) -> None:
        tracker = StateTracker()
        jira_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        github_time = datetime(2025, 1, 2, 12, 0, 0, tzinfo=UTC)

        tracker.last_jira_poll = jira_time
        tracker.last_github_poll = github_time

        assert tracker.last_jira_poll == jira_time
        assert tracker.last_github_poll == github_time


class TestThreadSafety:
    def test_concurrent_attempt_count_increments(self) -> None:
        tracker = StateTracker()
        results: list[int] = []
        num_threads = 20
        barrier = threading.Barrier(num_threads)

        def increment() -> None:
            barrier.wait()
            count = tracker.get_and_increment_attempt_count("TEST-1", "orch")
            results.append(count)

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(1, num_threads + 1))

    def test_concurrent_queue_operations(self) -> None:
        tracker = StateTracker(max_queue_size=1000)
        num_threads = 10
        items_per_thread = 10
        barrier = threading.Barrier(num_threads)

        def add_items(thread_id: int) -> None:
            barrier.wait()
            for i in range(items_per_thread):
                tracker.add_to_issue_queue(f"TEST-{thread_id}-{i}", f"orch-{thread_id}")

        threads = [threading.Thread(target=add_items, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        queue = tracker.get_issue_queue()
        assert len(queue) == num_threads * items_per_thread

    def test_concurrent_per_orch_count_operations(self) -> None:
        tracker = StateTracker()
        num_threads = 20
        barrier = threading.Barrier(num_threads)

        def increment_decrement() -> None:
            barrier.wait()
            tracker.increment_per_orch_count("my-orch")
            tracker.decrement_per_orch_count("my-orch")

        threads = [threading.Thread(target=increment_decrement) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        count = tracker.get_per_orch_count("my-orch")
        assert count == 0

    def test_concurrent_running_steps_operations(self) -> None:
        tracker = StateTracker()
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def add_remove(thread_id: int) -> None:
            barrier.wait()
            future_id = thread_id * 1000
            tracker.add_running_step(
                future_id=future_id,
                issue_key=f"TEST-{thread_id}",
                orchestration_name="orch",
                attempt_number=1,
                issue_url=f"https://example.com/TEST-{thread_id}",
            )
            tracker.remove_running_step(future_id)

        threads = [threading.Thread(target=add_remove, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with tracker._running_steps_lock:
            assert len(tracker._running_steps) == 0

    def test_concurrent_completed_execution_operations(self) -> None:
        tracker = StateTracker(max_completed_executions=1000)
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def add_execution(thread_id: int) -> None:
            barrier.wait()
            info = CompletedExecutionInfo(
                issue_key=f"TEST-{thread_id}",
                orchestration_name="orch",
                attempt_number=1,
                started_at=datetime.now(tz=UTC),
                completed_at=datetime.now(tz=UTC),
                status="success",
                input_tokens=100,
                output_tokens=50,
                total_cost_usd=0.01,
                issue_url=f"https://example.com/TEST-{thread_id}",
            )
            tracker.add_completed_execution(info)

        threads = [threading.Thread(target=add_execution, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        executions = tracker.get_completed_executions()
        assert len(executions) == num_threads

    def test_concurrent_poll_time_updates(self) -> None:
        tracker = StateTracker()
        num_threads = 20
        barrier = threading.Barrier(num_threads)

        def update_poll_times(thread_id: int) -> None:
            barrier.wait()
            now = datetime.now(tz=UTC)
            if thread_id % 2 == 0:
                tracker.last_jira_poll = now
            else:
                tracker.last_github_poll = now

        threads = [threading.Thread(target=update_poll_times, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.last_jira_poll is not None
        assert tracker.last_github_poll is not None
