"""Tests for Sentinel queue eviction and orchestration logging."""

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from sentinel.main import Sentinel

# Import shared fixtures and helpers from conftest.py
from tests.conftest import (
    MockAgentClient,
    MockJiraClient,
    MockTagClient,
    make_config,
    make_orchestration,
)


class TestQueueEvictionBehavior:
    """Tests for queue eviction behavior and logging."""

    def test_queue_evicts_oldest_item_when_full(self) -> None:
        """Test that the oldest item is evicted when queue reaches maxlen."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=3)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        sentinel._state_tracker.add_to_issue_queue("TEST-1", "orch-1")
        sentinel._state_tracker.add_to_issue_queue("TEST-2", "orch-2")
        sentinel._state_tracker.add_to_issue_queue("TEST-3", "orch-3")

        assert len(sentinel._state_tracker._issue_queue) == 3
        queue_keys = [item.issue_key for item in sentinel._state_tracker._issue_queue]
        assert queue_keys == ["TEST-1", "TEST-2", "TEST-3"]

        sentinel._state_tracker.add_to_issue_queue("TEST-4", "orch-4")

        assert len(sentinel._state_tracker._issue_queue) == 3
        queue_keys = [item.issue_key for item in sentinel._state_tracker._issue_queue]
        assert queue_keys == ["TEST-2", "TEST-3", "TEST-4"]
        assert "TEST-1" not in queue_keys

    def test_queue_maintains_fifo_ordering(self) -> None:
        """Test that the queue maintains FIFO ordering as items are added."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=5)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        for i in range(1, 6):
            sentinel._state_tracker.add_to_issue_queue(f"TEST-{i}", f"orch-{i}")

        queue_keys = [item.issue_key for item in sentinel._state_tracker._issue_queue]
        assert queue_keys == ["TEST-1", "TEST-2", "TEST-3", "TEST-4", "TEST-5"]

        assert sentinel._state_tracker._issue_queue[0].issue_key == "TEST-1"
        assert sentinel._state_tracker._issue_queue[-1].issue_key == "TEST-5"

    def test_eviction_logging_includes_evicted_item_key(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that eviction logging includes the evicted item's key."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=2)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        sentinel._state_tracker.add_to_issue_queue("EVICT-ME", "orch-old")
        sentinel._state_tracker.add_to_issue_queue("TEST-2", "orch-2")

        caplog.clear()

        with caplog.at_level(logging.DEBUG, logger="sentinel.state_tracker"):
            sentinel._state_tracker.add_to_issue_queue("NEW-ITEM", "orch-new")

        eviction_records = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == "sentinel.state_tracker"
            and "capacity" in r.message
            and "evicted" in r.message
        ]

        assert len(eviction_records) == 1, (
            f"Expected 1 eviction log record, got {len(eviction_records)}: "
            f"{[r.message for r in eviction_records]}"
        )

        eviction_record = eviction_records[0]
        assert (
            "EVICT-ME" in eviction_record.message
        ), f"Expected evicted key 'EVICT-ME' in log message: {eviction_record.message}"
        assert (
            "orch-old" in eviction_record.message
        ), f"Expected evicted orchestration 'orch-old' in log message: {eviction_record.message}"
        assert (
            "NEW-ITEM" in eviction_record.message
        ), f"Expected new key 'NEW-ITEM' in log message: {eviction_record.message}"

    def test_dashboard_shows_most_recent_queued_issues(self) -> None:
        """Test that get_issue_queue returns most recent items after eviction."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=3)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        for i in range(1, 6):
            sentinel._state_tracker.add_to_issue_queue(f"TEST-{i}", f"orch-{i}")

        queue = sentinel.get_issue_queue()
        assert len(queue) == 3

        queue_keys = [item.issue_key for item in queue]
        assert queue_keys == ["TEST-3", "TEST-4", "TEST-5"]
        assert "TEST-1" not in queue_keys
        assert "TEST-2" not in queue_keys

    def test_no_eviction_log_when_queue_not_full(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that no eviction log is produced when queue is not full."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=10)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        with caplog.at_level(logging.DEBUG, logger="sentinel.state_tracker"):
            sentinel._state_tracker.add_to_issue_queue("TEST-1", "orch-1")
            sentinel._state_tracker.add_to_issue_queue("TEST-2", "orch-2")
            sentinel._state_tracker.add_to_issue_queue("TEST-3", "orch-3")

        eviction_records = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == "sentinel.state_tracker"
            and "capacity" in r.message
            and "evicted" in r.message
        ]
        assert (
            len(eviction_records) == 0
        ), f"Expected no eviction logs, got: {[r.message for r in eviction_records]}"

    def test_multiple_evictions_log_each_evicted_item(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that each eviction produces a log with the correct evicted item."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=2)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        sentinel._state_tracker.add_to_issue_queue("ITEM-A", "orch-a")
        sentinel._state_tracker.add_to_issue_queue("ITEM-B", "orch-b")

        with caplog.at_level(logging.DEBUG, logger="sentinel.state_tracker"):
            sentinel._state_tracker.add_to_issue_queue("ITEM-C", "orch-c")
            sentinel._state_tracker.add_to_issue_queue("ITEM-D", "orch-d")

        eviction_records = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == "sentinel.state_tracker"
            and "capacity" in r.message
            and "evicted" in r.message
        ]

        assert len(eviction_records) == 2, (
            f"Expected 2 eviction log records, got {len(eviction_records)}: "
            f"{[r.message for r in eviction_records]}"
        )

        assert (
            "ITEM-A" in eviction_records[0].message
        ), f"Expected 'ITEM-A' in first eviction log: {eviction_records[0].message}"
        assert (
            "ITEM-B" in eviction_records[1].message
        ), f"Expected 'ITEM-B' in second eviction log: {eviction_records[1].message}"

    def test_queue_clear_resets_for_new_cycle(self) -> None:
        """Test that _clear_issue_queue properly resets the queue."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=3)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        for i in range(1, 4):
            sentinel._state_tracker.add_to_issue_queue(f"OLD-{i}", f"orch-{i}")

        assert len(sentinel._state_tracker._issue_queue) == 3

        sentinel._state_tracker.clear_issue_queue()

        assert len(sentinel._state_tracker._issue_queue) == 0

        sentinel._state_tracker.add_to_issue_queue("NEW-1", "new-orch-1")

        assert len(sentinel._state_tracker._issue_queue) == 1
        assert sentinel._state_tracker._issue_queue[0].issue_key == "NEW-1"

    def test_eviction_preserves_queued_at_timestamp_ordering(self) -> None:
        """Test that queued_at timestamps are preserved and ordered correctly."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=2)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        mock_times = [
            datetime(2026, 1, 15, 10, 0, 0),
            datetime(2026, 1, 15, 10, 0, 1),
            datetime(2026, 1, 15, 10, 0, 2),
        ]
        time_iterator = iter(mock_times)

        with patch("sentinel.state_tracker.datetime") as mock_datetime:
            mock_datetime.now.side_effect = lambda: next(time_iterator)
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            sentinel._state_tracker.add_to_issue_queue("TEST-1", "orch-1")
            sentinel._state_tracker.add_to_issue_queue("TEST-2", "orch-2")
            sentinel._state_tracker.add_to_issue_queue("TEST-3", "orch-3")

        queue_items = list(sentinel._state_tracker._issue_queue)
        assert len(queue_items) == 2

        assert queue_items[0].issue_key == "TEST-2"
        assert queue_items[1].issue_key == "TEST-3"
        assert queue_items[0].queued_at < queue_items[1].queued_at
        assert queue_items[0].queued_at == datetime(2026, 1, 15, 10, 0, 1)
        assert queue_items[1].queued_at == datetime(2026, 1, 15, 10, 0, 2)


class TestSentinelOrchestrationLogging:
    """Tests for Sentinel integration with per-orchestration logging."""

    def test_sentinel_initializes_orch_log_manager_when_configured(self, tmp_path: Path) -> None:
        """Test Sentinel initializes OrchestrationLogManager when configured."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        assert sentinel._orch_log_manager is not None
        sentinel.run_once_and_wait()

    def test_sentinel_does_not_initialize_orch_log_manager_when_not_configured(
        self,
    ) -> None:
        """Test Sentinel doesn't initialize OrchestrationLogManager when not set."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        assert config.orchestration_logs_dir is None
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        assert sentinel._orch_log_manager is None

    def test_sentinel_creates_log_files_via_log_for_orchestration(self, tmp_path: Path) -> None:
        """Test that _log_for_orchestration creates log files for orchestrations."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration(name="test-orchestration", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        sentinel._log_for_orchestration("test-orchestration", logging.INFO, "Test log entry")

        sentinel._orch_log_manager.close_all()

        log_file = logs_dir / "test-orchestration.log"
        assert log_file.exists()

    def test_sentinel_logs_contain_orchestration_activity(self, tmp_path: Path) -> None:
        """Test orchestration logs contain relevant execution activity."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration(name="log-test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        sentinel._log_for_orchestration(
            "log-test-orch",
            logging.INFO,
            "Polling Jira for orchestration 'log-test-orch'",
        )
        sentinel._log_for_orchestration(
            "log-test-orch", logging.INFO, "Submitting 'log-test-orch' for TEST-1"
        )

        sentinel._orch_log_manager.close_all()

        log_file = logs_dir / "log-test-orch.log"
        content = log_file.read_text()

        assert "Polling Jira" in content
        assert "TEST-1" in content or "log-test-orch" in content

    def test_sentinel_separate_orchestrations_have_separate_logs(self, tmp_path: Path) -> None:
        """Test different orchestrations write to separate log files."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [
            make_orchestration(name="orch-alpha", tags=["alpha"]),
            make_orchestration(name="orch-beta", tags=["beta"]),
        ]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        sentinel._log_for_orchestration("orch-alpha", logging.INFO, "Message for orch-alpha")
        sentinel._log_for_orchestration("orch-beta", logging.INFO, "Message for orch-beta")

        sentinel._orch_log_manager.close_all()

        log_alpha = logs_dir / "orch-alpha.log"
        log_beta = logs_dir / "orch-beta.log"
        assert log_alpha.exists()
        assert log_beta.exists()

        content_alpha = log_alpha.read_text()
        content_beta = log_beta.read_text()

        assert "orch-alpha" in content_alpha
        assert "orch-beta" in content_beta
        assert "Message for orch-beta" not in content_alpha
        assert "Message for orch-alpha" not in content_beta

    def test_sentinel_closes_log_manager_on_run_once_and_wait(self, tmp_path: Path) -> None:
        """Test that run_once_and_wait properly closes the log manager."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        assert sentinel._orch_log_manager is not None
        sentinel._orch_log_manager.get_logger("test-orch")

        sentinel.run_once_and_wait()

        assert len(sentinel._orch_log_manager._handlers) == 0
        assert len(sentinel._orch_log_manager._loggers) == 0

    def test_log_for_orchestration_logs_to_main_logger(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that _log_for_orchestration logs to the main logger."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        with caplog.at_level(logging.INFO):
            sentinel._log_for_orchestration("test-orch", logging.INFO, "Test log message")

        assert "Test log message" in caplog.text

    def test_log_for_orchestration_logs_to_both_when_configured(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test _log_for_orchestration logs to both main logger and orch file."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        with caplog.at_level(logging.INFO):
            sentinel._log_for_orchestration("dual-log-test", logging.INFO, "Dual log message")

        assert "Dual log message" in caplog.text

        sentinel.run_once_and_wait()
        log_file = logs_dir / "dual-log-test.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Dual log message" in content
