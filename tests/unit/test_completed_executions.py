"""Tests for completed executions tracking functionality (DS-523).

Also includes tests for usage data propagation (DS-528):
- ExecutionResult usage data fields
- AgentExecutor usage data extraction
- Sentinel._record_completed_execution usage data handling
"""

import threading
import time
from datetime import datetime

import pytest

from sentinel.agent_clients.base import UsageInfo
from sentinel.executor import AgentExecutor, ExecutionResult, ExecutionStatus
from sentinel.main import Sentinel
from sentinel.state_tracker import CompletedExecutionInfo, RunningStepInfo, StateTracker

# Import shared fixtures and helpers from conftest.py
from tests.conftest import (
    MockAgentClient,
    MockJiraPoller,
    MockTagClient,
    make_agent_factory,
    make_config,
    make_issue,
    make_orchestration,
)


class TestCompletedExecutionInfo:
    """Tests for CompletedExecutionInfo dataclass."""

    def test_completed_execution_info_fields(self) -> None:
        """Test that CompletedExecutionInfo has all required fields."""
        started = datetime.now()
        completed = datetime.now()

        info = CompletedExecutionInfo(
            issue_key="TEST-123",
            orchestration_name="test-orch",
            attempt_number=2,
            started_at=started,
            completed_at=completed,
            status="success",
            input_tokens=1000,
            output_tokens=500,
            total_cost_usd=0.05,
            issue_url="https://jira.example.com/TEST-123",
        )

        assert info.issue_key == "TEST-123"
        assert info.orchestration_name == "test-orch"
        assert info.attempt_number == 2
        assert info.started_at == started
        assert info.completed_at == completed
        assert info.status == "success"
        assert info.input_tokens == 1000
        assert info.output_tokens == 500
        assert info.total_cost_usd == 0.05
        assert info.issue_url == "https://jira.example.com/TEST-123"

    def test_completed_execution_info_failure_status(self) -> None:
        """Test CompletedExecutionInfo with failure status."""
        info = CompletedExecutionInfo(
            issue_key="TEST-456",
            orchestration_name="another-orch",
            attempt_number=3,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            status="failure",
            input_tokens=500,
            output_tokens=250,
            total_cost_usd=0.02,
            issue_url="https://github.com/org/repo/issues/456",
        )

        assert info.status == "failure"


class TestStateTrackerCompletedExecutions:
    """Tests for StateTracker completed executions tracking."""

    def test_add_completed_execution(self) -> None:
        """Test adding a completed execution."""
        tracker = StateTracker(max_completed_executions=10)

        info = CompletedExecutionInfo(
            issue_key="TEST-1",
            orchestration_name="orch-1",
            attempt_number=1,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            status="success",
            input_tokens=100,
            output_tokens=50,
            total_cost_usd=0.01,
            issue_url="https://jira.example.com/TEST-1",
        )

        tracker.add_completed_execution(info)
        executions = tracker.get_completed_executions()

        assert len(executions) == 1
        assert executions[0].issue_key == "TEST-1"

    def test_get_completed_executions_returns_most_recent_first(self) -> None:
        """Test that completed executions are ordered most recent first."""
        tracker = StateTracker(max_completed_executions=10)

        for i in range(3):
            info = CompletedExecutionInfo(
                issue_key=f"TEST-{i}",
                orchestration_name="orch-1",
                attempt_number=1,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                status="success",
                input_tokens=100,
                output_tokens=50,
                total_cost_usd=0.01,
                issue_url=f"https://jira.example.com/TEST-{i}",
            )
            tracker.add_completed_execution(info)
            time.sleep(0.01)  # Small delay to ensure ordering

        executions = tracker.get_completed_executions()

        assert len(executions) == 3
        # Most recent should be first (TEST-2 was added last)
        assert executions[0].issue_key == "TEST-2"
        assert executions[1].issue_key == "TEST-1"
        assert executions[2].issue_key == "TEST-0"

    def test_max_completed_executions_evicts_oldest(self) -> None:
        """Test that oldest entries are evicted when max size is exceeded."""
        tracker = StateTracker(max_completed_executions=3)

        for i in range(5):
            info = CompletedExecutionInfo(
                issue_key=f"TEST-{i}",
                orchestration_name="orch-1",
                attempt_number=1,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                status="success",
                input_tokens=100,
                output_tokens=50,
                total_cost_usd=0.01,
                issue_url=f"https://jira.example.com/TEST-{i}",
            )
            tracker.add_completed_execution(info)

        executions = tracker.get_completed_executions()

        # Should only have the 3 most recent
        assert len(executions) == 3
        # Most recent first: TEST-4, TEST-3, TEST-2
        assert executions[0].issue_key == "TEST-4"
        assert executions[1].issue_key == "TEST-3"
        assert executions[2].issue_key == "TEST-2"

    def test_completed_executions_thread_safety(self) -> None:
        """Test that completed executions tracking is thread-safe."""
        tracker = StateTracker(max_completed_executions=100)
        results: list[bool] = []
        num_threads = 10
        executions_per_thread = 10

        def add_executions(thread_id: int) -> None:
            try:
                for i in range(executions_per_thread):
                    info = CompletedExecutionInfo(
                        issue_key=f"TEST-{thread_id}-{i}",
                        orchestration_name="orch-1",
                        attempt_number=1,
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                        status="success",
                        input_tokens=100,
                        output_tokens=50,
                        total_cost_usd=0.01,
                        issue_url=f"https://jira.example.com/TEST-{thread_id}-{i}",
                    )
                    tracker.add_completed_execution(info)
                results.append(True)
            except Exception:
                results.append(False)

        threads = [
            threading.Thread(target=add_executions, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results), "Some threads failed"
        executions = tracker.get_completed_executions()
        assert len(executions) == num_threads * executions_per_thread

    def test_default_max_completed_executions(self) -> None:
        """Test that default max_completed_executions is 50."""
        tracker = StateTracker()
        # Add more than default max
        for i in range(60):
            info = CompletedExecutionInfo(
                issue_key=f"TEST-{i}",
                orchestration_name="orch-1",
                attempt_number=1,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                status="success",
                input_tokens=100,
                output_tokens=50,
                total_cost_usd=0.01,
                issue_url=f"https://jira.example.com/TEST-{i}",
            )
            tracker.add_completed_execution(info)

        executions = tracker.get_completed_executions()
        assert len(executions) == 50  # Default max


class TestSentinelCompletedExecutions:
    """Tests for Sentinel completed executions integration."""

    def test_sentinel_get_completed_executions(self) -> None:
        """Test that Sentinel exposes get_completed_executions method."""
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Should return empty list initially
        executions = sentinel.get_completed_executions()
        assert executions == []

    def test_sentinel_completed_execution_recorded(self) -> None:
        """Test that completed executions are recorded through the state tracker."""
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Manually add a completed execution to the state tracker
        info = CompletedExecutionInfo(
            issue_key="TEST-1",
            orchestration_name="test-orch",
            attempt_number=1,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            status="success",
            input_tokens=100,
            output_tokens=50,
            total_cost_usd=0.01,
            issue_url="https://jira.example.com/TEST-1",
        )
        sentinel._state_tracker.add_completed_execution(info)

        executions = sentinel.get_completed_executions()
        assert len(executions) == 1
        assert executions[0].issue_key == "TEST-1"


class TestExecutionResultUsageData:
    """Tests for ExecutionResult usage data fields (DS-528)."""

    def test_execution_result_has_usage_fields(self) -> None:
        """Test that ExecutionResult has usage data fields with defaults."""
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response="SUCCESS",
            attempts=1,
            issue_key="TEST-123",
            orchestration_name="test-orch",
        )

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_cost_usd == 0.0

    def test_execution_result_with_usage_data(self) -> None:
        """Test that ExecutionResult can store usage data."""
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response="SUCCESS",
            attempts=1,
            issue_key="TEST-123",
            orchestration_name="test-orch",
            input_tokens=1000,
            output_tokens=500,
            total_cost_usd=0.05,
        )

        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.total_cost_usd == 0.05


@pytest.mark.asyncio
class TestAgentExecutorUsageDataPropagation:
    """Tests for AgentExecutor usage data extraction from AgentRunResult (DS-528).

    Note: execute() is async (DS-509), so these tests are async.
    """

    async def test_executor_propagates_usage_data_on_success(self) -> None:
        """Test that executor extracts usage data from AgentRunResult on success."""
        usage = UsageInfo(
            input_tokens=1500,
            output_tokens=750,
            total_cost_usd=0.075,
        )
        agent_client = MockAgentClient(
            responses=["SUCCESS: Task completed"],
            usage=usage,
        )
        executor = AgentExecutor(agent_client)

        issue = make_issue("TEST-1", "Test summary")
        orchestration = make_orchestration()

        result = await executor.execute(issue, orchestration)

        assert result.succeeded
        assert result.input_tokens == 1500
        assert result.output_tokens == 750
        assert result.total_cost_usd == 0.075

    async def test_executor_propagates_usage_data_on_failure(self) -> None:
        """Test that executor extracts usage data from AgentRunResult even on failure."""
        usage = UsageInfo(
            input_tokens=500,
            output_tokens=250,
            total_cost_usd=0.025,
        )
        agent_client = MockAgentClient(
            responses=["FAILURE: Task failed"],
            usage=usage,
        )
        executor = AgentExecutor(agent_client)

        orchestration = make_orchestration()
        # Set max_attempts to 1 to avoid retries
        orchestration.retry.max_attempts = 1
        issue = make_issue("TEST-2", "Test summary")

        result = await executor.execute(issue, orchestration)

        assert not result.succeeded
        assert result.input_tokens == 500
        assert result.output_tokens == 250
        assert result.total_cost_usd == 0.025

    async def test_executor_handles_none_usage(self) -> None:
        """Test that executor handles None usage data gracefully."""
        agent_client = MockAgentClient(
            responses=["SUCCESS: Task completed"],
            usage=None,  # No usage data
        )
        executor = AgentExecutor(agent_client)

        issue = make_issue("TEST-3", "Test summary")
        orchestration = make_orchestration()

        result = await executor.execute(issue, orchestration)

        assert result.succeeded
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_cost_usd == 0.0


class TestSentinelRecordCompletedExecutionUsageData:
    """Tests for Sentinel._record_completed_execution usage data handling (DS-528)."""

    def test_record_completed_execution_uses_usage_data_from_result(self) -> None:
        """Test that _record_completed_execution extracts usage data from ExecutionResult."""
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Create an ExecutionResult with usage data
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response="SUCCESS",
            attempts=1,
            issue_key="TEST-123",
            orchestration_name="test-orch",
            input_tokens=2000,
            output_tokens=1000,
            total_cost_usd=0.10,
        )

        # Create a RunningStepInfo
        running_step = RunningStepInfo(
            issue_key="TEST-123",
            orchestration_name="test-orch",
            attempt_number=1,
            started_at=datetime.now(),
            issue_url="https://jira.example.com/TEST-123",
        )

        # Record the completed execution
        sentinel._record_completed_execution(result, running_step)

        # Verify the usage data was captured
        executions = sentinel.get_completed_executions()
        assert len(executions) == 1
        assert executions[0].input_tokens == 2000
        assert executions[0].output_tokens == 1000
        assert executions[0].total_cost_usd == 0.10

    def test_record_completed_execution_with_zero_usage(self) -> None:
        """Test that _record_completed_execution handles zero usage data."""
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Create an ExecutionResult with default (zero) usage data
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response="SUCCESS",
            attempts=1,
            issue_key="TEST-456",
            orchestration_name="test-orch",
            # No usage data specified - defaults to 0
        )

        running_step = RunningStepInfo(
            issue_key="TEST-456",
            orchestration_name="test-orch",
            attempt_number=1,
            started_at=datetime.now(),
            issue_url="https://jira.example.com/TEST-456",
        )

        sentinel._record_completed_execution(result, running_step)

        executions = sentinel.get_completed_executions()
        assert len(executions) == 1
        assert executions[0].input_tokens == 0
        assert executions[0].output_tokens == 0
        assert executions[0].total_cost_usd == 0.0
