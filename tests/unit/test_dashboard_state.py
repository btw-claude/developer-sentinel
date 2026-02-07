"""Tests for dashboard state execution summary statistics computation.

Tests for the _compute_execution_stats method of SentinelStateAccessor,
verifying global summary statistics and per-orchestration grouping.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sentinel.config import Config, ExecutionConfig
from sentinel.dashboard.state import (
    CompletedExecutionInfoView,
    ExecutionSummaryStats,
    SentinelStateAccessor,
)
from tests.unit.test_dashboard_routes import MockSentinel


def make_execution(
    *,
    orchestration_name: str = "test-orch",
    status: str = "success",
    duration_seconds: float = 120.0,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    total_cost_usd: float = 0.05,
    completed_at: datetime | None = None,
    issue_key: str = "TEST-1",
    issue_url: str = "https://jira.example.com/TEST-1",
) -> CompletedExecutionInfoView:
    """Create a CompletedExecutionInfoView instance for testing.

    Provides sensible defaults for all fields, with the ability to override
    any parameter to test specific scenarios.

    Args:
        orchestration_name: Name of the orchestration that ran.
        status: Execution status, either "success" or "failure".
        duration_seconds: Duration of the execution in seconds.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        total_cost_usd: Total cost in USD.
        completed_at: When the execution completed. Defaults to now.
        issue_key: The issue key that triggered the execution.
        issue_url: URL to the issue.

    Returns:
        A CompletedExecutionInfoView instance with the specified parameters.
    """
    if completed_at is None:
        completed_at = datetime.now()

    return CompletedExecutionInfoView(
        issue_key=issue_key,
        orchestration_name=orchestration_name,
        status=status,
        duration_seconds=duration_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=total_cost_usd,
        completed_at=completed_at,
        issue_url=issue_url,
    )


def _create_accessor() -> SentinelStateAccessor:
    """Create a SentinelStateAccessor with a MockSentinel for testing.

    Returns:
        A SentinelStateAccessor backed by a MockSentinel.
    """
    config = Config(execution=ExecutionConfig())
    sentinel = MockSentinel(config)
    return SentinelStateAccessor(sentinel)  # type: ignore[arg-type]


class TestComputeExecutionStatsEmpty:
    """Tests for _compute_execution_stats with empty input."""

    def test_empty_list_returns_zeroed_summary_stats(self) -> None:
        """Test that empty executions list returns all-zero summary stats."""
        accessor = _create_accessor()
        summary, orch_stats = accessor._compute_execution_stats([])

        assert summary.total_executions == 0
        assert summary.success_count == 0
        assert summary.failure_count == 0
        assert summary.success_rate == 0.0
        assert summary.avg_duration_seconds == 0.0
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0
        assert summary.total_tokens == 0
        assert summary.total_cost_usd == 0.0
        assert summary.avg_cost_usd == 0.0

    def test_empty_list_returns_empty_orchestration_stats(self) -> None:
        """Test that empty executions list returns empty orchestration stats list."""
        accessor = _create_accessor()
        summary, orch_stats = accessor._compute_execution_stats([])

        assert orch_stats == []

    def test_empty_list_returns_correct_types(self) -> None:
        """Test that empty list returns correct tuple types."""
        accessor = _create_accessor()
        summary, orch_stats = accessor._compute_execution_stats([])

        assert isinstance(summary, ExecutionSummaryStats)
        assert isinstance(orch_stats, list)


class TestComputeExecutionStatsSingleExecution:
    """Tests for _compute_execution_stats with a single execution."""

    def test_single_success_execution(self) -> None:
        """Test stats computation with a single successful execution."""
        accessor = _create_accessor()
        executions = [
            make_execution(
                status="success",
                duration_seconds=60.0,
                input_tokens=500,
                output_tokens=200,
                total_cost_usd=0.01,
            )
        ]

        summary, orch_stats = accessor._compute_execution_stats(executions)

        assert summary.total_executions == 1
        assert summary.success_count == 1
        assert summary.failure_count == 0
        assert summary.success_rate == 100.0
        assert summary.avg_duration_seconds == 60.0
        assert summary.total_input_tokens == 500
        assert summary.total_output_tokens == 200
        assert summary.total_tokens == 700
        assert summary.total_cost_usd == 0.01
        assert summary.avg_cost_usd == 0.01

    def test_single_failure_execution(self) -> None:
        """Test stats computation with a single failed execution."""
        accessor = _create_accessor()
        executions = [
            make_execution(
                status="failure",
                duration_seconds=30.0,
                input_tokens=300,
                output_tokens=100,
                total_cost_usd=0.005,
            )
        ]

        summary, orch_stats = accessor._compute_execution_stats(executions)

        assert summary.total_executions == 1
        assert summary.success_count == 0
        assert summary.failure_count == 1
        assert summary.success_rate == 0.0
        assert summary.avg_duration_seconds == 30.0
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 100
        assert summary.total_tokens == 400

    def test_single_execution_produces_one_orchestration_stat(self) -> None:
        """Test that single execution produces exactly one orchestration stat entry."""
        accessor = _create_accessor()
        executions = [make_execution(orchestration_name="my-orch")]

        summary, orch_stats = accessor._compute_execution_stats(executions)

        assert len(orch_stats) == 1
        assert orch_stats[0].orchestration_name == "my-orch"
        assert orch_stats[0].total_runs == 1


class TestComputeExecutionStatsMixed:
    """Tests for _compute_execution_stats with mixed statuses."""

    def test_mixed_statuses_correct_success_rate(self) -> None:
        """Test success rate calculation with mixed success and failure results."""
        accessor = _create_accessor()
        executions = [
            make_execution(status="success"),
            make_execution(status="success"),
            make_execution(status="failure"),
            make_execution(status="success"),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        assert summary.total_executions == 4
        assert summary.success_count == 3
        assert summary.failure_count == 1
        assert summary.success_rate == 75.0

    def test_mixed_statuses_with_two_thirds_success_rate(self) -> None:
        """Test success rate rounding with 2/3 success rate."""
        accessor = _create_accessor()
        executions = [
            make_execution(status="success"),
            make_execution(status="success"),
            make_execution(status="failure"),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        assert summary.total_executions == 3
        assert summary.success_count == 2
        assert summary.failure_count == 1
        # 2/3 * 100 = 66.666... rounded to 2 decimal places
        assert summary.success_rate == 66.67

    def test_all_failures_zero_success_rate(self) -> None:
        """Test that all failures produces 0% success rate."""
        accessor = _create_accessor()
        executions = [
            make_execution(status="failure"),
            make_execution(status="failure"),
            make_execution(status="failure"),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        assert summary.success_rate == 0.0
        assert summary.failure_count == 3
        assert summary.success_count == 0


class TestComputeExecutionStatsTokensAndCost:
    """Tests for token and cost aggregation in _compute_execution_stats."""

    def test_token_aggregation(self) -> None:
        """Test that input, output, and total tokens are correctly aggregated."""
        accessor = _create_accessor()
        executions = [
            make_execution(input_tokens=1000, output_tokens=500),
            make_execution(input_tokens=2000, output_tokens=800),
            make_execution(input_tokens=500, output_tokens=200),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        assert summary.total_input_tokens == 3500
        assert summary.total_output_tokens == 1500
        assert summary.total_tokens == 5000

    def test_cost_aggregation(self) -> None:
        """Test that total and average cost are correctly computed."""
        accessor = _create_accessor()
        executions = [
            make_execution(total_cost_usd=0.10),
            make_execution(total_cost_usd=0.20),
            make_execution(total_cost_usd=0.30),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        assert summary.total_cost_usd == 0.6
        assert summary.avg_cost_usd == 0.2

    def test_zero_tokens_not_treated_as_none(self) -> None:
        """Test that zero token values are treated as valid integers, not as None."""
        accessor = _create_accessor()
        executions = [
            make_execution(input_tokens=0, output_tokens=0, total_cost_usd=0.0),
            make_execution(input_tokens=1000, output_tokens=500, total_cost_usd=0.05),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        assert summary.total_input_tokens == 1000
        assert summary.total_output_tokens == 500
        assert summary.total_tokens == 1500
        assert summary.total_cost_usd == 0.05

    def test_zero_cost_not_treated_as_none(self) -> None:
        """Test that zero cost is a valid value and included in averages."""
        accessor = _create_accessor()
        executions = [
            make_execution(total_cost_usd=0.0),
            make_execution(total_cost_usd=0.10),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        assert summary.total_cost_usd == 0.1
        # Average: 0.1 / 2 = 0.05
        assert summary.avg_cost_usd == 0.05


class TestComputeExecutionStatsOrchestrationGrouping:
    """Tests for per-orchestration grouping in _compute_execution_stats."""

    def test_per_orchestration_grouping(self) -> None:
        """Test that executions are grouped correctly by orchestration name."""
        accessor = _create_accessor()
        executions = [
            make_execution(orchestration_name="orch-alpha", status="success"),
            make_execution(orchestration_name="orch-alpha", status="failure"),
            make_execution(orchestration_name="orch-beta", status="success"),
        ]

        _, orch_stats = accessor._compute_execution_stats(executions)

        assert len(orch_stats) == 2

        # First should be alpha (alphabetical sorting)
        alpha = orch_stats[0]
        assert alpha.orchestration_name == "orch-alpha"
        assert alpha.total_runs == 2
        assert alpha.success_count == 1
        assert alpha.failure_count == 1
        assert alpha.success_rate == 50.0

        # Second should be beta
        beta = orch_stats[1]
        assert beta.orchestration_name == "orch-beta"
        assert beta.total_runs == 1
        assert beta.success_count == 1
        assert beta.failure_count == 0
        assert beta.success_rate == 100.0

    def test_orchestration_stats_sorted_alphabetically(self) -> None:
        """Test that orchestration stats are sorted alphabetically by name."""
        accessor = _create_accessor()
        executions = [
            make_execution(orchestration_name="zebra-orch"),
            make_execution(orchestration_name="alpha-orch"),
            make_execution(orchestration_name="middle-orch"),
        ]

        _, orch_stats = accessor._compute_execution_stats(executions)

        assert len(orch_stats) == 3
        assert orch_stats[0].orchestration_name == "alpha-orch"
        assert orch_stats[1].orchestration_name == "middle-orch"
        assert orch_stats[2].orchestration_name == "zebra-orch"

    def test_per_orchestration_cost_aggregation(self) -> None:
        """Test that cost is aggregated correctly per orchestration."""
        accessor = _create_accessor()
        executions = [
            make_execution(orchestration_name="orch-a", total_cost_usd=0.10),
            make_execution(orchestration_name="orch-a", total_cost_usd=0.20),
            make_execution(orchestration_name="orch-b", total_cost_usd=0.50),
        ]

        _, orch_stats = accessor._compute_execution_stats(executions)

        orch_a = orch_stats[0]
        assert orch_a.orchestration_name == "orch-a"
        assert orch_a.total_cost_usd == 0.3

        orch_b = orch_stats[1]
        assert orch_b.orchestration_name == "orch-b"
        assert orch_b.total_cost_usd == 0.5

    def test_single_orchestration_with_multiple_executions(self) -> None:
        """Test stats for a single orchestration with multiple executions."""
        accessor = _create_accessor()
        executions = [
            make_execution(
                orchestration_name="only-orch",
                status="success",
                duration_seconds=100.0,
                total_cost_usd=0.10,
            ),
            make_execution(
                orchestration_name="only-orch",
                status="success",
                duration_seconds=200.0,
                total_cost_usd=0.20,
            ),
            make_execution(
                orchestration_name="only-orch",
                status="failure",
                duration_seconds=300.0,
                total_cost_usd=0.30,
            ),
        ]

        _, orch_stats = accessor._compute_execution_stats(executions)

        assert len(orch_stats) == 1
        stat = orch_stats[0]
        assert stat.orchestration_name == "only-orch"
        assert stat.total_runs == 3
        assert stat.success_count == 2
        assert stat.failure_count == 1
        assert stat.success_rate == 66.67
        assert stat.avg_duration_seconds == 200.0
        assert stat.total_cost_usd == 0.6


class TestComputeExecutionStatsLastRunAt:
    """Tests for last_run_at computation in per-orchestration stats."""

    def test_last_run_at_picks_most_recent_completed_at(self) -> None:
        """Test that last_run_at is the most recent completed_at per orchestration."""
        accessor = _create_accessor()
        now = datetime.now()
        earliest = now - timedelta(hours=3)
        middle = now - timedelta(hours=1)
        latest = now

        executions = [
            make_execution(orchestration_name="my-orch", completed_at=earliest),
            make_execution(orchestration_name="my-orch", completed_at=latest),
            make_execution(orchestration_name="my-orch", completed_at=middle),
        ]

        _, orch_stats = accessor._compute_execution_stats(executions)

        assert len(orch_stats) == 1
        assert orch_stats[0].last_run_at == latest

    def test_last_run_at_per_orchestration(self) -> None:
        """Test that last_run_at is computed independently per orchestration."""
        accessor = _create_accessor()
        now = datetime.now()
        time_a = now - timedelta(hours=2)
        time_b = now - timedelta(hours=1)

        executions = [
            make_execution(orchestration_name="orch-a", completed_at=time_a),
            make_execution(orchestration_name="orch-b", completed_at=time_b),
        ]

        _, orch_stats = accessor._compute_execution_stats(executions)

        assert len(orch_stats) == 2
        orch_a = orch_stats[0]
        orch_b = orch_stats[1]

        assert orch_a.orchestration_name == "orch-a"
        assert orch_a.last_run_at == time_a

        assert orch_b.orchestration_name == "orch-b"
        assert orch_b.last_run_at == time_b


class TestComputeExecutionStatsAvgDuration:
    """Tests for average duration computation in _compute_execution_stats."""

    def test_avg_duration_seconds_global(self) -> None:
        """Test global average duration calculation across all executions."""
        accessor = _create_accessor()
        executions = [
            make_execution(duration_seconds=60.0),
            make_execution(duration_seconds=120.0),
            make_execution(duration_seconds=180.0),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        # Average: (60 + 120 + 180) / 3 = 120.0
        assert summary.avg_duration_seconds == 120.0

    def test_avg_duration_seconds_per_orchestration(self) -> None:
        """Test per-orchestration average duration calculation."""
        accessor = _create_accessor()
        executions = [
            make_execution(orchestration_name="fast-orch", duration_seconds=10.0),
            make_execution(orchestration_name="fast-orch", duration_seconds=20.0),
            make_execution(orchestration_name="slow-orch", duration_seconds=300.0),
            make_execution(orchestration_name="slow-orch", duration_seconds=600.0),
        ]

        _, orch_stats = accessor._compute_execution_stats(executions)

        fast = orch_stats[0]
        assert fast.orchestration_name == "fast-orch"
        assert fast.avg_duration_seconds == 15.0

        slow = orch_stats[1]
        assert slow.orchestration_name == "slow-orch"
        assert slow.avg_duration_seconds == 450.0

    def test_avg_duration_rounding(self) -> None:
        """Test that average duration is rounded to 2 decimal places."""
        accessor = _create_accessor()
        executions = [
            make_execution(duration_seconds=10.0),
            make_execution(duration_seconds=20.0),
            make_execution(duration_seconds=30.0),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        # (10 + 20 + 30) / 3 = 20.0 - clean division
        assert summary.avg_duration_seconds == 20.0

    def test_avg_duration_rounding_with_remainder(self) -> None:
        """Test that average duration rounds fractional results to 2 decimal places."""
        accessor = _create_accessor()
        executions = [
            make_execution(duration_seconds=10.0),
            make_execution(duration_seconds=10.0),
            make_execution(duration_seconds=10.0),
            make_execution(duration_seconds=10.0),
            make_execution(duration_seconds=10.0),
            make_execution(duration_seconds=10.0),
            make_execution(duration_seconds=11.0),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        # (60 + 11) / 7 = 71/7 = 10.142857... rounded to 10.14
        assert summary.avg_duration_seconds == 10.14


class TestComputeExecutionStatsRounding:
    """Tests for rounding behavior in _compute_execution_stats."""

    def test_success_rate_rounded_to_two_decimals(self) -> None:
        """Test that success rate is rounded to 2 decimal places."""
        accessor = _create_accessor()
        # 1 out of 3 = 33.333...%
        executions = [
            make_execution(status="success"),
            make_execution(status="failure"),
            make_execution(status="failure"),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        assert summary.success_rate == 33.33

    def test_cost_rounded_to_six_decimals(self) -> None:
        """Test that cost values are rounded to 6 decimal places."""
        accessor = _create_accessor()
        executions = [
            make_execution(total_cost_usd=0.1111111),
            make_execution(total_cost_usd=0.2222222),
            make_execution(total_cost_usd=0.3333333),
        ]

        summary, _ = accessor._compute_execution_stats(executions)

        # Total: 0.6666666, rounded to 6 decimal places
        assert summary.total_cost_usd == 0.666667
        # Average: 0.6666666 / 3 = 0.2222222, rounded to 6 decimal places
        assert summary.avg_cost_usd == 0.222222


class TestMockSentinelProtocol:
    """Tests to verify MockSentinel fully implements SentinelStateProvider protocol."""

    def test_mock_sentinel_has_get_completed_executions(self) -> None:
        """Test that MockSentinel implements get_completed_executions method."""
        config = Config(execution=ExecutionConfig())
        sentinel = MockSentinel(config)
        result = sentinel.get_completed_executions()

        assert isinstance(result, list)
        assert result == []

    def test_mock_sentinel_state_accessor_works(self) -> None:
        """Test that SentinelStateAccessor can use MockSentinel to get full state."""
        config = Config(execution=ExecutionConfig())
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        # get_state should work without errors now that MockSentinel
        # implements get_completed_executions
        state = accessor.get_state()

        assert state is not None
        assert state.execution_summary is not None
        assert state.execution_summary.total_executions == 0
        assert state.orchestration_stats == []

    def test_mock_sentinel_has_all_protocol_methods(self) -> None:
        """Test that MockSentinel has all methods required by SentinelStateProvider."""
        config = Config(execution=ExecutionConfig())
        sentinel = MockSentinel(config)

        # Verify all SentinelStateProvider protocol methods exist
        assert hasattr(sentinel, "config")
        assert hasattr(sentinel, "orchestrations")
        assert callable(sentinel.get_hot_reload_metrics)
        assert callable(sentinel.get_running_steps)
        assert callable(sentinel.get_issue_queue)
        assert callable(sentinel.get_start_time)
        assert callable(sentinel.get_last_jira_poll)
        assert callable(sentinel.get_last_github_poll)
        assert callable(sentinel.get_active_versions)
        assert callable(sentinel.get_pending_removal_versions)
        assert callable(sentinel.get_execution_state)
        assert callable(sentinel.get_completed_executions)
        assert callable(sentinel.is_shutdown_requested)
