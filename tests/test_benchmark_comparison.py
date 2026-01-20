"""Benchmark comparison tests for CLI vs SDK streaming vs SDK non-streaming.

DS-172: Create baseline comparison tests to measure performance across configurations.

This module provides a benchmark suite comparing three execution paths:
1. Direct claude CLI command
2. SDK with streaming logs enabled
3. SDK with streaming logs disabled

Measures:
- Time to first response
- Total execution time
- Message count and timing
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from sentinel.config import Config
from sentinel.sdk_clients import ClaudeSdkAgentClient, TimingMetrics


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark runs.

    Attributes:
        time_to_first_response: Time in seconds to receive first response.
        total_execution_time: Total time in seconds for the execution.
        message_count: Number of messages received (for SDK paths).
        inter_message_times: List of times between messages.
        mode: The execution mode ('cli', 'sdk_streaming', 'sdk_non_streaming').
    """

    time_to_first_response: float | None = None
    total_execution_time: float = 0.0
    message_count: int = 0
    inter_message_times: list[float] = field(default_factory=list)
    mode: str = ""

    @property
    def avg_inter_message_time(self) -> float | None:
        """Calculate average time between messages."""
        if not self.inter_message_times:
            return None
        return sum(self.inter_message_times) / len(self.inter_message_times)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "mode": self.mode,
            "time_to_first_response": self.time_to_first_response,
            "total_execution_time": self.total_execution_time,
            "message_count": self.message_count,
            "avg_inter_message_time": self.avg_inter_message_time,
            "inter_message_times": self.inter_message_times,
        }


@dataclass
class BenchmarkResult:
    """Results from running a complete benchmark comparison.

    Attributes:
        cli_metrics: Metrics from direct CLI execution.
        sdk_streaming_metrics: Metrics from SDK with streaming enabled.
        sdk_non_streaming_metrics: Metrics from SDK with streaming disabled.
        prompt: The prompt used for all tests.
    """

    cli_metrics: BenchmarkMetrics
    sdk_streaming_metrics: BenchmarkMetrics
    sdk_non_streaming_metrics: BenchmarkMetrics
    prompt: str

    def get_comparison_summary(self) -> dict[str, Any]:
        """Generate a comparison summary of all execution paths."""
        return {
            "prompt": self.prompt,
            "cli": self.cli_metrics.to_dict(),
            "sdk_streaming": self.sdk_streaming_metrics.to_dict(),
            "sdk_non_streaming": self.sdk_non_streaming_metrics.to_dict(),
            "speedup": {
                "streaming_vs_cli": (
                    self.cli_metrics.total_execution_time
                    / self.sdk_streaming_metrics.total_execution_time
                    if self.sdk_streaming_metrics.total_execution_time > 0
                    else None
                ),
                "non_streaming_vs_cli": (
                    self.cli_metrics.total_execution_time
                    / self.sdk_non_streaming_metrics.total_execution_time
                    if self.sdk_non_streaming_metrics.total_execution_time > 0
                    else None
                ),
                "non_streaming_vs_streaming": (
                    self.sdk_streaming_metrics.total_execution_time
                    / self.sdk_non_streaming_metrics.total_execution_time
                    if self.sdk_non_streaming_metrics.total_execution_time > 0
                    else None
                ),
            },
        }


class MockMessage:
    """Mock message from the Claude Agent SDK query."""

    def __init__(self, text: str) -> None:
        self.text = text


def create_timed_mock_query(
    responses: list[str],
    delays: list[float] | None = None,
) -> MagicMock:
    """Create a mock query that yields responses with optional delays.

    Args:
        responses: List of response texts to yield.
        delays: Optional list of delays (in seconds) before each response.
            Must have the same length as responses if provided.

    Returns:
        A MagicMock that returns an async generator.

    Raises:
        ValueError: If delays is provided and has a different length than responses.
    """
    if delays is None:
        delays = [0.0] * len(responses)
    elif len(delays) != len(responses):
        raise ValueError(
            f"Length mismatch: responses has {len(responses)} items, "
            f"delays has {len(delays)} items. Lists must have matching lengths."
        )

    async def async_gen() -> AsyncIterator[MockMessage]:
        for response, delay in zip(responses, delays):
            if delay > 0:
                await asyncio.sleep(delay)
            yield MockMessage(response)

    mock = MagicMock(return_value=async_gen())
    return mock


class TestTimingMetrics:
    """Tests for TimingMetrics dataclass."""

    def test_start_query_records_time(self) -> None:
        """Should record start time when query begins."""
        metrics = TimingMetrics()
        metrics.start_query()

        assert metrics.query_start_time > 0

    def test_record_message_sets_first_message_time(self) -> None:
        """Should set first_message_time on first message."""
        metrics = TimingMetrics()
        metrics.start_query()
        time.sleep(0.01)
        metrics.record_message_received()

        assert metrics.first_message_time is not None
        assert metrics.first_message_time > metrics.query_start_time
        assert metrics.message_count == 1

    def test_record_message_tracks_inter_message_times(self) -> None:
        """Should track time between messages."""
        metrics = TimingMetrics()
        metrics.start_query()
        metrics.record_message_received()
        time.sleep(0.01)
        metrics.record_message_received()
        time.sleep(0.02)
        metrics.record_message_received()

        assert metrics.message_count == 3
        assert len(metrics.inter_message_times) == 2
        assert all(t > 0 for t in metrics.inter_message_times)

    def test_time_to_first_message_calculation(self) -> None:
        """Should correctly calculate time to first message."""
        metrics = TimingMetrics()
        metrics.start_query()
        time.sleep(0.05)
        metrics.record_message_received()

        ttfm = metrics.time_to_first_message
        assert ttfm is not None
        assert ttfm >= 0.05

    def test_time_to_first_message_none_without_message(self) -> None:
        """Should return None if no messages received."""
        metrics = TimingMetrics()
        metrics.start_query()
        metrics.finish()

        assert metrics.time_to_first_message is None

    def test_total_elapsed_time_calculation(self) -> None:
        """Should correctly calculate total elapsed time."""
        metrics = TimingMetrics()
        metrics.start_query()
        time.sleep(0.05)
        metrics.finish()

        assert metrics.total_elapsed_time >= 0.05

    def test_avg_inter_message_time_calculation(self) -> None:
        """Should correctly calculate average inter-message time."""
        metrics = TimingMetrics()
        metrics.inter_message_times = [0.1, 0.2, 0.3]

        avg = metrics.avg_inter_message_time
        assert avg is not None
        assert abs(avg - 0.2) < 0.001

    def test_avg_inter_message_time_none_without_messages(self) -> None:
        """Should return None if no inter-message times recorded."""
        metrics = TimingMetrics()

        assert metrics.avg_inter_message_time is None

    def test_file_io_time_accumulation(self) -> None:
        """Should accumulate file I/O time correctly."""
        metrics = TimingMetrics()
        metrics.add_file_io_time(0.1)
        metrics.add_file_io_time(0.2)
        metrics.add_file_io_time(0.15)

        assert abs(metrics.file_io_time - 0.45) < 0.001

    def test_api_wait_time_accumulation(self) -> None:
        """Should accumulate API wait time correctly."""
        metrics = TimingMetrics()
        metrics.add_api_wait_time(0.5)
        metrics.add_api_wait_time(0.3)

        assert abs(metrics.api_wait_time - 0.8) < 0.001

    def test_to_dict_returns_complete_metrics(self) -> None:
        """Should return all metrics as dictionary."""
        metrics = TimingMetrics()
        metrics.start_query()
        metrics.record_message_received()
        metrics.add_file_io_time(0.1)
        metrics.add_api_wait_time(0.2)
        metrics.finish()

        result = metrics.to_dict()

        assert "total_elapsed_time" in result
        assert "time_to_first_message" in result
        assert "message_count" in result
        assert "avg_inter_message_time" in result
        assert "file_io_time" in result
        assert "api_wait_time" in result
        assert "inter_message_times" in result

    # DS-189: Optimize inter_message_times storage for long-running operations

    def test_inter_message_times_below_threshold_returns_raw_array(self) -> None:
        """Should return raw inter_message_times array when below threshold (DS-189)."""
        metrics = TimingMetrics()
        # Add fewer times than the threshold
        metrics.inter_message_times = [0.1, 0.2, 0.3, 0.15, 0.25]

        result = metrics.to_dict()

        assert "inter_message_times" in result
        assert "inter_message_times_summary" not in result
        assert result["inter_message_times"] == [0.1, 0.2, 0.3, 0.15, 0.25]

    def test_inter_message_times_above_threshold_returns_summary(self) -> None:
        """Should return summary when inter_message_times exceeds threshold (DS-189)."""
        metrics = TimingMetrics()
        # Add more times than the threshold (default is 100)
        metrics.inter_message_times = [0.1 * i for i in range(1, 102)]  # 101 items

        result = metrics.to_dict()

        assert "inter_message_times_summary" in result
        assert "inter_message_times" not in result
        summary = result["inter_message_times_summary"]
        assert summary["count"] == 101
        assert "min" in summary
        assert "max" in summary
        assert "avg" in summary
        assert "p50" in summary
        assert "p95" in summary
        assert "p99" in summary

    def test_inter_message_times_at_threshold_returns_raw_array(self) -> None:
        """Should return raw array when exactly at threshold (DS-189)."""
        metrics = TimingMetrics()
        # Add exactly threshold number of items
        metrics.inter_message_times = [0.1] * TimingMetrics.INTER_MESSAGE_TIMES_THRESHOLD

        result = metrics.to_dict()

        # At threshold, should still return raw array (only exceeding triggers summary)
        assert "inter_message_times" in result
        assert "inter_message_times_summary" not in result

    def test_get_inter_message_times_summary_empty_data(self) -> None:
        """Should return appropriate summary for empty data (DS-189)."""
        metrics = TimingMetrics()

        summary = metrics.get_inter_message_times_summary()

        assert summary["count"] == 0
        assert summary["min"] is None
        assert summary["max"] is None
        assert summary["avg"] is None
        assert summary["p50"] is None
        assert summary["p95"] is None
        assert summary["p99"] is None

    def test_get_inter_message_times_summary_single_value(self) -> None:
        """Should handle single value correctly (DS-189)."""
        metrics = TimingMetrics()
        metrics.inter_message_times = [0.5]

        summary = metrics.get_inter_message_times_summary()

        assert summary["count"] == 1
        assert summary["min"] == 0.5
        assert summary["max"] == 0.5
        assert summary["avg"] == 0.5
        assert summary["p50"] == 0.5
        assert summary["p95"] == 0.5
        assert summary["p99"] == 0.5

    def test_get_inter_message_times_summary_known_values(self) -> None:
        """Should calculate correct statistics for known values (DS-189)."""
        metrics = TimingMetrics()
        # Create a list with known statistical properties
        # Values 1-100 (so p50=50.5, min=1, max=100)
        metrics.inter_message_times = [float(i) for i in range(1, 101)]

        summary = metrics.get_inter_message_times_summary()

        assert summary["count"] == 100
        assert summary["min"] == 1.0
        assert summary["max"] == 100.0
        assert summary["avg"] == 50.5
        # p50 should be around 50.5 (median of 1-100)
        assert 50 <= summary["p50"] <= 51
        # p95 should be around 95
        assert 94 <= summary["p95"] <= 96
        # p99 should be around 99
        assert 98 <= summary["p99"] <= 100

    def test_calculate_percentile_edge_cases(self) -> None:
        """Should calculate percentiles correctly for edge cases (DS-189)."""
        metrics = TimingMetrics()

        # Empty data should return 0.0
        assert metrics._calculate_percentile([], 50) == 0.0

        # Single element
        assert metrics._calculate_percentile([5.0], 0) == 5.0
        assert metrics._calculate_percentile([5.0], 50) == 5.0
        assert metrics._calculate_percentile([5.0], 100) == 5.0

        # Two elements - interpolation test
        data = [1.0, 3.0]
        assert metrics._calculate_percentile(data, 0) == 1.0
        assert metrics._calculate_percentile(data, 50) == 2.0  # interpolated
        assert metrics._calculate_percentile(data, 100) == 3.0

    def test_threshold_is_configurable(self) -> None:
        """Should respect the INTER_MESSAGE_TIMES_THRESHOLD value (DS-189)."""
        # Verify the default threshold value
        assert TimingMetrics.INTER_MESSAGE_TIMES_THRESHOLD == 100

        # Temporarily modify the threshold for this test
        original_threshold = TimingMetrics.INTER_MESSAGE_TIMES_THRESHOLD
        try:
            TimingMetrics.INTER_MESSAGE_TIMES_THRESHOLD = 5
            # Create metrics AFTER modifying threshold (dataclass copies class attr to instance)
            metrics = TimingMetrics()
            metrics.inter_message_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 6 items > 5

            result = metrics.to_dict()

            assert "inter_message_times_summary" in result
            assert "inter_message_times" not in result
        finally:
            TimingMetrics.INTER_MESSAGE_TIMES_THRESHOLD = original_threshold

    # Error scenario tests (DS-176 improvement #3)

    def test_record_message_before_start_query(self) -> None:
        """Should handle record_message_received() called before start_query().

        When record_message_received() is called before start_query(), the
        first_message_time will be set but query_start_time will be 0.0,
        resulting in potentially invalid time_to_first_message calculations.
        This test documents the current behavior.

        TODO: Consider whether TimingMetrics should raise an error or warning when
        record_message_received() is called before start_query(). The current behavior
        silently accepts this out-of-order call sequence, which could lead to misleading
        metrics. Options to consider:
        1. Raise ValueError if query_start_time is 0.0 (strict validation)
        2. Log a warning but continue (lenient with feedback)
        3. Keep current behavior if there's a valid use case for this pattern
        See: https://github.com/btw-claude/developer-sentinel/pull/178 for context.
        """
        metrics = TimingMetrics()
        # Call record_message_received without calling start_query first
        metrics.record_message_received()

        # first_message_time should still be set
        assert metrics.first_message_time is not None
        # But query_start_time is 0.0 (uninitialized)
        assert metrics.query_start_time == 0.0
        # time_to_first_message will be calculated but may be very large
        # since it's first_message_time - 0.0
        assert metrics.time_to_first_message is not None
        assert metrics.message_count == 1

    def test_finish_before_start_query(self) -> None:
        """Should handle finish() called before start_query().

        When finish() is called before start_query(), total_elapsed_time
        will be calculated as total_end_time - 0.0, which documents the
        current behavior.
        """
        metrics = TimingMetrics()
        # Call finish without calling start_query first
        metrics.finish()

        # total_end_time should be set
        assert metrics.total_end_time > 0
        # total_elapsed_time will be calculated but may be very large
        assert metrics.total_elapsed_time > 0

    def test_double_start_query(self) -> None:
        """Should handle start_query() called multiple times.

        When start_query() is called multiple times, it simply overwrites
        the query_start_time. This documents the current behavior.
        """
        metrics = TimingMetrics()
        metrics.start_query()
        first_start = metrics.query_start_time

        time.sleep(0.01)
        metrics.start_query()
        second_start = metrics.query_start_time

        # Second call should overwrite the first start time
        assert second_start > first_start
        assert metrics.query_start_time == second_start

    def test_double_finish(self) -> None:
        """Should handle finish() called multiple times.

        When finish() is called multiple times, it simply overwrites
        the total_end_time. This documents the current behavior.
        """
        metrics = TimingMetrics()
        metrics.start_query()
        metrics.finish()
        first_end = metrics.total_end_time

        time.sleep(0.01)
        metrics.finish()
        second_end = metrics.total_end_time

        # Second call should overwrite the first end time
        assert second_end > first_end
        assert metrics.total_end_time == second_end

    def test_add_times_with_negative_values(self) -> None:
        """Should accept negative time values (documents current behavior).

        The add_file_io_time and add_api_wait_time methods do not validate
        input values. This test documents that negative values are accepted.

        Design Note: Negative time values are intentionally accepted without validation.
        This design choice supports the following use cases:
        1. Clock adjustment corrections - When system clock adjustments occur during
           timing measurements, negative deltas may be legitimate.
        2. Time correction offsets - Callers may need to subtract previously added
           time to correct erroneous measurements.
        3. Simplicity - Adding validation would add complexity for an edge case that
           is unlikely to occur in normal operation.

        If input validation is desired in the future, consider:
        - Raising ValueError for negative values (strict)
        - Clamping to zero with a warning (lenient)
        - Adding a separate subtract_*_time() method for corrections
        """
        metrics = TimingMetrics()
        metrics.add_file_io_time(-0.1)
        metrics.add_api_wait_time(-0.2)

        # Negative values are accepted without validation
        assert metrics.file_io_time == -0.1
        assert metrics.api_wait_time == -0.2


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics dataclass."""

    def test_default_values(self) -> None:
        """Should have appropriate default values."""
        metrics = BenchmarkMetrics()

        assert metrics.time_to_first_response is None
        assert metrics.total_execution_time == 0.0
        assert metrics.message_count == 0
        assert metrics.inter_message_times == []
        assert metrics.mode == ""

    def test_avg_inter_message_time_calculation(self) -> None:
        """Should calculate average inter-message time correctly."""
        metrics = BenchmarkMetrics(
            inter_message_times=[0.1, 0.2, 0.3],
            mode="test",
        )

        assert metrics.avg_inter_message_time is not None
        assert abs(metrics.avg_inter_message_time - 0.2) < 0.001

    def test_avg_inter_message_time_none_without_data(self) -> None:
        """Should return None when no inter-message times."""
        metrics = BenchmarkMetrics()

        assert metrics.avg_inter_message_time is None

    def test_to_dict_format(self) -> None:
        """Should return properly formatted dictionary."""
        metrics = BenchmarkMetrics(
            time_to_first_response=0.5,
            total_execution_time=1.5,
            message_count=10,
            inter_message_times=[0.1, 0.2],
            mode="sdk_streaming",
        )

        result = metrics.to_dict()

        assert result["mode"] == "sdk_streaming"
        assert result["time_to_first_response"] == 0.5
        assert result["total_execution_time"] == 1.5
        assert result["message_count"] == 10
        assert result["avg_inter_message_time"] is not None


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_comparison_summary_format(self) -> None:
        """Should generate properly formatted comparison summary."""
        cli_metrics = BenchmarkMetrics(
            time_to_first_response=0.5,
            total_execution_time=2.0,
            mode="cli",
        )
        sdk_streaming = BenchmarkMetrics(
            time_to_first_response=0.3,
            total_execution_time=1.5,
            message_count=5,
            mode="sdk_streaming",
        )
        sdk_non_streaming = BenchmarkMetrics(
            time_to_first_response=0.2,
            total_execution_time=1.0,
            message_count=1,
            mode="sdk_non_streaming",
        )

        result = BenchmarkResult(
            cli_metrics=cli_metrics,
            sdk_streaming_metrics=sdk_streaming,
            sdk_non_streaming_metrics=sdk_non_streaming,
            prompt="test prompt",
        )

        summary = result.get_comparison_summary()

        assert summary["prompt"] == "test prompt"
        assert "cli" in summary
        assert "sdk_streaming" in summary
        assert "sdk_non_streaming" in summary
        assert "speedup" in summary

    def test_speedup_calculations(self) -> None:
        """Should calculate speedup ratios correctly."""
        cli_metrics = BenchmarkMetrics(total_execution_time=4.0, mode="cli")
        sdk_streaming = BenchmarkMetrics(total_execution_time=2.0, mode="sdk_streaming")
        sdk_non_streaming = BenchmarkMetrics(total_execution_time=1.0, mode="sdk_non_streaming")

        result = BenchmarkResult(
            cli_metrics=cli_metrics,
            sdk_streaming_metrics=sdk_streaming,
            sdk_non_streaming_metrics=sdk_non_streaming,
            prompt="test",
        )

        summary = result.get_comparison_summary()

        # CLI (4.0) / streaming (2.0) = 2.0x speedup
        assert summary["speedup"]["streaming_vs_cli"] == 2.0
        # CLI (4.0) / non-streaming (1.0) = 4.0x speedup
        assert summary["speedup"]["non_streaming_vs_cli"] == 4.0
        # streaming (2.0) / non-streaming (1.0) = 2.0x speedup
        assert summary["speedup"]["non_streaming_vs_streaming"] == 2.0


class TestSdkStreamingVsNonStreaming:
    """Tests comparing SDK streaming vs non-streaming execution paths."""

    @pytest.fixture
    def mock_config(self) -> Config:
        """Create a mock config for testing."""
        return Config()

    @pytest.fixture
    def temp_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create temporary directories for workdir and logs."""
        workdir = tmp_path / "workdir"
        logs = tmp_path / "logs"
        workdir.mkdir()
        logs.mkdir()
        return workdir, logs

    def test_streaming_enabled_uses_streaming_path(
        self, mock_config: Config, temp_dirs: tuple[Path, Path]
    ) -> None:
        """SDK client with streaming enabled should use _run_with_log."""
        workdir, logs = temp_dirs

        with patch("sentinel.sdk_clients.query", create_timed_mock_query(["response"])):
            client = ClaudeSdkAgentClient(
                mock_config,
                base_workdir=workdir,
                log_base_dir=logs,
                disable_streaming_logs=False,
            )

            result = client.run_agent(
                prompt="test prompt",
                tools=[],
                issue_key="TEST-1",
                orchestration_name="test_orch",
            )

            assert result.response == "response"
            # Verify log file was created (streaming path)
            log_files = list(logs.glob("test_orch/*.log"))
            assert len(log_files) == 1

    def test_streaming_disabled_uses_simple_path(
        self, mock_config: Config, temp_dirs: tuple[Path, Path]
    ) -> None:
        """SDK client with streaming disabled should use _run_simple."""
        workdir, logs = temp_dirs

        with patch("sentinel.sdk_clients.query", create_timed_mock_query(["response"])):
            client = ClaudeSdkAgentClient(
                mock_config,
                base_workdir=workdir,
                log_base_dir=logs,
                disable_streaming_logs=True,
            )

            result = client.run_agent(
                prompt="test prompt",
                tools=[],
                issue_key="TEST-1",
                orchestration_name="test_orch",
            )

            assert result.response == "response"
            # Verify log file was created (non-streaming path also writes logs after completion)
            log_files = list(logs.glob("test_orch/*.log"))
            assert len(log_files) == 1

    def test_config_disable_streaming_logs_respected(
        self, temp_dirs: tuple[Path, Path]
    ) -> None:
        """Should respect disable_streaming_logs from config."""
        workdir, logs = temp_dirs
        config = Config(disable_streaming_logs=True)

        with patch("sentinel.sdk_clients.query", create_timed_mock_query(["response"])):
            client = ClaudeSdkAgentClient(
                config,
                base_workdir=workdir,
                log_base_dir=logs,
            )

            # Verify the internal flag matches config
            assert client._disable_streaming_logs is True

    def test_explicit_disable_overrides_config(
        self, temp_dirs: tuple[Path, Path]
    ) -> None:
        """Explicit disable_streaming_logs parameter should override config."""
        workdir, logs = temp_dirs
        config = Config(disable_streaming_logs=False)

        client = ClaudeSdkAgentClient(
            config,
            base_workdir=workdir,
            log_base_dir=logs,
            disable_streaming_logs=True,
        )

        assert client._disable_streaming_logs is True

    def test_streaming_log_contains_timing_metrics(
        self, mock_config: Config, temp_dirs: tuple[Path, Path]
    ) -> None:
        """Streaming logs should contain timing metrics section."""
        workdir, logs = temp_dirs

        with patch("sentinel.sdk_clients.query", create_timed_mock_query(["response"])):
            client = ClaudeSdkAgentClient(
                mock_config,
                base_workdir=workdir,
                log_base_dir=logs,
                disable_streaming_logs=False,
            )

            client.run_agent(
                prompt="test prompt",
                tools=[],
                issue_key="TEST-1",
                orchestration_name="test_orch",
            )

            log_files = list(logs.glob("test_orch/*.log"))
            assert len(log_files) == 1

            log_content = log_files[0].read_text()
            assert "TIMING METRICS" in log_content
            assert "Total elapsed time" in log_content
            assert "Messages received" in log_content
            assert "File I/O time" in log_content
            assert "API wait time" in log_content

    def test_streaming_log_contains_json_metrics_export(
        self, mock_config: Config, temp_dirs: tuple[Path, Path]
    ) -> None:
        """Streaming logs should contain JSON metrics export for programmatic access (DS-173)."""
        workdir, logs = temp_dirs

        with patch("sentinel.sdk_clients.query", create_timed_mock_query(["response"])):
            client = ClaudeSdkAgentClient(
                mock_config,
                base_workdir=workdir,
                log_base_dir=logs,
                disable_streaming_logs=False,
            )

            client.run_agent(
                prompt="test prompt",
                tools=[],
                issue_key="TEST-1",
                orchestration_name="test_orch",
            )

            log_files = list(logs.glob("test_orch/*.log"))
            assert len(log_files) == 1

            log_content = log_files[0].read_text()
            assert "METRICS JSON" in log_content
            # Verify the JSON contains the expected keys from to_dict()
            assert '"total_elapsed_time"' in log_content
            assert '"time_to_first_message"' in log_content
            assert '"message_count"' in log_content
            assert '"file_io_time"' in log_content
            assert '"api_wait_time"' in log_content
            assert '"inter_message_times"' in log_content

    def test_non_streaming_log_format(
        self, mock_config: Config, temp_dirs: tuple[Path, Path]
    ) -> None:
        """Non-streaming logs should have correct format."""
        workdir, logs = temp_dirs

        with patch("sentinel.sdk_clients.query", create_timed_mock_query(["response"])):
            client = ClaudeSdkAgentClient(
                mock_config,
                base_workdir=workdir,
                log_base_dir=logs,
                disable_streaming_logs=True,
            )

            client.run_agent(
                prompt="test prompt",
                tools=[],
                issue_key="TEST-1",
                orchestration_name="test_orch",
            )

            log_files = list(logs.glob("test_orch/*.log"))
            assert len(log_files) == 1

            log_content = log_files[0].read_text()
            assert "non-streaming mode" in log_content
            assert "SENTINEL_DISABLE_STREAMING_LOGS=true" in log_content
            assert "COMPLETED" in log_content


class TestCliVsSdkComparison:
    """Tests for CLI vs SDK execution comparison.

    Note: These tests use mocks to simulate behavior.
    For real performance comparison, see the integration benchmarks.
    """

    def test_cli_execution_metrics_structure(self) -> None:
        """CLI metrics should have expected structure."""
        metrics = BenchmarkMetrics(
            time_to_first_response=0.5,
            total_execution_time=2.0,
            message_count=1,  # CLI returns single response
            mode="cli",
        )

        result = metrics.to_dict()

        assert result["mode"] == "cli"
        assert result["message_count"] == 1
        assert result["time_to_first_response"] == 0.5

    def test_sdk_streaming_metrics_structure(self) -> None:
        """SDK streaming metrics should include inter-message times."""
        metrics = BenchmarkMetrics(
            time_to_first_response=0.3,
            total_execution_time=1.5,
            message_count=10,
            inter_message_times=[0.1, 0.15, 0.12, 0.08, 0.1, 0.11, 0.09, 0.13, 0.12],
            mode="sdk_streaming",
        )

        result = metrics.to_dict()

        assert result["mode"] == "sdk_streaming"
        assert result["message_count"] == 10
        assert len(metrics.inter_message_times) == 9
        assert metrics.avg_inter_message_time is not None

    def test_sdk_non_streaming_metrics_structure(self) -> None:
        """SDK non-streaming metrics should have single message."""
        metrics = BenchmarkMetrics(
            time_to_first_response=0.2,
            total_execution_time=1.0,
            message_count=1,  # Non-streaming returns aggregated response
            mode="sdk_non_streaming",
        )

        result = metrics.to_dict()

        assert result["mode"] == "sdk_non_streaming"
        assert result["message_count"] == 1


class TestMultipleResponseSimulation:
    """Tests simulating multiple streaming responses."""

    def test_multiple_responses_tracked(self) -> None:
        """Should track timing for multiple streamed responses."""
        metrics = TimingMetrics()
        metrics.start_query()

        # Simulate receiving 5 messages with varying delays
        for i in range(5):
            time.sleep(0.01)  # Simulate network delay
            metrics.record_message_received()

        metrics.finish()

        assert metrics.message_count == 5
        assert len(metrics.inter_message_times) == 4  # n-1 intervals
        assert metrics.time_to_first_message is not None
        assert metrics.total_elapsed_time >= 0.05

    @pytest.fixture
    def mock_config(self) -> Config:
        """Create a mock config for testing."""
        return Config()

    @pytest.fixture
    def temp_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create temporary directories for workdir and logs."""
        workdir = tmp_path / "workdir"
        logs = tmp_path / "logs"
        workdir.mkdir()
        logs.mkdir()
        return workdir, logs

    def test_streaming_captures_all_messages(
        self, mock_config: Config, temp_dirs: tuple[Path, Path]
    ) -> None:
        """Streaming should capture timing for all messages."""
        workdir, logs = temp_dirs

        # Create mock that yields multiple messages
        responses = ["part1", "part2", "part3", "part4", "final"]
        delays = [0.01, 0.02, 0.01, 0.015, 0.01]

        with patch(
            "sentinel.sdk_clients.query",
            create_timed_mock_query(responses, delays),
        ):
            client = ClaudeSdkAgentClient(
                mock_config,
                base_workdir=workdir,
                log_base_dir=logs,
                disable_streaming_logs=False,
            )

            result = client.run_agent(
                prompt="test prompt",
                tools=[],
                issue_key="TEST-1",
                orchestration_name="test_orch",
            )

            # Final response should be the last message
            assert result.response == "final"

            # Check log file contains metrics
            log_files = list(logs.glob("test_orch/*.log"))
            assert len(log_files) == 1
            log_content = log_files[0].read_text()
            assert "Messages received:" in log_content


class TestBenchmarkHelpers:
    """Tests for benchmark helper functions."""

    def test_create_timed_mock_query_no_delays(self) -> None:
        """Mock query without delays should yield responses immediately."""

        async def run_test() -> list[str]:
            mock = create_timed_mock_query(["a", "b", "c"])
            results = []
            async for msg in mock():
                results.append(msg.text)
            return results

        results = asyncio.run(run_test())
        assert results == ["a", "b", "c"]

    def test_create_timed_mock_query_with_delays(self) -> None:
        """Mock query with delays should introduce timing."""

        async def run_test() -> tuple[list[str], float]:
            mock = create_timed_mock_query(["a", "b"], [0.05, 0.05])
            start = time.perf_counter()
            results = []
            async for msg in mock():
                results.append(msg.text)
            elapsed = time.perf_counter() - start
            return results, elapsed

        results, elapsed = asyncio.run(run_test())
        assert results == ["a", "b"]
        assert elapsed >= 0.1  # At least 50ms + 50ms

    # DS-176 improvement #1: Validation tests for create_timed_mock_query

    def test_create_timed_mock_query_mismatched_lengths_raises_error(self) -> None:
        """Should raise ValueError when responses and delays have different lengths."""
        with pytest.raises(ValueError) as exc_info:
            create_timed_mock_query(["a", "b", "c"], [0.1, 0.2])

        assert "Length mismatch" in str(exc_info.value)
        assert "responses has 3 items" in str(exc_info.value)
        assert "delays has 2 items" in str(exc_info.value)

    def test_create_timed_mock_query_more_delays_than_responses(self) -> None:
        """Should raise ValueError when delays has more items than responses."""
        with pytest.raises(ValueError) as exc_info:
            create_timed_mock_query(["a"], [0.1, 0.2, 0.3])

        assert "Length mismatch" in str(exc_info.value)
        assert "responses has 1 items" in str(exc_info.value)
        assert "delays has 3 items" in str(exc_info.value)

    def test_create_timed_mock_query_empty_lists(self) -> None:
        """Should handle empty response list."""

        async def run_test() -> list[str]:
            mock = create_timed_mock_query([])
            results = []
            async for msg in mock():
                results.append(msg.text)
            return results

        results = asyncio.run(run_test())
        assert results == []

    def test_create_timed_mock_query_matching_lengths(self) -> None:
        """Should succeed when responses and delays have matching lengths."""

        async def run_test() -> list[str]:
            mock = create_timed_mock_query(["a", "b"], [0.0, 0.0])
            results = []
            async for msg in mock():
                results.append(msg.text)
            return results

        results = asyncio.run(run_test())
        assert results == ["a", "b"]


class TestPerformanceInstrumentation:
    """Tests verifying performance instrumentation from DS-169."""

    def test_timing_metrics_integration_with_query(self) -> None:
        """TimingMetrics should integrate with _run_query pattern."""
        metrics = TimingMetrics()
        metrics.start_query()

        # Simulate the pattern in _run_query
        api_wait_start = time.perf_counter()
        time.sleep(0.02)  # Simulate API wait

        api_wait_end = time.perf_counter()
        metrics.add_api_wait_time(api_wait_end - api_wait_start)
        metrics.record_message_received()

        # Simulate file I/O
        io_start = time.perf_counter()
        time.sleep(0.01)  # Simulate file write
        metrics.add_file_io_time(time.perf_counter() - io_start)

        metrics.finish()

        # Verify metrics captured
        assert metrics.api_wait_time >= 0.02
        assert metrics.file_io_time >= 0.01
        assert metrics.message_count == 1
        assert metrics.total_elapsed_time >= 0.03

    def test_log_metrics_does_not_raise(self) -> None:
        """log_metrics should not raise even with partial data."""
        metrics = TimingMetrics()
        metrics.start_query()
        metrics.finish()

        # Should not raise
        metrics.log_metrics("test_operation")

    def test_log_metrics_with_complete_data(self) -> None:
        """log_metrics should handle complete data without raising."""
        metrics = TimingMetrics()
        metrics.start_query()
        metrics.record_message_received()
        time.sleep(0.01)
        metrics.record_message_received()
        metrics.add_file_io_time(0.005)
        metrics.add_api_wait_time(0.015)
        metrics.finish()

        # Should not raise
        metrics.log_metrics("complete_test")


# DS-176 improvement #4: Subprocess-based CLI integration tests
class TestCliSubprocessIntegration:
    """Subprocess-based CLI integration tests for real-world performance validation.

    These tests use actual subprocess calls to validate CLI behavior and measure
    real-world performance characteristics. They are marked with pytest.mark.integration
    to allow selective execution during CI/CD.

    Note: These tests require the claude CLI to be installed and configured.
    They are skipped if the CLI is not available.
    """

    @staticmethod
    def _is_claude_cli_available() -> bool:
        """Check if the claude CLI is available in PATH."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @pytest.fixture
    def cli_available(self) -> bool:
        """Fixture to check if CLI is available."""
        return self._is_claude_cli_available()

    @pytest.mark.integration
    def test_cli_subprocess_execution_timing(self, cli_available: bool) -> None:
        """Measure actual CLI subprocess execution timing.

        This test validates that we can properly measure CLI execution time
        using subprocess. It uses a simple prompt that should complete quickly.
        """
        if not cli_available:
            pytest.skip("claude CLI not available")

        start_time = time.perf_counter()

        result = subprocess.run(
            ["claude", "--print", "Say only: OK"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Validate the subprocess completed
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Record metrics for this execution
        metrics = BenchmarkMetrics(
            time_to_first_response=execution_time,  # CLI returns all at once
            total_execution_time=execution_time,
            message_count=1,
            mode="cli_subprocess",
        )

        # Verify metrics structure
        result_dict = metrics.to_dict()
        assert result_dict["mode"] == "cli_subprocess"
        assert result_dict["total_execution_time"] > 0

    @pytest.mark.integration
    def test_cli_subprocess_with_timeout(self, cli_available: bool) -> None:
        """Test CLI subprocess execution with timeout handling."""
        if not cli_available:
            pytest.skip("claude CLI not available")

        # Test that timeout mechanism works correctly
        # Note: Using 0.01s timeout instead of 0.001s for better CI stability.
        # The subprocess needs a small amount of time to start, so extremely
        # short timeouts may flake on slow CI runners.
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                # Use a prompt that would take a long time
                ["claude", "--print", "Write a 10000 word essay"],
                capture_output=True,
                text=True,
                timeout=0.01,  # Short timeout to force expiration (CI-stable)
            )

    @pytest.mark.integration
    def test_cli_subprocess_error_handling(self, cli_available: bool) -> None:
        """Test CLI subprocess error handling for invalid inputs."""
        if not cli_available:
            pytest.skip("claude CLI not available")

        # Test with invalid flag to verify error capture
        result = subprocess.run(
            ["claude", "--invalid-flag-that-does-not-exist"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # CLI should return non-zero exit code for invalid flags
        assert result.returncode != 0

    @pytest.mark.integration
    def test_cli_subprocess_captures_output(self, cli_available: bool) -> None:
        """Verify CLI subprocess properly captures stdout and stderr."""
        if not cli_available:
            pytest.skip("claude CLI not available")

        result = subprocess.run(
            ["claude", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Help output should be captured in stdout
        assert len(result.stdout) > 0 or len(result.stderr) > 0
        # Should complete successfully
        assert result.returncode == 0

    @pytest.mark.integration
    def test_cli_vs_sdk_timing_comparison_structure(
        self, cli_available: bool
    ) -> None:
        """Validate the structure for comparing CLI vs SDK timing.

        This test sets up the comparison structure without actually making
        API calls, validating that the benchmark comparison infrastructure
        works correctly.
        """
        # Create sample metrics for comparison structure validation
        cli_metrics = BenchmarkMetrics(
            time_to_first_response=1.5,
            total_execution_time=2.0,
            message_count=1,
            mode="cli",
        )

        # Simulate what SDK streaming metrics would look like
        sdk_streaming_metrics = BenchmarkMetrics(
            time_to_first_response=0.3,
            total_execution_time=1.8,
            message_count=15,
            inter_message_times=[0.1] * 14,
            mode="sdk_streaming",
        )

        # Simulate SDK non-streaming metrics
        sdk_non_streaming_metrics = BenchmarkMetrics(
            time_to_first_response=1.6,
            total_execution_time=1.6,
            message_count=1,
            mode="sdk_non_streaming",
        )

        # Create comparison result
        result = BenchmarkResult(
            cli_metrics=cli_metrics,
            sdk_streaming_metrics=sdk_streaming_metrics,
            sdk_non_streaming_metrics=sdk_non_streaming_metrics,
            prompt="test comparison",
        )

        # Validate comparison summary structure
        summary = result.get_comparison_summary()

        assert "prompt" in summary
        assert "cli" in summary
        assert "sdk_streaming" in summary
        assert "sdk_non_streaming" in summary
        assert "speedup" in summary

        # Validate speedup calculations make sense
        assert summary["speedup"]["streaming_vs_cli"] is not None
        assert summary["speedup"]["non_streaming_vs_cli"] is not None
        assert summary["speedup"]["non_streaming_vs_streaming"] is not None


# DS-176 improvement #2: Time-based test stability improvements
class TestDeterministicTiming:
    """Tests demonstrating deterministic timing patterns for CI stability.

    These tests use deterministic timing approaches instead of time.sleep()
    to improve CI stability. They demonstrate patterns that can be used
    with pytest-freezegun or time-machine for fully deterministic tests.

    Note: For tests that require actual wall-clock timing (e.g., performance
    benchmarks), time.sleep() may still be necessary. These tests focus on
    unit tests where timing can be mocked.
    """

    def test_timing_metrics_with_explicit_timestamps(self) -> None:
        """Demonstrate testing with explicit timestamp control.

        Instead of relying on time.sleep(), we can directly set timestamp
        values to test timing calculations deterministically.
        """
        metrics = TimingMetrics()

        # Directly set timestamps for deterministic testing
        metrics.query_start_time = 1000.0
        metrics.first_message_time = 1000.5
        metrics.last_message_time = 1001.0
        metrics.total_end_time = 1002.0
        metrics.message_count = 3
        metrics.inter_message_times = [0.2, 0.3]

        # Verify calculations work with explicit values
        assert metrics.time_to_first_message == 0.5
        assert metrics.total_elapsed_time == 2.0
        assert metrics.avg_inter_message_time == 0.25

    def test_timing_metrics_edge_cases(self) -> None:
        """Test timing calculations with edge case values."""
        metrics = TimingMetrics()

        # Test with zero elapsed time
        metrics.query_start_time = 1000.0
        metrics.total_end_time = 1000.0

        assert metrics.total_elapsed_time == 0.0

        # Test with very small inter-message times
        metrics.inter_message_times = [0.001, 0.001, 0.001]
        assert abs(metrics.avg_inter_message_time - 0.001) < 0.0001  # type: ignore

    def test_benchmark_metrics_timing_isolation(self) -> None:
        """Verify BenchmarkMetrics works independently of wall-clock time."""
        # All values are explicitly provided, no timing dependencies
        metrics = BenchmarkMetrics(
            time_to_first_response=0.123,
            total_execution_time=0.456,
            message_count=5,
            inter_message_times=[0.05, 0.06, 0.07, 0.08],
            mode="test",
        )

        # Calculations should be deterministic
        assert metrics.avg_inter_message_time == 0.065

        # to_dict should contain all explicit values
        result = metrics.to_dict()
        assert result["time_to_first_response"] == 0.123
        assert result["total_execution_time"] == 0.456

    def test_benchmark_result_speedup_calculation_deterministic(self) -> None:
        """Verify speedup calculations are deterministic with known values."""
        cli = BenchmarkMetrics(total_execution_time=10.0, mode="cli")
        streaming = BenchmarkMetrics(total_execution_time=5.0, mode="sdk_streaming")
        non_streaming = BenchmarkMetrics(total_execution_time=2.0, mode="sdk_non_streaming")

        result = BenchmarkResult(
            cli_metrics=cli,
            sdk_streaming_metrics=streaming,
            sdk_non_streaming_metrics=non_streaming,
            prompt="deterministic test",
        )

        summary = result.get_comparison_summary()

        # With explicit values, results should be exactly predictable
        assert summary["speedup"]["streaming_vs_cli"] == 2.0
        assert summary["speedup"]["non_streaming_vs_cli"] == 5.0
        assert summary["speedup"]["non_streaming_vs_streaming"] == 2.5
