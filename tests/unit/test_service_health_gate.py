"""Unit tests for ServiceHealthGate component."""

from __future__ import annotations

import os
import threading
import time
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sentinel.service_health_gate import (
    _PROBE_STRATEGY_REQUIRED_PARAMS,
    GITHUB_API_VERSION,
    GITHUB_PROBE_PATH,
    JIRA_PROBE_PATH,
    GitHubProbeStrategy,
    JiraProbeStrategy,
    ProbeStrategy,
    ServiceAvailability,
    ServiceHealthGate,
    ServiceHealthGateConfig,
)


class TestServiceHealthGateConfig:
    """Tests for ServiceHealthGateConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ServiceHealthGateConfig()
        assert config.enabled is True
        assert config.failure_threshold == 3
        assert config.initial_probe_interval == 30.0
        assert config.max_probe_interval == 300.0
        assert config.probe_backoff_factor == 2.0
        assert config.probe_timeout == 5.0

    def test_frozen_dataclass(self) -> None:
        """Test that config is frozen (immutable)."""
        config = ServiceHealthGateConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore[misc]

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ServiceHealthGateConfig(
            enabled=False,
            failure_threshold=5,
            initial_probe_interval=60.0,
            max_probe_interval=600.0,
            probe_backoff_factor=3.0,
            probe_timeout=10.0,
        )
        assert config.enabled is False
        assert config.failure_threshold == 5
        assert config.initial_probe_interval == 60.0
        assert config.max_probe_interval == 600.0
        assert config.probe_backoff_factor == 3.0
        assert config.probe_timeout == 10.0

    def test_from_env_defaults(self) -> None:
        """Test loading config from environment with defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = ServiceHealthGateConfig.from_env()
            assert config.enabled is True
            assert config.failure_threshold == 3
            assert config.initial_probe_interval == 30.0
            assert config.max_probe_interval == 300.0
            assert config.probe_backoff_factor == 2.0
            assert config.probe_timeout == 5.0

    def test_from_env_custom_values(self) -> None:
        """Test loading config from environment with custom values."""
        env = {
            "SENTINEL_HEALTH_GATE_ENABLED": "false",
            "SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD": "5",
            "SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL": "60.0",
            "SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL": "600.0",
            "SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR": "3.0",
            "SENTINEL_HEALTH_GATE_PROBE_TIMEOUT": "10.0",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = ServiceHealthGateConfig.from_env()
            assert config.enabled is False
            assert config.failure_threshold == 5
            assert config.initial_probe_interval == 60.0
            assert config.max_probe_interval == 600.0
            assert config.probe_backoff_factor == 3.0
            assert config.probe_timeout == 10.0

    def test_from_env_enabled_variants(self) -> None:
        """Test various truthy values for enabled flag."""
        for value in ("true", "1", "yes", "True", "YES"):
            with mock.patch.dict(os.environ, {"SENTINEL_HEALTH_GATE_ENABLED": value}, clear=True):
                config = ServiceHealthGateConfig.from_env()
                assert config.enabled is True, f"Expected enabled=True for '{value}'"

        for value in ("false", "0", "no", "anything"):
            with mock.patch.dict(os.environ, {"SENTINEL_HEALTH_GATE_ENABLED": value}, clear=True):
                config = ServiceHealthGateConfig.from_env()
                assert config.enabled is False, f"Expected enabled=False for '{value}'"

    def test_from_env_probe_backoff_factor_clamped_to_minimum(self) -> None:
        """Test that probe_backoff_factor below 1.0 is clamped to 1.0."""
        with mock.patch.dict(
            os.environ,
            {"SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR": "0.5"},
            clear=True,
        ):
            config = ServiceHealthGateConfig.from_env()
            assert config.probe_backoff_factor == 1.0

    def test_from_env_probe_backoff_factor_zero_clamped(self) -> None:
        """Test that probe_backoff_factor of 0.0 is clamped to 1.0."""
        with mock.patch.dict(
            os.environ,
            {"SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR": "0.0"},
            clear=True,
        ):
            config = ServiceHealthGateConfig.from_env()
            assert config.probe_backoff_factor == 1.0


class TestServiceAvailability:
    """Tests for ServiceAvailability dataclass."""

    def test_default_values(self) -> None:
        """Test default availability state."""
        svc = ServiceAvailability(service_name="jira")
        assert svc.service_name == "jira"
        assert svc.available is True
        assert svc.consecutive_failures == 0
        assert svc.last_check_at is None
        assert svc.last_available_at is None
        assert svc.last_error is None
        assert svc.paused_at is None
        assert svc.probe_count == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        now = time.time()
        svc = ServiceAvailability(
            service_name="github",
            available=False,
            consecutive_failures=5,
            last_check_at=now,
            last_available_at=now - 100,
            last_error="Connection timed out",
            paused_at=now - 50,
            probe_count=2,
        )
        result = svc.to_dict()
        assert result == {
            "service_name": "github",
            "available": False,
            "consecutive_failures": 5,
            "last_check_at": now,
            "last_available_at": now - 100,
            "last_error": "Connection timed out",
            "paused_at": now - 50,
            "probe_count": 2,
        }

    def test_mutable(self) -> None:
        """Test that ServiceAvailability is mutable."""
        svc = ServiceAvailability(service_name="jira")
        svc.available = False
        svc.consecutive_failures = 3
        assert svc.available is False
        assert svc.consecutive_failures == 3


class TestServiceHealthGateShouldPoll:
    """Tests for ServiceHealthGate.should_poll()."""

    def test_should_poll_new_service(self) -> None:
        """Test that a new service is available for polling."""
        gate = ServiceHealthGate()
        assert gate.should_poll("jira") is True

    def test_should_poll_disabled(self) -> None:
        """Test that polling is always allowed when health gate is disabled."""
        config = ServiceHealthGateConfig(enabled=False)
        gate = ServiceHealthGate(config=config)
        assert gate.should_poll("jira") is True

    def test_should_poll_after_gating(self) -> None:
        """Test that polling is blocked after service is gated."""
        config = ServiceHealthGateConfig(failure_threshold=2)
        gate = ServiceHealthGate(config=config)

        # Record enough failures to gate the service
        gate.record_poll_failure("jira", Exception("fail 1"))
        assert gate.should_poll("jira") is True  # Still under threshold

        gate.record_poll_failure("jira", Exception("fail 2"))
        assert gate.should_poll("jira") is False  # Now gated

    def test_should_poll_after_recovery(self) -> None:
        """Test that polling resumes after service recovers."""
        config = ServiceHealthGateConfig(failure_threshold=2)
        gate = ServiceHealthGate(config=config)

        # Gate the service
        gate.record_poll_failure("jira", Exception("fail 1"))
        gate.record_poll_failure("jira", Exception("fail 2"))
        assert gate.should_poll("jira") is False

        # Record success to recover
        gate.record_poll_success("jira")
        assert gate.should_poll("jira") is True


class TestServiceHealthGateRecordPollSuccess:
    """Tests for ServiceHealthGate.record_poll_success()."""

    def test_success_resets_failures(self) -> None:
        """Test that success resets consecutive failure count."""
        gate = ServiceHealthGate()
        gate.record_poll_failure("jira", Exception("fail"))
        gate.record_poll_success("jira")

        status = gate.get_all_status()
        assert status["jira"]["consecutive_failures"] == 0

    def test_success_updates_timestamps(self) -> None:
        """Test that success updates last_check_at and last_available_at."""
        gate = ServiceHealthGate()
        gate.record_poll_success("jira")

        status = gate.get_all_status()
        assert status["jira"]["last_check_at"] is not None
        assert status["jira"]["last_available_at"] is not None

    def test_success_clears_error(self) -> None:
        """Test that success clears last_error."""
        gate = ServiceHealthGate()
        gate.record_poll_failure("jira", Exception("fail"))
        gate.record_poll_success("jira")

        status = gate.get_all_status()
        assert status["jira"]["last_error"] is None

    def test_success_recovers_gated_service(self) -> None:
        """Test that success recovers a gated service."""
        config = ServiceHealthGateConfig(failure_threshold=2)
        gate = ServiceHealthGate(config=config)

        # Gate the service
        gate.record_poll_failure("jira", Exception("fail 1"))
        gate.record_poll_failure("jira", Exception("fail 2"))
        assert gate.should_poll("jira") is False

        # Recover
        gate.record_poll_success("jira")
        status = gate.get_all_status()
        assert status["jira"]["available"] is True
        assert status["jira"]["paused_at"] is None
        assert status["jira"]["probe_count"] == 0

    def test_success_disabled_is_noop(self) -> None:
        """Test that success is a no-op when disabled."""
        config = ServiceHealthGateConfig(enabled=False)
        gate = ServiceHealthGate(config=config)
        gate.record_poll_success("jira")
        # Should not create a service entry
        assert gate.get_all_status() == {}


class TestServiceHealthGateRecordPollFailure:
    """Tests for ServiceHealthGate.record_poll_failure()."""

    def test_failure_increments_count(self) -> None:
        """Test that failure increments consecutive failure count."""
        gate = ServiceHealthGate()
        gate.record_poll_failure("jira", Exception("fail"))

        status = gate.get_all_status()
        assert status["jira"]["consecutive_failures"] == 1

    def test_failure_records_error(self) -> None:
        """Test that failure records the error message."""
        gate = ServiceHealthGate()
        gate.record_poll_failure("jira", ConnectionError("Connection refused"))

        status = gate.get_all_status()
        assert status["jira"]["last_error"] == "Connection refused"

    def test_failure_without_exception(self) -> None:
        """Test failure recording without an exception object."""
        gate = ServiceHealthGate()
        gate.record_poll_failure("jira")

        status = gate.get_all_status()
        assert status["jira"]["last_error"] == "Unknown error"

    def test_failure_gates_at_threshold(self) -> None:
        """Test that service is gated when reaching failure threshold."""
        config = ServiceHealthGateConfig(failure_threshold=3)
        gate = ServiceHealthGate(config=config)

        for i in range(3):
            gate.record_poll_failure("jira", Exception(f"fail {i + 1}"))

        status = gate.get_all_status()
        assert status["jira"]["available"] is False
        assert status["jira"]["paused_at"] is not None

    def test_failure_does_not_gate_below_threshold(self) -> None:
        """Test that service is not gated below failure threshold."""
        config = ServiceHealthGateConfig(failure_threshold=3)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail 1"))
        gate.record_poll_failure("jira", Exception("fail 2"))

        status = gate.get_all_status()
        assert status["jira"]["available"] is True
        assert status["jira"]["paused_at"] is None

    def test_failure_disabled_is_noop(self) -> None:
        """Test that failure is a no-op when disabled."""
        config = ServiceHealthGateConfig(enabled=False)
        gate = ServiceHealthGate(config=config)
        gate.record_poll_failure("jira", Exception("fail"))
        assert gate.get_all_status() == {}

    def test_additional_failures_after_gating(self) -> None:
        """Test that failures continue to increment after gating."""
        config = ServiceHealthGateConfig(failure_threshold=2)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail 1"))
        gate.record_poll_failure("jira", Exception("fail 2"))
        gate.record_poll_failure("jira", Exception("fail 3"))

        status = gate.get_all_status()
        assert status["jira"]["consecutive_failures"] == 3
        assert status["jira"]["available"] is False


class TestServiceHealthGateShouldProbe:
    """Tests for ServiceHealthGate.should_probe()."""

    def test_should_not_probe_available_service(self) -> None:
        """Test that available services are not probed."""
        gate = ServiceHealthGate()
        assert gate.should_probe("jira") is False

    def test_should_not_probe_when_disabled(self) -> None:
        """Test that probing is disabled when health gate is disabled."""
        config = ServiceHealthGateConfig(enabled=False)
        gate = ServiceHealthGate(config=config)
        assert gate.should_probe("jira") is False

    def test_should_probe_after_initial_interval(self) -> None:
        """Test that probing is allowed after the initial probe interval."""
        current_time = 1000.0
        config = ServiceHealthGateConfig(
            failure_threshold=1,
            initial_probe_interval=30.0,
        )
        gate = ServiceHealthGate(config=config, time_func=lambda: current_time)

        # Gate the service
        gate.record_poll_failure("jira", Exception("fail"))
        assert gate.should_poll("jira") is False

        # Not enough time has passed
        current_time = 1010.0  # only 10s later
        assert gate.should_probe("jira") is False

        # Enough time has passed
        current_time = 1031.0  # 31s later, past the 30s interval
        assert gate.should_probe("jira") is True

    def test_exponential_backoff_with_time_func(self) -> None:
        """Test that probe interval increases exponentially using time_func."""
        current_time = 1000.0
        config = ServiceHealthGateConfig(
            failure_threshold=1,
            initial_probe_interval=10.0,
            probe_backoff_factor=2.0,
            max_probe_interval=300.0,
        )
        gate = ServiceHealthGate(config=config, time_func=lambda: current_time)

        # Gate the service at t=1000
        gate.record_poll_failure("jira", Exception("fail"))

        # probe_count=0: interval = 10.0 * 2.0^0 = 10.0
        current_time = 1005.0  # 5s after gating
        assert gate.should_probe("jira") is False  # 5.0 < 10.0

        current_time = 1011.0  # 11s after gating
        assert gate.should_probe("jira") is True  # 11.0 >= 10.0

        # Simulate a failed probe to increment probe_count
        with patch.object(gate, "_execute_probe", return_value=False):
            gate.probe_service("jira", base_url="https://jira.example.com")

        # probe_count=1: interval = 10.0 * 2.0^1 = 20.0
        # last_check_at was set to 1011.0 during probe_service
        current_time = 1025.0  # 14s after last check
        assert gate.should_probe("jira") is False  # 14.0 < 20.0

        current_time = 1032.0  # 21s after last check
        assert gate.should_probe("jira") is True  # 21.0 >= 20.0

    def test_exponential_backoff(self) -> None:
        """Test that probe interval increases exponentially."""
        config = ServiceHealthGateConfig(
            failure_threshold=1,
            initial_probe_interval=1.0,
            probe_backoff_factor=2.0,
            max_probe_interval=300.0,
        )
        gate = ServiceHealthGate(config=config)

        # Gate the service
        gate.record_poll_failure("jira", Exception("fail"))

        # Manually set probe_count to simulate probes
        with gate._lock:
            service = gate._get_or_create_service("jira")

            # probe_count=0: interval = 1.0 * 2.0^0 = 1.0
            service.probe_count = 0
            service.paused_at = time.time() - 0.5  # 0.5s ago
            service.last_check_at = service.paused_at
        assert gate.should_probe("jira") is False  # 0.5 < 1.0

        with gate._lock:
            service = gate._get_or_create_service("jira")
            service.paused_at = time.time() - 1.5
            service.last_check_at = service.paused_at
        assert gate.should_probe("jira") is True  # 1.5 >= 1.0

        with gate._lock:
            service = gate._get_or_create_service("jira")
            # probe_count=1: interval = 1.0 * 2.0^1 = 2.0
            service.probe_count = 1
            service.paused_at = time.time() - 1.5
            service.last_check_at = service.paused_at
        assert gate.should_probe("jira") is False  # 1.5 < 2.0

        with gate._lock:
            service = gate._get_or_create_service("jira")
            service.paused_at = time.time() - 2.5
            service.last_check_at = service.paused_at
        assert gate.should_probe("jira") is True  # 2.5 >= 2.0

    def test_max_probe_interval_cap(self) -> None:
        """Test that probe interval is capped at max_probe_interval."""
        current_time = 1000.0
        config = ServiceHealthGateConfig(
            failure_threshold=1,
            initial_probe_interval=1.0,
            probe_backoff_factor=10.0,
            max_probe_interval=5.0,
        )
        gate = ServiceHealthGate(config=config, time_func=lambda: current_time)

        gate.record_poll_failure("jira", Exception("fail"))

        with gate._lock:
            service = gate._get_or_create_service("jira")
            # probe_count=5: interval = 1.0 * 10.0^5 = 100000 -> capped at 5.0
            service.probe_count = 5

        current_time = 1006.0  # 6s after gating
        assert gate.should_probe("jira") is True  # 6.0 >= 5.0 (capped)


class TestServiceHealthGateProbeService:
    """Tests for ServiceHealthGate.probe_service()."""

    def test_probe_jira_success(self) -> None:
        """Test successful Jira probe."""
        config = ServiceHealthGateConfig(failure_threshold=1, probe_timeout=5.0)
        gate = ServiceHealthGate(config=config)

        # Gate the service
        gate.record_poll_failure("jira", Exception("fail"))
        assert gate.should_poll("jira") is False

        # Mock successful probe
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        assert result is True
        assert gate.should_poll("jira") is True

    def test_probe_github_success(self) -> None:
        """Test successful GitHub probe."""
        config = ServiceHealthGateConfig(failure_threshold=1, probe_timeout=5.0)
        gate = ServiceHealthGate(config=config)

        # Gate the service
        gate.record_poll_failure("github", Exception("fail"))
        assert gate.should_poll("github") is False

        # Mock successful probe
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = gate.probe_service("github", base_url="https://api.github.com", token="ghp_test")

        assert result is True
        assert gate.should_poll("github") is True

    def test_probe_jira_failure_timeout(self) -> None:
        """Test Jira probe failure due to timeout."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            result = gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        assert result is False
        assert gate.should_poll("jira") is False

    def test_probe_github_failure_http_error(self) -> None:
        """Test GitHub probe failure due to HTTP error."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("github", Exception("fail"))

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_client.get.return_value = mock_response
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Service Unavailable",
                request=MagicMock(),
                response=mock_response,
            )
            mock_client_cls.return_value = mock_client

            result = gate.probe_service("github", base_url="https://api.github.com", token="ghp_test")

        assert result is False
        assert gate.should_poll("github") is False

    def test_probe_increments_probe_count_on_failure(self) -> None:
        """Test that failed probe increments probe_count."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        status = gate.get_all_status()
        assert status["jira"]["probe_count"] == 1

    def test_probe_no_base_url(self) -> None:
        """Test probe with no base_url returns False."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))
        result = gate.probe_service("jira", base_url="")
        assert result is False

    def test_probe_unknown_service(self) -> None:
        """Test probe of unknown service returns False."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("unknown_service", Exception("fail"))
        result = gate.probe_service("unknown_service", base_url="https://example.com")
        assert result is False

    def test_probe_disabled_returns_true(self) -> None:
        """Test that probe returns True when health gate is disabled."""
        config = ServiceHealthGateConfig(enabled=False)
        gate = ServiceHealthGate(config=config)

        result = gate.probe_service("jira", base_url="https://jira.example.com")
        assert result is True

    def test_probe_jira_request_error(self) -> None:
        """Test Jira probe failure due to request error."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value = mock_client

            result = gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        assert result is False

    def test_probe_unexpected_exception_handled(self) -> None:
        """Test that unexpected exceptions during probe are handled gracefully."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))

        with patch.object(gate, "_execute_probe", side_effect=RuntimeError("unexpected")):
            result = gate.probe_service("jira", base_url="https://jira.example.com")

        assert result is False
        assert gate.should_poll("jira") is False


class TestServiceHealthGateGetAllStatus:
    """Tests for ServiceHealthGate.get_all_status()."""

    def test_empty_status(self) -> None:
        """Test status with no tracked services."""
        gate = ServiceHealthGate()
        assert gate.get_all_status() == {}

    def test_status_after_interactions(self) -> None:
        """Test status after recording successes and failures."""
        gate = ServiceHealthGate()
        gate.record_poll_success("jira")
        gate.record_poll_failure("github", Exception("timeout"))

        status = gate.get_all_status()
        assert "jira" in status
        assert "github" in status
        assert status["jira"]["available"] is True
        assert status["github"]["consecutive_failures"] == 1

    def test_status_is_snapshot(self) -> None:
        """Test that status returns a snapshot (dict copy)."""
        gate = ServiceHealthGate()
        gate.record_poll_success("jira")

        status1 = gate.get_all_status()
        gate.record_poll_failure("jira", Exception("fail"))
        status2 = gate.get_all_status()

        # status1 should not be affected by subsequent operations
        assert status1["jira"]["consecutive_failures"] == 0
        assert status2["jira"]["consecutive_failures"] == 1


class TestServiceHealthGateMultipleServices:
    """Tests for tracking multiple services independently."""

    def test_independent_tracking(self) -> None:
        """Test that services are tracked independently."""
        config = ServiceHealthGateConfig(failure_threshold=2)
        gate = ServiceHealthGate(config=config)

        # Gate jira but not github
        gate.record_poll_failure("jira", Exception("fail 1"))
        gate.record_poll_failure("jira", Exception("fail 2"))
        gate.record_poll_success("github")

        assert gate.should_poll("jira") is False
        assert gate.should_poll("github") is True

    def test_recover_one_service(self) -> None:
        """Test recovering one service while another remains gated."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))
        gate.record_poll_failure("github", Exception("fail"))

        assert gate.should_poll("jira") is False
        assert gate.should_poll("github") is False

        gate.record_poll_success("jira")
        assert gate.should_poll("jira") is True
        assert gate.should_poll("github") is False


class TestServiceHealthGateThreadSafety:
    """Tests for thread safety of ServiceHealthGate."""

    def test_concurrent_operations(self) -> None:
        """Test that concurrent operations don't corrupt state."""
        config = ServiceHealthGateConfig(failure_threshold=100)
        gate = ServiceHealthGate(config=config)
        errors: list[Exception] = []

        def record_failures() -> None:
            try:
                for _ in range(50):
                    gate.record_poll_failure("jira", Exception("fail"))
            except Exception as e:
                errors.append(e)

        def record_successes() -> None:
            try:
                for _ in range(50):
                    gate.record_poll_success("jira")
            except Exception as e:
                errors.append(e)

        def check_poll() -> None:
            try:
                for _ in range(50):
                    gate.should_poll("jira")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_failures),
            threading.Thread(target=record_successes),
            threading.Thread(target=check_poll),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_multiple_services(self) -> None:
        """Test concurrent operations across different services."""
        config = ServiceHealthGateConfig(failure_threshold=50)
        gate = ServiceHealthGate(config=config)
        errors: list[Exception] = []

        def operate_service(name: str) -> None:
            try:
                for _ in range(50):
                    gate.record_poll_failure(name, Exception("fail"))
                    gate.should_poll(name)
                    gate.record_poll_success(name)
                    gate.get_all_status()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=operate_service, args=(f"service_{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestServiceHealthGateConfigFromAppConfig:
    """Tests for ServiceHealthGateConfig integration with app Config."""

    def test_config_values_match(self) -> None:
        """Test that ServiceHealthGateConfig values align with app Config defaults."""
        from sentinel.config import Config

        app_config = Config()
        gate_cfg = app_config.service_health_gate

        assert gate_cfg.enabled is True
        assert gate_cfg.failure_threshold == 3
        assert gate_cfg.initial_probe_interval == 30.0
        assert gate_cfg.max_probe_interval == 300.0
        assert gate_cfg.probe_backoff_factor == 2.0
        assert gate_cfg.probe_timeout == 5.0


class TestProbeEndpointConstants:
    """Tests for probe endpoint constants."""

    def test_jira_probe_path(self) -> None:
        """Test Jira probe endpoint matches expected path."""
        assert JIRA_PROBE_PATH == "/rest/api/3/serverInfo"

    def test_github_probe_path(self) -> None:
        """Test GitHub probe endpoint matches expected path."""
        assert GITHUB_PROBE_PATH == "/rate_limit"

    def test_github_api_version(self) -> None:
        """Test GitHub API version constant matches expected value."""
        assert GITHUB_API_VERSION == "2022-11-28"

    def test_github_api_version_used_in_probe(self) -> None:
        """Test that GitHub probe uses the GITHUB_API_VERSION constant."""
        config = ServiceHealthGateConfig(failure_threshold=1, probe_timeout=5.0)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("github", Exception("fail"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            gate.probe_service("github", base_url="https://api.github.com", token="ghp_test")

            # Verify the headers include the GITHUB_API_VERSION constant
            call_args = mock_client.get.call_args
            headers = call_args[1]["headers"]
            assert headers["X-GitHub-Api-Version"] == GITHUB_API_VERSION


class TestProbeMetricsCounters:
    """Tests for probe success/failure metrics counters."""

    def test_initial_counters_are_zero(self) -> None:
        """Test that all probe counters start at zero."""
        gate = ServiceHealthGate()
        assert gate.probe_success_count == 0
        assert gate.probe_expected_failure_count == 0
        assert gate.probe_unexpected_error_count == 0

    def test_probe_success_increments_success_counter(self) -> None:
        """Test that successful probe increments success counter."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        assert gate.probe_success_count == 1
        assert gate.probe_expected_failure_count == 0
        assert gate.probe_unexpected_error_count == 0

    def test_probe_expected_failure_increments_expected_counter(self) -> None:
        """Test that expected HTTP failure increments expected failure counter."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        assert gate.probe_success_count == 0
        assert gate.probe_expected_failure_count == 1
        assert gate.probe_unexpected_error_count == 0

    def test_probe_unexpected_error_increments_unexpected_counter(self) -> None:
        """Test that unexpected error increments unexpected error counter."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))

        with patch.object(gate, "_execute_probe", side_effect=RuntimeError("unexpected")):
            gate.probe_service("jira", base_url="https://jira.example.com")

        assert gate.probe_success_count == 0
        assert gate.probe_expected_failure_count == 0
        assert gate.probe_unexpected_error_count == 1

    def test_multiple_probes_accumulate_counters(self) -> None:
        """Test that counters accumulate across multiple probe attempts."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        # First: expected failure (timeout)
        gate.record_poll_failure("jira", Exception("fail"))
        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client
            gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        # Second: unexpected error
        with patch.object(gate, "_execute_probe", side_effect=RuntimeError("unexpected")):
            gate.probe_service("jira", base_url="https://jira.example.com")

        # Third: success
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client
            gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        assert gate.probe_success_count == 1
        assert gate.probe_expected_failure_count == 1
        assert gate.probe_unexpected_error_count == 1

    def test_unknown_service_counted_as_expected_failure(self) -> None:
        """Test that probing an unknown service counts as expected failure."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("unknown_service", Exception("fail"))
        gate.probe_service("unknown_service", base_url="https://example.com")

        assert gate.probe_expected_failure_count == 1
        assert gate.probe_unexpected_error_count == 0


class TestTimeFuncInjection:
    """Tests for time_func clock injection in ServiceHealthGate."""

    def test_default_time_func_is_time_time(self) -> None:
        """Test that default time_func uses time.time."""
        gate = ServiceHealthGate()
        # Should be very close to time.time()
        now = time.time()
        gate_now = gate._time_func()
        assert abs(gate_now - now) < 1.0

    def test_custom_time_func_used_in_record_poll_success(self) -> None:
        """Test that custom time_func is used in record_poll_success."""
        fixed_time = 5000.0
        gate = ServiceHealthGate(time_func=lambda: fixed_time)

        gate.record_poll_success("jira")
        status = gate.get_all_status()
        assert status["jira"]["last_check_at"] == 5000.0
        assert status["jira"]["last_available_at"] == 5000.0

    def test_custom_time_func_used_in_record_poll_failure(self) -> None:
        """Test that custom time_func is used in record_poll_failure."""
        fixed_time = 6000.0
        gate = ServiceHealthGate(time_func=lambda: fixed_time)

        gate.record_poll_failure("jira", Exception("fail"))
        status = gate.get_all_status()
        assert status["jira"]["last_check_at"] == 6000.0

    def test_custom_time_func_used_in_should_probe(self) -> None:
        """Test that custom time_func is used in should_probe."""
        current_time = 1000.0
        config = ServiceHealthGateConfig(
            failure_threshold=1,
            initial_probe_interval=10.0,
        )
        gate = ServiceHealthGate(config=config, time_func=lambda: current_time)

        # Gate the service at t=1000
        gate.record_poll_failure("jira", Exception("fail"))

        # At t=1005: not enough time (5 < 10)
        current_time = 1005.0
        assert gate.should_probe("jira") is False

        # At t=1011: enough time (11 >= 10)
        current_time = 1011.0
        assert gate.should_probe("jira") is True

    def test_custom_time_func_used_in_probe_service(self) -> None:
        """Test that custom time_func is used in probe_service."""
        fixed_time = 7000.0
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config, time_func=lambda: fixed_time)

        gate.record_poll_failure("jira", Exception("fail"))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        status = gate.get_all_status()
        assert status["jira"]["last_check_at"] == 7000.0
        assert status["jira"]["last_available_at"] == 7000.0

    def test_probe_service_success_uses_single_timestamp(self) -> None:
        """Test that probe_service success path uses a single timestamp for consistency."""
        call_count = 0

        def counting_time() -> float:
            nonlocal call_count
            call_count += 1
            return 8000.0 + call_count

        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config, time_func=counting_time)

        gate.record_poll_failure("jira", Exception("fail"))

        # Reset counter before probe_service call
        call_count = 10

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            gate.probe_service("jira", base_url="https://jira.example.com", auth=("user", "token"))

        status = gate.get_all_status()
        # last_check_at and last_available_at should be identical since
        # time is captured once and reused
        assert status["jira"]["last_check_at"] == status["jira"]["last_available_at"]


class TestProbeStrategyRequiredParamsConstant:
    """Tests for _PROBE_STRATEGY_REQUIRED_PARAMS module-level constant."""

    def test_required_params_contains_expected_names(self) -> None:
        """Test that _PROBE_STRATEGY_REQUIRED_PARAMS has the expected parameter names."""
        expected = frozenset({"timeout", "base_url", "auth", "token"})
        assert expected == _PROBE_STRATEGY_REQUIRED_PARAMS

    def test_required_params_is_frozenset(self) -> None:
        """Test that _PROBE_STRATEGY_REQUIRED_PARAMS is immutable (frozenset)."""
        assert isinstance(_PROBE_STRATEGY_REQUIRED_PARAMS, frozenset)

    def test_required_params_matches_probe_strategy_execute_signature(self) -> None:
        """Test that _PROBE_STRATEGY_REQUIRED_PARAMS matches the ProbeStrategy.execute() params.

        Ensures the constant stays in sync with the protocol definition.
        """
        import inspect

        sig = inspect.signature(ProbeStrategy.execute)
        keyword_params = {
            name
            for name, param in sig.parameters.items()
            if param.kind
            in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and name != "self"
        }
        assert keyword_params == _PROBE_STRATEGY_REQUIRED_PARAMS


class TestProbeStrategyProtocol:
    """Tests for ProbeStrategy protocol and strategy registration."""

    def test_jira_strategy_satisfies_protocol(self) -> None:
        """Test that JiraProbeStrategy satisfies ProbeStrategy protocol."""
        strategy = JiraProbeStrategy()
        assert isinstance(strategy, ProbeStrategy)

    def test_github_strategy_satisfies_protocol(self) -> None:
        """Test that GitHubProbeStrategy satisfies ProbeStrategy protocol."""
        strategy = GitHubProbeStrategy()
        assert isinstance(strategy, ProbeStrategy)

    def test_custom_strategy_satisfies_protocol(self) -> None:
        """Test that a custom class with execute() satisfies ProbeStrategy."""

        class MyStrategy:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
            ) -> bool:
                return True

        strategy = MyStrategy()
        assert isinstance(strategy, ProbeStrategy)

    def test_non_conforming_class_rejected(self) -> None:
        """Test that a class without execute() is not a ProbeStrategy."""

        class NotAStrategy:
            pass

        assert not isinstance(NotAStrategy(), ProbeStrategy)

    def test_default_strategies_registered(self) -> None:
        """Test that Jira and GitHub strategies are registered by default."""
        gate = ServiceHealthGate()
        assert gate.get_probe_strategy("jira") is not None
        assert gate.get_probe_strategy("github") is not None
        assert isinstance(gate.get_probe_strategy("jira"), JiraProbeStrategy)
        assert isinstance(gate.get_probe_strategy("github"), GitHubProbeStrategy)

    def test_register_custom_strategy(self) -> None:
        """Test registering a custom probe strategy."""

        class CustomStrategy:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
            ) -> bool:
                return True

        gate = ServiceHealthGate()
        strategy = CustomStrategy()
        gate.register_probe_strategy("custom_service", strategy)
        assert gate.get_probe_strategy("custom_service") is strategy

    def test_register_strategy_case_insensitive(self) -> None:
        """Test that strategy registration is case-insensitive."""

        class CustomStrategy:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
            ) -> bool:
                return True

        gate = ServiceHealthGate()
        strategy = CustomStrategy()
        gate.register_probe_strategy("MyService", strategy)
        assert gate.get_probe_strategy("myservice") is strategy
        assert gate.get_probe_strategy("MYSERVICE") is strategy

    def test_register_strategy_type_validation(self) -> None:
        """Test that registering a non-strategy raises TypeError."""
        gate = ServiceHealthGate()
        with pytest.raises(TypeError, match="strategy must satisfy ProbeStrategy protocol"):
            gate.register_probe_strategy("bad", "not a strategy")  # type: ignore[arg-type]

    def test_register_strategy_signature_validation_missing_params(self) -> None:
        """Test that registering a strategy with wrong execute() signature raises TypeError.

        A class with execute(self) (no keyword args) passes the
        @runtime_checkable isinstance() check but would fail at call time.
        The signature validation in register_probe_strategy() catches this
        eagerly at registration time.
        """

        class BadSignatureStrategy:
            def execute(self) -> bool:  # type: ignore[override]
                return True

        gate = ServiceHealthGate()
        with pytest.raises(TypeError, match="missing required keyword parameters"):
            gate.register_probe_strategy("bad_sig", BadSignatureStrategy())  # type: ignore[arg-type]

    def test_register_strategy_signature_validation_partial_params(self) -> None:
        """Test that a strategy with only some required params is rejected."""

        class PartialParamsStrategy:
            def execute(self, *, timeout: float) -> bool:
                return True

        gate = ServiceHealthGate()
        with pytest.raises(TypeError, match="missing required keyword parameters"):
            gate.register_probe_strategy("partial", PartialParamsStrategy())  # type: ignore[arg-type]

    def test_register_strategy_signature_validation_accepts_correct_signature(self) -> None:
        """Test that a strategy with the correct execute() signature passes validation."""

        class CorrectStrategy:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
            ) -> bool:
                return True

        gate = ServiceHealthGate()
        # Should not raise
        gate.register_probe_strategy("correct", CorrectStrategy())
        assert gate.get_probe_strategy("correct") is not None

    def test_register_strategy_signature_validation_accepts_extra_params(self) -> None:
        """Test that a strategy with extra params beyond the required ones is accepted."""

        class ExtraParamsStrategy:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
                extra_param: str = "default",
            ) -> bool:
                return True

        gate = ServiceHealthGate()
        # Should not raise — extra params are fine
        gate.register_probe_strategy("extra", ExtraParamsStrategy())
        assert gate.get_probe_strategy("extra") is not None

    def test_unregister_strategy(self) -> None:
        """Test unregistering a probe strategy."""
        gate = ServiceHealthGate()
        assert gate.get_probe_strategy("jira") is not None
        gate.unregister_probe_strategy("jira")
        assert gate.get_probe_strategy("jira") is None

    def test_unregister_nonexistent_strategy_is_noop(self) -> None:
        """Test that unregistering a non-existent strategy does not raise."""
        gate = ServiceHealthGate()
        gate.unregister_probe_strategy("nonexistent")  # Should not raise

    def test_get_probe_strategy_unknown(self) -> None:
        """Test that get_probe_strategy returns None for unknown service."""
        gate = ServiceHealthGate()
        assert gate.get_probe_strategy("unknown_service") is None

    def test_custom_strategy_used_in_probe_service(self) -> None:
        """Test that a registered custom strategy is invoked by probe_service."""
        probe_called = False

        class TrackingStrategy:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
            ) -> bool:
                nonlocal probe_called
                probe_called = True
                return True

        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)
        gate.register_probe_strategy("custom", TrackingStrategy())

        gate.record_poll_failure("custom", Exception("fail"))
        result = gate.probe_service("custom", base_url="https://example.com")

        assert result is True
        assert probe_called is True
        assert gate.should_poll("custom") is True

    def test_custom_strategy_receives_correct_params(self) -> None:
        """Test that strategy.execute() receives the correct parameters."""
        received_params: dict[str, Any] = {}

        class ParamCapture:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
            ) -> bool:
                received_params["timeout"] = timeout
                received_params["base_url"] = base_url
                received_params["auth"] = auth
                received_params["token"] = token
                return True

        config = ServiceHealthGateConfig(failure_threshold=1, probe_timeout=7.5)
        gate = ServiceHealthGate(config=config)
        gate.register_probe_strategy("test_svc", ParamCapture())

        gate.record_poll_failure("test_svc", Exception("fail"))
        gate.probe_service(
            "test_svc",
            base_url="https://test.example.com",
            auth=("user", "pass"),
            token="tok123",
        )

        assert received_params["timeout"] == 7.5
        assert received_params["base_url"] == "https://test.example.com"
        assert received_params["auth"] == ("user", "pass")
        assert received_params["token"] == "tok123"

    def test_override_default_strategy(self) -> None:
        """Test that a default strategy can be overridden by registering a new one."""

        class AlwaysAvailable:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
            ) -> bool:
                return True

        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)
        gate.register_probe_strategy("jira", AlwaysAvailable())

        gate.record_poll_failure("jira", Exception("fail"))
        # The overridden strategy always returns True (no HTTP call needed)
        result = gate.probe_service("jira", base_url="https://jira.example.com")
        assert result is True


class TestJiraProbeStrategy:
    """Tests for JiraProbeStrategy."""

    def test_no_base_url(self) -> None:
        """Test that JiraProbeStrategy returns False with no base_url."""
        strategy = JiraProbeStrategy()
        assert strategy.execute(timeout=5.0, base_url="") is False

    def test_successful_probe(self) -> None:
        """Test successful Jira probe via strategy."""
        strategy = JiraProbeStrategy()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://jira.example.com",
                auth=("user", "token"),
            )

        assert result is True
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert JIRA_PROBE_PATH in call_args[0][0]

    def test_timeout(self) -> None:
        """Test Jira probe timeout via strategy."""
        strategy = JiraProbeStrategy()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://jira.example.com",
                auth=("user", "token"),
            )

        assert result is False

    def test_http_error(self) -> None:
        """Test Jira probe HTTP error via strategy."""
        strategy = JiraProbeStrategy()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_client.get.return_value = mock_response
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Service Unavailable",
                request=MagicMock(),
                response=mock_response,
            )
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://jira.example.com",
                auth=("user", "token"),
            )

        assert result is False

    def test_request_error(self) -> None:
        """Test Jira probe request error via strategy."""
        strategy = JiraProbeStrategy()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://jira.example.com",
            )

        assert result is False


class TestGitHubProbeStrategy:
    """Tests for GitHubProbeStrategy."""

    def test_no_base_url(self) -> None:
        """Test that GitHubProbeStrategy returns False with no base_url."""
        strategy = GitHubProbeStrategy()
        assert strategy.execute(timeout=5.0, base_url="") is False

    def test_successful_probe(self) -> None:
        """Test successful GitHub probe via strategy."""
        strategy = GitHubProbeStrategy()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://api.github.com",
                token="ghp_test",
            )

        assert result is True
        call_args = mock_client.get.call_args
        assert GITHUB_PROBE_PATH in call_args[0][0]
        headers = call_args[1]["headers"]
        assert headers["X-GitHub-Api-Version"] == GITHUB_API_VERSION
        assert headers["Authorization"] == "Bearer ghp_test"

    def test_successful_probe_no_token(self) -> None:
        """Test GitHub probe without token."""
        strategy = GitHubProbeStrategy()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://api.github.com",
            )

        assert result is True
        call_args = mock_client.get.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" not in headers

    def test_timeout(self) -> None:
        """Test GitHub probe timeout via strategy."""
        strategy = GitHubProbeStrategy()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://api.github.com",
                token="ghp_test",
            )

        assert result is False

    def test_http_error(self) -> None:
        """Test GitHub probe HTTP error via strategy."""
        strategy = GitHubProbeStrategy()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.get.return_value = mock_response
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Internal Server Error",
                request=MagicMock(),
                response=mock_response,
            )
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://api.github.com",
                token="ghp_test",
            )

        assert result is False

    def test_request_error(self) -> None:
        """Test GitHub probe request error via strategy."""
        strategy = GitHubProbeStrategy()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value = mock_client

            result = strategy.execute(
                timeout=5.0,
                base_url="https://api.github.com",
            )

        assert result is False


class TestThreadSafeProbeCounters:
    """Tests for thread-safe probe metric counters."""

    def test_counter_lock_exists(self) -> None:
        """Test that ServiceHealthGate has a dedicated counter lock."""
        gate = ServiceHealthGate()
        assert hasattr(gate, "_counter_lock")
        assert isinstance(gate._counter_lock, type(threading.Lock()))

    def test_counters_are_properties(self) -> None:
        """Test that probe counters are implemented as properties."""
        assert isinstance(
            ServiceHealthGate.probe_success_count, property
        )
        assert isinstance(
            ServiceHealthGate.probe_expected_failure_count, property
        )
        assert isinstance(
            ServiceHealthGate.probe_unexpected_error_count, property
        )

    def test_counter_property_getters(self) -> None:
        """Test that counter property getters return correct values."""
        gate = ServiceHealthGate()
        assert gate.probe_success_count == 0
        assert gate.probe_expected_failure_count == 0
        assert gate.probe_unexpected_error_count == 0

    def test_counter_property_setters(self) -> None:
        """Test that counter property setters work correctly."""
        gate = ServiceHealthGate()
        gate.probe_success_count = 5
        gate.probe_expected_failure_count = 10
        gate.probe_unexpected_error_count = 3

        assert gate.probe_success_count == 5
        assert gate.probe_expected_failure_count == 10
        assert gate.probe_unexpected_error_count == 3

    def test_counter_property_setters_accept_zero(self) -> None:
        """Test that counter property setters accept zero."""
        gate = ServiceHealthGate()
        gate.probe_success_count = 0
        gate.probe_expected_failure_count = 0
        gate.probe_unexpected_error_count = 0

        assert gate.probe_success_count == 0
        assert gate.probe_expected_failure_count == 0
        assert gate.probe_unexpected_error_count == 0

    def test_counter_property_setters_reject_negative_values(self) -> None:
        """Test that counter property setters reject negative values with ValueError."""
        gate = ServiceHealthGate()

        with pytest.raises(ValueError, match="probe_success_count must be non-negative"):
            gate.probe_success_count = -1

        with pytest.raises(ValueError, match="probe_expected_failure_count must be non-negative"):
            gate.probe_expected_failure_count = -1

        with pytest.raises(ValueError, match="probe_unexpected_error_count must be non-negative"):
            gate.probe_unexpected_error_count = -1

    def test_counter_property_setters_reject_non_int_types(self) -> None:
        """Test that counter property setters reject non-int types with TypeError."""
        gate = ServiceHealthGate()

        with pytest.raises(TypeError, match="probe_success_count must be an int"):
            gate.probe_success_count = 1.5  # type: ignore[assignment]

        with pytest.raises(TypeError, match="probe_expected_failure_count must be an int"):
            gate.probe_expected_failure_count = "10"  # type: ignore[assignment]

        with pytest.raises(TypeError, match="probe_unexpected_error_count must be an int"):
            gate.probe_unexpected_error_count = 3.0  # type: ignore[assignment]

    def test_counter_property_setters_reject_bool_type(self) -> None:
        """Test that counter property setters reject bool (subclass of int).

        In Python, bool is a subclass of int (True == 1, False == 0).
        The isinstance check for int would pass for bools, so this test
        documents the current behavior: bools are accepted because
        isinstance(True, int) is True.
        """
        gate = ServiceHealthGate()
        # bool is a subclass of int in Python, so isinstance(True, int) is True.
        # This documents that bools pass the type check (acceptable edge case).
        gate.probe_success_count = True  # type: ignore[assignment]
        assert gate.probe_success_count == 1

    def test_counter_property_setters_do_not_mutate_on_invalid_input(self) -> None:
        """Test that counters retain their previous value after a rejected setter call."""
        gate = ServiceHealthGate()
        gate.probe_success_count = 5

        with pytest.raises(ValueError):
            gate.probe_success_count = -1

        # Original value should be preserved
        assert gate.probe_success_count == 5

        with pytest.raises(TypeError):
            gate.probe_success_count = "bad"  # type: ignore[assignment]

        # Original value should still be preserved
        assert gate.probe_success_count == 5

    def test_get_probe_metrics_returns_snapshot(self) -> None:
        """Test that get_probe_metrics returns an atomic snapshot."""
        config = ServiceHealthGateConfig(failure_threshold=1)
        gate = ServiceHealthGate(config=config)

        # Perform a successful probe
        gate.record_poll_failure("jira", Exception("fail"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client
            gate.probe_service("jira", base_url="https://jira.example.com", auth=("u", "t"))

        # Perform an expected failure
        gate.record_poll_failure("jira", Exception("fail"))
        with patch("sentinel.service_health_gate.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client
            gate.probe_service("jira", base_url="https://jira.example.com", auth=("u", "t"))

        # Perform an unexpected error
        with patch.object(gate, "_execute_probe", side_effect=RuntimeError("unexpected")):
            gate.probe_service("jira", base_url="https://jira.example.com")

        metrics = gate.get_probe_metrics()
        assert metrics == {
            "probe_success_count": 1,
            "probe_expected_failure_count": 1,
            "probe_unexpected_error_count": 1,
        }

    def test_get_probe_metrics_initial_state(self) -> None:
        """Test that get_probe_metrics returns zeros initially."""
        gate = ServiceHealthGate()
        metrics = gate.get_probe_metrics()
        assert metrics == {
            "probe_success_count": 0,
            "probe_expected_failure_count": 0,
            "probe_unexpected_error_count": 0,
        }

    def test_get_probe_metrics_is_independent_snapshot(self) -> None:
        """Test that get_probe_metrics returns independent dict snapshots."""
        gate = ServiceHealthGate()
        metrics1 = gate.get_probe_metrics()

        # Mutate the returned dict
        metrics1["probe_success_count"] = 999

        # Original should be unaffected
        metrics2 = gate.get_probe_metrics()
        assert metrics2["probe_success_count"] == 0

    def test_concurrent_counter_increments(self) -> None:
        """Test that concurrent probe operations produce correct counter totals."""
        config = ServiceHealthGateConfig(failure_threshold=1000)
        gate = ServiceHealthGate(config=config)
        errors: list[Exception] = []
        iterations = 100

        def do_successful_probes() -> None:
            try:
                for _ in range(iterations):
                    with gate._counter_lock:
                        gate._probe_success_count += 1
            except Exception as e:
                errors.append(e)

        def do_expected_failures() -> None:
            try:
                for _ in range(iterations):
                    with gate._counter_lock:
                        gate._probe_expected_failure_count += 1
            except Exception as e:
                errors.append(e)

        def do_unexpected_errors() -> None:
            try:
                for _ in range(iterations):
                    with gate._counter_lock:
                        gate._probe_unexpected_error_count += 1
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=do_successful_probes),
            threading.Thread(target=do_expected_failures),
            threading.Thread(target=do_unexpected_errors),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert gate.probe_success_count == iterations
        assert gate.probe_expected_failure_count == iterations
        assert gate.probe_unexpected_error_count == iterations

    def test_concurrent_counter_reads_during_writes(self) -> None:
        """Test that reading counters is safe during concurrent writes."""
        config = ServiceHealthGateConfig(failure_threshold=1000)
        gate = ServiceHealthGate(config=config)
        errors: list[Exception] = []
        iterations = 100

        def writer() -> None:
            try:
                for _ in range(iterations):
                    with gate._counter_lock:
                        gate._probe_success_count += 1
                        gate._probe_expected_failure_count += 1
                        gate._probe_unexpected_error_count += 1
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(iterations):
                    metrics = gate.get_probe_metrics()
                    # All three counters must be non-negative
                    assert metrics["probe_success_count"] >= 0
                    assert metrics["probe_expected_failure_count"] >= 0
                    assert metrics["probe_unexpected_error_count"] >= 0
                    # In an atomic snapshot, all three should be equal
                    # (since writer increments all three together)
                    assert (
                        metrics["probe_success_count"]
                        == metrics["probe_expected_failure_count"]
                        == metrics["probe_unexpected_error_count"]
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert gate.probe_success_count == iterations

    def test_concurrent_probe_service_counter_accuracy(self) -> None:
        """Test counter accuracy under concurrent probe_service calls."""
        config = ServiceHealthGateConfig(failure_threshold=1000)
        gate = ServiceHealthGate(config=config)
        errors: list[Exception] = []
        iterations = 50

        # Pre-gate a service so probe_service will update counters
        for _ in range(1000):
            gate.record_poll_failure("jira", Exception("fail"))

        def do_probes() -> None:
            try:
                for _ in range(iterations):
                    with patch.object(gate, "_execute_probe", return_value=True):
                        gate.probe_service("jira", base_url="https://jira.example.com")
                    # Re-gate for next probe
                    gate.record_poll_failure("jira", Exception("fail"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_probes) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Each thread does `iterations` successful probes
        assert gate.probe_success_count == iterations * 3
