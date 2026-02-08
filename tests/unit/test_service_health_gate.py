"""Unit tests for ServiceHealthGate component."""

from __future__ import annotations

import os
import threading
import time
from unittest import mock
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sentinel.service_health_gate import (
    GITHUB_PROBE_PATH,
    JIRA_PROBE_PATH,
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
        config = ServiceHealthGateConfig(
            failure_threshold=1,
            initial_probe_interval=0.1,  # 100ms for fast testing
        )
        gate = ServiceHealthGate(config=config)

        # Gate the service
        gate.record_poll_failure("jira", Exception("fail"))
        assert gate.should_poll("jira") is False

        # Not enough time has passed
        assert gate.should_probe("jira") is False

        # Wait for the probe interval
        time.sleep(0.15)
        assert gate.should_probe("jira") is True

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
        config = ServiceHealthGateConfig(
            failure_threshold=1,
            initial_probe_interval=1.0,
            probe_backoff_factor=10.0,
            max_probe_interval=5.0,
        )
        gate = ServiceHealthGate(config=config)

        gate.record_poll_failure("jira", Exception("fail"))

        with gate._lock:
            service = gate._get_or_create_service("jira")
            # probe_count=5: interval = 1.0 * 10.0^5 = 100000 -> capped at 5.0
            service.probe_count = 5
            service.paused_at = time.time() - 6.0
            service.last_check_at = service.paused_at

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
