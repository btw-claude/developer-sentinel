"""Unit tests for circuit breaker pattern implementation."""

from __future__ import annotations

import os
import threading
import time
from unittest import mock

import pytest

from sentinel.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    get_circuit_breaker,
    get_circuit_breaker_registry,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=60.0,
            half_open_max_calls=5,
            enabled=False,
        )
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 5
        assert config.enabled is False

    def test_from_env_defaults(self) -> None:
        """Test loading config from environment with defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = CircuitBreakerConfig.from_env()
            assert config.failure_threshold == 5
            assert config.recovery_timeout == 30.0
            assert config.half_open_max_calls == 3
            assert config.enabled is True

    def test_from_env_global_overrides(self) -> None:
        """Test loading config from global environment variables."""
        env = {
            "SENTINEL_CIRCUIT_BREAKER_ENABLED": "false",
            "SENTINEL_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "10",
            "SENTINEL_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "60",
            "SENTINEL_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS": "5",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = CircuitBreakerConfig.from_env()
            assert config.failure_threshold == 10
            assert config.recovery_timeout == 60.0
            assert config.half_open_max_calls == 5
            assert config.enabled is False

    def test_from_env_service_specific_overrides(self) -> None:
        """Test loading config with service-specific overrides."""
        env = {
            "SENTINEL_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "5",
            "SENTINEL_JIRA_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "10",
            "SENTINEL_JIRA_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "45",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = CircuitBreakerConfig.from_env("jira")
            assert config.failure_threshold == 10
            assert config.recovery_timeout == 45.0
            # Half open max calls should use global default
            assert config.half_open_max_calls == 3


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    def test_initial_state_is_closed(self) -> None:
        """Test that circuit starts in closed state."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_open_after_failure_threshold(self) -> None:
        """Test circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        # Record failures up to threshold
        for i in range(3):
            assert cb.state == CircuitState.CLOSED
            cb.record_failure(Exception(f"Error {i}"))

        # Should now be open
        assert cb.state == CircuitState.OPEN

    def test_rejects_requests_when_open(self) -> None:
        """Test that requests are rejected when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        cb.record_failure(Exception("Error"))
        assert cb.state == CircuitState.OPEN

        # Should reject request
        assert cb.allow_request() is False
        assert cb.metrics.rejected_calls == 1

    def test_transitions_to_half_open_after_timeout(self) -> None:
        """Test circuit transitions to half-open after recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)

        cb.record_failure(Exception("Error"))
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should now be half-open
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_requests(self) -> None:
        """Test half-open state allows limited requests."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=2,
        )
        cb = CircuitBreaker("test", config)

        cb.record_failure(Exception("Error"))
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Should allow limited requests
        assert cb.allow_request() is True
        assert cb.allow_request() is True
        # Third request should be rejected
        assert cb.allow_request() is False

    def test_half_open_closes_on_success(self) -> None:
        """Test circuit closes after successful calls in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=2,
        )
        cb = CircuitBreaker("test", config)

        cb.record_failure(Exception("Error"))
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Record successful calls
        cb.allow_request()
        cb.record_success()
        cb.allow_request()
        cb.record_success()

        # Should now be closed
        assert cb.state == CircuitState.CLOSED

    def test_half_open_opens_on_failure(self) -> None:
        """Test circuit opens immediately on failure in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
        )
        cb = CircuitBreaker("test", config)

        cb.record_failure(Exception("Error"))
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Any failure should reopen
        cb.allow_request()
        cb.record_failure(Exception("Error"))
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator usage."""

    def test_decorator_allows_successful_calls(self) -> None:
        """Test decorator allows and tracks successful calls."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())

        @cb
        def successful_function() -> str:
            return "success"

        result = successful_function()
        assert result == "success"
        assert cb.metrics.successful_calls == 1

    def test_decorator_tracks_failures(self) -> None:
        """Test decorator tracks failed calls."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)

        @cb
        def failing_function() -> str:
            raise ValueError("Error")

        with pytest.raises(ValueError):
            failing_function()

        assert cb.metrics.failed_calls == 1

    def test_decorator_raises_circuit_breaker_error_when_open(self) -> None:
        """Test decorator raises error when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        @cb
        def some_function() -> str:
            raise ValueError("Error")

        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            some_function()

        # Next call should fail with CircuitBreakerError
        with pytest.raises(CircuitBreakerError) as exc_info:
            some_function()

        assert exc_info.value.service_name == "test"
        assert exc_info.value.state == CircuitState.OPEN


class TestCircuitBreakerContextManager:
    """Tests for circuit breaker context manager usage."""

    def test_context_manager_successful_call(self) -> None:
        """Test context manager with successful operation."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())

        with cb:
            result = "success"

        assert result == "success"
        assert cb.metrics.successful_calls == 1

    def test_context_manager_failed_call(self) -> None:
        """Test context manager with failed operation."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)

        with pytest.raises(ValueError):
            with cb:
                raise ValueError("Error")

        assert cb.metrics.failed_calls == 1

    def test_context_manager_raises_when_open(self) -> None:
        """Test context manager raises error when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        cb.record_failure(Exception("Error"))

        with pytest.raises(CircuitBreakerError):
            with cb:
                pass


class TestCircuitBreakerDisabled:
    """Tests for disabled circuit breaker."""

    def test_disabled_always_allows_requests(self) -> None:
        """Test disabled circuit breaker always allows requests."""
        config = CircuitBreakerConfig(enabled=False, failure_threshold=1)
        cb = CircuitBreaker("test", config)

        # Should always allow even after failures
        cb.record_failure(Exception("Error"))
        cb.record_failure(Exception("Error"))
        assert cb.allow_request() is True

    def test_disabled_does_not_record_metrics(self) -> None:
        """Test disabled circuit breaker doesn't track metrics."""
        config = CircuitBreakerConfig(enabled=False)
        cb = CircuitBreaker("test", config)

        cb.record_failure(Exception("Error"))
        cb.record_success()

        # Metrics should not be updated when disabled
        assert cb.metrics.failed_calls == 0
        assert cb.metrics.successful_calls == 0


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker metrics."""

    def test_metrics_tracking(self) -> None:
        """Test comprehensive metrics tracking."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)

        cb.allow_request()
        cb.record_success()
        cb.allow_request()
        cb.record_failure(Exception("Error"))

        metrics = cb.metrics
        assert metrics.total_calls == 2
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 1
        assert metrics.last_success_time is not None
        assert metrics.last_failure_time is not None

    def test_get_status(self) -> None:
        """Test get_status returns complete state."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        status = cb.get_status()

        assert status["service_name"] == "test"
        assert status["state"] == "closed"
        assert status["enabled"] is True
        assert "config" in status
        assert "metrics" in status


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset functionality."""

    def test_reset_closes_circuit(self) -> None:
        """Test reset returns circuit to closed state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        cb.record_failure(Exception("Error"))
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_reset_clears_failure_count(self) -> None:
        """Test reset clears the failure count."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)

        cb.record_failure(Exception("Error"))
        cb.record_failure(Exception("Error"))
        cb.reset()

        # Should be able to record failures again without opening
        cb.record_failure(Exception("Error"))
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_get_creates_new_breaker(self) -> None:
        """Test registry creates new breaker for unknown service."""
        registry = CircuitBreakerRegistry()
        cb = registry.get("new_service")

        assert cb.service_name == "new_service"

    def test_get_returns_existing_breaker(self) -> None:
        """Test registry returns same breaker for same service."""
        registry = CircuitBreakerRegistry()
        cb1 = registry.get("service")
        cb2 = registry.get("service")

        assert cb1 is cb2

    def test_get_with_custom_config(self) -> None:
        """Test registry accepts custom config for new breaker."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = registry.get("custom", config)

        assert cb.config.failure_threshold == 10

    def test_get_all_status(self) -> None:
        """Test getting status of all registered breakers."""
        registry = CircuitBreakerRegistry()
        registry.get("service1")
        registry.get("service2")

        status = registry.get_all_status()
        assert "service1" in status
        assert "service2" in status

    def test_reset_all(self) -> None:
        """Test resetting all registered breakers."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=1)

        cb1 = registry.get("service1", config)
        cb2 = registry.get("service2", config)

        cb1.record_failure(Exception("Error"))
        cb2.record_failure(Exception("Error"))

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        registry.reset_all()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED


class TestCircuitBreakerThreadSafety:
    """Tests for circuit breaker thread safety."""

    def test_concurrent_access(self) -> None:
        """Test circuit breaker handles concurrent access safely."""
        config = CircuitBreakerConfig(failure_threshold=100)
        cb = CircuitBreaker("test", config)

        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    if cb.allow_request():
                        if threading.current_thread().name.endswith("0"):
                            cb.record_success()
                        else:
                            cb.record_failure(Exception("Error"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, name=f"thread-{i}") for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Verify metrics are consistent
        assert cb.metrics.total_calls > 0


class TestGlobalHelperFunctions:
    """Tests for global helper functions."""

    def test_get_circuit_breaker_registry(self) -> None:
        """Test global registry accessor returns singleton."""
        reg1 = get_circuit_breaker_registry()
        reg2 = get_circuit_breaker_registry()

        assert reg1 is reg2

    def test_get_circuit_breaker(self) -> None:
        """Test global circuit breaker accessor."""
        cb = get_circuit_breaker("test_service")
        assert cb.service_name == "test_service"

        # Should return same instance
        cb2 = get_circuit_breaker("test_service")
        assert cb is cb2
