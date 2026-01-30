"""Fault injection tests for circuit breaker and rate limiter behavior.

These tests simulate failures and verify that the circuit breaker and rate
limiter properly handle error conditions under real-world scenarios.

The tests use HTTP mocking (respx) to simulate various failure conditions
while testing the actual circuit breaker and rate limiter implementations.

Run with: pytest tests/integration -m integration
Skip with: pytest -m "not integration"
"""

from __future__ import annotations

import contextlib
import time
from unittest.mock import MagicMock

import httpx
import pytest

from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from sentinel.github_poller import GitHubClientError
from sentinel.github_rest_client import (
    GitHubRateLimitError,
    GitHubRestClient,
    GitHubRetryConfig,
    _calculate_backoff_delay,
    _execute_with_retry,
)
from sentinel.poller import JiraClientError
from sentinel.rest_clients import JiraRestClient, JiraRestTagClient, RetryConfig
from sentinel.rest_clients import _calculate_backoff_delay as jira_calculate_backoff_delay


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions under failure conditions."""

    @pytest.mark.integration
    def test_circuit_opens_after_threshold_failures(self) -> None:
        """Circuit should open after reaching failure threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            half_open_max_calls=2,
        )
        breaker = CircuitBreaker(service_name="test", config=config)

        # Initially closed
        assert breaker.state == CircuitState.CLOSED

        # Record failures up to threshold
        for _ in range(3):
            breaker.record_failure(Exception("simulated failure"))

        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN
        assert breaker.metrics.failed_calls == 3
        assert breaker.metrics.state_changes == 1

    @pytest.mark.integration
    def test_open_circuit_rejects_requests(self) -> None:
        """Open circuit should reject requests without calling the service."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=60.0,
            half_open_max_calls=1,
        )
        breaker = CircuitBreaker(service_name="test", config=config)

        # Force circuit open
        for _ in range(2):
            breaker.record_failure(Exception("failure"))

        assert breaker.state == CircuitState.OPEN

        # Requests should be rejected
        assert not breaker.allow_request()
        assert breaker.metrics.rejected_calls == 1

    @pytest.mark.integration
    def test_circuit_transitions_to_half_open(self) -> None:
        """Circuit should transition to half-open after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # Very short for testing
            half_open_max_calls=2,
        )
        breaker = CircuitBreaker(service_name="test", config=config)

        # Force circuit open
        for _ in range(2):
            breaker.record_failure(Exception("failure"))

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next request should be allowed (half-open state)
        assert breaker.allow_request()
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.integration
    def test_half_open_closes_on_success(self) -> None:
        """Half-open circuit should close after successful call."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=1,
        )
        breaker = CircuitBreaker(service_name="test", config=config)

        # Force circuit open
        for _ in range(2):
            breaker.record_failure(Exception("failure"))

        # Wait for recovery timeout
        time.sleep(0.15)
        breaker.allow_request()  # Transition to half-open

        assert breaker.state == CircuitState.HALF_OPEN

        # Record success
        breaker.record_success()

        # Circuit should close
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.integration
    def test_half_open_reopens_on_failure(self) -> None:
        """Half-open circuit should reopen after failure."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2,
        )
        breaker = CircuitBreaker(service_name="test", config=config)

        # Force circuit open
        for _ in range(2):
            breaker.record_failure(Exception("failure"))

        # Wait for recovery timeout
        time.sleep(0.15)
        breaker.allow_request()  # Transition to half-open

        assert breaker.state == CircuitState.HALF_OPEN

        # Record failure in half-open
        breaker.record_failure(Exception("failure during recovery"))

        # Circuit should reopen
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerWithJiraClient:
    """Tests for circuit breaker integration with JiraRestClient."""

    @pytest.mark.integration
    def test_jira_client_circuit_breaker_opens_on_failures(self) -> None:
        """JiraRestClient circuit breaker should open after consecutive failures."""
        # Create client with custom circuit breaker
        breaker = CircuitBreaker(
            service_name="jira_test",
            config=CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=60.0,
                half_open_max_calls=1,
            ),
        )
        client = JiraRestClient(
            base_url="https://invalid-jira-instance.example.com",
            email="test@example.com",
            api_token="fake-token",
            circuit_breaker=breaker,
            retry_config=RetryConfig(max_retries=0),  # No retries for faster test
        )

        # Make failing requests
        for _ in range(2):
            with contextlib.suppress(JiraClientError):
                client.search_issues("project = TEST")

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN

        # Next request should fail immediately with circuit breaker error
        with pytest.raises(JiraClientError) as exc_info:
            client.search_issues("project = TEST")

        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.integration
    def test_jira_tag_client_shares_circuit_state(self) -> None:
        """JiraRestTagClient should share circuit breaker state."""
        # Create shared circuit breaker
        breaker = CircuitBreaker(
            service_name="jira_shared",
            config=CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=60.0,
                half_open_max_calls=1,
            ),
        )

        # Both clients share the same circuit breaker
        rest_client = JiraRestClient(
            base_url="https://invalid.example.com",
            email="test@example.com",
            api_token="fake-token",
            circuit_breaker=breaker,
            retry_config=RetryConfig(max_retries=0),
        )

        # Create tag client sharing the same circuit breaker
        # (verifying the breaker is shared even if tag_client isn't used directly)
        _tag_client = JiraRestTagClient(
            base_url="https://invalid.example.com",
            email="test@example.com",
            api_token="fake-token",
            circuit_breaker=breaker,
            retry_config=RetryConfig(max_retries=0),
        )

        # Failures in one client should affect the other
        for _ in range(2):
            with contextlib.suppress(JiraClientError):
                rest_client.search_issues("project = TEST")

        # Tag client should also see open circuit (via shared breaker)
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerWithGitHubClient:
    """Tests for circuit breaker integration with GitHubRestClient."""

    @pytest.mark.integration
    def test_github_client_circuit_breaker_opens_on_failures(self) -> None:
        """GitHubRestClient circuit breaker should open after consecutive failures."""
        breaker = CircuitBreaker(
            service_name="github_test",
            config=CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=60.0,
                half_open_max_calls=1,
            ),
        )
        client = GitHubRestClient(
            token="fake-token",
            base_url="https://invalid-github-api.example.com",
            circuit_breaker=breaker,
            retry_config=GitHubRetryConfig(max_retries=0),
        )

        # Make failing requests
        for _ in range(2):
            with contextlib.suppress(GitHubClientError):
                client.search_issues("is:issue")

        client.close()

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN


class TestRateLimiterBehavior:
    """Tests for rate limiter behavior under simulated rate limit conditions."""

    @pytest.mark.integration
    def test_backoff_delay_calculation_exponential(self) -> None:
        """Backoff delay should increase exponentially."""
        config = GitHubRetryConfig(
            initial_delay=1.0,
            max_delay=60.0,
            jitter_min=1.0,  # No jitter for deterministic testing
            jitter_max=1.0,
        )

        delay_0 = _calculate_backoff_delay(0, config)
        delay_1 = _calculate_backoff_delay(1, config)
        delay_2 = _calculate_backoff_delay(2, config)

        # Delays should double each attempt (1, 2, 4)
        assert delay_0 == pytest.approx(1.0, abs=0.01)
        assert delay_1 == pytest.approx(2.0, abs=0.01)
        assert delay_2 == pytest.approx(4.0, abs=0.01)

    @pytest.mark.integration
    def test_backoff_delay_respects_max_delay(self) -> None:
        """Backoff delay should not exceed max_delay."""
        config = GitHubRetryConfig(
            initial_delay=1.0,
            max_delay=5.0,
            jitter_min=1.0,
            jitter_max=1.0,
        )

        # After many attempts, should hit max_delay
        delay = _calculate_backoff_delay(10, config)

        assert delay == pytest.approx(5.0, abs=0.01)

    @pytest.mark.integration
    def test_backoff_delay_uses_retry_after_header(self) -> None:
        """Backoff should use Retry-After header when provided."""
        config = GitHubRetryConfig(
            initial_delay=1.0,
            max_delay=60.0,
            jitter_min=1.0,
            jitter_max=1.0,
        )

        # Retry-After header should override calculated delay
        delay = _calculate_backoff_delay(0, config, retry_after=30.0)

        assert delay == pytest.approx(30.0, abs=0.01)

    @pytest.mark.integration
    def test_backoff_jitter_applies_randomization(self) -> None:
        """Jitter should add randomization to delays."""
        config = GitHubRetryConfig(
            initial_delay=10.0,
            max_delay=60.0,
            jitter_min=0.5,
            jitter_max=1.5,
        )

        # Generate multiple delays
        delays = [_calculate_backoff_delay(0, config) for _ in range(10)]

        # Should have some variation
        assert min(delays) >= 5.0  # 10 * 0.5
        assert max(delays) <= 15.0  # 10 * 1.5

        # Should not all be identical (with very high probability)
        unique_delays = len(set(delays))
        assert unique_delays > 1

    @pytest.mark.integration
    def test_jira_backoff_delay_calculation(self) -> None:
        """Jira backoff delay should also increase exponentially."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=30.0,
            jitter_min=1.0,
            jitter_max=1.0,
        )

        delay_0 = jira_calculate_backoff_delay(0, config)
        delay_1 = jira_calculate_backoff_delay(1, config)

        assert delay_0 == pytest.approx(1.0, abs=0.01)
        assert delay_1 == pytest.approx(2.0, abs=0.01)


class TestRetryMechanism:
    """Tests for retry mechanism under simulated failure conditions."""

    @pytest.mark.integration
    def test_retry_succeeds_after_transient_failure(self) -> None:
        """Retry should succeed after transient failures."""
        call_count = 0

        def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate rate limit response
                response = MagicMock()
                response.status_code = 429
                response.headers = {"Retry-After": "0.01"}
                raise httpx.HTTPStatusError(
                    "Rate limited",
                    request=MagicMock(),
                    response=response,
                )
            return "success"

        config = GitHubRetryConfig(
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.1,
        )

        result = _execute_with_retry(operation, config, GitHubClientError)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.integration
    def test_retry_exhausted_raises_rate_limit_error(self) -> None:
        """Should raise GitHubRateLimitError when retries exhausted."""

        def always_fail() -> None:
            response = MagicMock()
            response.status_code = 429
            response.headers = {}
            raise httpx.HTTPStatusError(
                "Rate limited",
                request=MagicMock(),
                response=response,
            )

        config = GitHubRetryConfig(
            max_retries=2,
            initial_delay=0.01,
            max_delay=0.1,
        )

        with pytest.raises(GitHubRateLimitError) as exc_info:
            _execute_with_retry(always_fail, config, GitHubClientError)

        assert "rate limit exceeded" in str(exc_info.value).lower()

    @pytest.mark.integration
    def test_non_rate_limit_error_not_retried(self) -> None:
        """Non-rate-limit errors should not be retried."""
        call_count = 0

        def operation() -> None:
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.status_code = 404  # Not a rate limit error
            response.headers = {}
            raise httpx.HTTPStatusError(
                "Not found",
                request=MagicMock(),
                response=response,
            )

        config = GitHubRetryConfig(max_retries=3)

        with pytest.raises(httpx.HTTPStatusError):
            _execute_with_retry(operation, config)

        # Should not have retried
        assert call_count == 1


class TestCombinedCircuitBreakerAndRateLimiter:
    """Tests for combined circuit breaker and rate limiter behavior."""

    @pytest.mark.integration
    def test_circuit_breaker_opens_during_rate_limit_storm(self) -> None:
        """Circuit breaker should open when rate limit causes repeated failures."""
        breaker = CircuitBreaker(
            service_name="rate_limit_test",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
            ),
        )

        # Simulate repeated rate limit failures
        for _ in range(3):
            breaker.record_failure(GitHubRateLimitError("Rate limit exceeded"))

        # Circuit should be open, preventing further requests
        assert breaker.state == CircuitState.OPEN
        assert not breaker.allow_request()

    @pytest.mark.integration
    def test_metrics_track_all_failure_types(self) -> None:
        """Metrics should track all types of failures."""
        breaker = CircuitBreaker(
            service_name="metrics_test",
            config=CircuitBreakerConfig(
                failure_threshold=10,  # High threshold to avoid opening
                recovery_timeout=60.0,
            ),
        )

        # Record various failure types
        breaker.record_failure(TimeoutError("Connection timeout"))
        breaker.record_failure(GitHubRateLimitError("Rate limit"))
        breaker.record_failure(GitHubClientError("API error"))

        assert breaker.metrics.failed_calls == 3
        assert breaker.metrics.last_failure_time is not None


class TestFaultInjectionScenarios:
    """Fault injection tests simulating real-world failure scenarios."""

    @pytest.mark.integration
    def test_intermittent_failure_recovery(self) -> None:
        """System should recover from intermittent failures."""
        breaker = CircuitBreaker(
            service_name="intermittent_test",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=0.1,
                half_open_max_calls=2,
            ),
        )

        # Some failures, but not enough to open
        for _ in range(3):
            breaker.record_failure(Exception("intermittent"))

        assert breaker.state == CircuitState.CLOSED

        # Successes reset the failure count
        for _ in range(5):
            breaker.record_success()

        # Should still be closed with good metrics
        assert breaker.state == CircuitState.CLOSED
        assert breaker.metrics.successful_calls == 5

    @pytest.mark.integration
    def test_cascading_failure_prevention(self) -> None:
        """Circuit breaker should prevent cascading failures."""
        breakers = [
            CircuitBreaker(
                service_name=f"service_{i}",
                config=CircuitBreakerConfig(
                    failure_threshold=2,
                    recovery_timeout=60.0,
                ),
            )
            for i in range(3)
        ]

        # Simulate cascading failures starting from first service
        for breaker in breakers:
            for _ in range(2):
                breaker.record_failure(Exception("upstream failure"))

        # All circuits should be open
        assert all(b.state == CircuitState.OPEN for b in breakers)

        # No requests should be allowed
        assert all(not b.allow_request() for b in breakers)

    @pytest.mark.integration
    def test_recovery_after_prolonged_outage(self) -> None:
        """System should recover after prolonged outage."""
        breaker = CircuitBreaker(
            service_name="outage_test",
            config=CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=0.1,  # Short for testing
                half_open_max_calls=1,
            ),
        )

        # Simulate outage
        for _ in range(2):
            breaker.record_failure(Exception("outage"))

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should allow probe request
        assert breaker.allow_request()
        assert breaker.state == CircuitState.HALF_OPEN

        # Simulate successful recovery
        breaker.record_success()

        # Should be fully recovered
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request()
