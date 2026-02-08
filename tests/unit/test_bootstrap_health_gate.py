"""Tests for ServiceHealthGate wiring in bootstrap."""

from sentinel.bootstrap import BootstrapContext, create_sentinel_from_context
from sentinel.circuit_breaker import CircuitBreakerRegistry
from sentinel.service_health_gate import ServiceHealthGate
from tests.helpers import make_agent_factory, make_config
from tests.mocks import MockJiraClient, MockTagClient


class TestBootstrapServiceHealthGate:
    """Tests for ServiceHealthGate creation in bootstrap."""

    def test_create_sentinel_from_context_creates_health_gate(self) -> None:
        """Verify create_sentinel_from_context creates and injects ServiceHealthGate."""
        config = make_config()
        agent_factory, _ = make_agent_factory()
        context = BootstrapContext(
            config=config,
            orchestrations=[],
            jira_client=MockJiraClient(issues=[]),
            tag_client=MockTagClient(),
            agent_factory=agent_factory,
            agent_logger=None,  # type: ignore[arg-type]
            circuit_breaker_registry=CircuitBreakerRegistry(),
        )
        sentinel = create_sentinel_from_context(context)
        assert sentinel.service_health_gate is not None
        assert isinstance(sentinel.service_health_gate, ServiceHealthGate)

    def test_create_sentinel_from_context_health_gate_uses_config(self) -> None:
        """Verify ServiceHealthGate uses configuration from context."""
        config = make_config(
            health_gate_enabled=True,
            health_gate_failure_threshold=7,
        )
        agent_factory, _ = make_agent_factory()
        context = BootstrapContext(
            config=config,
            orchestrations=[],
            jira_client=MockJiraClient(issues=[]),
            tag_client=MockTagClient(),
            agent_factory=agent_factory,
            agent_logger=None,  # type: ignore[arg-type]
            circuit_breaker_registry=CircuitBreakerRegistry(),
        )
        sentinel = create_sentinel_from_context(context)
        assert sentinel.service_health_gate is not None
        assert sentinel.service_health_gate.enabled is True

    def test_create_sentinel_from_context_health_gate_disabled(self) -> None:
        """Verify ServiceHealthGate can be disabled via config."""
        config = make_config(health_gate_enabled=False)
        agent_factory, _ = make_agent_factory()
        context = BootstrapContext(
            config=config,
            orchestrations=[],
            jira_client=MockJiraClient(issues=[]),
            tag_client=MockTagClient(),
            agent_factory=agent_factory,
            agent_logger=None,  # type: ignore[arg-type]
            circuit_breaker_registry=CircuitBreakerRegistry(),
        )
        sentinel = create_sentinel_from_context(context)
        assert sentinel.service_health_gate is not None
        assert sentinel.service_health_gate.enabled is False
