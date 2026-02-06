"""Unit tests for AgentClientFactory.

Tests for the factory pattern for agent client instantiation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel.agent_clients import (
    AgentClient,
    AgentClientFactory,
    AgentRunResult,
    ClaudeSdkAgentClient,
    create_default_factory,
)
from sentinel.config import Config
from sentinel.types import AgentType
from tests.helpers import make_config


class MockAgentClient(AgentClient):
    """Mock agent client for testing."""

    def __init__(self, agent_type_value: AgentType = AgentType.CLAUDE) -> None:
        self._agent_type = agent_type_value
        self.config: Config | None = None

    @property
    def agent_type(self) -> AgentType:
        return self._agent_type

    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
    ) -> AgentRunResult:
        return AgentRunResult(response="Mock response", workdir=None)


@pytest.fixture
def factory_config() -> Config:
    """Create a standard test Config for factory tests."""
    return make_config(
        agent_workdir=Path("/tmp/test-workdir"),
        agent_logs_dir=Path("/tmp/test-logs"),
    )


class TestAgentClientFactory:
    """Tests for AgentClientFactory."""

    def test_register_and_create(self, factory_config: Config) -> None:
        """Test registering a builder and creating a client."""
        factory = AgentClientFactory()

        def mock_builder(cfg: Config) -> AgentClient:
            client = MockAgentClient("claude")
            client.config = cfg
            return client

        factory.register("claude", mock_builder)
        client = factory.create("claude", factory_config)

        assert isinstance(client, MockAgentClient)
        assert client.agent_type == "claude"
        assert client.config is factory_config

    def test_create_unregistered_type_raises_error(self, factory_config: Config) -> None:
        """Test that creating an unregistered type raises ValueError."""
        factory = AgentClientFactory()

        with pytest.raises(ValueError) as exc_info:
            factory.create("claude", factory_config)

        assert "No builder registered for agent type 'claude'" in str(exc_info.value)
        assert "Available types:" in str(exc_info.value)

    def test_create_defaults_to_claude_when_none(self, factory_config: Config) -> None:
        """Test that create() defaults to 'claude' when agent_type is None."""
        factory = AgentClientFactory()

        def mock_builder(cfg: Config) -> AgentClient:
            return MockAgentClient("claude")

        factory.register("claude", mock_builder)
        client = factory.create(None, factory_config)

        assert client.agent_type == "claude"

    def test_get_or_create_caches_clients(self, factory_config: Config) -> None:
        """Test that get_or_create caches client instances."""
        factory = AgentClientFactory()
        call_count = 0

        def counting_builder(cfg: Config) -> AgentClient:
            nonlocal call_count
            call_count += 1
            return MockAgentClient("claude")

        factory.register("claude", counting_builder)

        # First call should create
        client1 = factory.get_or_create("claude", factory_config)
        assert call_count == 1

        # Second call should return cached
        client2 = factory.get_or_create("claude", factory_config)
        assert call_count == 1  # Builder not called again
        assert client1 is client2  # Same instance

    def test_get_or_create_different_configs_not_cached(self) -> None:
        """Test that different configs get different cached instances."""
        factory = AgentClientFactory()
        config1 = make_config(
            agent_workdir=Path("/tmp/test-workdir"),
            agent_logs_dir=Path("/tmp/test-logs"),
        )
        config2 = make_config(
            agent_workdir=Path("/tmp/test-workdir"),
            agent_logs_dir=Path("/tmp/test-logs"),
        )

        def mock_builder(cfg: Config) -> AgentClient:
            client = MockAgentClient("claude")
            client.config = cfg
            return client

        factory.register("claude", mock_builder)

        client1 = factory.get_or_create("claude", config1)
        client2 = factory.get_or_create("claude", config2)

        assert client1 is not client2
        assert client1.config is config1
        assert client2.config is config2

    def test_get_or_create_defaults_to_claude(self, factory_config: Config) -> None:
        """Test that get_or_create defaults to 'claude' when agent_type is None."""
        factory = AgentClientFactory()

        def mock_builder(cfg: Config) -> AgentClient:
            return MockAgentClient("claude")

        factory.register("claude", mock_builder)

        client = factory.get_or_create(None, factory_config)
        assert client.agent_type == "claude"

    def test_create_for_orchestration(self, factory_config: Config) -> None:
        """Test create_for_orchestration delegates to create."""
        factory = AgentClientFactory()

        def mock_builder(cfg: Config) -> AgentClient:
            return MockAgentClient("claude")

        factory.register("claude", mock_builder)

        client = factory.create_for_orchestration("claude", factory_config)
        assert client.agent_type == "claude"

    def test_create_for_orchestration_defaults_to_claude(self, factory_config: Config) -> None:
        """Test create_for_orchestration defaults to 'claude' when None."""
        factory = AgentClientFactory()

        def mock_builder(cfg: Config) -> AgentClient:
            return MockAgentClient("claude")

        factory.register("claude", mock_builder)

        client = factory.create_for_orchestration(None, factory_config)
        assert client.agent_type == "claude"

    def test_clear_cache(self, factory_config: Config) -> None:
        """Test that clear_cache clears all cached clients."""
        factory = AgentClientFactory()
        call_count = 0

        def counting_builder(cfg: Config) -> AgentClient:
            nonlocal call_count
            call_count += 1
            return MockAgentClient("claude")

        factory.register("claude", counting_builder)

        # Create and cache
        factory.get_or_create("claude", factory_config)
        assert call_count == 1

        # Clear cache
        factory.clear_cache()

        # Should create new instance
        factory.get_or_create("claude", factory_config)
        assert call_count == 2

    def test_registered_types(self) -> None:
        """Test registered_types property returns registered types."""
        factory = AgentClientFactory()

        assert factory.registered_types == []

        factory.register("claude", lambda cfg: MockAgentClient("claude"))
        assert factory.registered_types == ["claude"]

    def test_multiple_agent_types(self, factory_config: Config) -> None:
        """Test factory can handle multiple agent types."""
        factory = AgentClientFactory()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))
        factory.register("cursor", lambda cfg: MockAgentClient("cursor"))

        claude_client = factory.create("claude", factory_config)
        cursor_client = factory.create("cursor", factory_config)

        assert claude_client.agent_type == "claude"
        assert cursor_client.agent_type == "cursor"
        assert set(factory.registered_types) == {"claude", "cursor"}


class TestCreateDefaultFactory:
    """Tests for the create_default_factory helper function."""

    def test_creates_factory_with_claude_registered(self, factory_config: Config) -> None:
        """Test that create_default_factory registers 'claude' type."""
        factory = create_default_factory(factory_config)

        assert "claude" in factory.registered_types

    @patch("sentinel.agent_clients.claude_sdk.ClaudeSdkAgentClient")
    def test_creates_claude_sdk_client(
        self, mock_claude_class: MagicMock, factory_config: Config
    ) -> None:
        """Test that the factory creates ClaudeSdkAgentClient for 'claude' type."""
        mock_instance = MagicMock(spec=ClaudeSdkAgentClient)
        mock_claude_class.return_value = mock_instance

        factory = create_default_factory(factory_config)
        client = factory.create("claude", factory_config)

        # Circuit breaker is None when no registry is provided
        mock_claude_class.assert_called_once_with(
            config=factory_config,
            base_workdir=factory_config.execution.agent_workdir,
            log_base_dir=factory_config.execution.agent_logs_dir,
            circuit_breaker=None,
        )
        assert client is mock_instance

    def test_default_factory_creates_real_claude_client(self, factory_config: Config) -> None:
        """Test that create_default_factory creates a real ClaudeSdkAgentClient."""
        factory = create_default_factory(factory_config)

        client = factory.create("claude", factory_config)

        assert isinstance(client, ClaudeSdkAgentClient)
        assert client.agent_type == "claude"
        assert client.config is factory_config
        assert client.base_workdir == factory_config.execution.agent_workdir
        assert client.log_base_dir == factory_config.execution.agent_logs_dir


class TestFactoryCacheKeyBehavior:
    """Tests for cache key behavior in get_or_create."""

    def test_same_config_same_type_returns_cached(self, factory_config: Config) -> None:
        """Test same config and type returns cached instance."""
        factory = AgentClientFactory()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        client1 = factory.get_or_create("claude", factory_config)
        client2 = factory.get_or_create("claude", factory_config)

        assert client1 is client2
        assert creation_count == 1

    def test_different_type_same_config_creates_separate(self, factory_config: Config) -> None:
        """Test different types with same config creates separate instances."""
        factory = AgentClientFactory()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))
        factory.register("cursor", lambda cfg: MockAgentClient("cursor"))

        claude_client = factory.get_or_create("claude", factory_config)
        cursor_client = factory.get_or_create("cursor", factory_config)

        assert claude_client is not cursor_client
        assert claude_client.agent_type == "claude"
        assert cursor_client.agent_type == "cursor"

    def test_none_type_uses_claude_cache_key(self, factory_config: Config) -> None:
        """Test that None type uses 'claude' cache key."""
        factory = AgentClientFactory()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        # First call with None
        client1 = factory.get_or_create(None, factory_config)
        # Second call with explicit "claude"
        client2 = factory.get_or_create("claude", factory_config)

        assert client1 is client2
        assert creation_count == 1


class TestOrchestrationKwargsCache:
    """Tests for kwargs impact on cache key in create_for_orchestration."""

    def test_get_or_create_for_orchestration_caches_with_kwargs(
        self, factory_config: Config
    ) -> None:
        """Test that get_or_create_for_orchestration caches based on kwargs."""
        factory = AgentClientFactory()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        # Same kwargs should return cached instance
        client1 = factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/tmp/test1"
        )
        client2 = factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/tmp/test1"
        )

        assert client1 is client2
        assert creation_count == 1

    def test_different_kwargs_create_separate_cached_instances(
        self, factory_config: Config
    ) -> None:
        """Test that different kwargs create separate cached instances."""
        factory = AgentClientFactory()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        client1 = factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/tmp/test1"
        )
        client2 = factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/tmp/test2"
        )

        assert client1 is not client2
        assert creation_count == 2

    def test_empty_kwargs_cached_separately_from_with_kwargs(
        self, factory_config: Config
    ) -> None:
        """Test that empty kwargs and kwargs with values cache separately."""
        factory = AgentClientFactory()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        client1 = factory.get_or_create_for_orchestration("claude", factory_config)
        client2 = factory.get_or_create_for_orchestration(
            "claude", factory_config, some_option="value"
        )

        assert client1 is not client2
        assert creation_count == 2

    def test_kwargs_order_does_not_affect_cache(self, factory_config: Config) -> None:
        """Test that kwargs order doesn't affect cache key."""
        factory = AgentClientFactory()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        # Pass kwargs in different orders
        client1 = factory.get_or_create_for_orchestration(
            "claude", factory_config, a="1", b="2"
        )
        client2 = factory.get_or_create_for_orchestration(
            "claude", factory_config, b="2", a="1"
        )

        assert client1 is client2
        assert creation_count == 1

    def test_unhashable_kwargs_raises_type_error(self, factory_config: Config) -> None:
        """Test that unhashable kwargs values raise TypeError."""
        factory = AgentClientFactory()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))

        with pytest.raises(TypeError) as exc_info:
            factory.get_or_create_for_orchestration(
                "claude", factory_config, unhashable_list=[1, 2, 3]
            )

        assert "hashable" in str(exc_info.value).lower()

    def test_get_or_create_for_orchestration_defaults_to_claude(
        self, factory_config: Config
    ) -> None:
        """Test get_or_create_for_orchestration defaults to 'claude' when None."""
        factory = AgentClientFactory()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))

        client = factory.get_or_create_for_orchestration(None, factory_config)
        assert client.agent_type == "claude"

    def test_clear_cache_clears_orchestration_cache(self, factory_config: Config) -> None:
        """Test that clear_cache also clears orchestration-cached clients."""
        factory = AgentClientFactory()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/tmp/test"
        )
        assert creation_count == 1

        factory.clear_cache()

        factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/tmp/test"
        )
        assert creation_count == 2

    def test_make_kwargs_hashable_creates_sorted_tuple(self) -> None:
        """Test that _make_kwargs_hashable creates a sorted tuple."""
        factory = AgentClientFactory()

        result = factory._make_kwargs_hashable({"b": 2, "a": 1, "c": 3})

        assert result == (("a", 1), ("b", 2), ("c", 3))

    def test_make_kwargs_hashable_empty_dict(self) -> None:
        """Test that _make_kwargs_hashable handles empty dict."""
        factory = AgentClientFactory()

        result = factory._make_kwargs_hashable({})

        assert result == ()


class TestGetOrCreateForOrchestrationCachingBehavior:
    """Tests for caching behavior of get_or_create_for_orchestration.

    These tests verify that the caching optimization continues to work as
    expected - calling get_or_create_for_orchestration with the same agent
    type returns the same client instance.
    """

    def test_same_agent_type_returns_same_instance(self, factory_config: Config) -> None:
        """Verify same agent type with same config returns identical instance.

        This test ensures the caching optimization works correctly -
        calling get_or_create_for_orchestration with the same agent type
        should return the exact same client instance (identity check).
        """
        factory = AgentClientFactory()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))

        # Call get_or_create_for_orchestration multiple times with same agent type
        client1 = factory.get_or_create_for_orchestration("claude", factory_config)
        client2 = factory.get_or_create_for_orchestration("claude", factory_config)
        client3 = factory.get_or_create_for_orchestration("claude", factory_config)

        # All should be the exact same instance (identity, not just equality)
        assert client1 is client2, "Expected same instance for repeated calls"
        assert client2 is client3, "Expected same instance for all calls"
        assert id(client1) == id(client2) == id(client3), "Memory addresses should match"

    def test_caching_prevents_redundant_client_creation(self, factory_config: Config) -> None:
        """Verify caching prevents redundant client instantiation.

        This test documents that the caching behavior prevents
        unnecessary overhead from creating multiple client instances.
        """
        factory = AgentClientFactory()
        instantiation_count = 0

        def tracking_builder(cfg: Config) -> MockAgentClient:
            nonlocal instantiation_count
            instantiation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", tracking_builder)

        # Multiple calls should only instantiate once
        factory.get_or_create_for_orchestration("claude", factory_config)
        factory.get_or_create_for_orchestration("claude", factory_config)
        factory.get_or_create_for_orchestration("claude", factory_config)

        assert instantiation_count == 1, (
            f"Expected exactly 1 instantiation but got {instantiation_count}. "
            "Caching should prevent redundant client creation."
        )

    def test_different_agent_types_return_different_instances(self, factory_config: Config) -> None:
        """Verify different agent types return separate cached instances.

        Confirms that caching is agent-type-specific - different
        agent types should have their own cached instances.
        """
        factory = AgentClientFactory()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))
        factory.register("cursor", lambda cfg: MockAgentClient("cursor"))

        claude_client = factory.get_or_create_for_orchestration("claude", factory_config)
        cursor_client = factory.get_or_create_for_orchestration("cursor", factory_config)

        # Different agent types should have different instances
        assert claude_client is not cursor_client
        assert claude_client.agent_type == "claude"
        assert cursor_client.agent_type == "cursor"

        # But same agent type should still return cached instance
        claude_client2 = factory.get_or_create_for_orchestration("claude", factory_config)
        assert claude_client is claude_client2


class TestKwargsBasedCacheKeyDifferentiation:
    """Tests for kwargs-based cache key differentiation in get_or_create_for_orchestration.

    These tests verify that get_or_create_for_orchestration properly
    differentiates cached clients based on the **kwargs parameter, ensuring that:
    - Same agent type with different kwargs returns different client instances
    - Same agent type with same kwargs returns the same cached instance
    """

    def test_kwargs_cache_key_differentiation(self, factory_config: Config) -> None:
        """Verify kwargs properly differentiate cache keys for same agent type.

        This comprehensive test validates that the kwargs parameter
        correctly influences cache key generation, ensuring:
        1. Same agent type + same kwargs = same cached instance
        2. Same agent type + different kwargs = different instances

        This provides confidence in AgentClientFactory.get_or_create_for_orchestration()
        cache key behavior when kwargs are used.
        """
        factory = AgentClientFactory()
        creation_count = 0

        def counting_builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", counting_builder)

        # --- Test 1: Same agent type with SAME kwargs returns SAME instance ---
        client_a1 = factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/project/a", timeout=30
        )
        client_a2 = factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/project/a", timeout=30
        )

        assert (
            client_a1 is client_a2
        ), "Same agent type with identical kwargs should return the same cached instance"
        assert creation_count == 1, "Builder should only be called once for identical kwargs"

        # --- Test 2: Same agent type with DIFFERENT kwargs returns DIFFERENT instance ---
        client_b = factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/project/b", timeout=30
        )

        assert (
            client_b is not client_a1
        ), "Same agent type with different kwargs should return a different instance"
        assert creation_count == 2, "Builder should be called again for different kwargs"

        # --- Test 3: Verify original cached instance is still returned ---
        client_a3 = factory.get_or_create_for_orchestration(
            "claude", factory_config, working_dir="/project/a", timeout=30
        )

        assert (
            client_a3 is client_a1
        ), "Original kwargs should still return the original cached instance"
        assert creation_count == 2, "No additional builder calls for previously cached kwargs"

        # --- Test 4: Different kwargs value for same key creates new instance ---
        client_c = factory.get_or_create_for_orchestration(
            "claude",
            factory_config,
            working_dir="/project/a",
            timeout=60,  # different timeout
        )

        assert (
            client_c is not client_a1
        ), "Same kwargs keys but different values should create a different instance"
        assert creation_count == 3, "Builder should be called for kwargs with different values"

    def test_kwargs_cache_key_with_multiple_kwargs_combinations(self, factory_config: Config) -> None:
        """Verify cache correctly handles multiple distinct kwargs combinations.

        This test validates that the cache maintains separate entries
        for each unique kwargs combination, and correctly returns cached
        instances when the same combination is requested again.
        """
        factory = AgentClientFactory()
        created_clients: list[MockAgentClient] = []

        def tracking_builder(cfg: Config) -> MockAgentClient:
            client = MockAgentClient("claude")
            created_clients.append(client)
            return client

        factory.register("claude", tracking_builder)

        # Create multiple clients with different kwargs combinations
        kwargs_set_1 = {"env": "dev", "region": "us-east"}
        kwargs_set_2 = {"env": "prod", "region": "us-east"}
        kwargs_set_3 = {"env": "dev", "region": "eu-west"}

        client1 = factory.get_or_create_for_orchestration("claude", factory_config, **kwargs_set_1)
        client2 = factory.get_or_create_for_orchestration("claude", factory_config, **kwargs_set_2)
        client3 = factory.get_or_create_for_orchestration("claude", factory_config, **kwargs_set_3)

        # All three should be different instances
        assert client1 is not client2, "Different kwargs should yield different instances"
        assert client2 is not client3, "Different kwargs should yield different instances"
        assert client1 is not client3, "Different kwargs should yield different instances"
        assert len(created_clients) == 3, "Three unique kwargs combinations = three creations"

        # Re-request each - should return cached instances
        client1_again = factory.get_or_create_for_orchestration("claude", factory_config, **kwargs_set_1)
        client2_again = factory.get_or_create_for_orchestration("claude", factory_config, **kwargs_set_2)
        client3_again = factory.get_or_create_for_orchestration("claude", factory_config, **kwargs_set_3)

        assert client1_again is client1, "Same kwargs should return cached instance"
        assert client2_again is client2, "Same kwargs should return cached instance"
        assert client3_again is client3, "Same kwargs should return cached instance"
        assert len(created_clients) == 3, "No new creations for cached kwargs"


class TestEmptyKwargsEdgeCases:
    """Tests for empty kwargs vs no kwargs edge cases in get_or_create_for_orchestration.

    These tests verify that get_or_create_for_orchestration handles
    the edge case of empty kwargs {} vs no kwargs consistently, ensuring that
    both calling patterns produce the same cache key and return the same
    cached client instance.
    """

    def test_no_kwargs_vs_empty_kwargs_dict_returns_same_instance(self, factory_config: Config) -> None:
        """Verify that no kwargs and explicit empty kwargs {} return the same cached instance.

        This test ensures that calling get_or_create_for_orchestration
        without any kwargs and calling it with an explicit empty dict (**{})
        produce identical cache keys and return the same cached instance.

        This edge case is important for consistency when callers might
        conditionally pass kwargs that could be empty.
        """
        factory = AgentClientFactory()
        creation_count = 0

        def counting_builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", counting_builder)

        # Call with no kwargs
        client_no_kwargs = factory.get_or_create_for_orchestration("claude", factory_config)
        assert creation_count == 1, "First call should create a new instance"

        # Call with explicit empty dict unpacked as kwargs
        empty_dict: dict[str, Any] = {}
        client_empty_kwargs = factory.get_or_create_for_orchestration(
            "claude", factory_config, **empty_dict
        )

        assert (
            client_no_kwargs is client_empty_kwargs
        ), "No kwargs and empty kwargs {} should return the same cached instance"
        assert creation_count == 1, (
            "Empty kwargs should use the same cache key as no kwargs - "
            "builder should not be called again"
        )

    def test_empty_kwargs_cache_key_generation_consistency(self) -> None:
        """Verify that _make_kwargs_hashable handles empty dict consistently.

        This test validates that the cache key generation produces
        consistent results for empty kwargs scenarios, ensuring the tuple
        representation is identical whether kwargs is implicitly or explicitly
        empty.
        """
        factory = AgentClientFactory()

        # Test that empty dict produces empty tuple
        empty_result = factory._make_kwargs_hashable({})
        assert empty_result == (), "Empty dict should produce empty tuple"

        # Verify the tuple is hashable (can be used as cache key component)
        assert hash(empty_result) is not None, "Result should be hashable"

    def test_multiple_calls_with_various_empty_kwargs_patterns(self, factory_config: Config) -> None:
        """Verify consistent caching across multiple empty kwargs call patterns.

        This comprehensive test verifies that all variations of
        "empty kwargs" calls return the same cached instance:
        1. No kwargs at all
        2. Unpacked empty dict
        3. Multiple sequential calls with mixed patterns
        """
        factory = AgentClientFactory()
        creation_count = 0

        def counting_builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", counting_builder)

        # Pattern 1: No kwargs
        client1 = factory.get_or_create_for_orchestration("claude", factory_config)

        # Pattern 2: Explicit empty dict
        client2 = factory.get_or_create_for_orchestration("claude", factory_config, **{})

        # Pattern 3: Variable containing empty dict
        empty_kwargs: dict[str, Any] = {}
        client3 = factory.get_or_create_for_orchestration("claude", factory_config, **empty_kwargs)

        # Pattern 4: Another no-kwargs call after the empty dict calls
        client4 = factory.get_or_create_for_orchestration("claude", factory_config)

        # All should be the exact same instance
        assert client1 is client2, "No kwargs and **{} should return same instance"
        assert client2 is client3, "All empty kwargs patterns should return same instance"
        assert client3 is client4, "Mixed call patterns should still return same instance"
        assert creation_count == 1, (
            "All empty kwargs variations should share the same cache key - "
            "only one instance should be created"
        )

    def test_empty_kwargs_distinct_from_kwargs_with_none_value(self, factory_config: Config) -> None:
        """Verify empty kwargs is distinct from kwargs containing None values.

        This test confirms that empty kwargs {} is correctly
        differentiated from kwargs that contain explicit None values,
        as these represent semantically different configurations.
        """
        factory = AgentClientFactory()
        creation_count = 0

        def counting_builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", counting_builder)

        # Empty kwargs
        client_empty = factory.get_or_create_for_orchestration("claude", factory_config)

        # Kwargs with explicit None value - should be different
        client_with_none = factory.get_or_create_for_orchestration(
            "claude", factory_config, some_option=None
        )

        assert (
            client_empty is not client_with_none
        ), "Empty kwargs should be distinct from kwargs containing None values"
        assert (
            creation_count == 2
        ), "Empty kwargs and kwargs-with-None should create separate instances"
