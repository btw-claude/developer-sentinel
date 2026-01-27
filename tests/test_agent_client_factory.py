"""Unit tests for AgentClientFactory.

DS-292: Tests for the factory pattern for agent client instantiation.
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
from sentinel.agent_clients.base import AgentType
from sentinel.config import Config


class MockAgentClient(AgentClient):
    """Mock agent client for testing."""

    def __init__(self, agent_type_value: AgentType = "claude") -> None:
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
    ) -> AgentRunResult:
        return AgentRunResult(response="Mock response", workdir=None)


def make_test_config() -> Config:
    """Create a Config for testing."""
    return Config(
        agent_workdir=Path("/tmp/test-workdir"),
        agent_logs_dir=Path("/tmp/test-logs"),
    )


class TestAgentClientFactory:
    """Tests for AgentClientFactory."""

    def test_register_and_create(self) -> None:
        """Test registering a builder and creating a client."""
        factory = AgentClientFactory()
        config = make_test_config()

        def mock_builder(cfg: Config) -> AgentClient:
            client = MockAgentClient("claude")
            client.config = cfg
            return client

        factory.register("claude", mock_builder)
        client = factory.create("claude", config)

        assert isinstance(client, MockAgentClient)
        assert client.agent_type == "claude"
        assert client.config is config

    def test_create_unregistered_type_raises_error(self) -> None:
        """Test that creating an unregistered type raises ValueError."""
        factory = AgentClientFactory()
        config = make_test_config()

        with pytest.raises(ValueError) as exc_info:
            factory.create("claude", config)

        assert "No builder registered for agent type 'claude'" in str(exc_info.value)
        assert "Available types:" in str(exc_info.value)

    def test_create_defaults_to_claude_when_none(self) -> None:
        """Test that create() defaults to 'claude' when agent_type is None."""
        factory = AgentClientFactory()
        config = make_test_config()

        def mock_builder(cfg: Config) -> AgentClient:
            return MockAgentClient("claude")

        factory.register("claude", mock_builder)
        client = factory.create(None, config)

        assert client.agent_type == "claude"

    def test_get_or_create_caches_clients(self) -> None:
        """Test that get_or_create caches client instances."""
        factory = AgentClientFactory()
        config = make_test_config()
        call_count = 0

        def counting_builder(cfg: Config) -> AgentClient:
            nonlocal call_count
            call_count += 1
            return MockAgentClient("claude")

        factory.register("claude", counting_builder)

        # First call should create
        client1 = factory.get_or_create("claude", config)
        assert call_count == 1

        # Second call should return cached
        client2 = factory.get_or_create("claude", config)
        assert call_count == 1  # Builder not called again
        assert client1 is client2  # Same instance

    def test_get_or_create_different_configs_not_cached(self) -> None:
        """Test that different configs get different cached instances."""
        factory = AgentClientFactory()
        config1 = make_test_config()
        config2 = make_test_config()

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

    def test_get_or_create_defaults_to_claude(self) -> None:
        """Test that get_or_create defaults to 'claude' when agent_type is None."""
        factory = AgentClientFactory()
        config = make_test_config()

        def mock_builder(cfg: Config) -> AgentClient:
            return MockAgentClient("claude")

        factory.register("claude", mock_builder)

        client = factory.get_or_create(None, config)
        assert client.agent_type == "claude"

    def test_create_for_orchestration(self) -> None:
        """Test create_for_orchestration delegates to create."""
        factory = AgentClientFactory()
        config = make_test_config()

        def mock_builder(cfg: Config) -> AgentClient:
            return MockAgentClient("claude")

        factory.register("claude", mock_builder)

        client = factory.create_for_orchestration("claude", config)
        assert client.agent_type == "claude"

    def test_create_for_orchestration_defaults_to_claude(self) -> None:
        """Test create_for_orchestration defaults to 'claude' when None."""
        factory = AgentClientFactory()
        config = make_test_config()

        def mock_builder(cfg: Config) -> AgentClient:
            return MockAgentClient("claude")

        factory.register("claude", mock_builder)

        client = factory.create_for_orchestration(None, config)
        assert client.agent_type == "claude"

    def test_clear_cache(self) -> None:
        """Test that clear_cache clears all cached clients."""
        factory = AgentClientFactory()
        config = make_test_config()
        call_count = 0

        def counting_builder(cfg: Config) -> AgentClient:
            nonlocal call_count
            call_count += 1
            return MockAgentClient("claude")

        factory.register("claude", counting_builder)

        # Create and cache
        factory.get_or_create("claude", config)
        assert call_count == 1

        # Clear cache
        factory.clear_cache()

        # Should create new instance
        factory.get_or_create("claude", config)
        assert call_count == 2

    def test_registered_types(self) -> None:
        """Test registered_types property returns registered types."""
        factory = AgentClientFactory()

        assert factory.registered_types == []

        factory.register("claude", lambda cfg: MockAgentClient("claude"))
        assert factory.registered_types == ["claude"]

    def test_multiple_agent_types(self) -> None:
        """Test factory can handle multiple agent types."""
        factory = AgentClientFactory()
        config = make_test_config()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))
        factory.register("cursor", lambda cfg: MockAgentClient("cursor"))

        claude_client = factory.create("claude", config)
        cursor_client = factory.create("cursor", config)

        assert claude_client.agent_type == "claude"
        assert cursor_client.agent_type == "cursor"
        assert set(factory.registered_types) == {"claude", "cursor"}


class TestCreateDefaultFactory:
    """Tests for the create_default_factory helper function."""

    def test_creates_factory_with_claude_registered(self) -> None:
        """Test that create_default_factory registers 'claude' type."""
        config = make_test_config()
        factory = create_default_factory(config)

        assert "claude" in factory.registered_types

    @patch("sentinel.agent_clients.claude_sdk.ClaudeSdkAgentClient")
    def test_creates_claude_sdk_client(self, mock_claude_class: MagicMock) -> None:
        """Test that the factory creates ClaudeSdkAgentClient for 'claude' type."""
        config = make_test_config()
        mock_instance = MagicMock(spec=ClaudeSdkAgentClient)
        mock_claude_class.return_value = mock_instance

        factory = create_default_factory(config)
        client = factory.create("claude", config)

        mock_claude_class.assert_called_once_with(
            config=config,
            base_workdir=config.agent_workdir,
            log_base_dir=config.agent_logs_dir,
        )
        assert client is mock_instance

    def test_default_factory_creates_real_claude_client(self) -> None:
        """Test that create_default_factory creates a real ClaudeSdkAgentClient."""
        config = make_test_config()
        factory = create_default_factory(config)

        client = factory.create("claude", config)

        assert isinstance(client, ClaudeSdkAgentClient)
        assert client.agent_type == "claude"
        assert client.config is config
        assert client.base_workdir == config.agent_workdir
        assert client.log_base_dir == config.agent_logs_dir


class TestFactoryCacheKeyBehavior:
    """Tests for cache key behavior in get_or_create."""

    def test_same_config_same_type_returns_cached(self) -> None:
        """Test same config and type returns cached instance."""
        factory = AgentClientFactory()
        config = make_test_config()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        client1 = factory.get_or_create("claude", config)
        client2 = factory.get_or_create("claude", config)

        assert client1 is client2
        assert creation_count == 1

    def test_different_type_same_config_creates_separate(self) -> None:
        """Test different types with same config creates separate instances."""
        factory = AgentClientFactory()
        config = make_test_config()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))
        factory.register("cursor", lambda cfg: MockAgentClient("cursor"))

        claude_client = factory.get_or_create("claude", config)
        cursor_client = factory.get_or_create("cursor", config)

        assert claude_client is not cursor_client
        assert claude_client.agent_type == "claude"
        assert cursor_client.agent_type == "cursor"

    def test_none_type_uses_claude_cache_key(self) -> None:
        """Test that None type uses 'claude' cache key."""
        factory = AgentClientFactory()
        config = make_test_config()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        # First call with None
        client1 = factory.get_or_create(None, config)
        # Second call with explicit "claude"
        client2 = factory.get_or_create("claude", config)

        assert client1 is client2
        assert creation_count == 1


class TestOrchestrationKwargsCache:
    """Tests for kwargs impact on cache key in create_for_orchestration (DS-299)."""

    def test_get_or_create_for_orchestration_caches_with_kwargs(self) -> None:
        """Test that get_or_create_for_orchestration caches based on kwargs."""
        factory = AgentClientFactory()
        config = make_test_config()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        # Same kwargs should return cached instance
        client1 = factory.get_or_create_for_orchestration(
            "claude", config, working_dir="/tmp/test1"
        )
        client2 = factory.get_or_create_for_orchestration(
            "claude", config, working_dir="/tmp/test1"
        )

        assert client1 is client2
        assert creation_count == 1

    def test_different_kwargs_create_separate_cached_instances(self) -> None:
        """Test that different kwargs create separate cached instances."""
        factory = AgentClientFactory()
        config = make_test_config()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        client1 = factory.get_or_create_for_orchestration(
            "claude", config, working_dir="/tmp/test1"
        )
        client2 = factory.get_or_create_for_orchestration(
            "claude", config, working_dir="/tmp/test2"
        )

        assert client1 is not client2
        assert creation_count == 2

    def test_empty_kwargs_cached_separately_from_with_kwargs(self) -> None:
        """Test that empty kwargs and kwargs with values cache separately."""
        factory = AgentClientFactory()
        config = make_test_config()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        client1 = factory.get_or_create_for_orchestration("claude", config)
        client2 = factory.get_or_create_for_orchestration(
            "claude", config, some_option="value"
        )

        assert client1 is not client2
        assert creation_count == 2

    def test_kwargs_order_does_not_affect_cache(self) -> None:
        """Test that kwargs order doesn't affect cache key."""
        factory = AgentClientFactory()
        config = make_test_config()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        # Pass kwargs in different orders
        client1 = factory.get_or_create_for_orchestration(
            "claude", config, a="1", b="2"
        )
        client2 = factory.get_or_create_for_orchestration(
            "claude", config, b="2", a="1"
        )

        assert client1 is client2
        assert creation_count == 1

    def test_unhashable_kwargs_raises_type_error(self) -> None:
        """Test that unhashable kwargs values raise TypeError."""
        factory = AgentClientFactory()
        config = make_test_config()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))

        with pytest.raises(TypeError) as exc_info:
            factory.get_or_create_for_orchestration(
                "claude", config, unhashable_list=[1, 2, 3]
            )

        assert "hashable" in str(exc_info.value).lower()

    def test_get_or_create_for_orchestration_defaults_to_claude(self) -> None:
        """Test get_or_create_for_orchestration defaults to 'claude' when None."""
        factory = AgentClientFactory()
        config = make_test_config()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))

        client = factory.get_or_create_for_orchestration(None, config)
        assert client.agent_type == "claude"

    def test_clear_cache_clears_orchestration_cache(self) -> None:
        """Test that clear_cache also clears orchestration-cached clients."""
        factory = AgentClientFactory()
        config = make_test_config()
        creation_count = 0

        def builder(cfg: Config) -> MockAgentClient:
            nonlocal creation_count
            creation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", builder)

        factory.get_or_create_for_orchestration(
            "claude", config, working_dir="/tmp/test"
        )
        assert creation_count == 1

        factory.clear_cache()

        factory.get_or_create_for_orchestration(
            "claude", config, working_dir="/tmp/test"
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
    """Tests for caching behavior of get_or_create_for_orchestration (DS-309).

    These tests verify that the caching optimization introduced in DS-303
    continues to work as expected - calling get_or_create_for_orchestration
    with the same agent type returns the same client instance.
    """

    def test_same_agent_type_returns_same_instance(self) -> None:
        """Verify same agent type with same config returns identical instance.

        DS-309: This test ensures the caching optimization works correctly -
        calling get_or_create_for_orchestration with the same agent type
        should return the exact same client instance (identity check).
        """
        factory = AgentClientFactory()
        config = make_test_config()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))

        # Call get_or_create_for_orchestration multiple times with same agent type
        client1 = factory.get_or_create_for_orchestration("claude", config)
        client2 = factory.get_or_create_for_orchestration("claude", config)
        client3 = factory.get_or_create_for_orchestration("claude", config)

        # All should be the exact same instance (identity, not just equality)
        assert client1 is client2, "Expected same instance for repeated calls"
        assert client2 is client3, "Expected same instance for all calls"
        assert id(client1) == id(client2) == id(client3), "Memory addresses should match"

    def test_caching_prevents_redundant_client_creation(self) -> None:
        """Verify caching prevents redundant client instantiation.

        DS-309: This test documents that the caching behavior prevents
        unnecessary overhead from creating multiple client instances.
        """
        factory = AgentClientFactory()
        config = make_test_config()
        instantiation_count = 0

        def tracking_builder(cfg: Config) -> MockAgentClient:
            nonlocal instantiation_count
            instantiation_count += 1
            return MockAgentClient("claude")

        factory.register("claude", tracking_builder)

        # Multiple calls should only instantiate once
        factory.get_or_create_for_orchestration("claude", config)
        factory.get_or_create_for_orchestration("claude", config)
        factory.get_or_create_for_orchestration("claude", config)

        assert instantiation_count == 1, (
            f"Expected exactly 1 instantiation but got {instantiation_count}. "
            "Caching should prevent redundant client creation."
        )

    def test_different_agent_types_return_different_instances(self) -> None:
        """Verify different agent types return separate cached instances.

        DS-309: Confirms that caching is agent-type-specific - different
        agent types should have their own cached instances.
        """
        factory = AgentClientFactory()
        config = make_test_config()

        factory.register("claude", lambda cfg: MockAgentClient("claude"))
        factory.register("cursor", lambda cfg: MockAgentClient("cursor"))

        claude_client = factory.get_or_create_for_orchestration("claude", config)
        cursor_client = factory.get_or_create_for_orchestration("cursor", config)

        # Different agent types should have different instances
        assert claude_client is not cursor_client
        assert claude_client.agent_type == "claude"
        assert cursor_client.agent_type == "cursor"

        # But same agent type should still return cached instance
        claude_client2 = factory.get_or_create_for_orchestration("claude", config)
        assert claude_client is claude_client2
