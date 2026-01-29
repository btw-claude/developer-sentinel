"""Factory pattern for agent client instantiation.

This module provides a factory for creating agent clients, supporting
multiple agent backends (Claude SDK, Cursor CLI, etc.) with caching
and orchestration-specific configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from sentinel.agent_clients.base import AgentClient, AgentType
from sentinel.config import Config
from sentinel.logging import get_logger

logger = get_logger(__name__)

# Default agent type used when None is provided to factory methods
DEFAULT_AGENT_TYPE: AgentType = "claude"

# Type alias for agent client builder functions
AgentClientBuilder = Callable[[Config], AgentClient]

# Cache key type aliases for documentation clarity:
# - SimpleCacheKey: Used by get_or_create() - tuple of (agent_type, config_id)
# - OrchestrationCacheKey: Used by get_or_create_for_orchestration() -
#   tuple of (agent_type, config_id, kwargs_key) where kwargs_key is a
#   hashable tuple representation of the kwargs dict
SimpleCacheKey: TypeAlias = tuple[AgentType, int]
OrchestrationCacheKey: TypeAlias = tuple[AgentType, int, tuple[tuple[str, Any], ...]]
CacheKey: TypeAlias = SimpleCacheKey | OrchestrationCacheKey


class AgentClientFactory:
    """Factory for creating and caching agent client instances.

    This factory supports:
    - Registration of agent client builders by type
    - Caching of client instances to avoid recreation
    - Orchestration-specific client configuration
    - Defaulting to 'claude' when agent_type is None

    Example:
        factory = create_default_factory(config)
        client = factory.create("claude", config)
        # Or with caching:
        client = factory.get_or_create("claude", config)
    """

    def __init__(self) -> None:
        """Initialize the factory with empty registries."""
        self._builders: dict[AgentType, AgentClientBuilder] = {}
        self._cache: dict[CacheKey, AgentClient] = {}

    def register(self, agent_type: AgentType, builder: AgentClientBuilder) -> None:
        """Register a builder function for an agent type.

        Args:
            agent_type: The type of agent ('claude' or 'cursor').
            builder: A callable that takes a Config and returns an AgentClient.
        """
        self._builders[agent_type] = builder
        logger.debug(f"Registered builder for agent type: {agent_type}")

    def create(self, agent_type: AgentType | None, config: Config) -> AgentClient:
        """Create a new agent client instance.

        Args:
            agent_type: The type of agent to create. Defaults to 'claude' if None.
            config: Configuration object for the client.

        Returns:
            A new AgentClient instance.

        Raises:
            ValueError: If no builder is registered for the agent type.
        """
        resolved_type: AgentType = agent_type if agent_type is not None else DEFAULT_AGENT_TYPE

        if resolved_type not in self._builders:
            available = list(self._builders.keys())
            raise ValueError(
                f"No builder registered for agent type '{resolved_type}'. "
                f"Available types: {available}"
            )

        builder = self._builders[resolved_type]
        client = builder(config)
        logger.debug(f"Created new {resolved_type} agent client")
        return client

    def get_or_create(self, agent_type: AgentType | None, config: Config) -> AgentClient:
        """Get a cached agent client or create a new one.

        Clients are cached by (agent_type, config_id) tuple, where config_id
        is the id() of the config object. This ensures that different configs
        get different client instances.

        Args:
            agent_type: The type of agent to get/create. Defaults to 'claude' if None.
            config: Configuration object for the client.

        Returns:
            An AgentClient instance (possibly cached).

        Raises:
            ValueError: If no builder is registered for the agent type.
        """
        resolved_type: AgentType = agent_type if agent_type is not None else DEFAULT_AGENT_TYPE
        cache_key: SimpleCacheKey = (resolved_type, id(config))

        if cache_key in self._cache:
            logger.debug(f"Returning cached {resolved_type} agent client")
            return self._cache[cache_key]

        client = self.create(resolved_type, config)
        self._cache[cache_key] = client
        return client

    def _make_kwargs_hashable(self, kwargs: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        """Convert kwargs dict to a hashable tuple for cache key use.

        Args:
            kwargs: Dictionary of keyword arguments.

        Returns:
            A sorted tuple of (key, value) pairs. Values must be hashable.

        Raises:
            TypeError: If any kwargs values are not hashable.
        """
        try:
            return tuple(sorted(kwargs.items()))
        except TypeError as e:
            raise TypeError(
                f"All kwargs values must be hashable for cache key generation: {e}"
            ) from e

    def get_or_create_for_orchestration(
        self,
        orch_agent_type: AgentType | None,
        config: Config,
        **kwargs: Any,
    ) -> AgentClient:
        """Get a cached client or create one for orchestration with kwargs support.

        Clients are cached by (agent_type, config_id, kwargs) tuple. This ensures
        that different orchestration configurations get different client instances.

        Args:
            orch_agent_type: The agent type specified by the orchestration.
                            Defaults to 'claude' if None.
            config: Base configuration object.
            **kwargs: Additional configuration options that influence the cache key.
                     All values must be hashable.

        Returns:
            An AgentClient instance (possibly cached).

        Raises:
            ValueError: If no builder is registered for the agent type.
            TypeError: If any kwargs values are not hashable.
        """
        resolved_type: AgentType = (
            orch_agent_type if orch_agent_type is not None else DEFAULT_AGENT_TYPE
        )
        kwargs_key = self._make_kwargs_hashable(kwargs)
        cache_key: OrchestrationCacheKey = (resolved_type, id(config), kwargs_key)

        if cache_key in self._cache:
            logger.debug(
                f"Returning cached {resolved_type} orchestration client " f"(kwargs: {kwargs_key})"
            )
            return self._cache[cache_key]

        client = self.create_for_orchestration(orch_agent_type, config, **kwargs)
        self._cache[cache_key] = client
        logger.debug(
            f"Created and cached {resolved_type} orchestration client " f"(kwargs: {kwargs_key})"
        )
        return client

    def create_for_orchestration(
        self,
        orch_agent_type: AgentType | None,
        config: Config,
        **kwargs: Any,
    ) -> AgentClient:
        """Create a client configured for a specific orchestration.

        This method is useful when orchestrations need agent clients with
        specific configurations (e.g., different working directories,
        logging paths, etc.).

        Note: For caching support with kwargs, use get_or_create_for_orchestration().
        The kwargs parameter influences cache key generation in that method.

        Args:
            orch_agent_type: The agent type specified by the orchestration.
                            Defaults to 'claude' if None.
            config: Base configuration object.
            **kwargs: Additional configuration options (reserved for future use).
                     When caching is desired, use get_or_create_for_orchestration()
                     to ensure kwargs are included in the cache key.

        Returns:
            An AgentClient configured for the orchestration.

        Raises:
            ValueError: If no builder is registered for the agent type.
        """
        # For now, this delegates to create() with the orchestration's agent type
        # Future enhancements could support per-orchestration config overrides
        return self.create(orch_agent_type, config)

    def clear_cache(self) -> None:
        """Clear the client cache.

        This is primarily useful for testing or when configuration changes.
        """
        self._cache.clear()
        logger.debug("Cleared agent client cache")

    @property
    def registered_types(self) -> list[AgentType]:
        """Return list of registered agent types."""
        return list(self._builders.keys())


def _build_claude_sdk_client(config: Config) -> AgentClient:
    """Builder function for ClaudeSdkAgentClient.

    Args:
        config: Configuration object.

    Returns:
        A configured ClaudeSdkAgentClient instance.
    """
    from sentinel.agent_clients.claude_sdk import ClaudeSdkAgentClient

    return ClaudeSdkAgentClient(
        config=config,
        base_workdir=config.agent_workdir,
        log_base_dir=config.agent_logs_dir,
    )


def _build_cursor_client(config: Config) -> AgentClient:
    """Builder function for CursorAgentClient.

    Args:
        config: Configuration object.

    Returns:
        A configured CursorAgentClient instance.
    """
    from sentinel.agent_clients.cursor import CursorAgentClient

    return CursorAgentClient(
        config=config,
        base_workdir=config.agent_workdir,
        log_base_dir=config.agent_logs_dir,
    )


def create_default_factory(config: Config) -> AgentClientFactory:
    """Create a factory with default builders registered.

    This is the recommended way to create a factory for production use.
    It registers builders for all supported agent types.

    Args:
        config: Configuration object (used for logging context).

    Returns:
        An AgentClientFactory with all default builders registered.
    """
    factory = AgentClientFactory()

    # Register the Claude SDK builder
    factory.register("claude", _build_claude_sdk_client)

    # Register the Cursor CLI builder
    factory.register("cursor", _build_cursor_client)

    logger.info(f"Created default factory with registered types: {factory.registered_types}")
    return factory
