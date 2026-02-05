"""Dependency Injection container for the Sentinel application.

This module provides a centralized dependency injection container using the
dependency-injector library. It enables:

- Testability: Easy swapping of implementations for testing
- Loose coupling: Components depend on abstractions, not concrete implementations
- Configuration management: Centralized configuration through providers
- Lifecycle management: Singleton instances managed by the container

Usage:
    # Production setup
    container = create_container()
    sentinel = container.sentinel()

    # Test setup with mocks
    container = create_container()
    container.jira_client.override(MockJiraClient())
    sentinel = container.sentinel()

See docs/dependency-injection.md for more patterns and examples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dependency_injector import containers, providers

if TYPE_CHECKING:
    from sentinel.agent_clients.factory import AgentClientFactory
    from sentinel.agent_logger import AgentLogger
    from sentinel.config import Config
    from sentinel.github_poller import GitHubClient, GitHubPoller
    from sentinel.github_rest_client import GitHubTagClient
    from sentinel.main import Sentinel
    from sentinel.orchestration import Orchestration
    from sentinel.poller import JiraClient, JiraPoller
    from sentinel.router import Router
    from sentinel.tag_manager import JiraTagClient, TagManager


class ClientsContainer(containers.DeclarativeContainer):
    """Container for API clients (Jira, GitHub).

    This sub-container groups all API client dependencies, making it easy
    to swap entire client sets for integration testing or different environments.
    """

    config: providers.Dependency[Config] = providers.Dependency()

    # Jira clients - provided lazily to support both REST and SDK implementations
    # Using Dependency() makes it explicit these must be provided at container creation
    jira_client: providers.Dependency[JiraClient] = providers.Dependency()
    jira_tag_client: providers.Dependency[JiraTagClient] = providers.Dependency()

    # GitHub clients - optional, may be None if not configured
    github_client: providers.Dependency[GitHubClient | None] = providers.Dependency()
    github_tag_client: providers.Dependency[GitHubTagClient | None] = providers.Dependency()


class PollersContainer(containers.DeclarativeContainer):
    """Container for pollers (JiraPoller, GitHubPoller).

    This sub-container groups poller dependencies that fetch issues from
    external systems.
    """

    config: providers.Dependency[Config] = providers.Dependency()
    clients = providers.DependenciesContainer()

    # JiraPoller - polls Jira for issues matching triggers
    # Using Dependency() makes it explicit these must be provided at container creation
    jira_poller: providers.Dependency[JiraPoller] = providers.Dependency()

    # GitHubPoller - polls GitHub for issues/PRs matching triggers (optional)
    github_poller: providers.Dependency[GitHubPoller | None] = providers.Dependency()


class ServicesContainer(containers.DeclarativeContainer):
    """Container for core services.

    This sub-container groups service-layer dependencies that coordinate
    between clients and business logic.
    """

    config: providers.Dependency[Config] = providers.Dependency()
    clients = providers.DependenciesContainer()

    # AgentClientFactory - creates agent clients based on orchestration config
    # Using Dependency() makes it explicit these must be provided at container creation
    agent_factory: providers.Dependency[AgentClientFactory] = providers.Dependency()

    # AgentLogger - logs agent execution output
    agent_logger: providers.Dependency[AgentLogger] = providers.Dependency()

    # TagManager - manages issue labels/tags
    tag_manager: providers.Dependency[TagManager] = providers.Dependency()

    # Router - routes issues to matching orchestrations
    router: providers.Dependency[Router] = providers.Dependency()


class SentinelContainer(containers.DeclarativeContainer):
    """Main dependency injection container for the Sentinel application.

    This is the root container that wires together all sub-containers and
    provides the main Sentinel instance. It follows a hierarchical structure:

    SentinelContainer
    ├── config (Config)
    ├── orchestrations (list[Orchestration])
    ├── clients (ClientsContainer)
    │   ├── jira_client
    │   ├── jira_tag_client
    │   ├── github_client
    │   └── github_tag_client
    └── services (ServicesContainer)
        ├── agent_factory
        ├── agent_logger
        └── tag_manager

    Usage:
        # Create container with providers
        container = create_container()

        # Override for testing
        container.clients.jira_client.override(mock_jira)

        # Get Sentinel instance
        sentinel = container.sentinel()
    """

    # Wiring configuration - modules that can use the @inject decorator.
    # This enables automatic dependency injection via function parameter annotations.
    #
    # Currently, the @inject decorator is not actively used in the codebase.
    # This configuration is included to support future adoption of the decorator
    # pattern for scenarios where explicit provider access is cumbersome.
    #
    # Example future usage:
    #     from dependency_injector.wiring import inject, Provide
    #
    #     @inject
    #     def process_issues(
    #         jira_client: JiraClient = Provide[SentinelContainer.clients.jira_client]
    #     ):
    #         issues = jira_client.search_issues(...)
    #
    # To enable wiring, call container.wire() after creating the container.
    # See: https://python-dependency-injector.ets-labs.org/wiring.html
    wiring_config = containers.WiringConfiguration(
        modules=[
            "sentinel.main",
            "sentinel.app",
            "sentinel.bootstrap",
            "sentinel.cli",
            "sentinel.shutdown",
            "sentinel.dashboard_server",
        ]
    )

    # Core configuration - using Dependency() for type safety
    config: providers.Dependency[Config] = providers.Dependency()

    # Orchestrations list - using Dependency() for type safety
    orchestrations: providers.Dependency[list[Orchestration]] = providers.Dependency()

    # Sub-containers
    clients = providers.Container(
        ClientsContainer,
        config=config,
    )

    pollers = providers.Container(
        PollersContainer,
        config=config,
        clients=clients,
    )

    services = providers.Container(
        ServicesContainer,
        config=config,
        clients=clients,
    )

    # Main Sentinel instance - Factory provider allows creating multiple instances
    # Using Dependency() for type safety - overridden at runtime with Factory
    sentinel: providers.Dependency[Sentinel] = providers.Dependency()


def create_jira_rest_client(config: Config) -> JiraClient:
    """Create a Jira REST API client.

    Args:
        config: Application configuration.

    Returns:
        JiraRestClient instance configured with credentials from config.
    """
    from sentinel.rest_clients import JiraRestClient

    return JiraRestClient(
        base_url=config.jira.base_url,
        email=config.jira.email,
        api_token=config.jira.api_token,
    )


def create_jira_sdk_client(config: Config) -> JiraClient:
    """Create a Jira SDK client (uses Claude for API calls).

    Args:
        config: Application configuration.

    Returns:
        JiraSdkClient instance.
    """
    from sentinel.sdk_clients import JiraSdkClient

    return JiraSdkClient(config)


def create_jira_rest_tag_client(config: Config) -> JiraTagClient:
    """Create a Jira REST API tag client.

    Args:
        config: Application configuration.

    Returns:
        JiraRestTagClient instance configured with credentials from config.
    """
    from sentinel.rest_clients import JiraRestTagClient

    return JiraRestTagClient(
        base_url=config.jira.base_url,
        email=config.jira.email,
        api_token=config.jira.api_token,
    )


def create_jira_sdk_tag_client(config: Config) -> JiraTagClient:
    """Create a Jira SDK tag client (uses Claude for API calls).

    Args:
        config: Application configuration.

    Returns:
        JiraSdkTagClient instance.
    """
    from sentinel.sdk_clients import JiraSdkTagClient

    return JiraSdkTagClient(config)


def create_github_rest_client(config: Config) -> GitHubClient | None:
    """Create a GitHub REST API client if configured.

    Args:
        config: Application configuration.

    Returns:
        GitHubRestClient instance if GitHub is configured, None otherwise.
    """
    if not config.github.configured:
        return None

    from sentinel.github_rest_client import GitHubRestClient

    return GitHubRestClient(
        token=config.github.token,
        base_url=config.github.api_url if config.github.api_url else None,
    )


def create_github_rest_tag_client(config: Config) -> GitHubTagClient | None:
    """Create a GitHub REST API tag client if configured.

    Args:
        config: Application configuration.

    Returns:
        GitHubRestTagClient instance if GitHub is configured, None otherwise.
    """
    if not config.github.configured:
        return None

    from sentinel.github_rest_client import GitHubRestTagClient

    return GitHubRestTagClient(
        token=config.github.token,
        base_url=config.github.api_url if config.github.api_url else None,
    )


def create_jira_poller(jira_client: JiraClient, config: Config) -> JiraPoller:
    """Create a JiraPoller with the configured client.

    Args:
        jira_client: Jira client for API operations.
        config: Application configuration.

    Returns:
        JiraPoller configured with the client and epic link field from config.
    """
    from sentinel.poller import JiraPoller

    return JiraPoller(
        jira_client,
        epic_link_field=config.jira.epic_link_field,
    )


def create_github_poller(github_client: GitHubClient | None) -> GitHubPoller | None:
    """Create a GitHubPoller if GitHub client is configured.

    Uses factory pattern since the poller may be None if GitHub is not configured.

    Args:
        github_client: Optional GitHub client for API operations.

    Returns:
        GitHubPoller if GitHub client is provided, None otherwise.
    """
    if github_client is None:
        return None

    from sentinel.github_poller import GitHubPoller

    return GitHubPoller(github_client)


def create_router(orchestrations: list[Orchestration]) -> Router:
    """Create a Router with the provided orchestrations.

    Args:
        orchestrations: List of orchestration configurations.

    Returns:
        Router configured with the orchestrations.
    """
    from sentinel.router import Router

    return Router(orchestrations)


def create_agent_factory(config: Config) -> AgentClientFactory:
    """Create the agent client factory with default builders.

    Args:
        config: Application configuration.

    Returns:
        AgentClientFactory with claude and cursor builders registered.
    """
    from sentinel.agent_clients.factory import create_default_factory

    return create_default_factory(config)


def create_agent_logger(config: Config) -> AgentLogger:
    """Create the agent execution logger.

    Args:
        config: Application configuration.

    Returns:
        AgentLogger configured with the logs directory from config.
    """
    from sentinel.agent_logger import AgentLogger

    return AgentLogger(base_dir=config.execution.agent_logs_dir)


def create_tag_manager(
    jira_tag_client: JiraTagClient,
    github_tag_client: GitHubTagClient | None,
) -> TagManager:
    """Create the tag manager with configured clients.

    Args:
        jira_tag_client: Jira tag client for label operations.
        github_tag_client: Optional GitHub tag client for label operations.

    Returns:
        TagManager configured with the provided clients.
    """
    from sentinel.tag_manager import TagManager

    return TagManager(
        client=jira_tag_client,
        github_client=github_tag_client,
    )


def create_sentinel(
    config: Config,
    orchestrations: list[Orchestration],
    jira_tag_client: JiraTagClient,
    agent_factory: AgentClientFactory,
    agent_logger: AgentLogger,
    jira_poller: JiraPoller,
    router: Router,
    github_poller: GitHubPoller | None = None,
    github_tag_client: GitHubTagClient | None = None,
) -> Sentinel:
    """Create a Sentinel instance with all dependencies.

    This factory function creates a Sentinel instance with all required
    dependencies injected. It's the primary way to instantiate Sentinel
    when using the DI container.

    Args:
        config: Application configuration.
        orchestrations: List of orchestration configurations.
        jira_tag_client: Jira client for tag operations.
        agent_factory: Factory for creating agent clients.
        agent_logger: Logger for agent execution.
        jira_poller: Poller for fetching Jira issues.
        router: Router for matching issues to orchestrations.
        github_poller: Optional poller for GitHub issues/PRs.
        github_tag_client: Optional GitHub client for tag operations.

    Returns:
        Configured Sentinel instance.
    """
    from sentinel.main import Sentinel

    return Sentinel(
        config=config,
        orchestrations=orchestrations,
        tag_client=jira_tag_client,
        agent_factory=agent_factory,
        agent_logger=agent_logger,
        jira_poller=jira_poller,
        router=router,
        github_poller=github_poller,
        github_tag_client=github_tag_client,
    )


def create_container(
    config: Config | None = None,
    orchestrations: list[Orchestration] | None = None,
) -> SentinelContainer:
    """Create and configure the main DI container.

    This is the recommended way to create a container for production use.
    It sets up all providers with their default implementations based on
    the provided configuration.

    Args:
        config: Optional configuration. If not provided, loads from environment.
        orchestrations: Optional list of orchestrations. If not provided,
                       loads from the orchestrations directory.

    Returns:
        Fully configured SentinelContainer ready for use.

    Example:
        # Production usage
        container = create_container()
        sentinel = container.sentinel()
        sentinel.run()

        # Testing with mock Jira client
        container = create_container()
        container.clients.jira_client.override(providers.Object(mock_jira))
        sentinel = container.sentinel()
    """
    from sentinel.config import load_config
    from sentinel.orchestration import load_orchestrations

    # Load config if not provided
    if config is None:
        config = load_config()

    # Load orchestrations if not provided
    if orchestrations is None:
        orchestrations = load_orchestrations(config.execution.orchestrations_dir)

    # Create container
    container = SentinelContainer()

    # Configure core providers
    container.config.override(providers.Object(config))
    container.orchestrations.override(providers.Object(orchestrations))

    # Configure Jira clients based on configuration
    if config.jira.configured:
        container.clients.jira_client.override(providers.Singleton(create_jira_rest_client, config))
        container.clients.jira_tag_client.override(
            providers.Singleton(create_jira_rest_tag_client, config)
        )
    else:
        container.clients.jira_client.override(providers.Singleton(create_jira_sdk_client, config))
        container.clients.jira_tag_client.override(
            providers.Singleton(create_jira_sdk_tag_client, config)
        )

    # Configure GitHub clients (may be None)
    container.clients.github_client.override(providers.Singleton(create_github_rest_client, config))
    container.clients.github_tag_client.override(
        providers.Singleton(create_github_rest_tag_client, config)
    )

    # Configure pollers
    container.pollers.jira_poller.override(
        providers.Singleton(
            create_jira_poller,
            container.clients.jira_client,
            config,
        )
    )
    container.pollers.github_poller.override(
        providers.Singleton(
            create_github_poller,
            container.clients.github_client,
        )
    )

    # Configure services
    container.services.agent_factory.override(providers.Singleton(create_agent_factory, config))
    container.services.agent_logger.override(providers.Singleton(create_agent_logger, config))
    container.services.tag_manager.override(
        providers.Singleton(
            create_tag_manager,
            container.clients.jira_tag_client,
            container.clients.github_tag_client,
        )
    )
    container.services.router.override(
        providers.Singleton(
            create_router,
            container.orchestrations,
        )
    )

    # Configure Sentinel factory
    container.sentinel.override(
        providers.Factory(
            create_sentinel,
            config=container.config,
            orchestrations=container.orchestrations,
            jira_tag_client=container.clients.jira_tag_client,
            agent_factory=container.services.agent_factory,
            agent_logger=container.services.agent_logger,
            jira_poller=container.pollers.jira_poller,
            router=container.services.router,
            github_poller=container.pollers.github_poller,
            github_tag_client=container.clients.github_tag_client,
        )
    )

    return container


def create_test_container(
    config: Config | None = None,
    orchestrations: list[Orchestration] | None = None,
) -> SentinelContainer:
    """Create a container pre-configured for testing.

    This creates a container with Dependency() providers that must be
    overridden with mocks before use. Unlike create_container(), this doesn't
    require valid Jira/GitHub credentials.

    Important:
        The container uses Dependency() providers for clients, pollers, services,
        and the sentinel instance. These providers will raise an error if accessed
        without being overridden first. Test authors must explicitly override all
        providers they intend to use with appropriate mock objects.

        For more information on Dependency() providers and how provider overrides
        work, see the dependency-injector library documentation:
        https://python-dependency-injector.ets-labs.org/providers/dependency.html

        Providers that need to be overridden before use:
        - container.clients.jira_client
        - container.clients.jira_tag_client
        - container.clients.github_client
        - container.clients.github_tag_client
        - container.pollers.jira_poller
        - container.pollers.github_poller
        - container.services.agent_factory
        - container.services.agent_logger
        - container.services.tag_manager
        - container.services.router
        - container.sentinel

    Args:
        config: Optional test configuration.
        orchestrations: Optional list of test orchestrations.

    Returns:
        SentinelContainer configured for testing with Dependency() providers.

    Raises:
        dependency_injector.errors.Error: When accessing a Dependency() provider
            that has not been overridden.

    Example:
        from unittest.mock import Mock

        container = create_test_container()
        container.clients.jira_client.override(providers.Object(Mock()))
        container.clients.jira_tag_client.override(providers.Object(Mock()))
        # ... override other providers as needed for your test ...
        sentinel = container.sentinel()
    """
    from sentinel.config import Config

    # Use default test config if not provided
    if config is None:
        config = Config()

    if orchestrations is None:
        orchestrations = []

    container = SentinelContainer()

    # Set up basic providers
    container.config.override(providers.Object(config))
    container.orchestrations.override(providers.Object(orchestrations))

    # Leave client providers as placeholders - tests should override them
    # This makes it explicit that tests need to provide their own mocks

    return container


# Convenience exports for common patterns
__all__ = [
    "SentinelContainer",
    "ClientsContainer",
    "PollersContainer",
    "ServicesContainer",
    "create_container",
    "create_test_container",
    "create_sentinel",
    "create_jira_rest_client",
    "create_jira_sdk_client",
    "create_jira_rest_tag_client",
    "create_jira_sdk_tag_client",
    "create_github_rest_client",
    "create_github_rest_tag_client",
    "create_jira_poller",
    "create_github_poller",
    "create_router",
    "create_agent_factory",
    "create_agent_logger",
    "create_tag_manager",
]
