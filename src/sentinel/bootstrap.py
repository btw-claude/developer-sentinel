"""Bootstrap and dependency wiring for Developer Sentinel.

This module provides the startup and initialization logic for the Sentinel
application, including:
- Configuration loading with CLI overrides
- Client initialization (Jira, GitHub)
- Agent factory creation
- Orchestration loading
- Sentinel instance assembly

The bootstrap module acts as the composition root, wiring together all
dependencies before the application starts running.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from sentinel.agent_clients.factory import create_default_factory
from sentinel.agent_logger import AgentLogger
from sentinel.config import Config, load_config
from sentinel.github_poller import GitHubClient
from sentinel.github_rest_client import GitHubRestClient, GitHubRestTagClient, GitHubTagClient
from sentinel.logging import get_logger, setup_logging
from sentinel.orchestration import Orchestration, OrchestrationError, load_orchestrations
from sentinel.poller import JiraClient
from sentinel.rest_clients import JiraRestClient, JiraRestTagClient
from sentinel.sdk_clients import JiraSdkClient, JiraSdkTagClient
from sentinel.tag_manager import JiraTagClient
from sentinel.types import TriggerSource

if TYPE_CHECKING:
    from sentinel.agent_clients.factory import AgentClientFactory
    from sentinel.main import Sentinel

logger = get_logger(__name__)


class BootstrapContext:
    """Container for all bootstrapped dependencies.

    This class holds all the initialized components needed to create
    a Sentinel instance. It provides a clean interface for dependency
    management during startup.
    """

    def __init__(
        self,
        config: Config,
        orchestrations: list[Orchestration],
        jira_client: JiraClient,
        tag_client: JiraTagClient,
        agent_factory: AgentClientFactory,
        agent_logger: AgentLogger,
        github_client: GitHubClient | None = None,
        github_tag_client: GitHubTagClient | None = None,
    ) -> None:
        """Initialize the bootstrap context.

        Args:
            config: Application configuration.
            orchestrations: List of loaded orchestrations.
            jira_client: Jira client for polling.
            tag_client: Jira client for tag operations.
            agent_factory: Factory for creating agent clients.
            agent_logger: Logger for agent execution.
            github_client: Optional GitHub client for polling.
            github_tag_client: Optional GitHub client for tag operations.
        """
        self.config = config
        self.orchestrations = orchestrations
        self.jira_client = jira_client
        self.tag_client = tag_client
        self.agent_factory = agent_factory
        self.agent_logger = agent_logger
        self.github_client = github_client
        self.github_tag_client = github_tag_client


def apply_cli_overrides(config: Config, parsed: argparse.Namespace) -> Config:
    """Apply CLI argument overrides to the configuration.

    Args:
        config: Base configuration loaded from environment.
        parsed: Parsed command-line arguments.

    Returns:
        New Config instance with CLI overrides applied.
    """
    overrides: dict[str, Any] = {}

    if parsed.config_dir:
        overrides["orchestrations_dir"] = parsed.config_dir
    if parsed.interval:
        overrides["poll_interval"] = parsed.interval
    if parsed.log_level:
        overrides["log_level"] = parsed.log_level

    if overrides:
        return replace(config, **overrides)
    return config


def create_jira_clients(config: Config) -> tuple[JiraClient, JiraTagClient]:
    """Create Jira clients based on configuration.

    Selects between REST API clients (preferred) and SDK clients
    based on whether Jira REST API credentials are configured.

    Args:
        config: Application configuration.

    Returns:
        Tuple of (jira_client, tag_client).
    """
    if config.jira_configured:
        logger.info("Using Jira REST API clients (direct HTTP)")
        jira_client: JiraClient = JiraRestClient(
            base_url=config.jira_base_url,
            email=config.jira_email,
            api_token=config.jira_api_token,
        )
        tag_client: JiraTagClient = JiraRestTagClient(
            base_url=config.jira_base_url,
            email=config.jira_email,
            api_token=config.jira_api_token,
        )
    else:
        logger.info("Using Jira SDK clients (via Claude Agent SDK)")
        logger.warning(
            "Jira REST API not configured. Set JIRA_BASE_URL, JIRA_EMAIL, "
            "and JIRA_API_TOKEN for faster polling."
        )
        jira_client = JiraSdkClient(config)
        tag_client = JiraSdkTagClient(config)

    return jira_client, tag_client


def create_github_clients(
    config: Config,
    orchestrations: list[Orchestration],
) -> tuple[GitHubClient | None, GitHubTagClient | None]:
    """Create GitHub clients if configured.

    Args:
        config: Application configuration.
        orchestrations: List of orchestrations (used for warning messages).

    Returns:
        Tuple of (github_client, github_tag_client), both may be None.
    """
    if config.github_configured:
        logger.info("Using GitHub REST API clients (direct HTTP)")
        github_client: GitHubClient | None = GitHubRestClient(
            token=config.github_token,
            base_url=config.github_api_url if config.github_api_url else None,
        )
        github_tag_client: GitHubTagClient | None = GitHubRestTagClient(
            token=config.github_token,
            base_url=config.github_api_url if config.github_api_url else None,
        )
    else:
        github_client = None
        github_tag_client = None

        # Warn if there are GitHub-triggered orchestrations but no GitHub config
        github_orchestrations = [
            o for o in orchestrations if o.trigger.source == TriggerSource.GITHUB.value
        ]
        if github_orchestrations:
            logger.warning(
                f"Found {len(github_orchestrations)} GitHub-triggered orchestrations "
                "but GitHub is not configured. Set GITHUB_TOKEN to enable GitHub polling."
            )

    return github_client, github_tag_client


def bootstrap(parsed: argparse.Namespace) -> BootstrapContext | None:
    """Bootstrap the application with all dependencies.

    This is the main entry point for application initialization. It:
    1. Loads and configures settings
    2. Sets up logging
    3. Loads orchestrations
    4. Initializes all clients

    Args:
        parsed: Parsed command-line arguments.

    Returns:
        BootstrapContext with all initialized dependencies, or None if
        initialization failed (e.g., no orchestrations found).
    """
    # Load configuration
    config = load_config(parsed.env_file)
    config = apply_cli_overrides(config, parsed)

    # Setup logging
    setup_logging(config.log_level, json_format=config.log_json)

    # Load orchestrations
    logger.info(f"Loading orchestrations from {config.orchestrations_dir}")
    try:
        orchestrations = load_orchestrations(config.orchestrations_dir)
    except OSError as e:
        logger.error(
            f"Failed to load orchestrations due to file system error: {e}",
            extra={"orchestrations_dir": str(config.orchestrations_dir)},
        )
        return None
    except (KeyError, ValueError) as e:
        logger.error(
            f"Failed to load orchestrations due to configuration error: {e}",
            extra={"orchestrations_dir": str(config.orchestrations_dir)},
        )
        return None
    except OrchestrationError as e:
        logger.error(
            f"Failed to load orchestrations: {e}",
            extra={"orchestrations_dir": str(config.orchestrations_dir)},
        )
        return None

    if not orchestrations:
        logger.warning("No orchestrations found, exiting")
        return None

    logger.info(f"Loaded {len(orchestrations)} orchestrations")

    # Initialize clients
    jira_client, tag_client = create_jira_clients(config)
    github_client, github_tag_client = create_github_clients(config, orchestrations)

    # Create agent factory and logger
    agent_factory = create_default_factory(config)
    agent_logger = AgentLogger(base_dir=config.agent_logs_dir)

    logger.info(f"Initialized agent factory with types: {agent_factory.registered_types}")

    return BootstrapContext(
        config=config,
        orchestrations=orchestrations,
        jira_client=jira_client,
        tag_client=tag_client,
        agent_factory=agent_factory,
        agent_logger=agent_logger,
        github_client=github_client,
        github_tag_client=github_tag_client,
    )


def create_sentinel_from_context(context: BootstrapContext) -> Sentinel:
    """Create a Sentinel instance from a bootstrap context.

    Args:
        context: Bootstrap context with all initialized dependencies.

    Returns:
        Configured Sentinel instance.
    """
    from sentinel.main import Sentinel

    return Sentinel(
        config=context.config,
        orchestrations=context.orchestrations,
        jira_client=context.jira_client,
        tag_client=context.tag_client,
        agent_factory=context.agent_factory,
        agent_logger=context.agent_logger,
        github_client=context.github_client,
        github_tag_client=context.github_tag_client,
    )


__all__ = [
    "BootstrapContext",
    "apply_cli_overrides",
    "bootstrap",
    "create_github_clients",
    "create_jira_clients",
    "create_sentinel_from_context",
]
