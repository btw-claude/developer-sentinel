"""Tests for the dependency injection container."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

from dependency_injector import providers

from sentinel.config import Config
from sentinel.container import (
    SentinelContainer,
    create_agent_factory,
    create_agent_logger,
    create_container,
    create_github_rest_client,
    create_github_rest_tag_client,
    create_jira_rest_client,
    create_jira_rest_tag_client,
    create_jira_sdk_client,
    create_jira_sdk_tag_client,
    create_sentinel,
    create_tag_manager,
    create_test_container,
)


class TestFactoryFunctions:
    """Tests for individual factory functions."""

    def test_create_jira_rest_client(self) -> None:
        """Test creating Jira REST client."""
        config = Config(
            jira_base_url="https://example.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
        )

        client = create_jira_rest_client(config)

        assert client is not None
        assert hasattr(client, "search_issues")

    def test_create_jira_sdk_client(self) -> None:
        """Test creating Jira SDK client."""
        config = Config()

        client = create_jira_sdk_client(config)

        assert client is not None
        assert hasattr(client, "search_issues")

    def test_create_jira_rest_tag_client(self) -> None:
        """Test creating Jira REST tag client."""
        config = Config(
            jira_base_url="https://example.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
        )

        client = create_jira_rest_tag_client(config)

        assert client is not None
        assert hasattr(client, "add_label")
        assert hasattr(client, "remove_label")

    def test_create_jira_sdk_tag_client(self) -> None:
        """Test creating Jira SDK tag client."""
        config = Config()

        client = create_jira_sdk_tag_client(config)

        assert client is not None
        assert hasattr(client, "add_label")
        assert hasattr(client, "remove_label")

    def test_create_github_rest_client_when_configured(self) -> None:
        """Test creating GitHub REST client when token is configured."""
        config = Config(github_token="test-token")

        client = create_github_rest_client(config)

        assert client is not None

    def test_create_github_rest_client_when_not_configured(self) -> None:
        """Test creating GitHub REST client returns None when not configured."""
        config = Config()

        client = create_github_rest_client(config)

        assert client is None

    def test_create_github_rest_tag_client_when_configured(self) -> None:
        """Test creating GitHub REST tag client when token is configured."""
        config = Config(github_token="test-token")

        client = create_github_rest_tag_client(config)

        assert client is not None

    def test_create_github_rest_tag_client_when_not_configured(self) -> None:
        """Test creating GitHub REST tag client returns None when not configured."""
        config = Config()

        client = create_github_rest_tag_client(config)

        assert client is None

    def test_create_agent_factory(self) -> None:
        """Test creating agent factory."""
        config = Config()

        factory = create_agent_factory(config)

        assert factory is not None
        assert "claude" in factory.registered_types
        assert "cursor" in factory.registered_types

    def test_create_agent_logger(self) -> None:
        """Test creating agent logger."""
        config = Config()

        logger = create_agent_logger(config)

        assert logger is not None

    def test_create_tag_manager(self) -> None:
        """Test creating tag manager."""
        mock_jira_tag = Mock()
        mock_github_tag = Mock()

        manager = create_tag_manager(mock_jira_tag, mock_github_tag)

        assert manager is not None
        assert manager.client is mock_jira_tag
        assert manager.github_client is mock_github_tag

    def test_create_tag_manager_without_github(self) -> None:
        """Test creating tag manager without GitHub client."""
        mock_jira_tag = Mock()

        manager = create_tag_manager(mock_jira_tag, None)

        assert manager is not None
        assert manager.client is mock_jira_tag
        assert manager.github_client is None


class TestCreateSentinel:
    """Tests for the create_sentinel factory function."""

    def test_create_sentinel_with_all_dependencies(self) -> None:
        """Test creating Sentinel with all dependencies (new DI pattern)."""
        config = Config()
        orchestrations: list = []
        mock_jira_tag = Mock()
        mock_agent_factory = MagicMock()
        mock_agent_factory.create_for_orchestration.return_value = Mock()
        mock_agent_logger = Mock()
        mock_jira_poller = Mock()
        mock_router = Mock()

        sentinel = create_sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_tag_client=mock_jira_tag,
            agent_factory=mock_agent_factory,
            agent_logger=mock_agent_logger,
            jira_poller=mock_jira_poller,
            router=mock_router,
        )

        assert sentinel is not None
        assert sentinel.config is config
        assert sentinel.orchestrations is orchestrations

    def test_create_sentinel_with_github_clients(self) -> None:
        """Test creating Sentinel with GitHub poller (new DI pattern)."""
        config = Config()
        orchestrations: list = []
        mock_jira_tag = Mock()
        mock_agent_factory = MagicMock()
        mock_agent_factory.create_for_orchestration.return_value = Mock()
        mock_agent_logger = Mock()
        mock_jira_poller = Mock()
        mock_router = Mock()
        mock_github_poller = Mock()
        mock_github_tag = Mock()

        sentinel = create_sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_tag_client=mock_jira_tag,
            agent_factory=mock_agent_factory,
            agent_logger=mock_agent_logger,
            jira_poller=mock_jira_poller,
            router=mock_router,
            github_poller=mock_github_poller,
            github_tag_client=mock_github_tag,
        )

        assert sentinel is not None
        assert sentinel.github_poller is mock_github_poller


class TestSentinelContainer:
    """Tests for the SentinelContainer class."""

    def test_container_has_expected_structure(self) -> None:
        """Test that container has expected sub-containers."""
        container = SentinelContainer()

        assert hasattr(container, "config")
        assert hasattr(container, "orchestrations")
        assert hasattr(container, "clients")
        assert hasattr(container, "services")
        assert hasattr(container, "sentinel")

    def test_override_config(self) -> None:
        """Test overriding config provider."""
        container = SentinelContainer()
        config = Config(poll_interval=999)

        container.config.override(providers.Object(config))

        assert container.config() is config
        assert container.config().poll_interval == 999


class TestCreateTestContainer:
    """Tests for the create_test_container function."""

    def test_create_test_container_with_defaults(self) -> None:
        """Test creating test container with default config."""
        container = create_test_container()

        assert container is not None
        assert container.config() is not None
        assert container.orchestrations() == []

    def test_create_test_container_with_custom_config(self) -> None:
        """Test creating test container with custom config."""
        config = Config(poll_interval=5)

        container = create_test_container(config=config)

        assert container.config() is config
        assert container.config().poll_interval == 5

    def test_create_test_container_with_orchestrations(self) -> None:
        """Test creating test container with orchestrations."""
        orchestrations = [Mock(), Mock()]

        container = create_test_container(orchestrations=orchestrations)

        assert container.orchestrations() == orchestrations

    def test_override_clients_in_test_container(self) -> None:
        """Test that clients can be overridden in test container."""
        container = create_test_container()
        mock_jira = Mock()

        container.clients.jira_client.override(providers.Object(mock_jira))

        assert container.clients.jira_client() is mock_jira


class TestCreateContainer:
    """Tests for the create_container function."""

    def test_create_container_with_jira_rest_config(self, tmp_path) -> None:
        """Test creating container with Jira REST configuration."""
        # Create empty orchestrations dir
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            jira_base_url="https://example.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])

        # Should use REST clients
        jira_client = container.clients.jira_client()
        assert jira_client is not None
        # Check it's the REST implementation
        assert jira_client.__class__.__name__ == "JiraRestClient"

    def test_create_container_with_jira_sdk_config(self, tmp_path) -> None:
        """Test creating container with Jira SDK configuration (no REST creds)."""
        # Create empty orchestrations dir
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])

        # Should use SDK clients
        jira_client = container.clients.jira_client()
        assert jira_client is not None
        # Check it's the SDK implementation
        assert jira_client.__class__.__name__ == "JiraSdkClient"

    def test_create_container_with_github_config(self, tmp_path) -> None:
        """Test creating container with GitHub configuration."""
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            github_token="test-token",
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])

        github_client = container.clients.github_client()
        assert github_client is not None

    def test_create_container_without_github_config(self, tmp_path) -> None:
        """Test creating container without GitHub configuration."""
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])

        github_client = container.clients.github_client()
        assert github_client is None

    def test_create_container_services_are_configured(self, tmp_path) -> None:
        """Test that services are properly configured."""
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            jira_base_url="https://example.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])

        # Check services are configured
        agent_factory = container.services.agent_factory()
        assert agent_factory is not None
        assert "claude" in agent_factory.registered_types

        agent_logger = container.services.agent_logger()
        assert agent_logger is not None

        tag_manager = container.services.tag_manager()
        assert tag_manager is not None


class TestContainerIntegration:
    """Integration tests for the DI container."""

    def test_create_sentinel_via_container(self, tmp_path) -> None:
        """Test creating Sentinel instance through the container."""
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            jira_base_url="https://example.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])
        sentinel = container.sentinel()

        assert sentinel is not None
        assert sentinel.config is config

    def test_container_provides_singletons(self, tmp_path) -> None:
        """Test that container provides singleton instances."""
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            jira_base_url="https://example.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])

        # Get clients multiple times
        jira1 = container.clients.jira_client()
        jira2 = container.clients.jira_client()

        # Should be the same instance
        assert jira1 is jira2

    def test_sentinel_is_factory_not_singleton(self, tmp_path) -> None:
        """Test that Sentinel is created fresh each time (Factory provider)."""
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            jira_base_url="https://example.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])

        sentinel1 = container.sentinel()
        sentinel2 = container.sentinel()

        # Should be different instances
        assert sentinel1 is not sentinel2

    def test_override_providers(self, tmp_path) -> None:
        """Test overriding providers with mocks."""
        orch_dir = tmp_path / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            jira_base_url="https://example.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
            orchestrations_dir=orch_dir,
        )

        container = create_container(config=config, orchestrations=[])

        # Get original client
        original_jira = container.clients.jira_client()
        assert original_jira.__class__.__name__ == "JiraRestClient"

        # Override with mock
        mock_jira = Mock()
        container.clients.jira_client.override(providers.Object(mock_jira))
        assert container.clients.jira_client() is mock_jira

        # Override again with a different mock
        mock_jira_2 = Mock()
        container.clients.jira_client.override(providers.Object(mock_jira_2))
        assert container.clients.jira_client() is mock_jira_2
        assert container.clients.jira_client() is not mock_jira
