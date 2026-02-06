"""Tests for Sentinel hot-reload observability metrics."""

from pathlib import Path

from sentinel.main import Sentinel
from sentinel.orchestration import Orchestration
from tests.helpers import make_agent_factory, make_config, set_mtime_in_future
from tests.mocks import MockAgentClient, MockAgentClientFactory, MockJiraPoller, MockTagClient


class TestSentinelHotReloadMetrics:
    """Tests for hot-reload observability metrics."""

    agent_factory: MockAgentClientFactory
    agent_client: MockAgentClient

    def setup_method(self) -> None:
        """Set up test fixtures shared across all test methods."""
        # agent_client is retained for future test convenience; it is not
        # currently referenced by any test methods in this class.
        self.agent_factory, self.agent_client = make_agent_factory()

    def test_get_hot_reload_metrics_returns_dict(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_poller: MockJiraPoller,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that get_hot_reload_metrics returns a dict with expected keys."""
        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=mock_jira_poller,
            agent_factory=self.agent_factory,
            tag_client=mock_tag_client,
        )

        metrics = sentinel.get_hot_reload_metrics()

        assert isinstance(metrics, dict)
        assert "orchestrations_loaded_total" in metrics
        assert "orchestrations_unloaded_total" in metrics
        assert "orchestrations_reloaded_total" in metrics
        assert metrics["orchestrations_loaded_total"] == 0
        assert metrics["orchestrations_unloaded_total"] == 0
        assert metrics["orchestrations_reloaded_total"] == 0

    def test_loaded_counter_increments_on_new_file(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_poller: MockJiraPoller,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that loaded counter increments when new files are detected."""
        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=mock_jira_poller,
            agent_factory=self.agent_factory,
            tag_client=mock_tag_client,
        )

        assert sentinel.get_hot_reload_metrics()["orchestrations_loaded_total"] == 0

        (temp_orchestrations_dir / "new.yaml").write_text(
            """orchestrations:
  - name: new-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
        )

        sentinel.run_once()

        assert sentinel.get_hot_reload_metrics()["orchestrations_loaded_total"] == 1

    def test_unloaded_counter_increments_on_file_deletion(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_poller: MockJiraPoller,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that unloaded counter increments when files are deleted."""
        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=mock_jira_poller,
            agent_factory=self.agent_factory,
            tag_client=mock_tag_client,
        )

        orch_file = temp_orchestrations_dir / "removable.yaml"
        orch_file.write_text(
            """orchestrations:
  - name: removable-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
        )
        sentinel.run_once()

        assert sentinel.get_hot_reload_metrics()["orchestrations_loaded_total"] == 1
        assert sentinel.get_hot_reload_metrics()["orchestrations_unloaded_total"] == 0

        orch_file.unlink()
        sentinel.run_once()

        assert sentinel.get_hot_reload_metrics()["orchestrations_unloaded_total"] == 1

    def test_reloaded_counter_increments_on_file_modification(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_poller: MockJiraPoller,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that reloaded counter increments when files are modified."""
        orch_file = temp_orchestrations_dir / "modifiable.yaml"
        orch_file.write_text(
            """orchestrations:
  - name: original-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Original
"""
        )

        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=mock_jira_poller,
            agent_factory=self.agent_factory,
            tag_client=mock_tag_client,
        )

        assert sentinel.get_hot_reload_metrics()["orchestrations_reloaded_total"] == 0

        orch_file.write_text(
            """orchestrations:
  - name: modified-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Modified
"""
        )
        set_mtime_in_future(orch_file)

        sentinel.run_once()

        assert sentinel.get_hot_reload_metrics()["orchestrations_reloaded_total"] == 1

    def test_metrics_accumulate_over_multiple_operations(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_poller: MockJiraPoller,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that metrics accumulate correctly over multiple operations."""
        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=mock_jira_poller,
            agent_factory=self.agent_factory,
            tag_client=mock_tag_client,
        )

        (temp_orchestrations_dir / "file1.yaml").write_text(
            """orchestrations:
  - name: orch-1
    trigger:
      project: TEST
    agent:
      prompt: Test 1
"""
        )
        (temp_orchestrations_dir / "file2.yaml").write_text(
            """orchestrations:
  - name: orch-2
    trigger:
      project: TEST
    agent:
      prompt: Test 2
"""
        )
        sentinel.run_once()

        metrics = sentinel.get_hot_reload_metrics()
        assert metrics["orchestrations_loaded_total"] == 2
        assert metrics["orchestrations_unloaded_total"] == 0
        assert metrics["orchestrations_reloaded_total"] == 0

        (temp_orchestrations_dir / "file1.yaml").unlink()
        sentinel.run_once()

        metrics = sentinel.get_hot_reload_metrics()
        assert metrics["orchestrations_loaded_total"] == 2
        assert metrics["orchestrations_unloaded_total"] == 1
        assert metrics["orchestrations_reloaded_total"] == 0

        (temp_orchestrations_dir / "file2.yaml").write_text(
            """orchestrations:
  - name: orch-2-modified
    trigger:
      project: TEST
    agent:
      prompt: Test 2 Modified
"""
        )
        set_mtime_in_future(temp_orchestrations_dir / "file2.yaml")
        sentinel.run_once()

        metrics = sentinel.get_hot_reload_metrics()
        assert metrics["orchestrations_loaded_total"] == 2
        assert metrics["orchestrations_unloaded_total"] == 1
        assert metrics["orchestrations_reloaded_total"] == 1
