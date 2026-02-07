"""Tests for Sentinel per-orchestration concurrency limits."""

import time
from typing import Any

from sentinel.agent_clients.base import AgentClient, AgentRunResult
from sentinel.main import Sentinel
from sentinel.orchestration import Orchestration
from sentinel.types import AgentType

# Import shared fixtures and helpers from conftest.py
from tests.conftest import (
    MockJiraPoller,
    MockTagClient,
    TrackingAgentClient,
    make_agent_factory,
    make_config,
    make_issue,
    make_orchestration,
)
from tests.mocks import MockAgentClientFactory


class TestPerOrchestrationConcurrencyLimits:
    """Tests for per-orchestration concurrency limits.

    These tests verify:
    - max_concurrent parsing and validation
    - Slot checking with per-orchestration limits
    - Integration with global concurrency limits
    - Per-orchestration count tracking
    """

    def test_orchestration_without_max_concurrent_uses_global_limit(self) -> None:
        """Test that orchestrations without max_concurrent use only global limit."""
        # Use shared TrackingAgentClient for concurrent execution tracking
        agent_client = TrackingAgentClient(execution_delay=0.05)
        agent_factory = MockAgentClientFactory(agent_client)

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[
                make_issue(key="TEST-1", summary="Issue 1", labels=["review"]),
                make_issue(key="TEST-2", summary="Issue 2", labels=["review"]),
                make_issue(key="TEST-3", summary="Issue 3", labels=["review"]),
            ],
            tag_client=tag_client,
        )
        config = make_config(max_concurrent_executions=2)
        # Orchestration without max_concurrent (uses None by default)
        orchestrations = [make_orchestration(tags=["review"], max_concurrent=None)]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # All 3 issues should be processed (global limit of 2 allows all eventually)
        results = sentinel.run_once_and_wait()
        assert len(results) == 3
        assert all(r.succeeded for r in results)
        # Global limit of 2 should allow up to 2 concurrent
        assert agent_client.max_concurrent_seen <= 2

    def test_per_orch_limit_stricter_than_global(self) -> None:
        """Test per-orchestration limit is respected when stricter than global."""
        # Use shared TrackingAgentClient for concurrent execution tracking
        agent_client = TrackingAgentClient(execution_delay=0.1)
        agent_factory = MockAgentClientFactory(agent_client)

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[
                make_issue(key="TEST-1", summary="Issue 1", labels=["review"]),
                make_issue(key="TEST-2", summary="Issue 2", labels=["review"]),
                make_issue(key="TEST-3", summary="Issue 3", labels=["review"]),
            ],
            tag_client=tag_client,
        )
        # Global limit of 5, but orchestration limit of 1
        config = make_config(max_concurrent_executions=5)
        orchestrations = [
            make_orchestration(name="limited-orch", tags=["review"], max_concurrent=1)
        ]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        assert len(results) == 3
        # Per-orch limit of 1 should cap concurrent executions
        assert agent_client.max_concurrent_seen == 1

    def test_global_limit_stricter_than_per_orch(self) -> None:
        """Test global limit is respected when stricter than per-orchestration."""
        # Use shared TrackingAgentClient for concurrent execution tracking
        agent_client = TrackingAgentClient(execution_delay=0.1)
        agent_factory = MockAgentClientFactory(agent_client)

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[
                make_issue(key="TEST-1", summary="Issue 1", labels=["review"]),
                make_issue(key="TEST-2", summary="Issue 2", labels=["review"]),
                make_issue(key="TEST-3", summary="Issue 3", labels=["review"]),
            ],
            tag_client=tag_client,
        )
        # Global limit of 1, per-orch limit of 10
        config = make_config(max_concurrent_executions=1)
        orchestrations = [
            make_orchestration(name="high-limit-orch", tags=["review"], max_concurrent=10)
        ]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        assert len(results) == 3
        # Global limit of 1 should cap concurrent executions
        assert agent_client.max_concurrent_seen == 1

    def test_multiple_orchestrations_with_different_limits(self) -> None:
        """Test multiple orchestrations respect their individual limits."""
        # Use shared TrackingAgentClient with per-orchestration tracking enabled
        agent_client = TrackingAgentClient(execution_delay=0.05, track_per_orch=True)
        agent_factory = MockAgentClientFactory(agent_client)

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[
                # Issues for orch-1 (max_concurrent=1)
                make_issue(key="TEST-1", summary="Issue 1", labels=["review"]),
                make_issue(key="TEST-2", summary="Issue 2", labels=["review"]),
                # Issues for orch-2 (max_concurrent=2)
                make_issue(key="TEST-3", summary="Issue 3", labels=["deploy"]),
                make_issue(key="TEST-4", summary="Issue 4", labels=["deploy"]),
            ],
            tag_client=tag_client,
        )
        # High global limit, different per-orch limits
        config = make_config(max_concurrent_executions=10)
        orchestrations = [
            make_orchestration(name="orch-1", tags=["review"], max_concurrent=1),
            make_orchestration(name="orch-2", tags=["deploy"], max_concurrent=2),
        ]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        # All issues should be processed
        assert len(results) == 4
        # orch-1 should never exceed its limit of 1
        assert agent_client.orch_max_concurrent.get("orch-1", 0) <= 1
        # orch-2 should never exceed its limit of 2
        assert agent_client.orch_max_concurrent.get("orch-2", 0) <= 2

    def test_per_orch_count_increment_and_decrement(self) -> None:
        """Test that per-orchestration counts are properly tracked."""
        # Use shared TrackingAgentClient for consistent test behavior
        agent_client = TrackingAgentClient(execution_delay=0.05)
        agent_factory = MockAgentClientFactory(agent_client)

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[
                make_issue(key="TEST-1", summary="Issue 1", labels=["review"]),
            ],
            tag_client=tag_client,
        )
        config = make_config(max_concurrent_executions=5)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"], max_concurrent=2)]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Before execution, count should be 0
        assert sentinel._state_tracker._per_orch_active_counts.get("test-orch", 0) == 0

        # Run and wait for completion
        results = sentinel.run_once_and_wait()

        assert len(results) == 1
        # After execution completes, count should return to 0
        assert sentinel._state_tracker._per_orch_active_counts.get("test-orch", 0) == 0

    def test_increment_per_orch_count_returns_new_count(self) -> None:
        """Test _increment_per_orch_count returns the new count."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # First increment
        count = sentinel._state_tracker.increment_per_orch_count("test-orch")
        assert count == 1
        assert sentinel._state_tracker._per_orch_active_counts["test-orch"] == 1

        # Second increment
        count = sentinel._state_tracker.increment_per_orch_count("test-orch")
        assert count == 2
        assert sentinel._state_tracker._per_orch_active_counts["test-orch"] == 2

        # Increment different orchestration
        count = sentinel._state_tracker.increment_per_orch_count("other-orch")
        assert count == 1
        assert sentinel._state_tracker._per_orch_active_counts["other-orch"] == 1
        assert sentinel._state_tracker._per_orch_active_counts["test-orch"] == 2

    def test_decrement_per_orch_count_returns_new_count(self) -> None:
        """Test _decrement_per_orch_count returns the new count."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Setup initial counts
        sentinel._state_tracker._per_orch_active_counts["test-orch"] = 3

        # First decrement
        count = sentinel._state_tracker.decrement_per_orch_count("test-orch")
        assert count == 2
        assert sentinel._state_tracker._per_orch_active_counts["test-orch"] == 2

        # Second decrement
        count = sentinel._state_tracker.decrement_per_orch_count("test-orch")
        assert count == 1

        # Third decrement
        count = sentinel._state_tracker.decrement_per_orch_count("test-orch")
        assert count == 0

    def test_decrement_per_orch_count_clamps_to_zero(self) -> None:
        """Test _decrement_per_orch_count does not go below zero."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Decrement when count is 0
        count = sentinel._state_tracker.decrement_per_orch_count("test-orch")
        assert count == 0
        assert sentinel._state_tracker._per_orch_active_counts["test-orch"] == 0

        # Decrement again
        count = sentinel._state_tracker.decrement_per_orch_count("test-orch")
        assert count == 0

    def test_decrement_per_orch_count_cleans_up_at_zero(self) -> None:
        """Test _decrement_per_orch_count removes entry when count reaches 0."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Increment to 1
        sentinel._state_tracker.increment_per_orch_count("test-orch")
        assert "test-orch" in sentinel._state_tracker._per_orch_active_counts

        # Decrement to 0 - should clean up entry
        count = sentinel._state_tracker.decrement_per_orch_count("test-orch")
        assert count == 0
        # Entry should be removed when count reaches 0
        assert "test-orch" not in sentinel._state_tracker._per_orch_active_counts

    def test_get_per_orch_count_returns_count(self) -> None:
        """Test get_per_orch_count returns the current count for observability."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Initially should return 0 for unknown orchestration
        assert sentinel.get_per_orch_count("test-orch") == 0

        # After incrementing, should return the count
        sentinel._state_tracker.increment_per_orch_count("test-orch")
        assert sentinel.get_per_orch_count("test-orch") == 1

        sentinel._state_tracker.increment_per_orch_count("test-orch")
        assert sentinel.get_per_orch_count("test-orch") == 2

        # Check a different orchestration
        assert sentinel.get_per_orch_count("other-orch") == 0

    def test_get_all_per_orch_counts_returns_all_counts(self) -> None:
        """Test get_all_per_orch_counts returns all counts for observability."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Initially should return empty dict
        counts = sentinel.get_all_per_orch_counts()
        assert counts == {}

        # After incrementing multiple orchestrations
        sentinel._state_tracker.increment_per_orch_count("orch-a")
        sentinel._state_tracker.increment_per_orch_count("orch-a")
        sentinel._state_tracker.increment_per_orch_count("orch-b")

        counts = sentinel.get_all_per_orch_counts()
        assert counts == {"orch-a": 2, "orch-b": 1}

        # Verify it returns a copy (not the original dict)
        counts["orch-c"] = 99
        assert "orch-c" not in sentinel.get_all_per_orch_counts()

    def test_get_available_slots_for_orchestration_no_limit(self) -> None:
        """Test _get_available_slots_for_orchestration with no per-orch limit."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config(max_concurrent_executions=5)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Orchestration without max_concurrent
        orch = make_orchestration(name="unlimited", max_concurrent=None)

        # Should return global available slots
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 5

    def test_get_available_slots_for_orchestration_with_limit(self) -> None:
        """Test _get_available_slots_for_orchestration with per-orch limit."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config(max_concurrent_executions=10)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Orchestration with max_concurrent=3
        orch = make_orchestration(name="limited", max_concurrent=3)

        # With 0 active executions, should return per-orch limit (3)
        # since it's lower than global (10)
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 3

    def test_get_available_slots_considers_current_per_orch_count(self) -> None:
        """Test _get_available_slots_for_orchestration considers current count."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config(max_concurrent_executions=10)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        orch = make_orchestration(name="limited", max_concurrent=3)

        # Simulate 2 active executions for this orchestration
        sentinel._state_tracker._per_orch_active_counts["limited"] = 2

        # Should return 1 (3 - 2)
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 1

        # Simulate 3 active executions (at limit)
        sentinel._state_tracker._per_orch_active_counts["limited"] = 3
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 0

    def test_get_available_slots_returns_min_of_global_and_per_orch(self) -> None:
        """Test that available slots is minimum of global and per-orch."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
        config = make_config(max_concurrent_executions=2)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        # Orchestration with max_concurrent=5, but global is 2
        orch = make_orchestration(name="high-limit", max_concurrent=5)

        # Should return global limit (2) since it's lower
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 2

    def test_per_orch_count_decremented_on_failure(self) -> None:
        """Test per-orchestration count is decremented even on execution failure."""

        class FailingAgentClient(AgentClient):
            @property
            def agent_type(self) -> AgentType:
                return AgentType.CLAUDE

            async def run_agent(
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
                agent_teams: bool = False,
            ) -> AgentRunResult:
                time.sleep(0.05)
                return AgentRunResult(response="FAILURE: Error occurred", workdir=None)

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[
                make_issue(key="TEST-1", summary="Issue 1", labels=["review"]),
            ],
            tag_client=tag_client,
        )
        agent_client = FailingAgentClient()
        agent_factory = MockAgentClientFactory(agent_client)
        config = make_config(max_concurrent_executions=5)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"], max_concurrent=2)]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        # Execution should complete (with failure after retries)
        assert len(results) == 1
        # Count should be decremented back to 0 even on failure
        assert sentinel._state_tracker._per_orch_active_counts.get("test-orch", 0) == 0

    def test_make_orchestration_helper_with_max_concurrent(self) -> None:
        """Test the make_orchestration helper supports max_concurrent."""
        # Without max_concurrent
        orch1 = make_orchestration(name="test1")
        assert orch1.max_concurrent is None

        # With max_concurrent
        orch2 = make_orchestration(name="test2", max_concurrent=5)
        assert orch2.max_concurrent == 5
