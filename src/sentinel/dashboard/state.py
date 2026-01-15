"""Read-only state accessor for the dashboard.

This module provides a thread-safe, read-only view of the Sentinel application state
for use by the dashboard. It ensures the dashboard cannot modify the orchestrator's
internal state while still providing access to monitoring data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentinel.main import Sentinel
    from sentinel.orchestration import Orchestration, OrchestrationVersion


@dataclass(frozen=True)
class OrchestrationInfo:
    """Read-only orchestration information for the dashboard."""

    name: str
    enabled: bool
    trigger_source: str
    trigger_project: str | None
    trigger_repo: str | None
    trigger_tags: list[str]
    agent_prompt_preview: str


@dataclass(frozen=True)
class OrchestrationVersionInfo:
    """Read-only orchestration version information for the dashboard."""

    name: str
    version_id: str
    source_file: str
    loaded_at: datetime
    active_executions: int


@dataclass(frozen=True)
class HotReloadMetrics:
    """Hot-reload metrics for the dashboard."""

    orchestrations_loaded_total: int
    orchestrations_unloaded_total: int
    orchestrations_reloaded_total: int


@dataclass(frozen=True)
class DashboardState:
    """Immutable snapshot of Sentinel state for dashboard rendering.

    This class provides a frozen view of the orchestrator's state at a specific
    point in time. It is safe to pass to templates and will not change during
    rendering.
    """

    # Configuration
    poll_interval: int
    max_concurrent_executions: int
    max_issues_per_poll: int

    # Orchestrations
    orchestrations: list[OrchestrationInfo] = field(default_factory=list)
    active_versions: list[OrchestrationVersionInfo] = field(default_factory=list)
    pending_removal_versions: list[OrchestrationVersionInfo] = field(default_factory=list)

    # Execution state
    active_execution_count: int = 0
    available_slots: int = 0

    # Hot-reload metrics
    hot_reload_metrics: HotReloadMetrics | None = None

    # Polling state
    shutdown_requested: bool = False
    consecutive_eager_polls: int = 0


class SentinelStateProvider(Protocol):
    """Protocol for objects that can provide Sentinel state.

    This protocol allows the dashboard to work with any object that exposes
    the required state attributes, enabling easier testing and decoupling.
    """

    @property
    def config(self) -> object:
        """Return the configuration object."""
        ...

    @property
    def orchestrations(self) -> list[Orchestration]:
        """Return the list of active orchestrations."""
        ...

    def get_hot_reload_metrics(self) -> dict[str, int]:
        """Return hot-reload metrics."""
        ...


class SentinelStateAccessor:
    """Thread-safe, read-only accessor for Sentinel state.

    This class wraps a Sentinel instance and provides read-only access to its
    state for the dashboard. All methods return immutable copies of the data
    to prevent accidental modification.
    """

    def __init__(self, sentinel: Sentinel) -> None:
        """Initialize the state accessor.

        Args:
            sentinel: The Sentinel instance to monitor.
        """
        self._sentinel = sentinel

    def get_state(self) -> DashboardState:
        """Get a snapshot of the current Sentinel state.

        Returns:
            An immutable DashboardState object containing current state.
        """
        sentinel = self._sentinel
        config = sentinel.config

        # Extract orchestration info
        orchestration_infos = [
            self._orchestration_to_info(orch) for orch in sentinel.orchestrations
        ]

        # Extract version info (thread-safe access)
        active_version_infos: list[OrchestrationVersionInfo] = []
        pending_removal_version_infos: list[OrchestrationVersionInfo] = []

        with sentinel._versions_lock:
            for version in sentinel._active_versions:
                active_version_infos.append(self._version_to_info(version))
            for version in sentinel._pending_removal_versions:
                pending_removal_version_infos.append(self._version_to_info(version))

        # Get execution state (thread-safe access)
        with sentinel._futures_lock:
            active_count = sum(1 for f in sentinel._active_futures if not f.done())

        available_slots = config.max_concurrent_executions - active_count

        # Get hot-reload metrics
        metrics = sentinel.get_hot_reload_metrics()
        hot_reload_metrics = HotReloadMetrics(
            orchestrations_loaded_total=metrics["orchestrations_loaded_total"],
            orchestrations_unloaded_total=metrics["orchestrations_unloaded_total"],
            orchestrations_reloaded_total=metrics["orchestrations_reloaded_total"],
        )

        return DashboardState(
            poll_interval=config.poll_interval,
            max_concurrent_executions=config.max_concurrent_executions,
            max_issues_per_poll=config.max_issues_per_poll,
            orchestrations=orchestration_infos,
            active_versions=active_version_infos,
            pending_removal_versions=pending_removal_version_infos,
            active_execution_count=active_count,
            available_slots=available_slots,
            hot_reload_metrics=hot_reload_metrics,
            shutdown_requested=sentinel._shutdown_requested,
            consecutive_eager_polls=sentinel._consecutive_eager_polls,
        )

    def _orchestration_to_info(self, orch: Orchestration) -> OrchestrationInfo:
        """Convert an Orchestration to OrchestrationInfo.

        Args:
            orch: The orchestration to convert.

        Returns:
            An OrchestrationInfo object with read-only data.
        """
        trigger = orch.trigger
        trigger_source = getattr(trigger, "source", "jira")
        trigger_project = getattr(trigger, "project", None)
        trigger_repo = getattr(trigger, "repo", None)
        trigger_tags = list(trigger.tags) if trigger.tags else []

        # Create a preview of the agent prompt (first 100 chars)
        prompt_preview = orch.agent.prompt[:100] if orch.agent.prompt else ""
        if len(orch.agent.prompt) > 100:
            prompt_preview += "..."

        return OrchestrationInfo(
            name=orch.name,
            enabled=orch.enabled,
            trigger_source=trigger_source,
            trigger_project=trigger_project,
            trigger_repo=trigger_repo,
            trigger_tags=trigger_tags,
            agent_prompt_preview=prompt_preview,
        )

    def _version_to_info(self, version: OrchestrationVersion) -> OrchestrationVersionInfo:
        """Convert an OrchestrationVersion to OrchestrationVersionInfo.

        Args:
            version: The version to convert.

        Returns:
            An OrchestrationVersionInfo object with read-only data.
        """
        return OrchestrationVersionInfo(
            name=version.name,
            version_id=version.version_id,
            source_file=str(version.source_file),
            loaded_at=version.loaded_at,
            active_executions=version.active_executions,
        )
