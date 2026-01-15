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
    from sentinel.main import QueuedIssueInfo, RunningStepInfo, Sentinel
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
class RunningStepInfoView:
    """Read-only running step information for the dashboard (DS-122).

    This provides an immutable view of a running execution step including
    calculated elapsed time for display purposes.
    """

    issue_key: str
    orchestration_name: str
    attempt_number: int
    started_at: datetime
    elapsed_seconds: float


@dataclass(frozen=True)
class QueuedIssueInfoView:
    """Read-only queued issue information for the dashboard (DS-123).

    This provides an immutable view of an issue waiting in queue for an
    execution slot, including calculated wait time for display purposes.
    """

    issue_key: str
    orchestration_name: str
    queued_at: datetime
    wait_seconds: float


@dataclass(frozen=True)
class SystemStatusInfo:
    """System status information for the dashboard (DS-124).

    This provides system-level metrics including thread pool usage,
    poll times, and process uptime.
    """

    # Thread pool usage
    used_slots: int
    max_slots: int

    # Poll times per source
    last_jira_poll: datetime | None
    last_github_poll: datetime | None

    # Configuration
    poll_interval: int

    # Process uptime
    start_time: datetime
    uptime_seconds: float


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

    # Running steps (DS-122) - active execution details for dashboard display
    running_steps: list[RunningStepInfoView] = field(default_factory=list)

    # Issue queue (DS-123) - issues waiting for execution slots
    issue_queue: list[QueuedIssueInfoView] = field(default_factory=list)

    # Hot-reload metrics
    hot_reload_metrics: HotReloadMetrics | None = None

    # Polling state
    shutdown_requested: bool = False
    pending_tasks: int = 0

    # System status (DS-124) - thread pool, poll times, uptime
    system_status: SystemStatusInfo | None = None


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

        # Get count of pending (not yet started) futures
        with sentinel._futures_lock:
            pending_count = sum(1 for f in sentinel._active_futures if not f.done())

        # Get running step info (DS-122)
        running_steps_raw = sentinel.get_running_steps()
        now = datetime.now()
        running_step_views = [
            RunningStepInfoView(
                issue_key=step.issue_key,
                orchestration_name=step.orchestration_name,
                attempt_number=step.attempt_number,
                started_at=step.started_at,
                elapsed_seconds=(now - step.started_at).total_seconds(),
            )
            for step in running_steps_raw
        ]

        # Get issue queue info (DS-123)
        issue_queue_raw = sentinel.get_issue_queue()
        issue_queue_views = [
            QueuedIssueInfoView(
                issue_key=item.issue_key,
                orchestration_name=item.orchestration_name,
                queued_at=item.queued_at,
                wait_seconds=(now - item.queued_at).total_seconds(),
            )
            for item in issue_queue_raw
        ]

        # Get system status info (DS-124)
        start_time = sentinel.get_start_time()
        system_status = SystemStatusInfo(
            used_slots=active_count,
            max_slots=config.max_concurrent_executions,
            last_jira_poll=sentinel.get_last_jira_poll(),
            last_github_poll=sentinel.get_last_github_poll(),
            poll_interval=config.poll_interval,
            start_time=start_time,
            uptime_seconds=(now - start_time).total_seconds(),
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
            running_steps=running_step_views,
            issue_queue=issue_queue_views,
            hot_reload_metrics=hot_reload_metrics,
            shutdown_requested=sentinel._shutdown_requested,
            pending_tasks=pending_count,
            system_status=system_status,
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
