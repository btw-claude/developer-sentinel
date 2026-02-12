"""Read-only state accessor for the dashboard.

This module provides a thread-safe, read-only view of the Sentinel application state
for use by the dashboard. It ensures the dashboard cannot modify the orchestrator's
internal state while still providing access to monitoring data.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from cachetools import TTLCache, cachedmethod
from cachetools.keys import hashkey

from sentinel.config import DEFAULT_GREEN_THRESHOLD, DEFAULT_YELLOW_THRESHOLD
from sentinel.logging import generate_log_filename, parse_log_filename_parts
from sentinel.types import TriggerSource

if TYPE_CHECKING:
    from sentinel.config import Config
    from sentinel.main import QueuedIssueInfo, RunningStepInfo
    from sentinel.orchestration import Orchestration
    from sentinel.state_tracker import CompletedExecutionInfo


@dataclass(frozen=True)
class OrchestrationInfo:
    """Read-only orchestration information for the dashboard."""

    name: str
    enabled: bool
    trigger_source: str
    trigger_project: str | None
    trigger_project_owner: str | None
    trigger_tags: list[str]
    agent_prompt_preview: str
    source_file: str


@dataclass(frozen=True)
class TriggerDetailInfo:
    """Read-only trigger detail information for the orchestration detail view.

    Contains all trigger fields from the orchestration configuration.
    """

    source: str
    project: str
    jql_filter: str
    tags: list[str]
    project_number: int | None
    project_scope: Literal["org", "user"]
    project_owner: str
    project_filter: str
    labels: list[str]


@dataclass(frozen=True)
class GitHubContextInfo:
    """Read-only GitHub context information for the orchestration detail view."""

    host: str
    org: str
    repo: str
    branch: str
    create_branch: bool
    base_branch: str


@dataclass(frozen=True)
class AgentDetailInfo:
    """Read-only agent detail information for the orchestration detail view.

    Contains the full (non-truncated) prompt and all agent configuration fields.
    """

    prompt: str
    github: GitHubContextInfo | None
    timeout_seconds: int | None
    model: str | None
    agent_type: str | None
    cursor_mode: str | None
    agent_teams: None  # Reserved for future use; not yet implemented in AgentConfig
    strict_template_variables: bool


@dataclass(frozen=True)
class RetryDetailInfo:
    """Read-only retry detail information for the orchestration detail view."""

    max_attempts: int
    success_patterns: list[str]
    failure_patterns: list[str]
    default_status: str
    default_outcome: str


@dataclass(frozen=True)
class OutcomeInfo:
    """Read-only outcome information for the orchestration detail view."""

    name: str
    patterns: list[str]
    add_tag: str


@dataclass(frozen=True)
class LifecycleInfo:
    """Read-only lifecycle information for the orchestration detail view.

    Combines on_start, on_complete, and on_failure configuration.
    """

    on_start_add_tag: str
    on_complete_remove_tag: str
    on_complete_add_tag: str
    on_failure_add_tag: str


@dataclass(frozen=True)
class OrchestrationDetailInfo:
    """Read-only orchestration detail information for the detail view.

    Contains the full orchestration configuration, including all trigger,
    agent, retry, outcome, and lifecycle fields.
    """

    name: str
    enabled: bool
    max_concurrent: int | None
    source_file: str
    trigger: TriggerDetailInfo
    agent: AgentDetailInfo
    retry: RetryDetailInfo
    outcomes: list[OutcomeInfo]
    lifecycle: LifecycleInfo


@dataclass(frozen=True)
class ProjectOrchestrations:
    """Orchestrations grouped by project or project owner.

    This groups orchestrations by their trigger project (for Jira) or
    project owner (for GitHub) for a more organized dashboard display.

    .. note:: API Stability
        This class is currently intended for internal dashboard use only.
        If this class is exposed as part of a public API in the future,
        consider the following deprecation and versioning practices:

        - Add a ``__version__`` attribute or use semantic versioning

          Example::

              @dataclass(frozen=True)
              class ProjectOrchestrations:
                  __version__ = "1.0.0"
                  # ... fields ...

          .. note:: Dataclass Behavior
              The ``__version__`` attribute is a class attribute, not a dataclass
              field. As such, it will **not** be included in the automatically
              generated ``__repr__``, ``__eq__``, ``__hash__``, or ``__init__``
              methods. This is intentional—version metadata is class-level, not
              instance-level. To access it, use ``ProjectOrchestrations.__version__``
              rather than ``instance.__version__`` (though both work, the former
              is more explicit about the class-level nature).

        - Document any planned changes in a CHANGELOG
        - Use ``warnings.warn()`` with ``DeprecationWarning`` for deprecated
          attributes or methods
        - Provide a migration path for breaking changes
        - Consider using ``typing_extensions.deprecated`` decorator (Python 3.13+)
          or a backport for earlier versions
    """

    identifier: str  # Project key or project owner, e.g., "DS" for Jira or "my-org" for GitHub
    orchestrations: list[OrchestrationInfo]

    @property
    def count(self) -> int:
        """Return the number of orchestrations in this project."""
        return len(self.orchestrations)


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
    """Read-only running step information for the dashboard.

    This provides an immutable view of a running execution step including
    calculated elapsed time for display purposes.
    """

    issue_key: str
    orchestration_name: str
    attempt_number: int
    started_at: datetime
    elapsed_seconds: float
    log_filename: str  # derived from started_at
    issue_url: str  # URL to Jira or GitHub issue


@dataclass(frozen=True)
class QueuedIssueInfoView:
    """Read-only queued issue information for the dashboard.

    This provides an immutable view of an issue waiting in queue for an
    execution slot, including calculated wait time for display purposes.
    """

    issue_key: str
    orchestration_name: str
    queued_at: datetime
    wait_seconds: float


@dataclass(frozen=True)
class CompletedExecutionInfoView:
    """Read-only completed execution information for the dashboard.

    This provides an immutable view of a completed execution for dashboard
    display, including computed duration and all usage fields (token counts
    and cost) from the Claude Agent SDK.
    """

    issue_key: str
    orchestration_name: str
    status: str  # "success" or "failure"
    duration_seconds: float  # computed from started_at and completed_at
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    completed_at: datetime
    issue_url: str


@dataclass(frozen=True)
class ExecutionSummaryStats:
    """Aggregated summary statistics across all completed executions.

    This provides global execution metrics for the dashboard Metrics tab,
    including success rates, token/cost totals, and average durations.
    """

    total_executions: int
    success_count: int
    failure_count: int
    success_rate: float  # percentage (0.0 - 100.0)
    avg_duration_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    avg_cost_usd: float

    @classmethod
    @lru_cache(maxsize=1)
    def empty(cls) -> ExecutionSummaryStats:
        """Create a zeroed ExecutionSummaryStats instance.

        Returns a cached singleton ExecutionSummaryStats with all counters,
        rates, and totals set to zero. Since ExecutionSummaryStats is a
        frozen dataclass, the result is always identical, so a cached
        singleton avoids redundant allocations.

        Returns:
            An ExecutionSummaryStats instance with all fields set to zero.
        """
        return cls(
            total_executions=0,
            success_count=0,
            failure_count=0,
            success_rate=0.0,
            avg_duration_seconds=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            avg_cost_usd=0.0,
        )


@dataclass(frozen=True)
class OrchestrationStats:
    """Per-orchestration execution statistics.

    This provides execution metrics grouped by orchestration name
    for the dashboard Orchestrations tab.
    """

    orchestration_name: str
    total_runs: int
    success_count: int
    failure_count: int
    success_rate: float  # percentage (0.0 - 100.0)
    avg_duration_seconds: float
    total_cost_usd: float
    last_run_at: datetime | None


@dataclass(frozen=True)
class SystemStatusInfo:
    """System status information for the dashboard.

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
class ServiceHealthInfo:
    """Read-only service health information for the dashboard.

    This provides a frozen snapshot of a service's health status
    from the ServiceHealthGate for dashboard display.
    """

    service_name: str
    available: bool
    consecutive_failures: int
    paused: bool
    paused_since_seconds: float | None
    last_error: str | None
    probe_count: int
    last_check_seconds_ago: float | None


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

    # Grouped orchestrations
    jira_projects: list[ProjectOrchestrations] = field(default_factory=list)
    github_repos: list[ProjectOrchestrations] = field(default_factory=list)

    # Active orchestration counts - count of projects/repos with running orchestrations
    active_jira_projects_count: int = 0
    active_github_repos_count: int = 0

    # Execution state
    active_execution_count: int = 0
    # Used by the Capacity display in both metrics.html and partials/status.html
    available_slots: int = 0

    # Running steps - active execution details for dashboard display
    running_steps: list[RunningStepInfoView] = field(default_factory=list)

    # Issue queue - issues waiting for execution slots
    issue_queue: list[QueuedIssueInfoView] = field(default_factory=list)

    # Completed executions - recent completed executions for dashboard display
    completed_executions: list[CompletedExecutionInfoView] = field(default_factory=list)

    # Hot-reload metrics
    hot_reload_metrics: HotReloadMetrics | None = None

    # Polling state
    shutdown_requested: bool = False
    active_incomplete_tasks: int = 0

    # Computed summary statistics
    execution_summary: ExecutionSummaryStats = field(
        default_factory=ExecutionSummaryStats.empty
    )
    orchestration_stats: list[OrchestrationStats] = field(default_factory=list)

    # System status - thread pool, poll times, uptime
    system_status: SystemStatusInfo | None = None

    # Service health status - external service availability
    service_health: list[ServiceHealthInfo] = field(default_factory=list)

    # Configurable success rate thresholds for display coloring
    success_rate_green_threshold: float = DEFAULT_GREEN_THRESHOLD
    success_rate_yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD


@dataclass(frozen=True)
class OrchestrationVersionSnapshot:
    """Snapshot of an orchestration version for dashboard display.

    This DTO captures version information at a point in time without
    exposing internal OrchestrationVersion implementation details.
    """

    name: str
    version_id: str
    source_file: str
    loaded_at: datetime
    active_executions: int


@dataclass(frozen=True)
class ExecutionStateSnapshot:
    """Snapshot of execution state for dashboard display.

    This DTO captures the current execution state without exposing
    internal threading primitives or Future objects.
    """

    active_count: int
    """Number of currently active executions."""


class SentinelStateProvider(Protocol):
    """Protocol for objects that can provide Sentinel state.

    This protocol defines the public interface that Sentinel must implement
    for the dashboard to access state. It enables decoupling between the
    dashboard and Sentinel internals - the dashboard only depends on this
    protocol, not on Sentinel's internal implementation details.

    All methods return DTOs or primitive types, never exposing internal
    state structures like locks, futures, or mutable collections.
    """

    @property
    def config(self) -> Config:
        """Return the configuration object."""
        ...

    @property
    def orchestrations(self) -> list[Orchestration]:
        """Return the list of active orchestrations."""
        ...

    def get_hot_reload_metrics(self) -> dict[str, int]:
        """Return hot-reload metrics."""
        ...

    def get_running_steps(self) -> list[RunningStepInfo]:
        """Return information about currently running execution steps."""
        ...

    def get_issue_queue(self) -> list[QueuedIssueInfo]:
        """Return information about issues waiting in queue."""
        ...

    def get_start_time(self) -> datetime:
        """Return the process start time."""
        ...

    def get_last_jira_poll(self) -> datetime | None:
        """Return the time of the last Jira poll, or None if never polled."""
        ...

    def get_last_github_poll(self) -> datetime | None:
        """Return the time of the last GitHub poll, or None if never polled."""
        ...

    def get_active_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Return snapshots of active orchestration versions.

        This method returns immutable DTOs instead of exposing internal
        OrchestrationVersion objects, maintaining encapsulation.
        """
        ...

    def get_pending_removal_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Return snapshots of versions pending removal.

        This method returns immutable DTOs instead of exposing internal
        OrchestrationVersion objects, maintaining encapsulation.
        """
        ...

    def get_execution_state(self) -> ExecutionStateSnapshot:
        """Return a snapshot of the current execution state.

        This method returns an immutable DTO instead of exposing internal
        threading primitives or Future objects.
        """
        ...

    def get_completed_executions(self) -> list[CompletedExecutionInfo]:
        """Return the list of completed executions.

        This method returns a list of completed execution info for recent
        executions, ordered with most recent first.
        """
        ...

    def is_shutdown_requested(self) -> bool:
        """Return whether shutdown has been requested."""
        ...

    def get_service_health_status(self) -> dict[str, dict[str, Any]]:
        """Return service health status from the health gate.

        Returns a dictionary mapping service names to their availability state,
        suitable for dashboard display.
        """
        ...


class SentinelStateAccessor:
    """Thread-safe, read-only accessor for Sentinel state.

    This class wraps a Sentinel instance (via the SentinelStateProvider protocol)
    and provides read-only access to its state for the dashboard. All methods
    return immutable copies of the data to prevent accidental modification.

    The accessor uses only public methods defined in SentinelStateProvider,
    never accessing private attributes directly. This decouples the dashboard
    from Sentinel's internal implementation details, making both components
    easier to test and maintain independently.
    """

    # TTL in seconds for the dashboard state cache. Rapid successive requests
    # within this window (e.g., HTMX auto-refresh every 2-5 seconds) reuse the
    # same computed DashboardState, avoiding redundant CPU and memory churn.
    _STATE_CACHE_TTL: float = 1.0

    def __init__(self, sentinel: SentinelStateProvider) -> None:
        """Initialize the state accessor.

        Args:
            sentinel: An object implementing SentinelStateProvider protocol.
        """
        self._sentinel = sentinel
        self._state_cache: TTLCache[str, DashboardState] = TTLCache(
            maxsize=1, ttl=self._STATE_CACHE_TTL
        )
        self._state_cache_lock = Lock()

    @cachedmethod(
        cache=lambda self: self._state_cache,
        lock=lambda self: self._state_cache_lock,
        key=lambda self: hashkey("state"),
    )
    def get_state(self) -> DashboardState:
        """Get a snapshot of the current Sentinel state.

        Returns a cached DashboardState if one exists within the TTL window,
        otherwise rebuilds the full state and caches the result. This avoids
        redundant computation under rapid successive requests (e.g., HTMX
        auto-refresh every 2-5 seconds).

        Thread-safety: The ``@cachedmethod`` decorator with a ``lock``
        parameter handles thread-safe cache access automatically. The lock
        guards all cache reads and writes, ensuring that concurrent threads
        cannot trigger redundant ``_build_state()`` calls. This replaces the
        previous manual double-checked locking pattern with the library's
        built-in mechanism, making the thread-safety guarantees more explicit.

        Returns:
            An immutable DashboardState object containing current state.
        """
        return self._build_state()

    def _build_state(self) -> DashboardState:
        """Build a fresh DashboardState snapshot from the provider.

        This method performs the full state computation: iterating all active
        orchestrations, completed executions, running steps, issue queues,
        and computing per-orchestration statistics.

        Returns:
            An immutable DashboardState object containing current state.
        """
        sentinel = self._sentinel
        config = sentinel.config

        # Get active version snapshots through public API
        # These are already DTOs, not internal objects
        active_version_snapshots = sentinel.get_active_versions()

        # Build a mapping from orchestration name to source_file for lookup.
        # This uses the public API instead of accessing internal _versions_lock
        # and _active_versions directly.
        orch_name_to_source_file: dict[str, str] = {
            v.name: v.source_file for v in active_version_snapshots
        }

        # Extract orchestration info
        orchestration_infos = [
            self._orchestration_to_info(orch, orch_name_to_source_file.get(orch.name, ""))
            for orch in sentinel.orchestrations
        ]

        # Group orchestrations by project/repo
        jira_projects, github_repos = self._group_orchestrations(orchestration_infos)

        # Convert version snapshots to OrchestrationVersionInfo (dashboard-specific DTO)
        active_version_infos = [
            self._snapshot_to_version_info(snapshot) for snapshot in active_version_snapshots
        ]

        pending_removal_version_infos = [
            self._snapshot_to_version_info(snapshot)
            for snapshot in sentinel.get_pending_removal_versions()
        ]

        # Get execution state through public API
        execution_state = sentinel.get_execution_state()
        active_count = execution_state.active_count

        available_slots = config.execution.max_concurrent_executions - active_count

        # Get hot-reload metrics
        metrics = sentinel.get_hot_reload_metrics()
        hot_reload_metrics = HotReloadMetrics(
            orchestrations_loaded_total=metrics["orchestrations_loaded_total"],
            orchestrations_unloaded_total=metrics["orchestrations_unloaded_total"],
            orchestrations_reloaded_total=metrics["orchestrations_reloaded_total"],
        )

        # pending_count is the same as active_count for dashboard purposes
        # (tracks incomplete tasks for display)
        pending_count = active_count

        # Get running step info
        running_steps_raw = sentinel.get_running_steps()
        now = datetime.now(tz=UTC)
        running_step_views = [
            RunningStepInfoView(
                issue_key=step.issue_key,
                orchestration_name=step.orchestration_name,
                attempt_number=step.attempt_number,
                started_at=step.started_at,
                elapsed_seconds=(now - step.started_at).total_seconds(),
                log_filename=generate_log_filename(
                    step.started_at,
                    issue_key=step.issue_key,
                    attempt=step.attempt_number,
                ),
                issue_url=step.issue_url,  # Pass through issue URL
            )
            for step in running_steps_raw
        ]

        # Get issue queue info
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

        # Get completed executions info
        completed_executions_raw = sentinel.get_completed_executions()
        completed_execution_views = [
            CompletedExecutionInfoView(
                issue_key=item.issue_key,
                orchestration_name=item.orchestration_name,
                status=item.status,
                duration_seconds=(item.completed_at - item.started_at).total_seconds(),
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                total_cost_usd=item.total_cost_usd,
                completed_at=item.completed_at,
                issue_url=item.issue_url,
            )
            for item in completed_executions_raw
        ]

        # Get system status info
        start_time = sentinel.get_start_time()
        system_status = SystemStatusInfo(
            used_slots=active_count,
            max_slots=config.execution.max_concurrent_executions,
            last_jira_poll=sentinel.get_last_jira_poll(),
            last_github_poll=sentinel.get_last_github_poll(),
            poll_interval=config.polling.interval,
            start_time=start_time,
            uptime_seconds=(now - start_time).total_seconds(),
        )

        # Get service health status
        service_health_raw = sentinel.get_service_health_status()
        service_health_views = [
            ServiceHealthInfo(
                service_name=svc_data.get("service_name", name),
                available=svc_data.get("available", True),
                consecutive_failures=svc_data.get("consecutive_failures", 0),
                paused=svc_data.get("paused_at") is not None,
                paused_since_seconds=(
                    now.timestamp() - svc_data["paused_at"]
                    if svc_data.get("paused_at") is not None
                    else None
                ),
                last_error=svc_data.get("last_error"),
                probe_count=svc_data.get("probe_count", 0),
                last_check_seconds_ago=(
                    now.timestamp() - svc_data["last_check_at"]
                    if svc_data.get("last_check_at") is not None
                    else None
                ),
            )
            for name, svc_data in service_health_raw.items()
        ]

        # Count projects/repos with active orchestrations
        # Build a mapping from orchestration name to its project/repo
        active_orchestration_names = {step.orchestration_name for step in running_step_views}
        active_jira_projects: set[str] = set()
        active_github_repos: set[str] = set()
        for orch in orchestration_infos:
            if orch.name in active_orchestration_names:
                if orch.trigger_source == TriggerSource.JIRA.value and orch.trigger_project:
                    active_jira_projects.add(orch.trigger_project)
                elif (
                    orch.trigger_source == TriggerSource.GITHUB.value
                    and orch.trigger_project_owner
                ):
                    active_github_repos.add(orch.trigger_project_owner)

        # Compute execution summary statistics
        execution_summary, orchestration_stats = self._compute_execution_stats(
            completed_execution_views
        )

        return DashboardState(
            poll_interval=config.polling.interval,
            max_concurrent_executions=config.execution.max_concurrent_executions,
            max_issues_per_poll=config.polling.max_issues_per_poll,
            orchestrations=orchestration_infos,
            active_versions=active_version_infos,
            pending_removal_versions=pending_removal_version_infos,
            jira_projects=jira_projects,
            github_repos=github_repos,
            active_jira_projects_count=len(active_jira_projects),
            active_github_repos_count=len(active_github_repos),
            active_execution_count=active_count,
            available_slots=available_slots,
            running_steps=running_step_views,
            issue_queue=issue_queue_views,
            completed_executions=completed_execution_views,
            hot_reload_metrics=hot_reload_metrics,
            shutdown_requested=sentinel.is_shutdown_requested(),
            active_incomplete_tasks=pending_count,
            execution_summary=execution_summary,
            orchestration_stats=orchestration_stats,
            system_status=system_status,
            service_health=service_health_views,
            success_rate_green_threshold=config.dashboard.success_rate_green_threshold,
            success_rate_yellow_threshold=config.dashboard.success_rate_yellow_threshold,
        )

    def _orchestration_to_info(self, orch: Orchestration, source_file: str) -> OrchestrationInfo:
        """Convert an Orchestration to OrchestrationInfo.

        Args:
            orch: The orchestration to convert.
            source_file: Path to the source file this orchestration was loaded from.

        Returns:
            An OrchestrationInfo object with read-only data.
        """
        trigger = orch.trigger
        trigger_source = trigger.source
        trigger_project = trigger.project or None
        trigger_project_owner = trigger.project_owner or None
        trigger_tags = list(trigger.tags) if trigger.tags else []

        # Create a preview of the agent prompt (first 100 chars)
        prompt_preview = orch.agent.prompt[:100] if orch.agent.prompt else ""
        if orch.agent.prompt and len(orch.agent.prompt) > 100:
            prompt_preview += "..."

        return OrchestrationInfo(
            name=orch.name,
            enabled=orch.enabled,
            trigger_source=trigger_source,
            trigger_project=trigger_project,
            trigger_project_owner=trigger_project_owner,
            trigger_tags=trigger_tags,
            agent_prompt_preview=prompt_preview,
            source_file=source_file,
        )

    def get_orchestration_detail(self, name: str) -> OrchestrationDetailInfo | None:
        """Get detailed orchestration information by name.

        Finds the orchestration by name and converts it to an
        OrchestrationDetailInfo containing the full configuration.

        Args:
            name: The orchestration name to look up.

        Returns:
            An OrchestrationDetailInfo with full configuration, or None
            if no orchestration with the given name is found.
        """
        sentinel = self._sentinel

        # Find the orchestration by name
        orch = None
        for o in sentinel.orchestrations:
            if o.name == name:
                orch = o
                break

        if orch is None:
            return None

        # Get the source file from active versions
        active_version_snapshots = sentinel.get_active_versions()
        orch_name_to_source_file: dict[str, str] = {
            v.name: v.source_file for v in active_version_snapshots
        }
        source_file = orch_name_to_source_file.get(orch.name, "")

        # Convert trigger
        trigger = orch.trigger
        trigger_detail = TriggerDetailInfo(
            source=trigger.source,
            project=trigger.project,
            jql_filter=trigger.jql_filter,
            tags=list(trigger.tags) if trigger.tags else [],
            project_number=trigger.project_number,
            project_scope=trigger.project_scope,
            project_owner=trigger.project_owner,
            project_filter=trigger.project_filter,
            labels=list(trigger.labels) if trigger.labels else [],
        )

        # Convert GitHub context
        github_info = None
        if orch.agent.github is not None:
            gh = orch.agent.github
            github_info = GitHubContextInfo(
                host=gh.host,
                org=gh.org,
                repo=gh.repo,
                branch=gh.branch,
                create_branch=gh.create_branch,
                base_branch=gh.base_branch,
            )

        # Convert agent
        agent_detail = AgentDetailInfo(
            prompt=orch.agent.prompt,
            github=github_info,
            timeout_seconds=orch.agent.timeout_seconds,
            model=orch.agent.model,
            agent_type=orch.agent.agent_type,
            cursor_mode=orch.agent.cursor_mode,
            agent_teams=None,
            strict_template_variables=orch.agent.strict_template_variables,
        )

        # Convert retry
        retry_detail = RetryDetailInfo(
            max_attempts=orch.retry.max_attempts,
            success_patterns=list(orch.retry.success_patterns),
            failure_patterns=list(orch.retry.failure_patterns),
            default_status=orch.retry.default_status,
            default_outcome=orch.retry.default_outcome,
        )

        # Convert outcomes
        outcomes = [
            OutcomeInfo(
                name=outcome.name,
                patterns=list(outcome.patterns),
                add_tag=outcome.add_tag,
            )
            for outcome in orch.outcomes
        ]

        # Convert lifecycle
        lifecycle = LifecycleInfo(
            on_start_add_tag=orch.on_start.add_tag,
            on_complete_remove_tag=orch.on_complete.remove_tag,
            on_complete_add_tag=orch.on_complete.add_tag,
            on_failure_add_tag=orch.on_failure.add_tag,
        )

        return OrchestrationDetailInfo(
            name=orch.name,
            enabled=orch.enabled,
            max_concurrent=orch.max_concurrent,
            source_file=source_file,
            trigger=trigger_detail,
            agent=agent_detail,
            retry=retry_detail,
            outcomes=outcomes,
            lifecycle=lifecycle,
        )

    def _snapshot_to_version_info(
        self, snapshot: OrchestrationVersionSnapshot
    ) -> OrchestrationVersionInfo:
        """Convert an OrchestrationVersionSnapshot to OrchestrationVersionInfo.

        Args:
            snapshot: The version snapshot to convert.

        Returns:
            An OrchestrationVersionInfo object with read-only data.
        """
        return OrchestrationVersionInfo(
            name=snapshot.name,
            version_id=snapshot.version_id,
            source_file=snapshot.source_file,
            loaded_at=snapshot.loaded_at,
            active_executions=snapshot.active_executions,
        )

    def _group_orchestrations(
        self, orchestrations: list[OrchestrationInfo]
    ) -> tuple[list[ProjectOrchestrations], list[ProjectOrchestrations]]:
        """Group orchestrations by Jira project and GitHub project owner.

        Args:
            orchestrations: List of orchestration info objects to group.

        Returns:
            A tuple of (jira_projects, github_repos) where each is a list of
            ProjectOrchestrations grouped by project key or project owner.
        """
        jira_groups: dict[str, list[OrchestrationInfo]] = defaultdict(list)
        github_groups: dict[str, list[OrchestrationInfo]] = defaultdict(list)

        for orch in orchestrations:
            if orch.trigger_source == TriggerSource.JIRA.value and orch.trigger_project:
                jira_groups[orch.trigger_project].append(orch)
            elif orch.trigger_source == TriggerSource.GITHUB.value and orch.trigger_project_owner:
                github_groups[orch.trigger_project_owner].append(orch)

        # Convert to sorted lists of ProjectOrchestrations
        jira_projects = [
            ProjectOrchestrations(
                identifier=key, orchestrations=sorted(orchs, key=lambda o: o.name)
            )
            for key, orchs in sorted(jira_groups.items())
        ]
        github_repos = [
            ProjectOrchestrations(
                identifier=key, orchestrations=sorted(orchs, key=lambda o: o.name)
            )
            for key, orchs in sorted(github_groups.items())
        ]

        return jira_projects, github_repos

    def _compute_execution_stats(
        self, executions: list[CompletedExecutionInfoView]
    ) -> tuple[ExecutionSummaryStats, list[OrchestrationStats]]:
        """Compute summary statistics from completed executions.

        Groups completed executions by orchestration name and computes
        success rates, average durations, and token/cost totals both
        globally and per-orchestration.

        Args:
            executions: List of completed execution info views.

        Returns:
            A tuple of (execution_summary, orchestration_stats) where
            execution_summary contains global aggregated stats and
            orchestration_stats contains per-orchestration stats.
        """
        if not executions:
            return ExecutionSummaryStats.empty(), []

        # Compute global summary stats
        total = len(executions)
        successes = sum(1 for e in executions if e.status == "success")
        failures = total - successes
        success_rate = (successes / total) * 100.0

        total_duration = sum(e.duration_seconds for e in executions)
        avg_duration = total_duration / total

        total_input = sum(e.input_tokens for e in executions)
        total_output = sum(e.output_tokens for e in executions)
        total_tokens = total_input + total_output
        total_cost = sum(e.total_cost_usd for e in executions)
        avg_cost = total_cost / total

        execution_summary = ExecutionSummaryStats(
            total_executions=total,
            success_count=successes,
            failure_count=failures,
            success_rate=round(success_rate, 2),
            avg_duration_seconds=round(avg_duration, 2),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 6),
            avg_cost_usd=round(avg_cost, 6),
        )

        # Compute per-orchestration stats
        orch_groups: dict[str, list[CompletedExecutionInfoView]] = defaultdict(list)
        for execution in executions:
            orch_groups[execution.orchestration_name].append(execution)

        orchestration_stats: list[OrchestrationStats] = []
        for orch_name, orch_executions in sorted(orch_groups.items()):
            orch_total = len(orch_executions)
            orch_successes = sum(1 for e in orch_executions if e.status == "success")
            orch_failures = orch_total - orch_successes
            orch_success_rate = (orch_successes / orch_total) * 100.0

            orch_total_duration = sum(e.duration_seconds for e in orch_executions)
            orch_avg_duration = orch_total_duration / orch_total

            orch_total_cost = sum(e.total_cost_usd for e in orch_executions)

            # Find most recent execution for last_run_at
            last_run_at = max(e.completed_at for e in orch_executions)

            orchestration_stats.append(
                OrchestrationStats(
                    orchestration_name=orch_name,
                    total_runs=orch_total,
                    success_count=orch_successes,
                    failure_count=orch_failures,
                    success_rate=round(orch_success_rate, 2),
                    avg_duration_seconds=round(orch_avg_duration, 2),
                    total_cost_usd=round(orch_total_cost, 6),
                    last_run_at=last_run_at,
                )
            )

        return execution_summary, orchestration_stats

    def get_log_files(self) -> list[dict[str, Any]]:
        """Get available log files grouped by step.

        Discovers log files in the agent_logs_dir, grouped by step
        name, with files sorted by modification time (newest first).

        Returns:
            List of dictionaries with step name and files.
        """
        logs_dir = self._sentinel.config.execution.agent_logs_dir

        if not logs_dir.exists():
            return []

        result = []

        # Iterate through step directories
        for step_dir in sorted(logs_dir.iterdir()):
            if not step_dir.is_dir():
                continue

            step_name = step_dir.name
            files = []

            # Get log files in this step directory
            for log_file in step_dir.glob("*.log"):
                if log_file.is_file():
                    try:
                        stat = log_file.stat()
                        # Parse datetime from filename (YYYYMMDD_HHMMSS.log)
                        display_name = self._format_log_display_name(log_file.name)
                        files.append(
                            {
                                "filename": log_file.name,
                                "display_name": display_name,
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(
                                    stat.st_mtime, tz=UTC
                                ).isoformat(),
                            }
                        )
                    except OSError:
                        # Skip files that can't be accessed due to permissions or I/O issues
                        pass
                    except (ValueError, OverflowError):
                        # Skip files with invalid timestamps
                        pass

            # Sort files by modified time (newest first)
            files.sort(key=lambda f: cast(str, f["modified"]), reverse=True)

            if files:
                result.append(
                    {
                        "step": step_name,
                        "files": files,
                    }
                )

        # Sort steps alphabetically
        result.sort(key=lambda x: cast(str, x["step"]))

        return result

    def _format_log_display_name(self, filename: str) -> str:
        """Format a log filename for display.

        Supports both new format ({issue_key}_{YYYYMMDD-HHMMSS}_a{N}.log) and
        legacy format (YYYYMMDD_HHMMSS.log). Extracts the timestamp and formats
        it as a human-readable string, preserving issue key and attempt info
        when present.

        Args:
            filename: The log filename.

        Returns:
            Human-readable display name.
        """
        parsed = parse_log_filename_parts(filename)
        if parsed is not None:
            issue_key, dt, attempt = parsed
            formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
            # Legacy filenames have no _a{N} suffix — show timestamp only
            if "_a" not in filename:
                return formatted_ts
            if issue_key is not None:
                return f"{issue_key} {formatted_ts} (attempt {attempt})"
            return f"{formatted_ts} (attempt {attempt})"
        return filename

    def get_log_file_path(self, step: str, filename: str) -> Path | None:
        """Get the full path to a log file.

        Args:
            step: The step name.
            filename: The log file name.

        Returns:
            Path to the log file, or None if not found.
        """
        logs_dir = self._sentinel.config.execution.agent_logs_dir
        log_path = logs_dir / step / filename

        # Security check: ensure path is within logs directory
        try:
            log_path.resolve().relative_to(logs_dir.resolve())
        except ValueError:
            return None

        if log_path.exists() and log_path.is_file():
            return log_path

        return None
