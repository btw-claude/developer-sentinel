"""Read-only state accessor for the dashboard.

This module provides a thread-safe, read-only view of the Sentinel application state
for use by the dashboard. It ensures the dashboard cannot modify the orchestrator's
internal state while still providing access to monitoring data.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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
    source_file: str


@dataclass(frozen=True)
class ProjectOrchestrations:
    """Orchestrations grouped by project or repository (DS-224, DS-226).

    This groups orchestrations by their trigger project (for Jira) or
    repository (for GitHub) for a more organized dashboard display.

    .. note:: API Stability (DS-231)
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

    identifier: str  # Project key or repo name, e.g., "DS" for Jira or "org/repo" for GitHub
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
    """Read-only running step information for the dashboard (DS-122).

    This provides an immutable view of a running execution step including
    calculated elapsed time for display purposes.
    """

    issue_key: str
    orchestration_name: str
    attempt_number: int
    started_at: datetime
    elapsed_seconds: float
    log_filename: str  # derived from started_at (DS-319)


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

    # Grouped orchestrations (DS-224)
    jira_projects: list[ProjectOrchestrations] = field(default_factory=list)
    github_repos: list[ProjectOrchestrations] = field(default_factory=list)

    # Active orchestration counts (DS-255) - count of projects/repos with running orchestrations
    active_jira_projects_count: int = 0
    active_github_repos_count: int = 0

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
    active_incomplete_tasks: int = 0

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

        # Build a mapping from orchestration to source_file for lookup.
        # This is needed because source_file is stored in OrchestrationVersion.
        #
        # We use id(orch) as keys because Orchestration objects are identity-based
        # (each instance is unique) and the _versions_lock ensures the orchestration
        # objects referenced here remain valid and unchanged for the duration of
        # this method. This approach avoids requiring Orchestration to be hashable
        # while still providing O(1) lookup performance.
        orch_to_source_file: dict[int, str] = {}
        with sentinel._versions_lock:
            for version in sentinel._active_versions:
                orch_to_source_file[id(version.orchestration)] = str(version.source_file)

        # Extract orchestration info
        orchestration_infos = [
            self._orchestration_to_info(orch, orch_to_source_file.get(id(orch), ""))
            for orch in sentinel.orchestrations
        ]

        # Group orchestrations by project/repo (DS-224)
        jira_projects, github_repos = self._group_orchestrations(orchestration_infos)

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
                log_filename=step.started_at.strftime("%Y%m%d_%H%M%S") + ".log",
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

        # Count projects/repos with active orchestrations (DS-255)
        # Build a mapping from orchestration name to its project/repo
        active_orchestration_names = {step.orchestration_name for step in running_step_views}
        active_jira_projects: set[str] = set()
        active_github_repos: set[str] = set()
        for orch in orchestration_infos:
            if orch.name in active_orchestration_names:
                if orch.trigger_source == "jira" and orch.trigger_project:
                    active_jira_projects.add(orch.trigger_project)
                elif orch.trigger_source == "github" and orch.trigger_repo:
                    active_github_repos.add(orch.trigger_repo)

        return DashboardState(
            poll_interval=config.poll_interval,
            max_concurrent_executions=config.max_concurrent_executions,
            max_issues_per_poll=config.max_issues_per_poll,
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
            hot_reload_metrics=hot_reload_metrics,
            shutdown_requested=sentinel._shutdown_requested,
            active_incomplete_tasks=pending_count,
            system_status=system_status,
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
        trigger_source = getattr(trigger, "source", "jira")
        trigger_project = getattr(trigger, "project", None)
        trigger_repo = getattr(trigger, "repo", None)
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
            trigger_repo=trigger_repo,
            trigger_tags=trigger_tags,
            agent_prompt_preview=prompt_preview,
            source_file=source_file,
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

    def _group_orchestrations(
        self, orchestrations: list[OrchestrationInfo]
    ) -> tuple[list[ProjectOrchestrations], list[ProjectOrchestrations]]:
        """Group orchestrations by Jira project and GitHub repository (DS-224).

        Args:
            orchestrations: List of orchestration info objects to group.

        Returns:
            A tuple of (jira_projects, github_repos) where each is a list of
            ProjectOrchestrations grouped by project key or repo name.
        """
        jira_groups: dict[str, list[OrchestrationInfo]] = defaultdict(list)
        github_groups: dict[str, list[OrchestrationInfo]] = defaultdict(list)

        for orch in orchestrations:
            if orch.trigger_source == "jira" and orch.trigger_project:
                jira_groups[orch.trigger_project].append(orch)
            elif orch.trigger_source == "github" and orch.trigger_repo:
                github_groups[orch.trigger_repo].append(orch)

        # Convert to sorted lists of ProjectOrchestrations
        jira_projects = [
            ProjectOrchestrations(identifier=key, orchestrations=sorted(orchs, key=lambda o: o.name))
            for key, orchs in sorted(jira_groups.items())
        ]
        github_repos = [
            ProjectOrchestrations(identifier=key, orchestrations=sorted(orchs, key=lambda o: o.name))
            for key, orchs in sorted(github_groups.items())
        ]

        return jira_projects, github_repos

    def get_log_files(self) -> list[dict]:
        """Get available log files grouped by orchestration (DS-127).

        Discovers log files in the agent_logs_dir, grouped by orchestration
        name, with files sorted by modification time (newest first).

        Returns:
            List of dictionaries with orchestration name and files.
        """
        logs_dir = self._sentinel.config.agent_logs_dir

        if not logs_dir.exists():
            return []

        result = []

        # Iterate through orchestration directories
        for orch_dir in sorted(logs_dir.iterdir()):
            if not orch_dir.is_dir():
                continue

            orchestration_name = orch_dir.name
            files = []

            # Get log files in this orchestration directory
            for log_file in orch_dir.glob("*.log"):
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
                                    stat.st_mtime
                                ).isoformat(),
                            }
                        )
                    except Exception:
                        # Skip files that can't be accessed
                        pass

            # Sort files by modified time (newest first)
            files.sort(key=lambda f: f["modified"], reverse=True)

            if files:
                result.append(
                    {
                        "orchestration": orchestration_name,
                        "files": files,
                    }
                )

        # Sort orchestrations alphabetically
        result.sort(key=lambda x: x["orchestration"])

        return result

    def _format_log_display_name(self, filename: str) -> str:
        """Format a log filename for display (DS-127).

        Converts YYYYMMDD_HHMMSS.log to a human-readable format.

        Args:
            filename: The log filename.

        Returns:
            Human-readable display name.
        """
        try:
            # Remove .log extension
            name = filename.rsplit(".", 1)[0]
            # Parse datetime
            dt = datetime.strptime(name, "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return filename

    def get_log_file_path(self, orchestration: str, filename: str) -> Path | None:
        """Get the full path to a log file (DS-127).

        Args:
            orchestration: The orchestration name.
            filename: The log file name.

        Returns:
            Path to the log file, or None if not found.
        """
        logs_dir = self._sentinel.config.agent_logs_dir
        log_path = logs_dir / orchestration / filename

        # Security check: ensure path is within logs directory
        try:
            log_path.resolve().relative_to(logs_dir.resolve())
        except ValueError:
            return None

        if log_path.exists() and log_path.is_file():
            return log_path

        return None
