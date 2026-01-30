"""Route handlers for the dashboard.

This module defines the HTTP route handlers for the dashboard web interface.
All handlers receive state through the SentinelStateAccessor to ensure
read-only access to the orchestrator's state.

Health check endpoints provide:
- /health: Legacy health endpoint (deprecated, use /health/live)
- /health/live: Liveness probe (basic health check)
- /health/ready: Readiness probe (checks external service dependencies)
- /health/dashboard: Dashboard status endpoint (verifies dashboard is operational)

The /health/dashboard endpoint is specifically designed to verify that the
dashboard component is running and accessible. This is useful for monitoring
dashboard availability separately from the core Sentinel health, especially
since the dashboard can fail to start while Sentinel continues operating
in dashboard-less mode.

Deprecated module-level constants:
    The following constants have been deprecated and moved to Config class.
    Accessing them will emit a DeprecationWarning:
    - _DEFAULT_TOGGLE_COOLDOWN: Use Config.toggle_cooldown_seconds instead
    - _DEFAULT_RATE_LIMIT_CACHE_TTL: Use Config.rate_limit_cache_ttl instead
    - _DEFAULT_RATE_LIMIT_CACHE_MAXSIZE: Use Config.rate_limit_cache_maxsize instead
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from sentinel.types import TriggerSource
from sentinel.yaml_writer import OrchestrationYamlWriter, OrchestrationYamlWriterError

if TYPE_CHECKING:
    from sentinel.config import Config
    from sentinel.dashboard.state import SentinelStateAccessor
    from sentinel.health import HealthChecker

logger = logging.getLogger(__name__)


# Sunset date for the deprecated /health endpoint (RFC 8594 HTTP-date format)
# Used in both Deprecation and Sunset headers
HEALTH_ENDPOINT_SUNSET_DATE = "Sat, 01 Jun 2026 00:00:00 GMT"


# Legacy module-level constants (deprecated)
# These are kept for backward compatibility but will emit deprecation warnings
# when accessed. New code should use Config class values instead.
_DEPRECATED_CONSTANTS = {
    "_DEFAULT_TOGGLE_COOLDOWN": (2.0, "Config.toggle_cooldown_seconds"),
    "_DEFAULT_RATE_LIMIT_CACHE_TTL": (3600, "Config.rate_limit_cache_ttl"),
    "_DEFAULT_RATE_LIMIT_CACHE_MAXSIZE": (10000, "Config.rate_limit_cache_maxsize"),
}


def __getattr__(name: str) -> Any:
    """Emit deprecation warning when accessing legacy module-level constants.

    This function intercepts attribute access on the module and emits a
    DeprecationWarning when accessing the deprecated constants while still
    returning their values for backward compatibility.

    Args:
        name: The attribute name being accessed.

    Returns:
        The deprecated constant value if it exists.

    Raises:
        AttributeError: If the attribute is not a deprecated constant.
    """
    if name in _DEPRECATED_CONSTANTS:
        value, replacement = _DEPRECATED_CONSTANTS[name]
        warnings.warn(
            f"{name} is deprecated and will be removed in a future release. "
            f"Use {replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class RateLimiter:
    """Rate limiter for file write operations.

    This class provides rate limiting for toggle endpoints to prevent rapid
    file writes. It uses Config values for configuration instead of module-level
    constants.

    The rate limiter tracks the last write time per file path using a TTLCache
    to automatically clean up stale entries.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the rate limiter with configuration.

        Args:
            config: Application configuration containing rate limiting settings.
        """
        self._toggle_cooldown_seconds = config.toggle_cooldown_seconds
        self._last_write_times: TTLCache[str, float] = TTLCache(
            maxsize=config.rate_limit_cache_maxsize,
            ttl=config.rate_limit_cache_ttl,
        )

    def check_rate_limit(self, file_path: str) -> None:
        """Check if a file write is allowed based on rate limiting.

        Raises HTTPException with 429 status if the cooldown period has not elapsed
        since the last write to the same file.

        Args:
            file_path: The path to the file being written.

        Raises:
            HTTPException: 429 if the cooldown period has not elapsed.
        """
        current_time = time.monotonic()
        last_write = self._last_write_times.get(file_path)

        if last_write is not None:
            elapsed = current_time - last_write
            if elapsed < self._toggle_cooldown_seconds:
                remaining = self._toggle_cooldown_seconds - elapsed
                logger.warning(
                    "Rate limit exceeded for file %s, %.1fs remaining in cooldown",
                    file_path,
                    remaining,
                )
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit exceeded. Please wait {remaining:.1f} "
                        "seconds before toggling again."
                    ),
                )

    def record_write(self, file_path: str) -> None:
        """Record the write time for a file for rate limiting purposes.

        Args:
            file_path: The path to the file that was written.
        """
        self._last_write_times[file_path] = time.monotonic()


# Request/Response models for orchestration toggle endpoints
class ToggleRequest(BaseModel):
    """Request model for single orchestration toggle."""

    enabled: bool


class ToggleResponse(BaseModel):
    """Response model for single orchestration toggle."""

    success: bool
    enabled: bool
    name: str


class BulkToggleRequest(BaseModel):
    """Request model for bulk orchestration toggle."""

    source: Literal["jira", "github"]
    identifier: str
    enabled: bool


class BulkToggleResponse(BaseModel):
    """Response model for bulk orchestration toggle."""

    success: bool
    toggled_count: int


def create_routes(
    state_accessor: SentinelStateAccessor,
    health_checker: HealthChecker | None = None,
    config: Config | None = None,
) -> APIRouter:
    """Create dashboard routes with the given state accessor.

    This factory function creates an APIRouter with all dashboard routes
    configured to use the provided state accessor for data retrieval.

    Args:
        state_accessor: The state accessor for retrieving Sentinel state.
        health_checker: Optional health checker for dependency checks.
            If not provided, readiness checks will return basic status.
        config: Optional application configuration for rate limiting settings.
            If not provided, default Config values will be used.

    Returns:
        An APIRouter with all dashboard routes configured.
    """
    # Import Config here to avoid circular imports at module level
    from sentinel.config import Config as ConfigClass

    # Use provided config or create a default one
    effective_config = config if config is not None else ConfigClass()
    config_source = "provided" if config is not None else "default"
    logger.debug(
        "create_routes using %s Config: toggle_cooldown=%.1fs, " "cache_ttl=%ds, cache_maxsize=%d",
        config_source,
        effective_config.toggle_cooldown_seconds,
        effective_config.rate_limit_cache_ttl,
        effective_config.rate_limit_cache_maxsize,
    )

    # Create rate limiter with config values
    rate_limiter = RateLimiter(effective_config)

    dashboard_router = APIRouter()

    @dashboard_router.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Render the main dashboard page.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the rendered dashboard.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"state": state},
            ),
        )

    @dashboard_router.get("/orchestrations", response_class=HTMLResponse)
    async def orchestrations(request: Request) -> HTMLResponse:
        """Render the orchestrations list page.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the orchestrations list.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="orchestrations.html",
                context={"state": state},
            ),
        )

    @dashboard_router.get("/metrics", response_class=HTMLResponse)
    async def metrics(request: Request) -> HTMLResponse:
        """Render the metrics page.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with metrics display.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="metrics.html",
                context={"state": state},
            ),
        )

    @dashboard_router.get("/api/state")
    async def api_state(request: Request) -> dict[str, Any]:
        """Return the current state as JSON.

        This endpoint is useful for HTMX partial updates and JavaScript clients.

        Args:
            request: The incoming HTTP request.

        Returns:
            JSON representation of the current state.
        """
        state = state_accessor.get_state()
        return {
            "poll_interval": state.poll_interval,
            "max_concurrent_executions": state.max_concurrent_executions,
            "max_issues_per_poll": state.max_issues_per_poll,
            "orchestrations_count": len(state.orchestrations),
            "active_versions_count": len(state.active_versions),
            "pending_removal_count": len(state.pending_removal_versions),
            "active_execution_count": state.active_execution_count,
            "available_slots": state.available_slots,
            "shutdown_requested": state.shutdown_requested,
            "active_incomplete_tasks": state.active_incomplete_tasks,
            "hot_reload_metrics": (
                {
                    "loaded_total": state.hot_reload_metrics.orchestrations_loaded_total,
                    "unloaded_total": state.hot_reload_metrics.orchestrations_unloaded_total,
                    "reloaded_total": state.hot_reload_metrics.orchestrations_reloaded_total,
                }
                if state.hot_reload_metrics
                else None
            ),
        }

    @dashboard_router.get("/partials/status", response_class=HTMLResponse)
    async def partial_status(request: Request) -> HTMLResponse:
        """Render the status partial for HTMX updates.

        This endpoint returns just the status section HTML for efficient
        live updates without full page reloads.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the status partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/status.html",
                context={"state": state},
            ),
        )

    @dashboard_router.get("/partials/orchestrations", response_class=HTMLResponse)
    async def partial_orchestrations(
        request: Request, source: str = TriggerSource.JIRA.value
    ) -> HTMLResponse:
        """Render the orchestrations partial for HTMX updates.

        Args:
            request: The incoming HTTP request.
            source: Filter by source type - "jira" or "github".

        Returns:
            HTML response with the orchestrations partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates

        # Select the appropriate grouped data based on source
        if source == TriggerSource.GITHUB.value:
            projects = state.github_repos
            source_type = TriggerSource.GITHUB.value
        else:
            projects = state.jira_projects
            source_type = TriggerSource.JIRA.value

        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/orchestrations.html",
                context={"projects": projects, "source_type": source_type, "state": state},
            ),
        )

    @dashboard_router.get("/partials/running_steps", response_class=HTMLResponse)
    async def partial_running_steps(request: Request) -> HTMLResponse:
        """Render the running steps partial for HTMX updates.

        This endpoint returns the running steps section HTML for efficient
        live updates without full page reloads. Auto-refreshes every 1s.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the running steps partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/running_steps.html",
                context={"state": state},
            ),
        )

    @dashboard_router.get("/partials/issue_queue", response_class=HTMLResponse)
    async def partial_issue_queue(request: Request) -> HTMLResponse:
        """Render the issue queue partial for HTMX updates.

        This endpoint returns the issue queue section HTML for efficient
        live updates without full page reloads. Shows issues waiting for
        execution slots.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the issue queue partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/issue_queue.html",
                context={"state": state},
            ),
        )

    @dashboard_router.get("/partials/system_status", response_class=HTMLResponse)
    async def partial_system_status(request: Request) -> HTMLResponse:
        """Render the system status partial for HTMX updates.

        This endpoint returns the system status section HTML for efficient
        live updates without full page reloads. Shows thread pool usage,
        last poll times, poll interval, and process uptime.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the system status partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/system_status.html",
                context={"state": state},
            ),
        )

    @dashboard_router.get("/partials/recent_executions", response_class=HTMLResponse)
    async def partial_recent_executions(request: Request) -> HTMLResponse:
        """Render the recent executions partial for HTMX updates.

        This endpoint returns the recent executions section HTML for efficient
        live updates without full page reloads. Shows completed executions
        with their token usage and costs.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the recent executions partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/recent_executions.html",
                context={"state": state},
            ),
        )

    @dashboard_router.get("/health")
    async def health(response: Response) -> dict[str, str]:
        """Legacy health check endpoint.

        Deprecated: Use /health/live for liveness checks or
        /health/ready for readiness checks with dependency verification.

        This endpoint includes a Deprecation header per RFC 8594 to signal
        to API consumers that this endpoint is deprecated. The Sunset header
        indicates the planned removal date.

        Returns:
            JSON with health status.
        """
        # Add Deprecation header per RFC 8594 to signal this endpoint is deprecated
        # Using HTTP-date format for the sunset date per RFC 8594
        response.headers["Deprecation"] = HEALTH_ENDPOINT_SUNSET_DATE
        response.headers["Sunset"] = HEALTH_ENDPOINT_SUNSET_DATE
        response.headers["Link"] = (
            '</health/live>; rel="successor-version", </health/ready>; rel="successor-version"'
        )
        return {"status": "healthy"}

    @dashboard_router.get("/health/live")
    async def health_live() -> dict[str, Any]:
        """Liveness probe endpoint.

        Basic health check to verify the service is running.
        Does not check external dependencies.

        Returns:
            JSON with liveness status.
        """
        if health_checker is not None:
            result = health_checker.check_liveness()
            return result.to_dict()
        return {"status": "healthy", "timestamp": time.time(), "checks": {}}

    @dashboard_router.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        """Readiness probe endpoint.

        Checks connectivity to configured external services:
        - Jira (if configured)
        - GitHub (if configured)
        - Claude API (if API key is configured)

        Returns:
            JSON with readiness status and individual service checks.

        Example response:
            {
                "status": "healthy",
                "timestamp": 1706472123.456,
                "checks": {
                    "jira": {"status": "up", "latency_ms": 45},
                    "github": {"status": "up", "latency_ms": 32},
                    "claude": {"status": "up", "latency_ms": 120}
                }
            }
        """
        if health_checker is not None:
            result = await health_checker.check_readiness()
            return result.to_dict()
        return {"status": "healthy", "timestamp": time.time(), "checks": {}}

    @dashboard_router.get("/health/dashboard")
    async def health_dashboard() -> dict[str, Any]:
        """Dashboard status endpoint.

        Verifies that the dashboard component is operational. This endpoint
        is useful for monitoring dashboard availability separately from the
        core Sentinel health.

        Since the dashboard is an optional component that can fail to start
        while Sentinel continues operating, this endpoint provides a way to
        specifically check if the dashboard is available.

        If you can reach this endpoint, the dashboard is running.

        Returns:
            JSON with dashboard status.

        Example response:
            {
                "status": "up",
                "component": "dashboard",
                "timestamp": 1706472123.456,
                "message": "Dashboard is operational"
            }
        """
        return {
            "status": "up",
            "component": "dashboard",
            "timestamp": time.time(),
            "message": "Dashboard is operational",
        }

    @dashboard_router.get("/logs", response_class=HTMLResponse)
    async def logs(request: Request) -> HTMLResponse:
        """Render the log viewer page.

        This page allows users to view log files from orchestration executions
        with real-time updates via SSE.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the log viewer page.
        """
        log_files = state_accessor.get_log_files()
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="log_viewer.html",
                context={"log_files": log_files},
            ),
        )

    @dashboard_router.get("/api/logs/files")
    async def api_logs_files() -> list[dict[str, Any]]:
        """Return the list of available log files per orchestration.

        This endpoint discovers log files in the agent_logs_dir, grouped by
        orchestration name. Files are sorted by modification time with most
        recent logs first. Directory structure expected:
        {base_dir}/{orchestration_name}/{timestamp}.log

        Returns:
            List of dictionaries containing orchestration name and log files.
            Each entry has:
            - orchestration: The orchestration name
            - files: List of file info (filename, display_name, size, modified)
        """
        return state_accessor.get_log_files()

    @dashboard_router.get("/api/logs/stream/{orchestration}/{filename}")
    async def stream_log(
        request: Request,
        orchestration: str,
        filename: str,
    ) -> EventSourceResponse:
        """Stream log file content via SSE.

        This endpoint streams the content of a log file in real-time using
        Server-Sent Events. It sends the initial content and then watches
        for updates.

        Args:
            request: The incoming HTTP request.
            orchestration: The orchestration name.
            filename: The log file name.

        Returns:
            EventSourceResponse streaming log content.
        """

        async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
            """Generate SSE events for log file content."""
            log_path = state_accessor.get_log_file_path(orchestration, filename)

            if log_path is None or not log_path.exists():
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Log file not found"}),
                }
                return

            # Send initial content
            try:
                content = log_path.read_text()
                yield {
                    "event": "initial",
                    "data": json.dumps({"type": "initial", "content": content}),
                }
            except OSError as e:
                logger.warning(
                    "Failed to read log file %s/%s: %s",
                    orchestration,
                    filename,
                    e,
                )
                yield {
                    "event": "error",
                    "data": json.dumps({"error": f"I/O error: {e}"}),
                }
                return
            except UnicodeDecodeError as e:
                logger.warning(
                    "Failed to decode log file %s/%s: %s",
                    orchestration,
                    filename,
                    e,
                )
                yield {
                    "event": "error",
                    "data": json.dumps({"error": f"Encoding error: {e}"}),
                }
                return

            # Watch for updates
            last_size = log_path.stat().st_size
            last_mtime = log_path.stat().st_mtime

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                await asyncio.sleep(1)  # Poll every second

                try:
                    if not log_path.exists():
                        break

                    current_mtime = log_path.stat().st_mtime
                    current_size = log_path.stat().st_size

                    if current_mtime > last_mtime or current_size > last_size:
                        # File has been modified, read new content
                        with open(log_path) as f:
                            if current_size > last_size:
                                # Append mode - seek to last position
                                f.seek(last_size)
                                new_content = f.read()
                                if new_content:
                                    yield {
                                        "event": "append",
                                        "data": json.dumps(
                                            {"type": "append", "content": new_content}
                                        ),
                                    }
                            else:
                                # File might have been truncated/rewritten
                                content = f.read()
                                yield {
                                    "event": "initial",
                                    "data": json.dumps({"type": "initial", "content": content}),
                                }

                        last_size = current_size
                        last_mtime = current_mtime

                except OSError as e:
                    # Log file access errors for debugging
                    logger.warning(
                        "I/O error accessing log file %s/%s: %s",
                        orchestration,
                        filename,
                        e,
                    )
                except UnicodeDecodeError as e:
                    # Log encoding errors for debugging
                    logger.warning(
                        "Encoding error reading log file %s/%s: %s",
                        orchestration,
                        filename,
                        e,
                    )

        return EventSourceResponse(event_generator())

    @dashboard_router.post(
        "/api/orchestrations/{name}/toggle",
        response_model=ToggleResponse,
        summary="Toggle orchestration enabled status",
        description="Toggle the enabled status of a single orchestration by name. "
        "Modifies the orchestration's YAML file and changes take effect on "
        "the next hot-reload cycle. Rate limited to prevent rapid file writes.",
    )
    async def toggle_orchestration(name: str, request_body: ToggleRequest) -> ToggleResponse:
        """Toggle the enabled status of a single orchestration.

        This endpoint toggles the enabled field in the orchestration's YAML file.
        The change takes effect on the next hot-reload cycle.

        Args:
            name: The name of the orchestration to toggle.
            request_body: The toggle request containing the new enabled status.

        Returns:
            ToggleResponse with success status, new enabled state, and name.

        Raises:
            HTTPException: 404 if orchestration not found, 429 for rate limit, 500 for YAML errors.
        """
        logger.debug(
            "toggle_orchestration called for '%s' with enabled=%s", name, request_body.enabled
        )
        state = state_accessor.get_state()

        # Find the orchestration to get its source file
        orch_info = None
        for orch in state.orchestrations:
            if orch.name == name:
                orch_info = orch
                break

        if orch_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Orchestration '{name}' not found",
            )

        if not orch_info.source_file:
            raise HTTPException(
                status_code=404,
                detail=f"Source file not found for orchestration '{name}'",
            )

        source_file = Path(orch_info.source_file)
        if not source_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Orchestration file not found: {source_file}",
            )

        # Check rate limit before writing
        rate_limiter.check_rate_limit(str(source_file))

        try:
            writer = OrchestrationYamlWriter()
            result = writer.toggle_orchestration(source_file, name, request_body.enabled)

            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Orchestration '{name}' not found in file {source_file}",
                )

            # Record successful write for rate limiting
            rate_limiter.record_write(str(source_file))

            return ToggleResponse(
                success=True,
                enabled=request_body.enabled,
                name=name,
            )

        except OrchestrationYamlWriterError as e:
            logger.error("Failed to toggle orchestration '%s': %s", name, e)
            raise HTTPException(
                status_code=500,
                detail=str(e),
            ) from e

    @dashboard_router.post(
        "/api/orchestrations/bulk-toggle",
        response_model=BulkToggleResponse,
        summary="Bulk toggle orchestrations by source",
        description="Toggle the enabled status of multiple orchestrations by their "
        "source type (jira or github) and identifier (project key or org/repo). "
        "Rate limited per file to prevent rapid writes.",
    )
    async def bulk_toggle_orchestrations(request_body: BulkToggleRequest) -> BulkToggleResponse:
        """Toggle the enabled status of orchestrations by source and identifier.

        This endpoint bulk toggles orchestrations based on their source type
        (jira or github) and identifier (project key or org/repo).

        Args:
            request_body: The bulk toggle request containing source, identifier,
                and new enabled status.

        Returns:
            BulkToggleResponse with success status and count of toggled orchestrations.

        Raises:
            HTTPException: 404 if no matching orchestrations found,
                429 for rate limit, 500 for YAML errors.
        """
        logger.debug(
            "bulk_toggle_orchestrations called for %s '%s' with enabled=%s",
            request_body.source,
            request_body.identifier,
            request_body.enabled,
        )
        state = state_accessor.get_state()

        # Build a mapping of orchestration names to their source files
        orch_files: dict[str, Path] = {}

        for orch in state.orchestrations:
            if not orch.source_file:
                continue

            source_file = Path(orch.source_file)
            if not source_file.exists():
                continue

            # Filter by source type and identifier
            if request_body.source == TriggerSource.JIRA.value:
                if (
                    orch.trigger_source == TriggerSource.JIRA.value
                    and orch.trigger_project == request_body.identifier
                ):
                    orch_files[orch.name] = source_file
            elif (
                request_body.source == TriggerSource.GITHUB.value
                and orch.trigger_source == TriggerSource.GITHUB.value
                and orch.trigger_repo == request_body.identifier
            ):
                orch_files[orch.name] = source_file

        if not orch_files:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No orchestrations found for {request_body.source} "
                    f"'{request_body.identifier}'"
                ),
            )

        # Check rate limit for all unique files before writing
        unique_files = {str(f) for f in orch_files.values()}
        for file_path in unique_files:
            rate_limiter.check_rate_limit(file_path)

        try:
            writer = OrchestrationYamlWriter()

            if request_body.source == "jira":
                toggled_count = writer.toggle_by_project(
                    orch_files, request_body.identifier, request_body.enabled
                )
            else:  # github
                toggled_count = writer.toggle_by_repo(
                    orch_files, request_body.identifier, request_body.enabled
                )

            # Record successful writes for rate limiting
            for file_path in unique_files:
                rate_limiter.record_write(file_path)

            return BulkToggleResponse(
                success=True,
                toggled_count=toggled_count,
            )

        except OrchestrationYamlWriterError as e:
            logger.error(
                "Failed to bulk toggle orchestrations for %s '%s': %s",
                request_body.source,
                request_body.identifier,
                e,
            )
            raise HTTPException(
                status_code=500,
                detail=str(e),
            ) from e

    return dashboard_router
