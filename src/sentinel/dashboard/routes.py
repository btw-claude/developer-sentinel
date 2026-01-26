"""Route handlers for the dashboard.

This module defines the HTTP route handlers for the dashboard web interface.
All handlers receive state through the SentinelStateAccessor to ensure
read-only access to the orchestrator's state.

DS-250: Add API endpoints for orchestration toggle actions.
DS-259: Add rate limiting and OpenAPI documentation to toggle endpoints.
DS-282: Refactored to use validation helpers from sentinel.validation module.
DS-285: Add deprecation warnings to backwards compatibility aliases.
DS-287: Add version numbers to deprecation warnings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Literal

from cachetools import TTLCache

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from sentinel.validation import (
    MAX_CACHE_MAXSIZE,
    MAX_CACHE_TTL,
    MAX_TOGGLE_COOLDOWN,
    MIN_CACHE_MAXSIZE,
    MIN_CACHE_TTL,
    MIN_TOGGLE_COOLDOWN,
    validate_positive_float,
    validate_positive_int,
)
from sentinel.yaml_writer import OrchestrationYamlWriter, OrchestrationYamlWriterError

# Rate limiting configuration for toggle endpoints (DS-259, DS-268, DS-274, DS-278, DS-282)
# Cooldown period in seconds between writes to the same file
# Configurable via environment variable for operational flexibility (DS-268)
# Default values and reasonable bounds for each configuration (DS-278)
# Bounds constants moved to sentinel.validation module (DS-282)
_DEFAULT_TOGGLE_COOLDOWN: float = 2.0
_DEFAULT_RATE_LIMIT_CACHE_TTL: int = 3600  # 1 hour
_DEFAULT_RATE_LIMIT_CACHE_MAXSIZE: int = 10000  # 10k entries

# Backwards compatibility aliases for bounds constants (DS-282, DS-285)
# These are deprecated and will be removed in a future release.
# Users should import directly from sentinel.validation instead.
def __getattr__(name: str) -> float | int:
    """Provide deprecated access to bounds constants (DS-285).

    This function intercepts attribute access for deprecated module-level
    constants and emits a deprecation warning, guiding users to import
    from sentinel.validation directly.
    """
    _deprecated_constants = {
        "_MIN_TOGGLE_COOLDOWN": ("MIN_TOGGLE_COOLDOWN", MIN_TOGGLE_COOLDOWN),
        "_MAX_TOGGLE_COOLDOWN": ("MAX_TOGGLE_COOLDOWN", MAX_TOGGLE_COOLDOWN),
        "_MIN_CACHE_TTL": ("MIN_CACHE_TTL", MIN_CACHE_TTL),
        "_MAX_CACHE_TTL": ("MAX_CACHE_TTL", MAX_CACHE_TTL),
        "_MIN_CACHE_MAXSIZE": ("MIN_CACHE_MAXSIZE", MIN_CACHE_MAXSIZE),
        "_MAX_CACHE_MAXSIZE": ("MAX_CACHE_MAXSIZE", MAX_CACHE_MAXSIZE),
    }
    if name in _deprecated_constants:
        canonical_name, value = _deprecated_constants[name]
        warnings.warn(
            f"{name} is deprecated and will be removed in version 2.0. "
            f"Import {canonical_name} from sentinel.validation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _validate_positive_float(
    env_var: str, value_str: str, default: float, min_val: float, max_val: float
) -> float:
    """Deprecated wrapper for validate_positive_float (DS-285).

    This function is deprecated. Import validate_positive_float from
    sentinel.validation directly instead.
    """
    warnings.warn(
        "_validate_positive_float is deprecated and will be removed in version 2.0. "
        "Import validate_positive_float from sentinel.validation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_positive_float(env_var, value_str, default, min_val, max_val)


def _validate_positive_int(
    env_var: str, value_str: str, default: int, min_val: int, max_val: int
) -> int:
    """Deprecated wrapper for validate_positive_int (DS-285).

    This function is deprecated. Import validate_positive_int from
    sentinel.validation directly instead.
    """
    warnings.warn(
        "_validate_positive_int is deprecated and will be removed in version 2.0. "
        "Import validate_positive_int from sentinel.validation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_positive_int(env_var, value_str, default, min_val, max_val)


# Parse and validate environment variables with input validation (DS-278)
_toggle_cooldown_env = os.environ.get("SENTINEL_TOGGLE_COOLDOWN")
if _toggle_cooldown_env is not None:
    TOGGLE_COOLDOWN_SECONDS: float = validate_positive_float(
        "SENTINEL_TOGGLE_COOLDOWN",
        _toggle_cooldown_env,
        _DEFAULT_TOGGLE_COOLDOWN,
        MIN_TOGGLE_COOLDOWN,
        MAX_TOGGLE_COOLDOWN,
    )
else:
    TOGGLE_COOLDOWN_SECONDS = _DEFAULT_TOGGLE_COOLDOWN

# Track last write time per file path for rate limiting (DS-268, DS-274)
# Using TTLCache to automatically clean up stale entries
# Cache TTL and maxsize are configurable via environment variables (DS-274)
# Allows operators to tune cache behavior for different memory constraints and usage patterns
_cache_ttl_env = os.environ.get("SENTINEL_RATE_LIMIT_CACHE_TTL")
if _cache_ttl_env is not None:
    _RATE_LIMIT_CACHE_TTL: int = validate_positive_int(
        "SENTINEL_RATE_LIMIT_CACHE_TTL",
        _cache_ttl_env,
        _DEFAULT_RATE_LIMIT_CACHE_TTL,
        MIN_CACHE_TTL,
        MAX_CACHE_TTL,
    )
else:
    _RATE_LIMIT_CACHE_TTL = _DEFAULT_RATE_LIMIT_CACHE_TTL

_cache_maxsize_env = os.environ.get("SENTINEL_RATE_LIMIT_CACHE_MAXSIZE")
if _cache_maxsize_env is not None:
    _RATE_LIMIT_CACHE_MAXSIZE: int = validate_positive_int(
        "SENTINEL_RATE_LIMIT_CACHE_MAXSIZE",
        _cache_maxsize_env,
        _DEFAULT_RATE_LIMIT_CACHE_MAXSIZE,
        MIN_CACHE_MAXSIZE,
        MAX_CACHE_MAXSIZE,
    )
else:
    _RATE_LIMIT_CACHE_MAXSIZE = _DEFAULT_RATE_LIMIT_CACHE_MAXSIZE

_last_write_times: TTLCache[str, float] = TTLCache(
    maxsize=_RATE_LIMIT_CACHE_MAXSIZE, ttl=_RATE_LIMIT_CACHE_TTL
)


def _check_rate_limit(file_path: str) -> None:
    """Check if a file write is allowed based on rate limiting.

    Raises HTTPException with 429 status if the cooldown period has not elapsed
    since the last write to the same file. (DS-259)

    Args:
        file_path: The path to the file being written.

    Raises:
        HTTPException: 429 if the cooldown period has not elapsed.
    """
    current_time = time.monotonic()
    last_write = _last_write_times.get(file_path)

    if last_write is not None:
        elapsed = current_time - last_write
        if elapsed < TOGGLE_COOLDOWN_SECONDS:
            remaining = TOGGLE_COOLDOWN_SECONDS - elapsed
            logger.warning(
                "Rate limit exceeded for file %s, %.1fs remaining in cooldown",
                file_path,
                remaining,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait {remaining:.1f} seconds before toggling again.",
            )


def _record_write(file_path: str) -> None:
    """Record the write time for a file for rate limiting purposes. (DS-259)

    Args:
        file_path: The path to the file that was written.
    """
    _last_write_times[file_path] = time.monotonic()


if TYPE_CHECKING:
    from sentinel.dashboard.state import SentinelStateAccessor


# Request/Response models for orchestration toggle endpoints (DS-250)
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


def create_routes(state_accessor: SentinelStateAccessor) -> APIRouter:
    """Create dashboard routes with the given state accessor.

    This factory function creates an APIRouter with all dashboard routes
    configured to use the provided state accessor for data retrieval.

    Args:
        state_accessor: The state accessor for retrieving Sentinel state.

    Returns:
        An APIRouter with all dashboard routes configured.
    """
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
        return await templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"state": state},
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
        return await templates.TemplateResponse(
            request=request,
            name="orchestrations.html",
            context={"state": state},
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
        return await templates.TemplateResponse(
            request=request,
            name="metrics.html",
            context={"state": state},
        )

    @dashboard_router.get("/api/state")
    async def api_state(request: Request) -> dict:
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
        return await templates.TemplateResponse(
            request=request,
            name="partials/status.html",
            context={"state": state},
        )

    @dashboard_router.get("/partials/orchestrations", response_class=HTMLResponse)
    async def partial_orchestrations(request: Request, source: str = "jira") -> HTMLResponse:
        """Render the orchestrations partial for HTMX updates (DS-224).

        Args:
            request: The incoming HTTP request.
            source: Filter by source type - "jira" or "github".

        Returns:
            HTML response with the orchestrations partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates

        # Select the appropriate grouped data based on source
        if source == "github":
            projects = state.github_repos
            source_type = "github"
        else:
            projects = state.jira_projects
            source_type = "jira"

        return await templates.TemplateResponse(
            request=request,
            name="partials/orchestrations.html",
            context={"projects": projects, "source_type": source_type, "state": state},
        )

    @dashboard_router.get("/partials/running_steps", response_class=HTMLResponse)
    async def partial_running_steps(request: Request) -> HTMLResponse:
        """Render the running steps partial for HTMX updates (DS-122).

        This endpoint returns the running steps section HTML for efficient
        live updates without full page reloads. Auto-refreshes every 1s.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the running steps partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return await templates.TemplateResponse(
            request=request,
            name="partials/running_steps.html",
            context={"state": state},
        )

    @dashboard_router.get("/partials/issue_queue", response_class=HTMLResponse)
    async def partial_issue_queue(request: Request) -> HTMLResponse:
        """Render the issue queue partial for HTMX updates (DS-123).

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
        return await templates.TemplateResponse(
            request=request,
            name="partials/issue_queue.html",
            context={"state": state},
        )

    @dashboard_router.get("/partials/system_status", response_class=HTMLResponse)
    async def partial_system_status(request: Request) -> HTMLResponse:
        """Render the system status partial for HTMX updates (DS-124).

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
        return await templates.TemplateResponse(
            request=request,
            name="partials/system_status.html",
            context={"state": state},
        )

    @dashboard_router.get("/health")
    async def health() -> dict:
        """Health check endpoint.

        Returns:
            JSON with health status.
        """
        return {"status": "healthy"}

    @dashboard_router.get("/logs", response_class=HTMLResponse)
    async def logs(request: Request) -> HTMLResponse:
        """Render the log viewer page (DS-127).

        This page allows users to view log files from orchestration executions
        with real-time updates via SSE.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the log viewer page.
        """
        log_files = state_accessor.get_log_files()
        templates = request.app.state.templates
        return await templates.TemplateResponse(
            request=request,
            name="log_viewer.html",
            context={"log_files": log_files},
        )

    @dashboard_router.get("/api/logs/files")
    async def api_logs_files() -> list[dict]:
        """Return the list of available log files per orchestration (DS-125).

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
        """Stream log file content via SSE (DS-127).

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

        async def event_generator() -> AsyncGenerator[dict, None]:
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
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}),
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
                        with open(log_path, "r") as f:
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
                                    "data": json.dumps(
                                        {"type": "initial", "content": content}
                                    ),
                                }

                        last_size = current_size
                        last_mtime = current_mtime

                except Exception:
                    # Log file access errors for debugging (DS-144)
                    logger.warning(
                        "Error accessing log file %s/%s",
                        orchestration,
                        filename,
                        exc_info=True,
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
        """Toggle the enabled status of a single orchestration (DS-250, DS-259).

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
        logger.debug("toggle_orchestration called for '%s' with enabled=%s", name, request_body.enabled)
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

        # Check rate limit before writing (DS-259)
        _check_rate_limit(str(source_file))

        try:
            writer = OrchestrationYamlWriter()
            result = writer.toggle_orchestration(source_file, name, request_body.enabled)

            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Orchestration '{name}' not found in file {source_file}",
                )

            # Record successful write for rate limiting (DS-259)
            _record_write(str(source_file))

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
        """Toggle the enabled status of orchestrations by source and identifier (DS-250, DS-259).

        This endpoint bulk toggles orchestrations based on their source type
        (jira or github) and identifier (project key or org/repo).

        Args:
            request_body: The bulk toggle request containing source, identifier,
                and new enabled status.

        Returns:
            BulkToggleResponse with success status and count of toggled orchestrations.

        Raises:
            HTTPException: 404 if no matching orchestrations found, 429 for rate limit, 500 for YAML errors.
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
            if request_body.source == "jira":
                if orch.trigger_source == "jira" and orch.trigger_project == request_body.identifier:
                    orch_files[orch.name] = source_file
            elif request_body.source == "github":
                if orch.trigger_source == "github" and orch.trigger_repo == request_body.identifier:
                    orch_files[orch.name] = source_file

        if not orch_files:
            raise HTTPException(
                status_code=404,
                detail=f"No orchestrations found for {request_body.source} '{request_body.identifier}'",
            )

        # Check rate limit for all unique files before writing (DS-259)
        unique_files = set(str(f) for f in orch_files.values())
        for file_path in unique_files:
            _check_rate_limit(file_path)

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

            # Record successful writes for rate limiting (DS-259)
            for file_path in unique_files:
                _record_write(file_path)

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
