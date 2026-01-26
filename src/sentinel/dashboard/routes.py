"""Route handlers for the dashboard.

This module defines the HTTP route handlers for the dashboard web interface.
All handlers receive state through the SentinelStateAccessor to ensure
read-only access to the orchestrator's state.

DS-250: Add API endpoints for orchestration toggle actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Literal

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from sentinel.yaml_writer import OrchestrationYamlWriter, OrchestrationYamlWriterError

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

    @dashboard_router.post("/api/orchestrations/{name}/toggle", response_model=ToggleResponse)
    async def toggle_orchestration(name: str, request_body: ToggleRequest) -> ToggleResponse:
        """Toggle the enabled status of a single orchestration (DS-250).

        This endpoint toggles the enabled field in the orchestration's YAML file.
        The change takes effect on the next hot-reload cycle.

        Args:
            name: The name of the orchestration to toggle.
            request_body: The toggle request containing the new enabled status.

        Returns:
            ToggleResponse with success status, new enabled state, and name.

        Raises:
            HTTPException: 404 if orchestration not found, 500 for YAML errors.
        """
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

        try:
            writer = OrchestrationYamlWriter()
            result = writer.toggle_orchestration(source_file, name, request_body.enabled)

            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Orchestration '{name}' not found in file {source_file}",
                )

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

    @dashboard_router.post("/api/orchestrations/bulk-toggle", response_model=BulkToggleResponse)
    async def bulk_toggle_orchestrations(request_body: BulkToggleRequest) -> BulkToggleResponse:
        """Toggle the enabled status of orchestrations by source and identifier (DS-250).

        This endpoint bulk toggles orchestrations based on their source type
        (jira or github) and identifier (project key or org/repo).

        Args:
            request_body: The bulk toggle request containing source, identifier,
                and new enabled status.

        Returns:
            BulkToggleResponse with success status and count of toggled orchestrations.

        Raises:
            HTTPException: 404 if no matching orchestrations found, 500 for YAML errors.
        """
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
