"""Route handlers for the dashboard.

This module defines the HTTP route handlers for the dashboard web interface.
All handlers receive state through the SentinelStateAccessor to ensure
read-only access to the orchestrator's state.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from sentinel.dashboard.state import SentinelStateAccessor


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
    async def partial_orchestrations(request: Request) -> HTMLResponse:
        """Render the orchestrations partial for HTMX updates.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the orchestrations partial.
        """
        state = state_accessor.get_state()
        templates = request.app.state.templates
        return await templates.TemplateResponse(
            request=request,
            name="partials/orchestrations.html",
            context={"state": state},
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

    return dashboard_router
