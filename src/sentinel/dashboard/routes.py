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
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from cachetools import TTLCache
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from sentinel.orchestration import OrchestrationError, _parse_orchestration
from sentinel.types import AgentTypeLiteral, CursorModeLiteral, TriggerSource, TriggerSourceLiteral
from sentinel.yaml_writer import OrchestrationYamlWriter, OrchestrationYamlWriterError

if TYPE_CHECKING:
    from sentinel.config import Config
    from sentinel.dashboard.state import SentinelStateAccessor
    from sentinel.health import HealthChecker

logger = logging.getLogger(__name__)


# Sunset date for the deprecated /health endpoint (RFC 8594 HTTP-date format)
# Used in both Deprecation and Sunset headers
HEALTH_ENDPOINT_SUNSET_DATE = "Sat, 01 Jun 2026 00:00:00 GMT"

# Maximum CSRF token requests allowed per client IP per minute (DS-738).
# Guards against TTLCache exhaustion in the csrf_tokens cache (maxsize=1000).
CSRF_TOKEN_RATE_LIMIT_PER_MINUTE = 30


class RateLimiter:
    """Rate limiter for file write operations.

    This class provides rate limiting for toggle endpoints to prevent rapid
    file writes. It reads configuration from the ``Config`` object (cooldown
    interval, cache TTL, and cache max-size).

    The rate limiter tracks the last write time per file path using a TTLCache
    to automatically clean up stale entries.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the rate limiter with configuration.

        Args:
            config: Application configuration containing rate limiting settings.
        """
        self._toggle_cooldown_seconds = config.dashboard.toggle_cooldown_seconds
        self._last_write_times: TTLCache[str, float] = TTLCache(
            maxsize=config.dashboard.rate_limit_cache_maxsize,
            ttl=config.dashboard.rate_limit_cache_ttl,
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


# Pydantic models for orchestration edit endpoints (DS-727)
class TriggerEditRequest(BaseModel):
    """Request model for editing trigger configuration."""

    source: TriggerSourceLiteral | None = None
    project: str | None = None
    jql_filter: str | None = None
    tags: list[str] | None = None
    project_number: int | None = None
    project_scope: Literal["org", "user"] | None = None
    project_owner: str | None = None
    project_filter: str | None = None
    labels: list[str] | None = None


class GitHubContextEditRequest(BaseModel):
    """Request model for editing GitHub context configuration."""

    host: str | None = None
    org: str | None = None
    repo: str | None = None
    branch: str | None = None
    create_branch: bool | None = None
    base_branch: str | None = None


class AgentEditRequest(BaseModel):
    """Request model for editing agent configuration."""

    prompt: str | None = None
    github: GitHubContextEditRequest | None = None
    timeout_seconds: int | None = None
    model: str | None = None
    agent_type: AgentTypeLiteral | None = None
    cursor_mode: CursorModeLiteral | None = None
    agent_teams: bool | None = None
    strict_template_variables: bool | None = None


class RetryEditRequest(BaseModel):
    """Request model for editing retry configuration."""

    max_attempts: int | None = None
    success_patterns: list[str] | None = None
    failure_patterns: list[str] | None = None
    default_status: Literal["success", "failure"] | None = None
    default_outcome: str | None = None


class OutcomeEditRequest(BaseModel):
    """Request model for editing outcome configuration."""

    name: str | None = None
    patterns: list[str] | None = None
    add_tag: str | None = None


class LifecycleEditRequest(BaseModel):
    """Request model for editing lifecycle configuration."""

    on_start_add_tag: str | None = None
    on_complete_remove_tag: str | None = None
    on_complete_add_tag: str | None = None
    on_failure_add_tag: str | None = None


class OrchestrationEditRequest(BaseModel):
    """Request model for editing an orchestration configuration.

    All fields are optional. Only provided fields will be updated.
    The 'name' field is read-only and cannot be edited.
    """

    enabled: bool | None = None
    max_concurrent: int | None = None
    trigger: TriggerEditRequest | None = None
    agent: AgentEditRequest | None = None
    retry: RetryEditRequest | None = None
    outcomes: list[OutcomeEditRequest] | None = None
    lifecycle: LifecycleEditRequest | None = None


class OrchestrationEditResponse(BaseModel):
    """Response model for orchestration edit."""

    success: bool
    name: str
    errors: list[str] = []


class DeleteResponse(BaseModel):
    """Response model for orchestration deletion."""

    success: bool
    name: str


class OrchestrationCreateRequest(BaseModel):
    """Request model for creating a new orchestration."""

    name: str
    target_file: str
    enabled: bool | None = None
    max_concurrent: int | None = None
    trigger: TriggerEditRequest | None = None
    agent: AgentEditRequest | None = None
    retry: RetryEditRequest | None = None
    outcomes: list[OutcomeEditRequest] | None = None
    lifecycle: LifecycleEditRequest | None = None


class OrchestrationCreateResponse(BaseModel):
    """Response model for orchestration creation."""

    success: bool
    name: str
    errors: list[str] = []


def _build_yaml_updates(request: OrchestrationEditRequest) -> dict[str, Any]:
    """Convert an OrchestrationEditRequest to a YAML-compatible update dict.

    Uses Pydantic v2 model_dump(exclude_none=True) for top-level and nested
    models to avoid manual is-not-None checks. The lifecycle section requires
    special post-processing to map flat Pydantic fields to the nested YAML
    structure (on_start, on_complete, on_failure).

    Args:
        request: The edit request containing fields to update.

    Returns:
        A dictionary mirroring the YAML structure with only the
        fields that should be updated.
    """
    # Use model_dump to get all non-None fields automatically
    raw = request.model_dump(exclude_none=True)

    updates: dict[str, Any] = {}

    # Copy top-level scalar fields directly
    for key in ("enabled", "max_concurrent"):
        if key in raw:
            updates[key] = raw[key]

    # Nested models: model_dump already excludes None recursively
    if "trigger" in raw and raw["trigger"]:
        updates["trigger"] = raw["trigger"]

    if "agent" in raw and raw["agent"]:
        updates["agent"] = raw["agent"]

    if "retry" in raw and raw["retry"]:
        updates["retry"] = raw["retry"]

    if "outcomes" in raw:
        updates["outcomes"] = raw["outcomes"]

    # Lifecycle requires special handling: flat Pydantic fields map to nested YAML
    if "lifecycle" in raw and raw["lifecycle"]:
        lc = raw["lifecycle"]
        if "on_start_add_tag" in lc:
            updates.setdefault("on_start", {})["add_tag"] = lc["on_start_add_tag"]
        if "on_complete_remove_tag" in lc:
            updates.setdefault("on_complete", {})["remove_tag"] = lc["on_complete_remove_tag"]
        if "on_complete_add_tag" in lc:
            updates.setdefault("on_complete", {})["add_tag"] = lc["on_complete_add_tag"]
        if "on_failure_add_tag" in lc:
            updates.setdefault("on_failure", {})["add_tag"] = lc["on_failure_add_tag"]

    return updates


def _validate_orchestration_updates(
    orch_name: str,
    current_data: dict[str, Any],
    updates: dict[str, Any],
) -> list[str]:
    """Validate orchestration updates by passing through _parse_orchestration.

    Merges the updates into a copy of the current orchestration data and
    validates the result through the existing _parse_orchestration() function.
    This reuses ALL existing validation (branch patterns, agent_type/cursor_mode
    combos, project keys, etc.) without duplication.

    Note: _parse_orchestration() raises on the first validation error, so this
    function returns at most one error string. Users making multiple mistakes
    will need to fix them one at a time.

    Args:
        orch_name: The name of the orchestration being updated.
        current_data: The current orchestration data as a plain dict.
        updates: The updates to merge.

    Returns:
        A list of validation error messages. Empty list means validation passed.
    """
    # Deep merge updates into a copy of current data
    merged = _deep_merge_dicts(current_data, updates)

    # Ensure name is preserved in merged data
    merged["name"] = orch_name

    try:
        _parse_orchestration(merged)
        return []
    except OrchestrationError as e:
        return [str(e)]


def _deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep merge updates into a copy of base dict.

    Creates a new dict with base values overridden by updates.
    Nested dicts are recursively merged. Lists are replaced entirely,
    not appended or element-wise merged.

    Args:
        base: The base dictionary to merge into.
        updates: The updates to apply.

    Returns:
        A new dictionary with merged values.
    """
    result = dict(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


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
        effective_config.dashboard.toggle_cooldown_seconds,
        effective_config.dashboard.rate_limit_cache_ttl,
        effective_config.dashboard.rate_limit_cache_maxsize,
    )

    # Create rate limiter with config values
    rate_limiter = RateLimiter(effective_config)

    # CSRF protection for state-changing endpoints (DS-736)
    csrf_tokens: TTLCache[str, bool] = TTLCache(maxsize=1000, ttl=3600)

    # Rate limiting for CSRF token endpoint to prevent TTLCache exhaustion (DS-737)
    csrf_rate_limit: TTLCache[str, int] = TTLCache(maxsize=256, ttl=60)

    def _generate_csrf_token() -> str:
        """Generate a CSRF token and store it for validation."""
        token = secrets.token_urlsafe(32)
        csrf_tokens[token] = True
        return token

    def _validate_csrf_token(token: str | None) -> None:
        """Validate and consume a CSRF token.

        Args:
            token: The CSRF token to validate.

        Raises:
            HTTPException: 403 if token is missing or invalid.
        """
        if not token or token not in csrf_tokens:
            raise HTTPException(
                status_code=403,
                detail="Invalid or missing CSRF token",
            )
        # Consume token (single-use)
        del csrf_tokens[token]

    dashboard_router = APIRouter()

    @dashboard_router.get("/api/csrf-token")
    async def api_csrf_token(request: Request) -> dict[str, str]:
        """Generate and return a CSRF token.

        Provides a CSRF token for use in state-changing API requests.
        Tokens are single-use and expire after 1 hour.

        Rate limited to CSRF_TOKEN_RATE_LIMIT_PER_MINUTE requests per minute
        per client IP to prevent potential TTLCache exhaustion (DS-737). The
        csrf_tokens cache has maxsize=1000, so unbounded generation could fill
        and evict valid tokens.

        Args:
            request: The incoming HTTP request (used for client IP extraction).

        Returns:
            Dict containing the CSRF token.

        Raises:
            HTTPException: 429 if rate limit is exceeded.
        """
        client_ip = request.client.host if request.client else "unknown"
        if client_ip == "unknown":
            logger.warning(
                "CSRF token request from unknown client IP (request.client is None); "
                "this may indicate the app is behind a proxy that does not set "
                "X-Forwarded-For or similar headers"
            )
        current_count = csrf_rate_limit.get(client_ip, 0)
        if current_count >= CSRF_TOKEN_RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="CSRF token request rate limit exceeded. Please try again later.",
            )
        csrf_rate_limit[client_ip] = current_count + 1
        return {"csrf_token": _generate_csrf_token()}

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
                context={"state": state, "csrf_token": _generate_csrf_token()},
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

    @dashboard_router.get("/api/orchestrations/files")
    async def api_orchestrations_files() -> list[str]:
        """Return list of YAML files in the orchestrations directory.

        Recursively scans the orchestrations directory for .yaml and .yml files
        and returns their relative paths for use in the file selector.

        Returns:
            List of relative file paths (strings) from orchestrations_dir.
        """
        orchestrations_dir = effective_config.execution.orchestrations_dir

        yaml_files = []
        try:
            if orchestrations_dir.exists() and orchestrations_dir.is_dir():
                for pattern in ("*.yaml", "*.yml"):
                    for file_path in orchestrations_dir.rglob(pattern):
                        if file_path.is_file():
                            # Return relative path from orchestrations_dir
                            relative_path = file_path.relative_to(orchestrations_dir)
                            yaml_files.append(str(relative_path))
        except OSError as e:
            logger.warning("Error scanning orchestrations directory: %s", e)

        # Sort for consistent ordering
        yaml_files.sort()
        return yaml_files

    @dashboard_router.get("/api/orchestrations/{name}/detail")
    async def api_orchestration_detail(name: str) -> dict[str, Any]:
        """Return full orchestration configuration as JSON.

        Retrieves detailed orchestration configuration including trigger,
        agent, retry, outcome, and lifecycle settings.

        Args:
            name: The orchestration name.

        Returns:
            JSON representation of the full orchestration configuration.

        Raises:
            HTTPException: 404 if orchestration not found.
        """
        detail = state_accessor.get_orchestration_detail(name)
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"Orchestration '{name}' not found",
            )
        return {
            "name": detail.name,
            "enabled": detail.enabled,
            "max_concurrent": detail.max_concurrent,
            "source_file": detail.source_file,
            "trigger": {
                "source": detail.trigger.source,
                "project": detail.trigger.project,
                "jql_filter": detail.trigger.jql_filter,
                "tags": detail.trigger.tags,
                "project_number": detail.trigger.project_number,
                "project_scope": detail.trigger.project_scope,
                "project_owner": detail.trigger.project_owner,
                "project_filter": detail.trigger.project_filter,
                "labels": detail.trigger.labels,
            },
            "agent": {
                "prompt": detail.agent.prompt,
                "github": (
                    {
                        "host": detail.agent.github.host,
                        "org": detail.agent.github.org,
                        "repo": detail.agent.github.repo,
                        "branch": detail.agent.github.branch,
                        "create_branch": detail.agent.github.create_branch,
                        "base_branch": detail.agent.github.base_branch,
                    }
                    if detail.agent.github
                    else None
                ),
                "timeout_seconds": detail.agent.timeout_seconds,
                "model": detail.agent.model,
                "agent_type": detail.agent.agent_type,
                "cursor_mode": detail.agent.cursor_mode,
                "strict_template_variables": detail.agent.strict_template_variables,
            },
            "retry": {
                "max_attempts": detail.retry.max_attempts,
                "success_patterns": detail.retry.success_patterns,
                "failure_patterns": detail.retry.failure_patterns,
                "default_status": detail.retry.default_status,
                "default_outcome": detail.retry.default_outcome,
            },
            "outcomes": [
                {
                    "name": outcome.name,
                    "patterns": outcome.patterns,
                    "add_tag": outcome.add_tag,
                }
                for outcome in detail.outcomes
            ],
            "lifecycle": {
                "on_start_add_tag": detail.lifecycle.on_start_add_tag,
                "on_complete_remove_tag": detail.lifecycle.on_complete_remove_tag,
                "on_complete_add_tag": detail.lifecycle.on_complete_add_tag,
                "on_failure_add_tag": detail.lifecycle.on_failure_add_tag,
            },
        }

    @dashboard_router.get(
        "/partials/orchestration_detail/{name}", response_class=HTMLResponse
    )
    async def partial_orchestration_detail(request: Request, name: str) -> HTMLResponse:
        """Render the orchestration detail partial for HTMX inline expansion.

        Returns rendered HTML for the full orchestration configuration,
        displayed inline when a user clicks an orchestration row.

        Args:
            request: The incoming HTTP request.
            name: The orchestration name.

        Returns:
            HTML response with the orchestration detail partial.

        Raises:
            HTTPException: 404 if orchestration not found.
        """
        detail = state_accessor.get_orchestration_detail(name)
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"Orchestration '{name}' not found",
            )
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/orchestration_detail.html",
                context={"detail": detail},
            ),
        )

    @dashboard_router.get(
        "/partials/orchestration_edit/{name}", response_class=HTMLResponse
    )
    async def partial_orchestration_edit(request: Request, name: str) -> HTMLResponse:
        """Render the orchestration edit form partial for HTMX inline editing.

        Returns a pre-populated edit form that replaces the detail view
        when a user clicks the Edit button.

        Args:
            request: The incoming HTTP request.
            name: The orchestration name.

        Returns:
            HTML response with the orchestration edit form partial.

        Raises:
            HTTPException: 404 if orchestration not found.
        """
        detail = state_accessor.get_orchestration_detail(name)
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"Orchestration '{name}' not found",
            )
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/orchestration_edit_form.html",
                context={"detail": detail},
            ),
        )

    @dashboard_router.get(
        "/partials/orchestration_create", response_class=HTMLResponse
    )
    async def partial_orchestration_create(request: Request) -> HTMLResponse:
        """Render the orchestration creation form partial for HTMX.

        Returns an empty creation form with a name field and file selector.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTML response with the orchestration creation form partial.
        """
        templates = request.app.state.templates
        return cast(
            HTMLResponse,
            await templates.TemplateResponse(
                request=request,
                name="partials/orchestration_create_form.html",
                context={"csrf_token": _generate_csrf_token()},
            ),
        )

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
                and orch.trigger_project_owner == request_body.identifier
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

            toggled_count = writer.toggle_by_project(
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

    @dashboard_router.put(
        "/api/orchestrations/{name}",
        response_model=OrchestrationEditResponse,
        summary="Edit orchestration configuration",
        description="Update the configuration of an orchestration by name. "
        "Only provided fields are updated; omitted fields are left unchanged. "
        "The 'name' field is read-only. Validates changes through the existing "
        "orchestration parser before writing to YAML. "
        "Rate limited to prevent rapid file writes.",
    )
    async def edit_orchestration(
        name: str, request_body: OrchestrationEditRequest
    ) -> OrchestrationEditResponse:
        """Edit the configuration of an orchestration.

        This endpoint updates orchestration fields in the YAML file.
        Changes are validated through _parse_orchestration() before writing.
        The change takes effect on the next hot-reload cycle.

        Args:
            name: The name of the orchestration to edit.
            request_body: The edit request containing fields to update.

        Returns:
            OrchestrationEditResponse with success status, name, and any errors.

        Raises:
            HTTPException: 404 if orchestration not found, 422 for validation errors,
                429 for rate limit, 500 for YAML errors.
        """
        logger.debug("edit_orchestration called for '%s'", name)
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

        # Build YAML-compatible update dict from Pydantic model
        updates = _build_yaml_updates(request_body)

        if not updates:
            return OrchestrationEditResponse(success=True, name=name)

        # Read current orchestration data for validation
        try:
            writer = OrchestrationYamlWriter()
            current_data = writer.read_orchestration(source_file, name)
        except OrchestrationYamlWriterError as e:
            logger.error("Failed to read orchestration '%s': %s", name, e)
            raise HTTPException(status_code=500, detail=str(e)) from e

        if current_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Orchestration '{name}' not found in file {source_file}",
            )

        # Validate the merged result through _parse_orchestration()
        errors = _validate_orchestration_updates(name, current_data, updates)
        if errors:
            raise HTTPException(status_code=422, detail=errors[0])

        # Check rate limit before writing
        rate_limiter.check_rate_limit(str(source_file))

        try:
            result = writer.update_orchestration(source_file, name, updates)

            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Orchestration '{name}' not found in file {source_file}",
                )

            # Record successful write for rate limiting
            rate_limiter.record_write(str(source_file))

            return OrchestrationEditResponse(success=True, name=name)

        except OrchestrationYamlWriterError as e:
            logger.error("Failed to edit orchestration '%s': %s", name, e)
            raise HTTPException(
                status_code=500,
                detail=str(e),
            ) from e

    @dashboard_router.delete(
        "/api/orchestrations/{name}",
        response_model=DeleteResponse,
        summary="Delete an orchestration",
        description="Delete an orchestration by name. Removes the orchestration from "
        "its YAML file and hot-reload picks up the change. Rate limited to "
        "prevent rapid file writes.",
    )
    async def delete_orchestration(name: str) -> DeleteResponse:
        """Delete an orchestration by name.

        This endpoint removes the orchestration from its YAML source file.
        The change takes effect on the next hot-reload cycle.

        Args:
            name: The name of the orchestration to delete.

        Returns:
            DeleteResponse with success status and orchestration name.

        Raises:
            HTTPException: 404 if orchestration not found, 429 for rate limit, 500 for YAML errors.
        """
        logger.debug("delete_orchestration called for '%s'", name)
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
            result = writer.delete_orchestration(source_file, name)

            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Orchestration '{name}' not found in file {source_file}",
                )

            # Record successful write for rate limiting
            rate_limiter.record_write(str(source_file))

            return DeleteResponse(
                success=True,
                name=name,
            )

        except OrchestrationYamlWriterError as e:
            logger.error("Failed to delete orchestration '%s': %s", name, e)
            raise HTTPException(
                status_code=500,
                detail=str(e),
            ) from e

    @dashboard_router.post(
        "/api/orchestrations",
        response_model=OrchestrationCreateResponse,
        summary="Create a new orchestration",
        description="Create a new orchestration in the specified target file. "
        "Validates name uniqueness and target file path. Rate limited to "
        "prevent rapid file writes.",
    )
    async def create_orchestration(
        request_body: OrchestrationCreateRequest,
        x_csrf_token: str | None = Header(None),
    ) -> OrchestrationCreateResponse:
        """Create a new orchestration.

        This endpoint creates a new orchestration in the target YAML file.
        The change takes effect on the next hot-reload cycle.

        Args:
            request_body: The create request containing orchestration configuration.
            x_csrf_token: CSRF token from X-CSRF-Token header.

        Returns:
            OrchestrationCreateResponse with success status, name, and any errors.

        Raises:
            HTTPException: 403 for CSRF validation failure,
                422 for validation errors (duplicate name, invalid path),
                429 for rate limit, 500 for YAML errors.
        """
        _validate_csrf_token(x_csrf_token)
        logger.debug("create_orchestration called for '%s'", request_body.name)
        state = state_accessor.get_state()

        # Validate name uniqueness across all loaded orchestrations
        for orch in state.orchestrations:
            if orch.name == request_body.name:
                raise HTTPException(
                    status_code=422,
                    detail=f"Orchestration with name '{request_body.name}' already exists",
                )

        # Get orchestrations_dir from config
        orchestrations_dir = effective_config.execution.orchestrations_dir

        # Validate target_file is within orchestrations_dir
        target_file_path = Path(request_body.target_file)
        if not target_file_path.is_absolute():
            target_file_path = orchestrations_dir / target_file_path

        try:
            target_file_path.resolve().relative_to(orchestrations_dir.resolve())
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Target file '{request_body.target_file}' is not within "
                    f"orchestrations directory"
                ),
            ) from None

        # Build orchestration data dict from request
        orch_data: dict[str, Any] = {"name": request_body.name}

        # Add optional fields using _build_yaml_updates pattern
        updates = _build_yaml_updates(
            OrchestrationEditRequest(
                enabled=request_body.enabled,
                max_concurrent=request_body.max_concurrent,
                trigger=request_body.trigger,
                agent=request_body.agent,
                retry=request_body.retry,
                outcomes=request_body.outcomes,
                lifecycle=request_body.lifecycle,
            )
        )
        orch_data.update(updates)

        # Validate via _parse_orchestration
        try:
            _parse_orchestration(orch_data)
        except OrchestrationError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        # Check rate limit before writing
        rate_limiter.check_rate_limit(str(target_file_path))

        try:
            writer = OrchestrationYamlWriter()
            writer.add_orchestration(target_file_path, orch_data, orchestrations_dir)

            # Record successful write for rate limiting
            rate_limiter.record_write(str(target_file_path))

            return OrchestrationCreateResponse(
                success=True,
                name=request_body.name,
            )

        except OrchestrationYamlWriterError as e:
            logger.error("Failed to create orchestration '%s': %s", request_body.name, e)
            raise HTTPException(
                status_code=500,
                detail=str(e),
            ) from e

    return dashboard_router
