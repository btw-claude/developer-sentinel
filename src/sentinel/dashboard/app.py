"""FastAPI application factory for the dashboard.

This module provides a factory function for creating the FastAPI application
that serves the dashboard. The application is configured with Jinja2 templates
and routes for displaying Sentinel state.

Custom Jinja2 filters:
    format_duration: Formats a duration in seconds as a human-readable string.
        Example: 125 -> "2m 5s", 45 -> "45s"
    url_quote: URL-encodes a string using percent-encoding (%20 for spaces).
        Unlike Jinja2's built-in |urlencode (which uses quote_plus and encodes
        spaces as +), this filter uses urllib.parse.quote which produces %20.
        Preferred for values embedded in JavaScript string contexts where the
        value may be used for display via decodeURIComponent().
        Example: "my orchestration" -> "my%20orchestration"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from sentinel.dashboard.routes import create_routes
from sentinel.dashboard.state import SentinelStateAccessor

if TYPE_CHECKING:
    from sentinel.main import Sentinel


def format_duration(seconds: float | int | None) -> str:
    """Format a duration in seconds as a human-readable string.

    Converts a duration value to a friendly minutes/seconds format.
    If the duration is 60 seconds or more, displays as "Xm Ys".
    Otherwise, displays as "Xs".

    Args:
        seconds: Duration in seconds, or None.

    Returns:
        Formatted duration string, or "0s" if None or negative.

    Examples:
        >>> format_duration(125)
        '2m 5s'
        >>> format_duration(45)
        '45s'
        >>> format_duration(None)
        '0s'
    """
    if seconds is None or seconds < 0:
        return "0s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def url_quote(value: str) -> str:
    """URL-encode a string using percent-encoding (RFC 3986).

    Unlike Jinja2's built-in ``|urlencode`` filter which uses
    ``urllib.parse.quote_plus`` (encoding spaces as ``+``), this filter uses
    ``urllib.parse.quote`` which encodes spaces as ``%20``.

    This is preferred for values embedded in JavaScript string contexts
    because ``decodeURIComponent()`` correctly decodes ``%20`` to spaces
    but does **not** decode ``+`` to spaces. Using ``%20`` ensures that
    values round-trip correctly between server-side encoding and
    client-side decoding.

    Args:
        value: The string to URL-encode.

    Returns:
        The percent-encoded string with spaces as %20.

    Examples:
        >>> url_quote("my orchestration")
        'my%20orchestration'
        >>> url_quote("hello+world")
        'hello%2Bworld'
        >>> url_quote("simple")
        'simple'
    """
    return quote(str(value), safe="")


class TemplateEnvironmentWrapper:
    """Wrapper to make Jinja2 Environment work with FastAPI's TemplateResponse.

    FastAPI's Jinja2Templates expects a specific interface. This wrapper
    provides that interface while using our configured Environment.
    """

    def __init__(self, env: Environment) -> None:
        """Initialize the wrapper.

        Args:
            env: The Jinja2 Environment to wrap.
        """
        self._env = env

    async def template_response(
        self,
        *,
        request: object,
        name: str,
        context: dict[str, Any] | None = None,
    ) -> HTMLResponse:
        """Asynchronously render a template and return an HTML response.

        This method is async and must be awaited. It uses Jinja2's async
        template rendering (render_async) which is required when the
        Environment is created with enable_async=True.

        Warning:
            This method must be awaited. Calling without await returns a
            coroutine object instead of the expected HTMLResponse.

            Example::

                # Correct usage:
                response = await templates.template_response(...)
                # Returns: <HTMLResponse status_code=200 ...>

                # Incorrect (returns coroutine object):
                response = templates.template_response(...)
                # Returns: <coroutine object template_response at 0x...>

        Note:
            This method uses Python's async/await pattern. When calling from
            synchronous code, you must run it within an async context (e.g.,
            using asyncio.run() or from within another async function).

            Example::

                # Calling from synchronous code:
                import asyncio
                response = asyncio.run(templates.template_response(request=req, name='index.html'))

            For more information on async/await patterns, see:
            https://docs.python.org/3/library/asyncio-task.html

        Args:
            request: The incoming HTTP request.
            name: The template name to render.
            context: Template context variables.

        Returns:
            An HTMLResponse with the rendered template.
        """
        template = self._env.get_template(name)
        context = context or {}
        context["request"] = request
        content = await template.render_async(**context)
        return HTMLResponse(content=content)

    # Alias for backwards compatibility with the expected interface
    TemplateResponse = template_response


def create_app(
    sentinel: Sentinel,
    *,
    templates_dir: Path | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI dashboard application.

    This factory function creates a FastAPI application configured with:
    - Jinja2 templates for rendering HTML pages
    - Static file serving (if static_dir is provided)
    - Routes for dashboard pages and API endpoints
    - Read-only state access through SentinelStateAccessor

    Args:
        sentinel: The Sentinel instance to monitor.
        templates_dir: Optional custom templates directory. Defaults to
            the templates/ directory within this module.
        static_dir: Optional static files directory for CSS/JS assets.

    Returns:
        A configured FastAPI application ready to serve the dashboard.
    """
    app = FastAPI(
        title="Developer Sentinel Dashboard",
        description="Monitoring dashboard for the Developer Sentinel orchestrator",
        version="0.1.0",
    )

    # Configure templates directory
    if templates_dir is None:
        templates_dir = Path(__file__).parent / "templates"

    # Configure static directory - defaults to static/ alongside templates/
    if static_dir is None:
        static_dir = Path(__file__).parent / "static"

    # Ensure templates directory and required subdirectories exist.
    # Note: This creates directories as a side effect if they don't exist.
    # This is intentional for development convenience but callers should be
    # aware that file system modifications occur during app initialization.
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "partials").mkdir(exist_ok=True)

    # Create Jinja2 environment with autoescape for security.
    # DS-825 audit: select_autoescape(["html", "xml"]) ensures all .html and
    # .xml templates have auto-escaping enabled by default.  This covers both
    # element content and attribute contexts (e.g. title="{{ svc.last_error }}"
    # in service_health.html).  No templates in this project use |safe or
    # {% autoescape false %}, so all dynamic output is escaped.
    template_env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=True,
    )

    # Register custom filters
    template_env.filters["format_duration"] = format_duration
    template_env.filters["url_quote"] = url_quote

    # Store wrapped templates in app state for access in routes
    app.state.templates = TemplateEnvironmentWrapper(template_env)

    # Mount static files if directory provided
    if static_dir is not None and static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Create state accessor for read-only access to Sentinel state
    # Type ignore: Sentinel returns dict[str, Any] instead of DTO types for version methods
    # This is an architectural mismatch between runtime behavior and protocol definition
    state_accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

    # Create and include routes with config for rate limiting
    dashboard_routes = create_routes(state_accessor, config=sentinel.config)
    app.include_router(dashboard_routes)

    return app
