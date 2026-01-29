"""FastAPI application factory for the dashboard.

This module provides a factory function for creating the FastAPI application
that serves the dashboard. The application is configured with Jinja2 templates
and routes for displaying Sentinel state.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from sentinel.dashboard.routes import create_routes
from sentinel.dashboard.state import SentinelStateAccessor

if TYPE_CHECKING:
    from sentinel.main import Sentinel


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
        context: dict | None = None,
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

    # Create Jinja2 environment with autoescape for security
    template_env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=True,
    )

    # Store wrapped templates in app state for access in routes
    app.state.templates = TemplateEnvironmentWrapper(template_env)

    # Mount static files if directory provided
    if static_dir is not None and static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Create state accessor for read-only access to Sentinel state
    state_accessor = SentinelStateAccessor(sentinel)

    # Create and include routes with config for rate limiting
    dashboard_routes = create_routes(state_accessor, config=sentinel.config)
    app.include_router(dashboard_routes)

    return app
