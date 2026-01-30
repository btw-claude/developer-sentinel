"""Exception handling utilities for agent tools.

This module provides a decorator and helper functions that standardize
exception handling across tool execute methods. Instead of repeating
the same exception handling pattern in every tool, tools can use the
handle_tool_exceptions decorator.

The pattern handles three categories of exceptions:
- I/O errors (OSError, TimeoutError): Network/filesystem issues
- Data errors (KeyError, TypeError, ValueError): Invalid data from APIs
- Runtime errors (RuntimeError): Unexpected runtime issues
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from sentinel.agents.base import ToolResult

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_tool_exceptions(
    error_code: str,
    error_message_template: str,
) -> Callable[[F], F]:
    """Decorator to handle common exceptions in tool execute methods.

    This decorator wraps a tool's execute method and catches common exceptions,
    logging them appropriately and converting them to ToolResult failures.

    Args:
        error_code: The error code to use in ToolResult.fail() (e.g., "JIRA_ERROR").
        error_message_template: A template string for the error message.
            Use {context} as a placeholder for context info passed to the wrapper,
            and {error} for the exception message.

    Returns:
        A decorator that wraps the execute method with exception handling.

    Example:
        @handle_tool_exceptions(
            error_code="JIRA_ERROR",
            error_message_template="Failed to get issue {context}: {error}",
        )
        def execute(self, **kwargs: Any) -> ToolResult:
            # ... implementation ...
            return ToolResult.ok(result), issue_key  # Return context as second value

    Note:
        The decorated method should return a tuple of (ToolResult, context_string)
        where context_string is used in error messages. If the method returns
        just a ToolResult, an empty context is used.
    """
    # Import here to avoid circular imports
    from sentinel.agents.base import ToolResult

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, **kwargs: Any) -> ToolResult:
            tool_name = getattr(self, "name", "unknown_tool")
            logger.debug(f"Executing {tool_name}", extra={"tool": tool_name, "params": kwargs})

            # Validate params if the method exists
            if hasattr(self, "validate_params"):
                validation_error: ToolResult | None = self.validate_params(**kwargs)
                if validation_error:
                    return validation_error

            try:
                result: ToolResult = func(self, **kwargs)
                # Handle case where function returns (result, context) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    tool_result: ToolResult = result[0]
                    if tool_result.success:
                        logger.info(f"Tool {tool_name} succeeded", extra={"tool": tool_name})
                    return tool_result
                # Handle case where function returns just the result
                if result.success:
                    logger.info(f"Tool {tool_name} succeeded", extra={"tool": tool_name})
                return result
            except (OSError, TimeoutError) as e:
                context = _extract_context(kwargs)
                logger.error(
                    f"Tool {tool_name} failed due to I/O or timeout: {e}",
                    extra={"tool": tool_name, "error": str(e), "error_type": type(e).__name__},
                )
                return ToolResult.fail(
                    error_message_template.format(context=context, error=e),
                    error_code,
                )
            except (KeyError, TypeError, ValueError) as e:
                context = _extract_context(kwargs)
                logger.error(
                    f"Tool {tool_name} failed due to data error: {e}",
                    extra={"tool": tool_name, "error": str(e), "error_type": type(e).__name__},
                )
                return ToolResult.fail(
                    error_message_template.format(context=context, error=e),
                    error_code,
                )
            except RuntimeError as e:
                context = _extract_context(kwargs)
                logger.error(
                    f"Tool {tool_name} failed due to runtime error: {e}",
                    extra={"tool": tool_name, "error": str(e), "error_type": type(e).__name__},
                )
                return ToolResult.fail(
                    error_message_template.format(context=context, error=e),
                    error_code,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def _extract_context(kwargs: dict[str, Any]) -> str:
    """Extract a meaningful context string from kwargs for error messages.

    Looks for common key identifiers in the kwargs to build a context string.

    Args:
        kwargs: The keyword arguments passed to the tool.

    Returns:
        A context string suitable for error messages.
    """
    # Common identifier keys in order of preference
    context_keys = [
        "issue_key",  # Jira
        "page_id",  # Confluence
        "pr_number",  # GitHub PR
        "issue_number",  # GitHub issue
        "path",  # File path
        "cql",  # Confluence query
        "jql",  # Jira query
    ]

    for key in context_keys:
        if key in kwargs:
            value = kwargs[key]
            # Format PR/issue numbers with #
            if key in ("pr_number", "issue_number"):
                return f"#{value}"
            return str(value)

    # Fallback: return empty string if no context found
    return ""
