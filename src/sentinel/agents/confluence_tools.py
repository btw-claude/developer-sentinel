"""Confluence tool definitions for Claude agents.

This module defines tools that allow agents to interact with Confluence:
- Create pages
- Update pages
- Get page content
- Search content
- Add comments to pages
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sentinel.agents.base import (
    ParameterType,
    Tool,
    ToolParameter,
    ToolResult,
    ToolSchema,
)
from sentinel.logging import get_logger

logger = get_logger(__name__)


class ConfluenceToolClient(ABC):
    """Abstract interface for Confluence operations.

    This allows tools to work with different implementations:
    - MCP-based client (production)
    - Mock client (testing)
    """

    @abstractmethod
    def get_page(self, page_id: str) -> dict[str, Any]:
        """Get page content and metadata."""
        pass

    @abstractmethod
    def create_page(
        self,
        space_key: str,
        title: str,
        body: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new page."""
        pass

    @abstractmethod
    def update_page(
        self,
        page_id: str,
        title: str,
        body: str,
        version: int,
    ) -> dict[str, Any]:
        """Update an existing page."""
        pass

    @abstractmethod
    def search_content(self, cql: str, limit: int = 25) -> list[dict[str, Any]]:
        """Search for content using CQL."""
        pass

    @abstractmethod
    def add_comment(self, page_id: str, body: str) -> dict[str, Any]:
        """Add a comment to a page."""
        pass

    @abstractmethod
    def get_page_by_title(self, space_key: str, title: str) -> dict[str, Any] | None:
        """Get a page by its title in a space."""
        pass


class GetPageTool(Tool):
    """Tool to get Confluence page content."""

    def __init__(self, client: ConfluenceToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="confluence_get_page",
            description=(
                "Get a Confluence page's content and metadata by its ID. "
                "Returns the page title, body content, version, and space information."
            ),
            category="confluence",
            parameters=[
                ToolParameter(
                    name="page_id",
                    description="The Confluence page ID",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        logger.debug(f"Executing {self.name}", extra={"tool": self.name, "params": kwargs})
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        page_id = kwargs["page_id"]
        try:
            result = self.client.get_page(page_id)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except Exception as e:
            logger.error(
                f"Tool {self.name} failed: {e}", extra={"tool": self.name, "error": str(e)}
            )
            return ToolResult.fail(f"Failed to get page {page_id}: {e}", "CONFLUENCE_ERROR")


class GetPageByTitleTool(Tool):
    """Tool to get a Confluence page by title."""

    def __init__(self, client: ConfluenceToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="confluence_get_page_by_title",
            description=(
                "Find a Confluence page by its title within a specific space. "
                "Returns the page if found, or null if not found."
            ),
            category="confluence",
            parameters=[
                ToolParameter(
                    name="space_key",
                    description="The Confluence space key (e.g., 'DEV', 'DOCS')",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="title",
                    description="The page title to search for (exact match)",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        logger.debug(f"Executing {self.name}", extra={"tool": self.name, "params": kwargs})
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        space_key = kwargs["space_key"]
        title = kwargs["title"]
        try:
            result = self.client.get_page_by_title(space_key, title)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            if result is None:
                return ToolResult.ok({"found": False, "page": None})
            return ToolResult.ok({"found": True, "page": result})
        except Exception as e:
            logger.error(
                f"Tool {self.name} failed: {e}", extra={"tool": self.name, "error": str(e)}
            )
            return ToolResult.fail(
                f"Failed to find page '{title}' in space {space_key}: {e}",
                "CONFLUENCE_ERROR",
            )


class CreatePageTool(Tool):
    """Tool to create a new Confluence page."""

    def __init__(self, client: ConfluenceToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="confluence_create_page",
            description=(
                "Create a new Confluence page in a space. "
                "Optionally specify a parent page to create a child page."
            ),
            category="confluence",
            parameters=[
                ToolParameter(
                    name="space_key",
                    description="The Confluence space key (e.g., 'DEV', 'DOCS')",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="title",
                    description="The page title",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="body",
                    description="The page content in Confluence storage format (XHTML)",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="parent_id",
                    description="Optional parent page ID to create this as a child page",
                    type=ParameterType.STRING,
                    required=False,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        logger.debug(f"Executing {self.name}", extra={"tool": self.name, "params": kwargs})
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        try:
            result = self.client.create_page(
                space_key=kwargs["space_key"],
                title=kwargs["title"],
                body=kwargs["body"],
                parent_id=kwargs.get("parent_id"),
            )
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except Exception as e:
            logger.error(
                f"Tool {self.name} failed: {e}", extra={"tool": self.name, "error": str(e)}
            )
            return ToolResult.fail(f"Failed to create page: {e}", "CONFLUENCE_ERROR")


class UpdatePageTool(Tool):
    """Tool to update an existing Confluence page."""

    def __init__(self, client: ConfluenceToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="confluence_update_page",
            description=(
                "Update an existing Confluence page. You must provide the current "
                "version number (get it from confluence_get_page) to prevent conflicts."
            ),
            category="confluence",
            parameters=[
                ToolParameter(
                    name="page_id",
                    description="The Confluence page ID to update",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="title",
                    description="The new page title",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="body",
                    description="The new page content in Confluence storage format (XHTML)",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="version",
                    description="The current version number (for optimistic locking)",
                    type=ParameterType.INTEGER,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        logger.debug(f"Executing {self.name}", extra={"tool": self.name, "params": kwargs})
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        page_id = kwargs["page_id"]
        try:
            result = self.client.update_page(
                page_id=page_id,
                title=kwargs["title"],
                body=kwargs["body"],
                version=kwargs["version"],
            )
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except Exception as e:
            logger.error(
                f"Tool {self.name} failed: {e}", extra={"tool": self.name, "error": str(e)}
            )
            return ToolResult.fail(f"Failed to update page {page_id}: {e}", "CONFLUENCE_ERROR")


class SearchContentTool(Tool):
    """Tool to search Confluence content using CQL."""

    def __init__(self, client: ConfluenceToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="confluence_search_content",
            description=(
                "Search for Confluence content using CQL (Confluence Query Language). "
                "Examples: 'space = DEV and type = page', 'text ~ \"API documentation\"'"
            ),
            category="confluence",
            parameters=[
                ToolParameter(
                    name="cql",
                    description="CQL query string",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum number of results to return (default: 25)",
                    type=ParameterType.INTEGER,
                    required=False,
                    default=25,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        logger.debug(f"Executing {self.name}", extra={"tool": self.name, "params": kwargs})
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        cql = kwargs["cql"]
        limit = kwargs.get("limit", 25)
        try:
            results = self.client.search_content(cql, limit)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok({"results": results, "count": len(results)})
        except Exception as e:
            logger.error(
                f"Tool {self.name} failed: {e}", extra={"tool": self.name, "error": str(e)}
            )
            return ToolResult.fail(f"Failed to search content: {e}", "CONFLUENCE_ERROR")


class AddCommentTool(Tool):
    """Tool to add a comment to a Confluence page."""

    def __init__(self, client: ConfluenceToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="confluence_add_comment",
            description="Add a comment to a Confluence page.",
            category="confluence",
            parameters=[
                ToolParameter(
                    name="page_id",
                    description="The Confluence page ID",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="body",
                    description="The comment text (supports Confluence markup)",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        logger.debug(f"Executing {self.name}", extra={"tool": self.name, "params": kwargs})
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        page_id = kwargs["page_id"]
        body = kwargs["body"]
        try:
            result = self.client.add_comment(page_id, body)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except Exception as e:
            logger.error(
                f"Tool {self.name} failed: {e}", extra={"tool": self.name, "error": str(e)}
            )
            return ToolResult.fail(
                f"Failed to add comment to page {page_id}: {e}", "CONFLUENCE_ERROR"
            )


def get_confluence_tools(client: ConfluenceToolClient) -> list[Tool]:
    """Get all Confluence tools configured with the given client.

    Args:
        client: ConfluenceToolClient implementation.

    Returns:
        List of all Confluence tools.
    """
    return [
        GetPageTool(client),
        GetPageByTitleTool(client),
        CreatePageTool(client),
        UpdatePageTool(client),
        SearchContentTool(client),
        AddCommentTool(client),
    ]
