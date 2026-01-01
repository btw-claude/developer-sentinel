"""Tool registry for managing and enabling agent tools.

This module provides the ToolRegistry class that manages tools and enables
them conditionally based on orchestration configuration.
"""

from __future__ import annotations

from typing import Any

from sentinel.agents.base import Tool
from sentinel.agents.confluence_tools import ConfluenceToolClient, get_confluence_tools
from sentinel.agents.github_tools import GitHubToolClient, get_github_tools
from sentinel.agents.jira_tools import JiraToolClient, get_jira_tools


class ToolRegistry:
    """Registry for managing agent tools.

    The registry holds all available tools and provides methods to filter
    them based on orchestration configuration.
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}
        self._tools_by_category: dict[str, list[Tool]] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        Args:
            tool: The tool to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool

        category = tool.category
        if category not in self._tools_by_category:
            self._tools_by_category[category] = []
        self._tools_by_category[category].append(tool)

    def register_jira_tools(self, client: JiraToolClient) -> None:
        """Register all Jira tools with the given client.

        Args:
            client: JiraToolClient implementation.
        """
        for tool in get_jira_tools(client):
            self.register(tool)

    def register_confluence_tools(self, client: ConfluenceToolClient) -> None:
        """Register all Confluence tools with the given client.

        Args:
            client: ConfluenceToolClient implementation.
        """
        for tool in get_confluence_tools(client):
            self.register(tool)

    def register_github_tools(self, client: GitHubToolClient) -> None:
        """Register all GitHub tools with the given client.

        Args:
            client: GitHubToolClient implementation.
        """
        for tool in get_github_tools(client):
            self.register(tool)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: The tool name.

        Returns:
            The tool if found, None otherwise.
        """
        return self._tools.get(name)

    def get_tools_by_category(self, category: str) -> list[Tool]:
        """Get all tools in a category.

        Args:
            category: The category name (e.g., "jira", "confluence", "github").

        Returns:
            List of tools in the category.
        """
        return self._tools_by_category.get(category, [])

    def get_tools_for_categories(self, categories: list[str]) -> list[Tool]:
        """Get all tools for a list of categories.

        Args:
            categories: List of category names.

        Returns:
            List of all tools in any of the categories.
        """
        tools: list[Tool] = []
        for category in categories:
            tools.extend(self.get_tools_by_category(category))
        return tools

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools.

        Returns:
            List of all tools.
        """
        return list(self._tools.values())

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Get function schemas for all registered tools.

        Returns:
            List of function schemas in Claude's expected format.
        """
        return [tool.to_function_schema() for tool in self._tools.values()]

    def get_schemas_for_categories(self, categories: list[str]) -> list[dict[str, Any]]:
        """Get function schemas for tools in the given categories.

        Args:
            categories: List of category names.

        Returns:
            List of function schemas.
        """
        tools = self.get_tools_for_categories(categories)
        return [tool.to_function_schema() for tool in tools]

    @property
    def categories(self) -> list[str]:
        """Get all registered category names."""
        return list(self._tools_by_category.keys())

    @property
    def tool_count(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)


def get_tools_for_orchestration(
    registry: ToolRegistry,
    enabled_tools: list[str],
) -> list[Tool]:
    """Get tools enabled for an orchestration.

    This function filters the registry's tools based on the orchestration's
    tool configuration. Tool names in the config correspond to categories
    (e.g., "jira", "confluence", "github").

    Args:
        registry: The tool registry.
        enabled_tools: List of tool category names enabled for the orchestration.

    Returns:
        List of tools enabled for the orchestration.
    """
    return registry.get_tools_for_categories(enabled_tools)


def get_tool_schemas_for_orchestration(
    registry: ToolRegistry,
    enabled_tools: list[str],
) -> list[dict[str, Any]]:
    """Get tool schemas enabled for an orchestration.

    This function returns the function calling schemas for tools enabled
    in the orchestration configuration.

    Args:
        registry: The tool registry.
        enabled_tools: List of tool category names enabled for the orchestration.

    Returns:
        List of function schemas in Claude's expected format.
    """
    return registry.get_schemas_for_categories(enabled_tools)
