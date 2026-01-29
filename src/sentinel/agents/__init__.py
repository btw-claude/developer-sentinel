"""Agent tools package for Developer Sentinel.

This package provides tool definitions that Claude agents can use to interact with
external services like Jira, Confluence, and GitHub.
"""

from sentinel.agents.base import ParameterType, Tool, ToolParameter, ToolResult, ToolSchema
from sentinel.agents.confluence_tools import ConfluenceToolClient, get_confluence_tools
from sentinel.agents.github_tools import GitHubToolClient, get_github_tools
from sentinel.agents.jira_tools import JiraToolClient, get_jira_tools
from sentinel.agents.registry import (
    ToolRegistry,
    get_tool_schemas_for_orchestration,
    get_tools_for_orchestration,
)

__all__ = [
    # Base classes
    "ParameterType",
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolSchema",
    # Client interfaces
    "ConfluenceToolClient",
    "GitHubToolClient",
    "JiraToolClient",
    # Tool factories
    "get_confluence_tools",
    "get_github_tools",
    "get_jira_tools",
    # Registry
    "ToolRegistry",
    "get_tool_schemas_for_orchestration",
    "get_tools_for_orchestration",
]
