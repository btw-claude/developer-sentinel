"""Jira tool definitions for Claude agents.

This module defines tools that allow agents to interact with Jira:
- Get issue details
- Update issues
- Add comments
- Transition issues
- Manage labels
- Search issues with JQL
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


class JiraToolClient(ABC):
    """Abstract interface for Jira operations.

    This allows tools to work with different implementations:
    - MCP-based client (production)
    - Mock client (testing)
    """

    @abstractmethod
    def get_issue(self, issue_key: str) -> dict[str, Any]:
        """Get issue details."""
        pass

    @abstractmethod
    def update_issue(
        self,
        issue_key: str,
        summary: str | None = None,
        description: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        """Update issue fields."""
        pass

    @abstractmethod
    def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        """Add a comment to an issue."""
        pass

    @abstractmethod
    def transition_issue(self, issue_key: str, transition_id: str) -> dict[str, Any]:
        """Transition an issue to a new status."""
        pass

    @abstractmethod
    def add_label(self, issue_key: str, label: str) -> dict[str, Any]:
        """Add a label to an issue."""
        pass

    @abstractmethod
    def remove_label(self, issue_key: str, label: str) -> dict[str, Any]:
        """Remove a label from an issue."""
        pass

    @abstractmethod
    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues using JQL."""
        pass

    @abstractmethod
    def get_transitions(self, issue_key: str) -> list[dict[str, Any]]:
        """Get available transitions for an issue."""
        pass


class GetIssueTool(Tool):
    """Tool to get Jira issue details."""

    def __init__(self, client: JiraToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="jira_get_issue",
            description=(
                "Get detailed information about a Jira issue including summary, "
                "description, status, assignee, labels, and comments."
            ),
            category="jira",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="The Jira issue key (e.g., 'PROJ-123')",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        issue_key = kwargs["issue_key"]
        try:
            result = self.client.get_issue(issue_key)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(f"Failed to get issue {issue_key}: {e}", "JIRA_ERROR")


class UpdateIssueTool(Tool):
    """Tool to update Jira issue fields."""

    def __init__(self, client: JiraToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="jira_update_issue",
            description=(
                "Update fields on a Jira issue. You can update the summary, "
                "description, and/or priority. Only provide fields you want to change."
            ),
            category="jira",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="The Jira issue key (e.g., 'PROJ-123')",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="summary",
                    description="New summary/title for the issue",
                    type=ParameterType.STRING,
                    required=False,
                ),
                ToolParameter(
                    name="description",
                    description="New description for the issue",
                    type=ParameterType.STRING,
                    required=False,
                ),
                ToolParameter(
                    name="priority",
                    description="New priority (e.g., 'High', 'Medium', 'Low')",
                    type=ParameterType.STRING,
                    required=False,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        issue_key = kwargs["issue_key"]
        try:
            result = self.client.update_issue(
                issue_key=issue_key,
                summary=kwargs.get("summary"),
                description=kwargs.get("description"),
                priority=kwargs.get("priority"),
            )
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(f"Failed to update issue {issue_key}: {e}", "JIRA_ERROR")


class AddCommentTool(Tool):
    """Tool to add a comment to a Jira issue."""

    def __init__(self, client: JiraToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="jira_add_comment",
            description="Add a comment to a Jira issue.",
            category="jira",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="The Jira issue key (e.g., 'PROJ-123')",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="body",
                    description="The comment text (supports Jira markdown)",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        issue_key = kwargs["issue_key"]
        body = kwargs["body"]
        try:
            result = self.client.add_comment(issue_key, body)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(f"Failed to add comment to {issue_key}: {e}", "JIRA_ERROR")


class TransitionIssueTool(Tool):
    """Tool to transition a Jira issue to a new status."""

    def __init__(self, client: JiraToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="jira_transition_issue",
            description=(
                "Transition a Jira issue to a new status. Use jira_get_transitions "
                "first to get available transition IDs."
            ),
            category="jira",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="The Jira issue key (e.g., 'PROJ-123')",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="transition_id",
                    description="The transition ID (get from jira_get_transitions)",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        issue_key = kwargs["issue_key"]
        transition_id = kwargs["transition_id"]
        try:
            result = self.client.transition_issue(issue_key, transition_id)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(f"Failed to transition issue {issue_key}: {e}", "JIRA_ERROR")


class GetTransitionsTool(Tool):
    """Tool to get available transitions for a Jira issue."""

    def __init__(self, client: JiraToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="jira_get_transitions",
            description=(
                "Get available status transitions for a Jira issue. "
                "Use this before transitioning to find valid transition IDs."
            ),
            category="jira",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="The Jira issue key (e.g., 'PROJ-123')",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        issue_key = kwargs["issue_key"]
        try:
            result = self.client.get_transitions(issue_key)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(f"Failed to get transitions for {issue_key}: {e}", "JIRA_ERROR")


class AddLabelTool(Tool):
    """Tool to add a label to a Jira issue."""

    def __init__(self, client: JiraToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="jira_add_label",
            description="Add a label to a Jira issue.",
            category="jira",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="The Jira issue key (e.g., 'PROJ-123')",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="label",
                    description="The label to add (no spaces allowed)",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        issue_key = kwargs["issue_key"]
        label = kwargs["label"]
        try:
            result = self.client.add_label(issue_key, label)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(f"Failed to add label to {issue_key}: {e}", "JIRA_ERROR")


class RemoveLabelTool(Tool):
    """Tool to remove a label from a Jira issue."""

    def __init__(self, client: JiraToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="jira_remove_label",
            description="Remove a label from a Jira issue.",
            category="jira",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="The Jira issue key (e.g., 'PROJ-123')",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="label",
                    description="The label to remove",
                    type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        issue_key = kwargs["issue_key"]
        label = kwargs["label"]
        try:
            result = self.client.remove_label(issue_key, label)
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(f"Failed to remove label from {issue_key}: {e}", "JIRA_ERROR")


class SearchIssuesTool(Tool):
    """Tool to search for Jira issues using JQL."""

    def __init__(self, client: JiraToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="jira_search_issues",
            description=(
                "Search for Jira issues using JQL (Jira Query Language). "
                "Returns a list of matching issues with key, summary, and status."
            ),
            category="jira",
            parameters=[
                ToolParameter(
                    name="jql",
                    description=("JQL query string (e.g., 'project = PROJ AND status = Open')"),
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    description="Maximum number of results to return (default: 50)",
                    type=ParameterType.INTEGER,
                    required=False,
                    default=50,
                ),
            ],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        validation_error = self.validate_params(**kwargs)
        if validation_error:
            return validation_error

        jql = kwargs["jql"]
        max_results = kwargs.get("max_results", 50)
        try:
            results = self.client.search_issues(jql, max_results)
            return ToolResult.ok({"issues": results, "count": len(results)})
        except Exception as e:
            return ToolResult.fail(f"Failed to search issues: {e}", "JIRA_ERROR")


def get_jira_tools(client: JiraToolClient) -> list[Tool]:
    """Get all Jira tools configured with the given client.

    Args:
        client: JiraToolClient implementation.

    Returns:
        List of all Jira tools.
    """
    return [
        GetIssueTool(client),
        UpdateIssueTool(client),
        AddCommentTool(client),
        TransitionIssueTool(client),
        GetTransitionsTool(client),
        AddLabelTool(client),
        RemoveLabelTool(client),
        SearchIssuesTool(client),
    ]
