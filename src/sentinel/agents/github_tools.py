"""GitHub tool definitions for Claude agents.

This module defines tools that allow agents to interact with GitHub:
- Get issues and pull requests
- List PR files
- Create comments and reviews
- Get file contents
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sentinel.agents.base import ParameterType, Tool, ToolParameter, ToolResult, ToolSchema
from sentinel.logging import get_logger

logger = get_logger(__name__)


class GitHubToolClient(ABC):
    """Abstract interface for GitHub operations.

    This allows tools to work with different implementations:
    - SDK-based client (Claude Agent SDK)
    - Mock client (testing)

    All methods operate on a specific repository configured in the client.
    """

    @abstractmethod
    def get_issue(self, issue_number: int) -> dict[str, Any]:
        """Get issue details."""
        pass

    @abstractmethod
    def get_pull_request(self, pr_number: int) -> dict[str, Any]:
        """Get pull request details."""
        pass

    @abstractmethod
    def list_pr_files(self, pr_number: int) -> list[dict[str, Any]]:
        """List files changed in a pull request."""
        pass

    @abstractmethod
    def get_file_contents(self, path: str, ref: str | None = None) -> dict[str, Any]:
        """Get file contents from the repository."""
        pass

    @abstractmethod
    def create_issue_comment(self, issue_number: int, body: str) -> dict[str, Any]:
        """Create a comment on an issue or PR."""
        pass

    @abstractmethod
    def create_pr_review(
        self,
        pr_number: int,
        body: str,
        event: str,
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a review on a pull request."""
        pass

    @abstractmethod
    def list_pr_comments(self, pr_number: int) -> list[dict[str, Any]]:
        """List review comments on a pull request."""
        pass


class GetIssueTool(Tool):
    """Tool to get GitHub issue details."""

    def __init__(self, client: GitHubToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="github_get_issue",
            description=(
                "Get details about a GitHub issue including title, body, state, "
                "labels, assignees, and comments."
            ),
            category="github",
            parameters=[
                ToolParameter(
                    name="issue_number",
                    description="The issue number",
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

        issue_number = kwargs["issue_number"]
        try:
            result = self.client.get_issue(issue_number)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except (OSError, TimeoutError) as e:
            logger.error(
                f"Tool {self.name} failed due to I/O or timeout: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get issue #{issue_number}: {e}", "GITHUB_ERROR")
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Tool {self.name} failed due to data error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get issue #{issue_number}: {e}", "GITHUB_ERROR")
        except RuntimeError as e:
            logger.error(
                f"Tool {self.name} failed due to runtime error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get issue #{issue_number}: {e}", "GITHUB_ERROR")


class GetPullRequestTool(Tool):
    """Tool to get GitHub pull request details."""

    def __init__(self, client: GitHubToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="github_get_pull_request",
            description=(
                "Get details about a GitHub pull request including title, body, "
                "state, head/base branches, mergeable status, and review status."
            ),
            category="github",
            parameters=[
                ToolParameter(
                    name="pr_number",
                    description="The pull request number",
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

        pr_number = kwargs["pr_number"]
        try:
            result = self.client.get_pull_request(pr_number)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except (OSError, TimeoutError) as e:
            logger.error(
                f"Tool {self.name} failed due to I/O or timeout: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get PR #{pr_number}: {e}", "GITHUB_ERROR")
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Tool {self.name} failed due to data error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get PR #{pr_number}: {e}", "GITHUB_ERROR")
        except RuntimeError as e:
            logger.error(
                f"Tool {self.name} failed due to runtime error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get PR #{pr_number}: {e}", "GITHUB_ERROR")


class ListPRFilesTool(Tool):
    """Tool to list files changed in a pull request."""

    def __init__(self, client: GitHubToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="github_list_pr_files",
            description=(
                "List all files changed in a pull request with their status "
                "(added, modified, removed), additions, deletions, and patch."
            ),
            category="github",
            parameters=[
                ToolParameter(
                    name="pr_number",
                    description="The pull request number",
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

        pr_number = kwargs["pr_number"]
        try:
            files = self.client.list_pr_files(pr_number)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok({"files": files, "count": len(files)})
        except (OSError, TimeoutError) as e:
            logger.error(
                f"Tool {self.name} failed due to I/O or timeout: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to list files for PR #{pr_number}: {e}", "GITHUB_ERROR")
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Tool {self.name} failed due to data error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to list files for PR #{pr_number}: {e}", "GITHUB_ERROR")
        except RuntimeError as e:
            logger.error(
                f"Tool {self.name} failed due to runtime error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to list files for PR #{pr_number}: {e}", "GITHUB_ERROR")


class GetFileContentsTool(Tool):
    """Tool to get file contents from the repository."""

    def __init__(self, client: GitHubToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="github_get_file_contents",
            description=(
                "Get the contents of a file from the repository. "
                "Optionally specify a branch, tag, or commit SHA."
            ),
            category="github",
            parameters=[
                ToolParameter(
                    name="path",
                    description="Path to the file in the repository (e.g., 'src/main.py')",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="ref",
                    description="Branch, tag, or commit SHA (default: default branch)",
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

        path = kwargs["path"]
        ref = kwargs.get("ref")
        try:
            result = self.client.get_file_contents(path, ref)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except (OSError, TimeoutError) as e:
            logger.error(
                f"Tool {self.name} failed due to I/O or timeout: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get file contents for {path}: {e}", "GITHUB_ERROR")
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Tool {self.name} failed due to data error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get file contents for {path}: {e}", "GITHUB_ERROR")
        except RuntimeError as e:
            logger.error(
                f"Tool {self.name} failed due to runtime error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(f"Failed to get file contents for {path}: {e}", "GITHUB_ERROR")


class CreateCommentTool(Tool):
    """Tool to create a comment on an issue or PR."""

    def __init__(self, client: GitHubToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="github_create_comment",
            description=(
                "Create a comment on a GitHub issue or pull request. "
                "This creates a general comment, not a code review comment."
            ),
            category="github",
            parameters=[
                ToolParameter(
                    name="issue_number",
                    description="The issue or PR number",
                    type=ParameterType.INTEGER,
                    required=True,
                ),
                ToolParameter(
                    name="body",
                    description="The comment text (supports GitHub Markdown)",
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

        issue_number = kwargs["issue_number"]
        body = kwargs["body"]
        try:
            result = self.client.create_issue_comment(issue_number, body)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except (OSError, TimeoutError) as e:
            logger.error(
                f"Tool {self.name} failed due to I/O or timeout: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to create comment on #{issue_number}: {e}", "GITHUB_ERROR"
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Tool {self.name} failed due to data error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to create comment on #{issue_number}: {e}", "GITHUB_ERROR"
            )
        except RuntimeError as e:
            logger.error(
                f"Tool {self.name} failed due to runtime error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to create comment on #{issue_number}: {e}", "GITHUB_ERROR"
            )


class CreatePRReviewTool(Tool):
    """Tool to create a review on a pull request."""

    def __init__(self, client: GitHubToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="github_create_pr_review",
            description=(
                "Create a review on a GitHub pull request. You can approve, "
                "request changes, or just comment. Optionally include inline comments."
            ),
            category="github",
            parameters=[
                ToolParameter(
                    name="pr_number",
                    description="The pull request number",
                    type=ParameterType.INTEGER,
                    required=True,
                ),
                ToolParameter(
                    name="body",
                    description="The review summary text",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="event",
                    description="The review action",
                    type=ParameterType.STRING,
                    required=True,
                    enum=["APPROVE", "REQUEST_CHANGES", "COMMENT"],
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

        pr_number = kwargs["pr_number"]
        body = kwargs["body"]
        event = kwargs["event"]
        try:
            result = self.client.create_pr_review(pr_number, body, event)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok(result)
        except (OSError, TimeoutError) as e:
            logger.error(
                f"Tool {self.name} failed due to I/O or timeout: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to create review on PR #{pr_number}: {e}", "GITHUB_ERROR"
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Tool {self.name} failed due to data error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to create review on PR #{pr_number}: {e}", "GITHUB_ERROR"
            )
        except RuntimeError as e:
            logger.error(
                f"Tool {self.name} failed due to runtime error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to create review on PR #{pr_number}: {e}", "GITHUB_ERROR"
            )


class ListPRCommentsTool(Tool):
    """Tool to list review comments on a pull request."""

    def __init__(self, client: GitHubToolClient) -> None:
        self.client = client
        self._schema = ToolSchema(
            name="github_list_pr_comments",
            description=(
                "List all review comments on a pull request. "
                "These are inline comments on specific lines of code."
            ),
            category="github",
            parameters=[
                ToolParameter(
                    name="pr_number",
                    description="The pull request number",
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

        pr_number = kwargs["pr_number"]
        try:
            comments = self.client.list_pr_comments(pr_number)
            logger.info(f"Tool {self.name} succeeded", extra={"tool": self.name})
            return ToolResult.ok({"comments": comments, "count": len(comments)})
        except (OSError, TimeoutError) as e:
            logger.error(
                f"Tool {self.name} failed due to I/O or timeout: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to list comments for PR #{pr_number}: {e}", "GITHUB_ERROR"
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Tool {self.name} failed due to data error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to list comments for PR #{pr_number}: {e}", "GITHUB_ERROR"
            )
        except RuntimeError as e:
            logger.error(
                f"Tool {self.name} failed due to runtime error: {e}",
                extra={"tool": self.name, "error": str(e), "error_type": type(e).__name__},
            )
            return ToolResult.fail(
                f"Failed to list comments for PR #{pr_number}: {e}", "GITHUB_ERROR"
            )


def get_github_tools(client: GitHubToolClient) -> list[Tool]:
    """Get all GitHub tools configured with the given client.

    Args:
        client: GitHubToolClient implementation.

    Returns:
        List of all GitHub tools.
    """
    return [
        GetIssueTool(client),
        GetPullRequestTool(client),
        ListPRFilesTool(client),
        GetFileContentsTool(client),
        CreateCommentTool(client),
        CreatePRReviewTool(client),
        ListPRCommentsTool(client),
    ]
