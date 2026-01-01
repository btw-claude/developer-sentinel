"""Claude Agent SDK executor for running agents on matched issues."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sentinel.orchestration import Orchestration, RetryConfig
from sentinel.poller import JiraIssue

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of an agent execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of executing an agent on an issue."""

    status: ExecutionStatus
    response: str
    attempts: int
    issue_key: str
    orchestration_name: str

    @property
    def succeeded(self) -> bool:
        """Return True if the execution was successful."""
        return self.status == ExecutionStatus.SUCCESS


class AgentClientError(Exception):
    """Raised when an agent client operation fails."""

    pass


class AgentClient(ABC):
    """Abstract interface for Claude Agent SDK operations.

    This allows the executor to work with different implementations:
    - Real Claude Agent SDK client (production)
    - Mock client (testing)
    """

    @abstractmethod
    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Run a Claude agent with the given prompt and tools.

        Args:
            prompt: The prompt to send to the agent.
            tools: List of tool names the agent can use.
            context: Optional context dict (e.g., GitHub repo info).

        Returns:
            The agent's response text.

        Raises:
            AgentClientError: If the agent execution fails.
        """
        pass


class AgentExecutor:
    """Executes Claude agents on Jira issues with retry logic."""

    def __init__(self, client: AgentClient) -> None:
        """Initialize the executor with an agent client.

        Args:
            client: Agent client implementation.
        """
        self.client = client

    def build_prompt(self, issue: JiraIssue, orchestration: Orchestration) -> str:
        """Build the agent prompt from issue and orchestration config.

        Args:
            issue: The Jira issue to process.
            orchestration: The orchestration configuration.

        Returns:
            The complete prompt for the agent.
        """
        # Start with the orchestration's base prompt
        prompt_parts = [orchestration.agent.prompt]

        # Add issue context
        prompt_parts.append("\n\n## Issue Context\n")
        prompt_parts.append(f"**Issue Key:** {issue.key}\n")
        prompt_parts.append(f"**Summary:** {issue.summary}\n")

        if issue.description:
            prompt_parts.append(f"\n**Description:**\n{issue.description}\n")

        if issue.status:
            prompt_parts.append(f"\n**Status:** {issue.status}\n")

        if issue.assignee:
            prompt_parts.append(f"**Assignee:** {issue.assignee}\n")

        if issue.labels:
            prompt_parts.append(f"**Labels:** {', '.join(issue.labels)}\n")

        if issue.comments:
            prompt_parts.append("\n**Recent Comments:**\n")
            for i, comment in enumerate(issue.comments[-3:], 1):  # Last 3 comments
                if len(comment) > 200:
                    prompt_parts.append(f"{i}. {comment[:200]}...\n")
                else:
                    prompt_parts.append(f"{i}. {comment}\n")

        if issue.links:
            prompt_parts.append(f"\n**Linked Issues:** {', '.join(issue.links)}\n")

        # Add GitHub context if available
        github = orchestration.agent.github
        if github and github.org and github.repo:
            prompt_parts.append("\n## GitHub Context\n")
            prompt_parts.append(f"**Repository:** {github.org}/{github.repo}\n")
            if github.host != "github.com":
                prompt_parts.append(f"**Host:** {github.host}\n")

        # Add success/failure pattern instructions
        prompt_parts.append("\n## Response Format\n")
        prompt_parts.append(
            "When you complete the task, include 'SUCCESS' in your response. "
            "If you encounter an error or cannot complete the task, include 'FAILURE' "
            "followed by a brief explanation.\n"
        )

        return "".join(prompt_parts)

    def _matches_pattern(self, response: str, patterns: list[str]) -> bool:
        """Check if the response matches any of the patterns.

        Args:
            response: The agent's response text.
            patterns: List of patterns to match (case-insensitive).

        Returns:
            True if any pattern matches.
        """
        response_lower = response.lower()
        for pattern in patterns:
            # Support simple patterns and basic regex
            if pattern.startswith("^") or pattern.endswith("$") or "*" in pattern:
                # Treat as regex
                try:
                    if re.search(pattern, response, re.IGNORECASE):
                        return True
                except re.error:
                    # If invalid regex, fall back to substring match
                    if pattern.lower() in response_lower:
                        return True
            else:
                # Simple substring match
                if pattern.lower() in response_lower:
                    return True
        return False

    def _determine_status(
        self, response: str, retry_config: RetryConfig
    ) -> ExecutionStatus:
        """Determine the execution status from the agent response.

        Args:
            response: The agent's response text.
            retry_config: Retry configuration with success/failure patterns.

        Returns:
            The determined execution status.
        """
        # Check success patterns first
        if self._matches_pattern(response, retry_config.success_patterns):
            return ExecutionStatus.SUCCESS

        # Check failure patterns
        if self._matches_pattern(response, retry_config.failure_patterns):
            return ExecutionStatus.FAILURE

        # Default to success if no patterns matched
        # (assumes agent completed without explicit failure)
        logger.warning(
            "Response did not match success or failure patterns, defaulting to SUCCESS"
        )
        return ExecutionStatus.SUCCESS

    def execute(
        self,
        issue: JiraIssue,
        orchestration: Orchestration,
    ) -> ExecutionResult:
        """Execute an agent on an issue with retry logic.

        Args:
            issue: The Jira issue to process.
            orchestration: The orchestration configuration.

        Returns:
            ExecutionResult with status, response, and attempt count.
        """
        retry_config = orchestration.retry
        max_attempts = retry_config.max_attempts
        tools = orchestration.agent.tools

        # Build context for the agent
        context: dict[str, Any] = {}
        github = orchestration.agent.github
        if github:
            context["github"] = {
                "host": github.host,
                "org": github.org,
                "repo": github.repo,
            }

        prompt = self.build_prompt(issue, orchestration)
        last_response = ""
        last_status = ExecutionStatus.ERROR

        for attempt in range(1, max_attempts + 1):
            logger.info(
                f"Executing agent for {issue.key} with orchestration "
                f"'{orchestration.name}' (attempt {attempt}/{max_attempts})"
            )

            try:
                response = self.client.run_agent(prompt, tools, context)
                last_response = response
                status = self._determine_status(response, retry_config)
                last_status = status

                if status == ExecutionStatus.SUCCESS:
                    logger.info(
                        f"Agent execution succeeded for {issue.key} on attempt {attempt}"
                    )
                    return ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        response=response,
                        attempts=attempt,
                        issue_key=issue.key,
                        orchestration_name=orchestration.name,
                    )

                if status == ExecutionStatus.FAILURE:
                    if attempt < max_attempts:
                        logger.warning(
                            f"Agent execution failed for {issue.key} on attempt {attempt}, "
                            f"retrying..."
                        )
                    else:
                        logger.error(
                            f"Agent execution failed for {issue.key} after {attempt} attempts"
                        )

            except AgentClientError as e:
                last_response = str(e)
                last_status = ExecutionStatus.ERROR
                if attempt < max_attempts:
                    logger.warning(
                        f"Agent client error for {issue.key} on attempt {attempt}: {e}, "
                        f"retrying..."
                    )
                else:
                    logger.error(
                        f"Agent client error for {issue.key} after {attempt} attempts: {e}"
                    )

        # All attempts exhausted
        return ExecutionResult(
            status=last_status,
            response=last_response,
            attempts=max_attempts,
            issue_key=issue.key,
            orchestration_name=orchestration.name,
        )
