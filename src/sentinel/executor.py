"""Claude Agent SDK executor for running agents on matched issues."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from sentinel.logging import get_logger, log_agent_summary
from sentinel.orchestration import Orchestration, Outcome, RetryConfig
from sentinel.poller import JiraIssue

if TYPE_CHECKING:
    from sentinel.agent_logger import AgentLogger

logger = get_logger(__name__)


class ExecutionStatus(Enum):
    """Status of an agent execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of executing an agent on an issue.

    Attributes:
        status: The execution status (SUCCESS, FAILURE, or ERROR).
        response: The agent's response text.
        attempts: Number of attempts made.
        issue_key: The Jira issue key.
        orchestration_name: Name of the orchestration.
        matched_outcome: Name of the matched outcome (if outcomes are configured).
            None if no outcomes matched or outcomes are not configured.
    """

    status: ExecutionStatus
    response: str
    attempts: int
    issue_key: str
    orchestration_name: str
    matched_outcome: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return True if the execution was successful."""
        return self.status == ExecutionStatus.SUCCESS


class AgentClientError(Exception):
    """Raised when an agent client operation fails."""

    pass


class AgentTimeoutError(AgentClientError):
    """Raised when an agent execution times out."""

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
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
    ) -> str:
        """Run a Claude agent with the given prompt and tools.

        Args:
            prompt: The prompt to send to the agent.
            tools: List of tool names the agent can use.
            context: Optional context dict (e.g., GitHub repo info).
            timeout_seconds: Optional timeout in seconds. If None, no timeout is applied.
            issue_key: Optional issue key for creating a unique working directory.

        Returns:
            The agent's response text.

        Raises:
            AgentClientError: If the agent execution fails.
            AgentTimeoutError: If the agent execution times out.
        """
        pass


class AgentExecutor:
    """Executes Claude agents on Jira issues with retry logic."""

    def __init__(
        self,
        client: AgentClient,
        agent_logger: AgentLogger | None = None,
    ) -> None:
        """Initialize the executor with an agent client.

        Args:
            client: Agent client implementation.
            agent_logger: Optional logger for writing agent execution logs.
        """
        self.client = client
        self.agent_logger = agent_logger

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

        Patterns can be:
        - Simple substring matches (default): "SUCCESS", "error"
        - Regex patterns (with prefix): "regex:error.*\\d+", "regex:^Task completed"

        The "regex:" prefix explicitly indicates a regex pattern. Without
        the prefix, patterns are matched as case-insensitive substrings.

        Args:
            response: The agent's response text.
            patterns: List of patterns to match. Use "regex:" prefix for regex.

        Returns:
            True if any pattern matches.
        """
        response_lower = response.lower()
        for pattern in patterns:
            if pattern.startswith("regex:"):
                # Explicit regex pattern
                regex_pattern = pattern[6:]  # Strip "regex:" prefix
                try:
                    if re.search(regex_pattern, response, re.IGNORECASE):
                        return True
                except re.error:
                    # If invalid regex, fall back to substring match
                    if regex_pattern.lower() in response_lower:
                        return True
            else:
                # Simple substring match (case-insensitive)
                if pattern.lower() in response_lower:
                    return True
        return False

    def _log_execution(
        self,
        issue_key: str,
        orchestration_name: str,
        prompt: str,
        response: str,
        status: ExecutionStatus,
        attempts: int,
        start_time: datetime,
    ) -> None:
        """Log the agent execution if a logger is configured.

        Args:
            issue_key: The Jira issue key.
            orchestration_name: Name of the orchestration.
            prompt: The prompt sent to the agent.
            response: The agent's response.
            status: The execution status.
            attempts: Number of attempts made.
            start_time: When execution started.
        """
        if self.agent_logger is None:
            return

        try:
            self.agent_logger.log_execution(
                issue_key=issue_key,
                orchestration_name=orchestration_name,
                prompt=prompt,
                response=response,
                status=status,
                attempts=attempts,
                start_time=start_time,
                end_time=datetime.now(),
            )
        except Exception as e:
            # Log but don't fail the execution due to logging errors
            logger.warning(f"Failed to write agent execution log: {e}")

    def _determine_outcome(self, response: str, outcomes: list[Outcome]) -> Outcome | None:
        """Determine which outcome matches the response.

        Args:
            response: The agent's response text.
            outcomes: List of outcome configurations.

        Returns:
            The matched Outcome, or None if no outcome patterns match.
        """
        for outcome in outcomes:
            if self._matches_pattern(response, outcome.patterns):
                return outcome
        return None

    def _determine_status(
        self,
        response: str,
        retry_config: RetryConfig,
        outcomes: list[Outcome] | None = None,
    ) -> tuple[ExecutionStatus, str | None]:
        """Determine the execution status and matched outcome from the agent response.

        Args:
            response: The agent's response text.
            retry_config: Retry configuration with success/failure patterns and default_status.
            outcomes: Optional list of outcome configurations.

        Returns:
            Tuple of (ExecutionStatus, matched_outcome_name or None).
        """
        # If outcomes are configured, use outcome-based logic
        if outcomes:
            # First check failure patterns (these trigger retries)
            if self._matches_pattern(response, retry_config.failure_patterns):
                return ExecutionStatus.FAILURE, None

            # Check each outcome's patterns
            matched_outcome = self._determine_outcome(response, outcomes)
            if matched_outcome:
                return ExecutionStatus.SUCCESS, matched_outcome.name

            # No outcome matched - use default_outcome if configured
            if retry_config.default_outcome:
                # Find the default outcome
                for outcome in outcomes:
                    if outcome.name == retry_config.default_outcome:
                        logger.warning(
                            "Response did not match any outcome patterns, "
                            "using default_outcome: %s",
                            retry_config.default_outcome,
                        )
                        return ExecutionStatus.SUCCESS, outcome.name
                # Default outcome name not found in outcomes
                logger.warning(
                    "default_outcome '%s' not found in outcomes, using first outcome",
                    retry_config.default_outcome,
                )

            # Fall back to first outcome
            if outcomes:
                logger.warning(
                    "Response did not match any outcome patterns, using first outcome: %s",
                    outcomes[0].name,
                )
                return ExecutionStatus.SUCCESS, outcomes[0].name

        # Legacy logic when outcomes are not configured
        # Check success patterns first
        if self._matches_pattern(response, retry_config.success_patterns):
            return ExecutionStatus.SUCCESS, None

        # Check failure patterns
        if self._matches_pattern(response, retry_config.failure_patterns):
            return ExecutionStatus.FAILURE, None

        # Use configured default status when no patterns match
        default = retry_config.default_status.upper()
        logger.warning(
            "Response did not match success or failure patterns, using default_status: %s",
            default,
        )
        status = ExecutionStatus.SUCCESS if default == "SUCCESS" else ExecutionStatus.FAILURE
        return status, None

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
        timeout_seconds = orchestration.agent.timeout_seconds

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
        last_matched_outcome: str | None = None
        start_time = datetime.now()

        # Get outcomes if configured
        outcomes = orchestration.outcomes if orchestration.outcomes else None

        for attempt in range(1, max_attempts + 1):
            logger.info(
                f"Executing agent for {issue.key} with orchestration "
                f"'{orchestration.name}' (attempt {attempt}/{max_attempts})",
                extra={
                    "issue_key": issue.key,
                    "orchestration": orchestration.name,
                    "attempt": attempt,
                },
            )

            try:
                response = self.client.run_agent(
                    prompt, tools, context, timeout_seconds, issue_key=issue.key
                )
                last_response = response
                status, matched_outcome = self._determine_status(response, retry_config, outcomes)
                last_status = status
                last_matched_outcome = matched_outcome

                # Log agent summary
                log_agent_summary(
                    logger=logger,
                    issue_key=issue.key,
                    orchestration=orchestration.name,
                    status=status.value.upper(),
                    response=response,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )

                if status == ExecutionStatus.SUCCESS:
                    result = ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        response=response,
                        attempts=attempt,
                        issue_key=issue.key,
                        orchestration_name=orchestration.name,
                        matched_outcome=matched_outcome,
                    )
                    self._log_execution(
                        issue.key,
                        orchestration.name,
                        prompt,
                        response,
                        status,
                        attempt,
                        start_time,
                    )
                    return result

                if status == ExecutionStatus.FAILURE and attempt < max_attempts:
                    logger.warning(
                        f"Agent execution failed for {issue.key} on attempt {attempt}, retrying...",
                        extra={
                            "issue_key": issue.key,
                            "orchestration": orchestration.name,
                            "attempt": attempt,
                        },
                    )

            except AgentTimeoutError as e:
                last_response = f"Timeout: {e}"
                last_status = ExecutionStatus.ERROR
                log_agent_summary(
                    logger=logger,
                    issue_key=issue.key,
                    orchestration=orchestration.name,
                    status="TIMEOUT",
                    response=str(e),
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if attempt < max_attempts:
                    logger.warning(
                        f"Agent timed out for {issue.key} on attempt {attempt}, retrying...",
                        extra={
                            "issue_key": issue.key,
                            "orchestration": orchestration.name,
                            "attempt": attempt,
                            "timeout_seconds": timeout_seconds,
                        },
                    )
            except AgentClientError as e:
                last_response = str(e)
                last_status = ExecutionStatus.ERROR
                log_agent_summary(
                    logger=logger,
                    issue_key=issue.key,
                    orchestration=orchestration.name,
                    status="ERROR",
                    response=str(e),
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if attempt < max_attempts:
                    logger.warning(
                        f"Agent client error for {issue.key} on attempt {attempt}: {e}, "
                        f"retrying...",
                        extra={
                            "issue_key": issue.key,
                            "orchestration": orchestration.name,
                            "attempt": attempt,
                        },
                    )

        # All attempts exhausted
        result = ExecutionResult(
            status=last_status,
            response=last_response,
            attempts=max_attempts,
            issue_key=issue.key,
            orchestration_name=orchestration.name,
            matched_outcome=last_matched_outcome,
        )
        self._log_execution(
            issue.key,
            orchestration.name,
            prompt,
            last_response,
            last_status,
            max_attempts,
            start_time,
        )
        return result
