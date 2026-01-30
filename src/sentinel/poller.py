"""Jira poller module for querying issues matching orchestration triggers."""

from __future__ import annotations

import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from sentinel.logging import get_logger
from sentinel.orchestration import TriggerConfig

logger = get_logger(__name__)


class JqlSanitizationError(ValueError):
    """Raised when a value cannot be safely used in JQL."""

    pass


def sanitize_jql_string(value: str, field_name: str = "value") -> str:
    """Sanitize a string value for safe use in JQL queries.

    This function validates and escapes string values to prevent JQL injection attacks.
    Values are checked for dangerous patterns and properly escaped for use in quoted
    JQL string literals.

    Args:
        value: The string value to sanitize.
        field_name: Name of the field being sanitized (for error messages).

    Returns:
        The sanitized string safe for use in JQL quoted literals.

    Raises:
        JqlSanitizationError: If the value contains patterns that cannot be safely escaped.
    """
    if not value:
        raise JqlSanitizationError(f"{field_name} cannot be empty")

    # Check for null bytes which could cause issues
    if "\x00" in value:
        raise JqlSanitizationError(f"{field_name} contains invalid null character")

    # Escape backslashes first (before escaping quotes)
    sanitized = value.replace("\\", "\\\\")

    # Escape double quotes
    sanitized = sanitized.replace('"', '\\"')

    # Check for excessively long values that could be DoS attempts
    max_length = 1000
    if len(sanitized) > max_length:
        raise JqlSanitizationError(
            f"{field_name} exceeds maximum length of {max_length} characters"
        )

    return sanitized


def validate_jql_identifier(value: str, field_name: str = "identifier") -> str:
    """Validate a value for use as a JQL identifier (like project key).

    JQL identifiers like project keys should only contain alphanumeric characters,
    underscores, and hyphens. This provides stricter validation than string sanitization.

    Args:
        value: The identifier value to validate.
        field_name: Name of the field being validated (for error messages).

    Returns:
        The validated identifier (unchanged if valid).

    Raises:
        JqlSanitizationError: If the value contains invalid characters.
    """
    if not value:
        raise JqlSanitizationError(f"{field_name} cannot be empty")

    # Project keys and similar identifiers should be alphanumeric with underscores/hyphens
    # Common format: letters followed by optional numbers, e.g., "TEST", "PROJ-1", "MY_PROJECT"
    if not re.match(r"^[A-Za-z][A-Za-z0-9_-]*$", value):
        raise JqlSanitizationError(
            f"{field_name} contains invalid characters. "
            f"Only alphanumeric characters, underscores, and hyphens are allowed "
            f"(must start with a letter): {value!r}"
        )

    # Check for reasonable length
    max_length = 255
    if len(value) > max_length:
        raise JqlSanitizationError(
            f"{field_name} exceeds maximum length of {max_length} characters"
        )

    return value


def _count_unescaped_chars(value: str, char: str) -> int:
    """Count occurrences of a character that are not backslash-escaped.

    A character is considered escaped if it is preceded by an odd number of
    backslashes. For example, in the string 'test\\"value', the double quote
    is escaped (preceded by one backslash), but in 'test\\\\"value', the
    double quote is not escaped (preceded by two backslashes, which form an
    escaped backslash).

    Args:
        value: The string to search in.
        char: The single character to count (must be exactly one character).

    Returns:
        The count of unescaped occurrences of the character.

    Raises:
        ValueError: If char is not exactly one character.
    """
    if len(char) != 1:
        raise ValueError("char must be exactly one character")
    count = 0
    i = 0
    while i < len(value):
        if value[i] == char:
            # Check if this character is escaped
            num_backslashes = 0
            j = i - 1
            while j >= 0 and value[j] == "\\":
                num_backslashes += 1
                j -= 1
            # Character is escaped only if preceded by odd number of backslashes
            if num_backslashes % 2 == 0:
                count += 1
        i += 1
    return count


def validate_jql_filter(value: str) -> str:
    """Validate a custom JQL filter fragment for safe use.

    This function performs security validation on custom JQL filter fragments to
    prevent potential injection attacks while allowing legitimate JQL syntax.

    The validation checks for:
    - Null bytes which could cause parsing issues
    - Excessively long values that could be DoS attempts
    - Unbalanced parentheses which indicate malformed JQL
    - Unbalanced quotes which could lead to injection

    Note: This is a security-focused validation, not a full JQL syntax validator.
    Valid JQL syntax errors will be caught by the Jira API.

    Args:
        value: The JQL filter fragment to validate.

    Returns:
        The validated JQL filter (unchanged if valid).

    Raises:
        JqlSanitizationError: If the value contains potentially dangerous patterns.

    Security Considerations:
        - jql_filter is expected to be configured by trusted administrators
        - This validation provides defense-in-depth against misuse
        - The Jira API provides additional server-side validation
    """
    if not value:
        return value  # Empty filter is valid (optional field)

    # Check for null bytes which could cause issues
    if "\x00" in value:
        error_msg = "jql_filter contains invalid null character"
        logger.debug("JQL filter validation failed: %s. Filter: %r", error_msg, value)
        raise JqlSanitizationError(error_msg)

    # Check for excessively long values that could be DoS attempts
    max_length = 10000
    if len(value) > max_length:
        error_msg = f"jql_filter exceeds maximum length of {max_length} characters"
        logger.debug(
            "JQL filter validation failed: %s. Filter length: %d", error_msg, len(value)
        )
        raise JqlSanitizationError(error_msg)

    # Check for unbalanced parentheses (basic structure validation)
    paren_count = 0
    for char in value:
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
        if paren_count < 0:
            error_msg = (
                "jql_filter contains unbalanced parentheses (unexpected closing parenthesis)"
            )
            logger.debug("JQL filter validation failed: %s. Filter: %r", error_msg, value)
            raise JqlSanitizationError(error_msg)
    if paren_count != 0:
        error_msg = (
            "jql_filter contains unbalanced parentheses (unclosed opening parenthesis)"
        )
        logger.debug("JQL filter validation failed: %s. Filter: %r", error_msg, value)
        raise JqlSanitizationError(error_msg)

    # Check for unbalanced double quotes (basic string literal validation)
    double_quote_count = _count_unescaped_chars(value, '"')
    if double_quote_count % 2 != 0:
        error_msg = "jql_filter contains unbalanced double quotes (unclosed string literal)"
        logger.debug("JQL filter validation failed: %s. Filter: %r", error_msg, value)
        raise JqlSanitizationError(error_msg)

    # Check for unbalanced single quotes (JQL supports both quote styles)
    single_quote_count = _count_unescaped_chars(value, "'")
    if single_quote_count % 2 != 0:
        error_msg = "jql_filter contains unbalanced single quotes (unclosed string literal)"
        logger.debug("JQL filter validation failed: %s. Filter: %r", error_msg, value)
        raise JqlSanitizationError(error_msg)

    return value


@dataclass
class JiraIssue:
    """Represents a Jira issue with relevant fields for orchestration."""

    key: str
    summary: str
    description: str = ""
    status: str = ""
    assignee: str | None = None
    labels: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    epic_key: str | None = None  # Parent epic key
    parent_key: str | None = None  # Parent issue key (for sub-tasks)

    @classmethod
    def from_api_response(
        cls, data: dict[str, Any], epic_link_field: str = "customfield_10014"
    ) -> JiraIssue:
        """Create a JiraIssue from Jira API response data.

        Args:
            data: Raw issue data from Jira API.
            epic_link_field: The custom field ID for epic links (varies by Jira instance).
                Defaults to "customfield_10014" which is common for classic Jira projects.

        Returns:
            JiraIssue instance.
        """
        fields = data.get("fields", {})

        # Extract description text
        description = ""
        desc_data = fields.get("description")
        if desc_data and isinstance(desc_data, dict):
            # Atlassian Document Format - extract text content
            description = _extract_adf_text(desc_data)
        elif isinstance(desc_data, str):
            description = desc_data

        # Extract assignee
        assignee = None
        assignee_data = fields.get("assignee")
        if assignee_data:
            assignee = assignee_data.get("displayName") or assignee_data.get("name")

        # Extract status
        status = ""
        status_data = fields.get("status")
        if status_data:
            status = status_data.get("name", "")

        # Extract comments
        comments: list[str] = []
        comment_data = fields.get("comment", {})
        if isinstance(comment_data, dict):
            for comment in comment_data.get("comments", []):
                body = comment.get("body")
                if isinstance(body, dict):
                    comments.append(_extract_adf_text(body))
                elif isinstance(body, str):
                    comments.append(body)

        # Extract issue links
        links: list[str] = []
        for link in fields.get("issuelinks", []):
            if "outwardIssue" in link:
                links.append(link["outwardIssue"]["key"])
            if "inwardIssue" in link:
                links.append(link["inwardIssue"]["key"])

        # Extract epic link and parent key, distinguishing by parent issue type
        # For next-gen projects, the parent field can represent either an epic or a story/task
        epic_key = None
        parent_key = None
        parent_data = fields.get("parent")
        if parent_data:
            parent_issue_key = parent_data.get("key")
            parent_type_data = parent_data.get("fields", {}).get("issuetype", {})
            parent_type_name = parent_type_data.get("name", "").lower()

            # Check if parent is an Epic type
            if parent_type_name == "epic":
                epic_key = parent_issue_key
            else:
                # Non-epic parent (Story, Task, etc.) - this is a sub-task relationship
                parent_key = parent_issue_key

        # Fallback to classic epic link field if no epic found from parent
        if not epic_key:
            epic_key = fields.get(epic_link_field)

        return cls(
            key=data.get("key", ""),
            summary=fields.get("summary", ""),
            description=description,
            status=status,
            assignee=assignee,
            labels=fields.get("labels", []),
            comments=comments,
            links=links,
            epic_key=epic_key,
            parent_key=parent_key,
        )


def _extract_adf_text(adf: dict[str, Any]) -> str:
    """Extract plain text from Atlassian Document Format.

    Args:
        adf: Atlassian Document Format data.

    Returns:
        Plain text content.
    """
    texts: list[str] = []

    def extract(node: dict[str, Any]) -> None:
        if node.get("type") == "text":
            texts.append(node.get("text", ""))
        for child in node.get("content", []):
            if isinstance(child, dict):
                extract(child)

    extract(adf)
    return " ".join(texts)


class JiraClientError(Exception):
    """Raised when a Jira API operation fails."""

    pass


class JiraClient(ABC):
    """Abstract interface for Jira operations.

    This allows the poller to work with different implementations:
    - SDK-based client (Claude Agent SDK)
    - REST client (direct HTTP)
    - Mock client (testing)
    """

    @abstractmethod
    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues using JQL.

        Args:
            jql: JQL query string.
            max_results: Maximum number of results to return.

        Returns:
            List of raw issue data from Jira API.

        Raises:
            JiraClientError: If the search fails.
        """
        pass


class JiraPoller:
    """Polls Jira for issues matching orchestration triggers."""

    # Default maximum delay cap (5 minutes) to prevent excessively long waits
    DEFAULT_MAX_DELAY: float = 300.0

    # Default jitter range (0.5-1.5x multiplier) to prevent thundering herd
    DEFAULT_JITTER_MIN: float = 0.5
    DEFAULT_JITTER_MAX: float = 1.5

    def __init__(
        self,
        client: JiraClient,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        epic_link_field: str = "customfield_10014",
        max_delay: float | None = None,
        jitter_min: float | None = None,
        jitter_max: float | None = None,
    ) -> None:
        """Initialize the Jira poller.

        Args:
            client: Jira client implementation.
            max_retries: Maximum number of retry attempts for API calls.
            retry_delay: Base delay between retries (exponential backoff).
            epic_link_field: The custom field ID for epic links (varies by Jira instance).
                Defaults to "customfield_10014" which is common for classic Jira projects.
                Can be configured via JIRA_EPIC_LINK_FIELD environment variable.
            max_delay: Maximum delay in seconds between retries (default: 300.0 / 5 minutes).
                Caps the exponential backoff to prevent excessively long waits.
            jitter_min: Minimum jitter multiplier (default: 0.5).
                Applied to delay to prevent thundering herd problem.
            jitter_max: Maximum jitter multiplier (default: 1.5).
                Applied to delay to prevent thundering herd problem.
        """
        self.client = client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.epic_link_field = epic_link_field
        self.max_delay = max_delay if max_delay is not None else self.DEFAULT_MAX_DELAY
        self.jitter_min = jitter_min if jitter_min is not None else self.DEFAULT_JITTER_MIN
        self.jitter_max = jitter_max if jitter_max is not None else self.DEFAULT_JITTER_MAX

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before the next retry attempt with jitter and max cap.

        Implements exponential backoff with random jitter to prevent the
        thundering herd problem where all failing clients retry at exactly
        the same time.

        Args:
            attempt: Current retry attempt number (0-indexed).

        Returns:
            Delay in seconds before next retry, with jitter applied and
            capped at max_delay.
        """
        # Calculate base exponential backoff: retry_delay * 2^attempt
        base_delay = self.retry_delay * (2**attempt)

        # Cap at maximum delay to prevent excessively long waits
        capped_delay = min(base_delay, self.max_delay)

        # Apply random jitter (0.5-1.5x by default) to prevent thundering herd
        jitter: float = random.uniform(self.jitter_min, self.jitter_max)
        final_delay: float = capped_delay * jitter
        return final_delay

    def build_jql(self, trigger: TriggerConfig) -> str:
        """Build a JQL query from trigger configuration.

        Args:
            trigger: Trigger configuration from orchestration.

        Returns:
            JQL query string.

        Raises:
            JqlSanitizationError: If trigger values contain invalid characters.
        """
        conditions: list[str] = []

        # Project filter - validate as identifier
        if trigger.project:
            validated_project = validate_jql_identifier(trigger.project, "project")
            conditions.append(f'project = "{validated_project}"')

        # Tag/label filter - must have ALL specified tags
        # Labels are sanitized as strings since they can contain more characters
        for tag in trigger.tags:
            sanitized_tag = sanitize_jql_string(tag, "tag")
            conditions.append(f'labels = "{sanitized_tag}"')

        # Custom JQL filter - validated for structural integrity
        # Note: jql_filter is expected to be configured by trusted administrators.
        # Validation provides defense-in-depth against misuse (null bytes, unbalanced
        # parentheses/quotes, excessive length). Full JQL syntax is validated by Jira API.
        if trigger.jql_filter:
            validated_filter = validate_jql_filter(trigger.jql_filter)
            conditions.append(f"({validated_filter})")

        # Default: exclude resolved/closed issues
        if not any("status" in c.lower() for c in conditions):
            conditions.append('status NOT IN ("Done", "Closed", "Resolved")')

        return " AND ".join(conditions) if conditions else ""

    def poll(self, trigger: TriggerConfig, max_results: int = 50) -> list[JiraIssue]:
        """Poll Jira for issues matching the trigger configuration.

        Args:
            trigger: Trigger configuration from orchestration.
            max_results: Maximum number of issues to return.

        Returns:
            List of matching JiraIssue objects.

        Raises:
            JiraClientError: If polling fails after retries.
        """
        jql = self.build_jql(trigger)
        if not jql:
            logger.warning("Empty JQL query generated from trigger config")
            return []

        logger.info(f"Polling Jira with JQL: {jql}")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw_issues = self.client.search_issues(jql, max_results)
                issues = [
                    JiraIssue.from_api_response(issue, self.epic_link_field)
                    for issue in raw_issues
                ]
                logger.info(f"Found {len(issues)} matching issues")

                # Warn if epic_link_field is configured but not found on any issues
                if (
                    issues
                    and self.epic_link_field
                    and all(issue.epic_key is None for issue in issues)
                ):
                    # Check if the field exists on any raw issue
                    field_found = any(
                        self.epic_link_field in raw_issue.get("fields", {})
                        for raw_issue in raw_issues
                    )
                    if not field_found:
                        logger.warning(
                            f"Configured epic_link_field '{self.epic_link_field}' was not "
                            f"found on any of the {len(issues)} polled issues. "
                            f"Verify JIRA_EPIC_LINK_FIELD is set to the correct custom "
                            f"field ID for your Jira instance."
                        )

                return issues
            except JiraClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Jira API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Jira API error after {self.max_retries} attempts: {e}")

        raise JiraClientError(
            f"Failed to poll Jira after {self.max_retries} attempts"
        ) from last_error
