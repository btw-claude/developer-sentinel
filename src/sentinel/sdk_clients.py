"""Claude Agent SDK client implementations.

These clients use the Claude Agent SDK for Jira and agent operations.

Note: ClaudeSdkAgentClient, TimingMetrics, and related functions have been
moved to sentinel.agent_clients.claude_sdk. They are re-exported here for
backward compatibility.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from sentinel.config import Config
from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

# Backward-compatible re-exports from agent_clients.claude_sdk
# Note: _run_query is intentionally not re-exported as it's a private function
# that consumers should import directly from agent_clients.claude_sdk if needed.
from sentinel.agent_clients.claude_sdk import (
    ClaudeProcessInterruptedError,
    ClaudeSdkAgentClient,
    TimingMetrics,
    _run_query,
    get_shutdown_event,
    is_shutdown_requested,
    request_shutdown,
    reset_shutdown,
)

logger = get_logger(__name__)

# Re-export for backward compatibility
# Note: _run_query is excluded as it's a private function (underscore prefix)
__all__ = [
    "ClaudeProcessInterruptedError",
    "ClaudeSdkAgentClient",
    "JiraSdkClient",
    "JiraSdkTagClient",
    "TimingMetrics",
    "get_shutdown_event",
    "is_shutdown_requested",
    "request_shutdown",
    "reset_shutdown",
]


class JiraSdkClient(JiraClient):
    """Jira client that uses Claude Agent SDK."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues using JQL via Claude Agent SDK."""
        return asyncio.run(self._search_issues_async(jql, max_results))

    async def _search_issues_async(self, jql: str, max_results: int) -> list[dict[str, Any]]:
        prompt = f"""Search Jira for issues using this JQL query and return the results as JSON.

JQL: {jql}
Max results: {max_results}

Return ONLY a valid JSON array of issues. Each issue should have at minimum:
- key: the issue key (e.g., "PROJ-123")
- fields: object containing summary, description, status, assignee, labels, comment, issuelinks

Do not include any explanation, just the JSON array."""

        try:
            response = await _run_query(prompt)

            # Parse JSON response, handling markdown code blocks
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            issues = json.loads(json_str.strip())
            if not isinstance(issues, list):
                issues = [issues] if issues else []

            logger.info(f"JQL search returned {len(issues)} issues")
            return issues

        except ClaudeProcessInterruptedError:
            raise
        except asyncio.TimeoutError as e:
            raise JiraClientError(f"Jira search timed out: {e}") from e
        except json.JSONDecodeError as e:
            raise JiraClientError(f"Failed to parse Jira response as JSON: {e}") from e
        except Exception as e:
            raise JiraClientError(f"Jira search failed: {e}") from e


class JiraSdkTagClient(JiraTagClient):
    """Jira tag client that uses Claude Agent SDK."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def _update_label(self, issue_key: str, label: str, action: str) -> None:
        prompt = f"""{action.capitalize()} the label "{label}" {"to" if action == "add" else "from"} Jira issue {issue_key}.

If successful, respond with exactly: SUCCESS
If there's an error, respond with: ERROR: <description>"""

        try:
            response = await _run_query(prompt)

            if "ERROR" in response.upper() and "SUCCESS" not in response.upper():
                raise JiraTagClientError(f"Failed to {action} label: {response}")

            logger.info(f"{action.capitalize()}ed label '{label}' {'to' if action == 'add' else 'from'} {issue_key}")

        except ClaudeProcessInterruptedError:
            raise
        except JiraTagClientError:
            raise
        except asyncio.TimeoutError as e:
            raise JiraTagClientError(f"{action.capitalize()} label timed out: {e}") from e
        except Exception as e:
            raise JiraTagClientError(f"{action.capitalize()} label failed: {e}") from e

    def add_label(self, issue_key: str, label: str) -> None:
        """Add a label to a Jira issue."""
        asyncio.run(self._update_label(issue_key, label, "add"))

    def remove_label(self, issue_key: str, label: str) -> None:
        """Remove a label from a Jira issue."""
        asyncio.run(self._update_label(issue_key, label, "remove"))
