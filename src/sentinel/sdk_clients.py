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

# Re-exports from agent_clients.claude_sdk.
#
# Public backward-compat API (included in __all__):
#   ClaudeProcessInterruptedError, ClaudeSdkAgentClient,
#   ShutdownController, TimingMetrics
#
# Internal re-exports (not part of public API):
#   _run_query - used by other modules in this package;
#   excluded from __all__ and should not be relied upon by external consumers.
from sentinel.agent_clients.claude_sdk import (
    ClaudeProcessInterruptedError,
    ClaudeSdkAgentClient,
    ShutdownController,
    TimingMetrics,
    _run_query,
)
from sentinel.config import Config
from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

logger = get_logger(__name__)

# NOTE: Update this list when adding new exports to this module.
__all__ = [
    "ClaudeProcessInterruptedError",
    "ClaudeSdkAgentClient",
    "JiraSdkClient",
    "JiraSdkTagClient",
    "ShutdownController",
    "TimingMetrics",
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
            response, _usage = await _run_query(prompt)

            # Parse JSON response, handling markdown code blocks
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            issues = json.loads(json_str.strip())
            if not isinstance(issues, list):
                issues = [issues] if issues else []

            logger.info("JQL search returned %s issues", len(issues))
            return issues

        except ClaudeProcessInterruptedError:
            raise
        except TimeoutError as e:
            raise JiraClientError(f"Jira search timed out: {e}") from e
        except json.JSONDecodeError as e:
            raise JiraClientError(f"Failed to parse Jira response as JSON: {e}") from e
        except (KeyError, ValueError) as e:
            raise JiraClientError(f"Jira search failed due to data error: {e}") from e
        except OSError as e:
            raise JiraClientError(f"Jira search failed due to I/O error: {e}") from e
        except RuntimeError as e:
            raise JiraClientError(f"Jira search failed due to runtime error: {e}") from e


class JiraSdkTagClient(JiraTagClient):
    """Jira tag client that uses Claude Agent SDK."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def _update_label(self, issue_key: str, label: str, action: str) -> None:
        preposition = "to" if action == "add" else "from"
        prompt = f"""{action.capitalize()} the label "{label}" {preposition} Jira issue {issue_key}.

If successful, respond with exactly: SUCCESS
If there's an error, respond with: ERROR: <description>"""

        try:
            response, _usage = await _run_query(prompt)

            if "ERROR" in response.upper() and "SUCCESS" not in response.upper():
                raise JiraTagClientError(f"Failed to {action} label: {response}")

            logger.info(
                "%sed label '%s' %s %s",
                action.capitalize(), label, preposition, issue_key
            )

        except ClaudeProcessInterruptedError:
            raise
        except JiraTagClientError:
            raise
        except TimeoutError as e:
            raise JiraTagClientError(f"{action.capitalize()} label timed out: {e}") from e
        except (KeyError, ValueError) as e:
            raise JiraTagClientError(
                f"{action.capitalize()} label failed due to data error: {e}"
            ) from e
        except OSError as e:
            raise JiraTagClientError(
                f"{action.capitalize()} label failed due to I/O error: {e}"
            ) from e
        except RuntimeError as e:
            raise JiraTagClientError(
                f"{action.capitalize()} label failed due to runtime error: {e}"
            ) from e

    def add_label(self, issue_key: str, label: str) -> None:
        """Add a label to a Jira issue."""
        asyncio.run(self._update_label(issue_key, label, "add"))

    def remove_label(self, issue_key: str, label: str) -> None:
        """Remove a label from a Jira issue."""
        asyncio.run(self._update_label(issue_key, label, "remove"))
