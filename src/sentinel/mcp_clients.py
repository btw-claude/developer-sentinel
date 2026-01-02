"""MCP-based client implementations using Claude Code CLI.

These clients use subprocess calls to the `claude` CLI to leverage MCP tools
for Jira and other operations, avoiding the need for separate API configuration.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinel.executor import AgentClient, AgentClientError, AgentTimeoutError
from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

logger = get_logger(__name__)


def _run_claude(
    prompt: str,
    timeout_seconds: int | None = None,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
    model: str | None = None,
) -> str:
    """Run a prompt through Claude Code CLI.

    Args:
        prompt: The prompt to send to Claude.
        timeout_seconds: Optional timeout in seconds.
        allowed_tools: Optional list of MCP tool names to pre-authorize.
        cwd: Optional working directory for the subprocess.
        model: Optional model identifier (e.g., "claude-opus-4-5-20251101").

    Returns:
        Claude's response text.

    Raises:
        subprocess.TimeoutExpired: If the command times out.
        subprocess.CalledProcessError: If the command fails.
    """
    cmd = ["claude", "--print", "--output-format", "text"]

    # Add model if specified
    if model:
        cmd.extend(["--model", model])

    # Add allowed tools if specified
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])

    cmd.extend(["-p", prompt])

    logger.debug(f"Running claude command with prompt length {len(prompt)}, cwd={cwd}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=True,
        cwd=cwd,
    )

    return result.stdout.strip()


class JiraMcpClient(JiraClient):
    """Jira client that uses Claude Code's MCP tools for API operations."""

    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues using JQL via MCP.

        Args:
            jql: JQL query string.
            max_results: Maximum number of results to return.

        Returns:
            List of raw issue data from Jira API.

        Raises:
            JiraClientError: If the search fails.
        """
        prompt = f"""Search Jira for issues using this JQL query and return the results as JSON.

JQL: {jql}
Max results: {max_results}

Use the mcp__jira-agent__search_issues tool to search.

Return ONLY a valid JSON array of issues. Each issue should have at minimum:
- key: the issue key (e.g., "PROJ-123")
- fields: object containing summary, description, status, assignee, labels, comment, issuelinks

Do not include any explanation, just the JSON array."""

        try:
            response = _run_claude(
                prompt,
                timeout_seconds=120,
                allowed_tools=["mcp__jira-agent__search_issues"],
            )

            # Try to parse as JSON
            # Claude might include markdown code blocks, so strip those
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

        except subprocess.TimeoutExpired as e:
            raise JiraClientError(f"Jira search timed out: {e}") from e
        except subprocess.CalledProcessError as e:
            raise JiraClientError(f"Jira search failed: {e.stderr}") from e
        except json.JSONDecodeError as e:
            raise JiraClientError(f"Failed to parse Jira response as JSON: {e}") from e
        except Exception as e:
            raise JiraClientError(f"Jira search failed: {e}") from e


class JiraMcpTagClient(JiraTagClient):
    """Jira tag client that uses Claude Code's MCP tools for label operations."""

    def add_label(self, issue_key: str, label: str) -> None:
        """Add a label to a Jira issue via MCP.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
            label: The label to add.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        prompt = f"""Add the label "{label}" to Jira issue {issue_key}.

Use the mcp__jira-agent__update_issue tool to add this label.

If successful, respond with exactly: SUCCESS
If there's an error, respond with: ERROR: <description>"""

        try:
            response = _run_claude(
                prompt,
                timeout_seconds=60,
                allowed_tools=["mcp__jira-agent__update_issue"],
            )

            if "ERROR" in response.upper() and "SUCCESS" not in response.upper():
                raise JiraTagClientError(f"Failed to add label: {response}")

            logger.info(f"Added label '{label}' to {issue_key}")

        except subprocess.TimeoutExpired as e:
            raise JiraTagClientError(f"Add label timed out: {e}") from e
        except subprocess.CalledProcessError as e:
            raise JiraTagClientError(f"Add label failed: {e.stderr}") from e
        except JiraTagClientError:
            raise
        except Exception as e:
            raise JiraTagClientError(f"Add label failed: {e}") from e

    def remove_label(self, issue_key: str, label: str) -> None:
        """Remove a label from a Jira issue via MCP.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
            label: The label to remove.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        prompt = f"""Remove the label "{label}" from Jira issue {issue_key}.

Use the mcp__jira-agent__update_issue tool to remove this label.

If successful, respond with exactly: SUCCESS
If there's an error, respond with: ERROR: <description>"""

        try:
            response = _run_claude(
                prompt,
                timeout_seconds=60,
                allowed_tools=["mcp__jira-agent__update_issue"],
            )

            if "ERROR" in response.upper() and "SUCCESS" not in response.upper():
                raise JiraTagClientError(f"Failed to remove label: {response}")

            logger.info(f"Removed label '{label}' from {issue_key}")

        except subprocess.TimeoutExpired as e:
            raise JiraTagClientError(f"Remove label timed out: {e}") from e
        except subprocess.CalledProcessError as e:
            raise JiraTagClientError(f"Remove label failed: {e.stderr}") from e
        except JiraTagClientError:
            raise
        except Exception as e:
            raise JiraTagClientError(f"Remove label failed: {e}") from e


class ClaudeMcpAgentClient(AgentClient):
    """Agent client that uses Claude Code CLI with MCP tools."""

    def __init__(self, base_workdir: Path | None = None) -> None:
        """Initialize the agent client.

        Args:
            base_workdir: Base directory for creating agent working directories.
                         If None, agents run in the current directory.
        """
        self.base_workdir = base_workdir

    def _create_workdir(self, issue_key: str) -> Path:
        """Create a unique working directory for an agent execution.

        Args:
            issue_key: The Jira issue key (e.g., "DS-123").

        Returns:
            Path to the created directory.
        """
        if self.base_workdir is None:
            raise AgentClientError("base_workdir not configured")

        # Create base directory if it doesn't exist
        self.base_workdir.mkdir(parents=True, exist_ok=True)

        # Create unique directory: {issue_key}_{timestamp}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = self.base_workdir / f"{issue_key}_{timestamp}"
        workdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created agent working directory: {workdir}")
        return workdir

    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
    ) -> str:
        """Run a Claude agent with the given prompt and tools via Claude Code CLI.

        Args:
            prompt: The prompt to send to the agent.
            tools: List of tool names the agent can use (e.g., ["jira", "github"]).
            context: Optional context dict (e.g., GitHub repo info).
            timeout_seconds: Optional timeout in seconds.
            issue_key: Optional issue key for creating a unique working directory.
            model: Optional model identifier. If None, uses the CLI's default model.

        Returns:
            The agent's response text.

        Raises:
            AgentClientError: If the agent execution fails.
            AgentTimeoutError: If the agent execution times out.
        """
        # Create working directory if base_workdir is configured and issue_key provided
        workdir: str | None = None
        if self.base_workdir is not None and issue_key is not None:
            workdir = str(self._create_workdir(issue_key))

        # Build context section if provided
        context_section = ""
        if context:
            context_section = "\n\nContext:\n"
            for key, value in context.items():
                context_section += f"- {key}: {value}\n"

        # Build tools section and allowed tools list
        tools_section = ""
        allowed_tools: list[str] = []
        if tools:
            tools_section = "\n\nAvailable tools:\n"
            for tool in tools:
                if tool == "jira":
                    tools_section += "- Jira: mcp__jira-agent__* tools for issue operations\n"
                    allowed_tools.append("mcp__jira-agent__*")
                elif tool == "confluence":
                    tools_section += "- Confluence: mcp__confluence-wiki__* tools for wiki pages\n"
                    allowed_tools.append("mcp__confluence-wiki__*")
                elif tool == "github":
                    tools_section += "- GitHub: mcp__github__* tools for repository operations\n"
                    allowed_tools.append("mcp__github__*")

        full_prompt = f"{prompt}{context_section}{tools_section}"

        try:
            response = _run_claude(
                full_prompt,
                timeout_seconds=timeout_seconds,
                allowed_tools=allowed_tools if allowed_tools else None,
                cwd=workdir,
                model=model,
            )
            logger.info(f"Agent execution completed, response length: {len(response)}")
            return response

        except subprocess.TimeoutExpired as e:
            raise AgentTimeoutError(f"Agent execution timed out after {timeout_seconds}s") from e
        except subprocess.CalledProcessError as e:
            raise AgentClientError(f"Agent execution failed: {e.stderr}") from e
        except Exception as e:
            raise AgentClientError(f"Agent execution failed: {e}") from e
