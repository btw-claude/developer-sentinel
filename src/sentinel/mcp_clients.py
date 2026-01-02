"""MCP-based client implementations using Claude Code CLI.

These clients use subprocess calls to the `claude` CLI to leverage MCP tools
for Jira and other operations, avoiding the need for separate API configuration.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinel.executor import AgentClient, AgentClientError, AgentTimeoutError
from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

logger = get_logger(__name__)

# Module-level shutdown event for interrupting Claude subprocesses
_shutdown_event = threading.Event()


def request_shutdown() -> None:
    """Request shutdown of any running Claude subprocesses.

    This sets a flag that causes _run_claude to terminate its subprocess
    and raise an exception.
    """
    logger.debug("Shutdown requested for Claude subprocesses")
    _shutdown_event.set()


def reset_shutdown() -> None:
    """Reset the shutdown flag. Used for testing."""
    _shutdown_event.clear()


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_event.is_set()


class ClaudeProcessInterruptedError(Exception):
    """Raised when a Claude subprocess is interrupted by shutdown request."""

    pass


def _run_claude(
    prompt: str,
    timeout_seconds: int | None = None,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
    model: str | None = None,
) -> str:
    """Run a prompt through Claude Code CLI.

    Uses Popen with polling to allow for graceful shutdown on Ctrl-C.

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
        ClaudeProcessInterruptedError: If shutdown was requested during execution.
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

    # Use Popen with start_new_session to isolate from parent's signals
    # This prevents Ctrl-C from being sent directly to the subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=True,
    )

    start_time = time.time()
    poll_interval = 0.5  # Check for shutdown every 500ms

    try:
        while True:
            # Check if shutdown was requested
            if _shutdown_event.is_set():
                logger.info("Shutdown requested, terminating Claude subprocess")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Claude subprocess did not terminate, killing")
                    process.kill()
                    process.wait()
                raise ClaudeProcessInterruptedError(
                    "Claude subprocess interrupted by shutdown request"
                )

            # Check if process completed
            return_code = process.poll()
            if return_code is not None:
                stdout, stderr = process.communicate()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd, stdout, stderr)
                return stdout.strip()

            # Check for timeout
            if timeout_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.warning(f"Claude subprocess timed out after {timeout_seconds}s")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout_seconds)

            # Wait before polling again
            time.sleep(poll_interval)

    except ClaudeProcessInterruptedError:
        raise
    except Exception:
        # Ensure process is cleaned up on any error
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


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

        except ClaudeProcessInterruptedError:
            raise  # Let shutdown interruption propagate
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

        except ClaudeProcessInterruptedError:
            raise  # Let shutdown interruption propagate
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

        except ClaudeProcessInterruptedError:
            raise  # Let shutdown interruption propagate
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

        except ClaudeProcessInterruptedError:
            raise  # Let shutdown interruption propagate
        except subprocess.TimeoutExpired as e:
            raise AgentTimeoutError(f"Agent execution timed out after {timeout_seconds}s") from e
        except subprocess.CalledProcessError as e:
            raise AgentClientError(f"Agent execution failed: {e.stderr}") from e
        except Exception as e:
            raise AgentClientError(f"Agent execution failed: {e}") from e
