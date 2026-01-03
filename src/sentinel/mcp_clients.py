"""MCP-based client implementations using Claude Code CLI.

These clients use subprocess calls to the `claude` CLI to leverage MCP tools
for Jira and other operations, avoiding the need for separate API configuration.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinel.agent_logger import StreamingLogWriter
from sentinel.executor import AgentClient, AgentClientError, AgentTimeoutError
from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

logger = get_logger(__name__)

# Type alias for output callback function
OutputCallback = Callable[[str], None]


def _parse_stream_json_line(line: str) -> tuple[str | None, str | None]:
    """Parse a line of stream-json output from Claude CLI.

    Args:
        line: A JSON line from the stream-json output.

    Returns:
        A tuple of (text_delta, final_result):
        - text_delta: Extracted text if this is a content_block_delta event, else None
        - final_result: The complete result if this is a result message, else None
    """
    if not line.strip():
        return None, None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None, None

    msg_type = data.get("type")

    # Handle streaming text deltas
    if msg_type == "stream_event":
        event = data.get("event", {})
        if event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                return delta.get("text"), None

    # Handle final result
    if msg_type == "result":
        return None, data.get("result", "")

    return None, None


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


def _stream_reader(
    stream: Any,
    output_lines: list[str],
    callback: OutputCallback | None,
    prefix: str = "",
    parse_json: bool = False,
    result_holder: list[str] | None = None,
) -> None:
    """Read lines from a stream and optionally call a callback.

    This function runs in a separate thread to avoid blocking.

    Args:
        stream: The stream to read from (stdout or stderr).
        output_lines: List to accumulate output lines (raw lines or extracted text).
        callback: Optional callback to call with each line/text.
        prefix: Optional prefix to add to lines (e.g., "[stderr] ").
        parse_json: If True, parse stream-json format and extract text deltas.
        result_holder: If provided, stores the final result from stream-json.
    """
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break

            if parse_json:
                # Parse stream-json format and extract text
                text_delta, final_result = _parse_stream_json_line(line)
                if text_delta:
                    output_lines.append(text_delta)
                    if callback:
                        callback(f"{prefix}{text_delta}" if prefix else text_delta)
                if final_result is not None and result_holder is not None:
                    result_holder.append(final_result)
            else:
                # Raw line mode (for stderr)
                output_lines.append(line)
                if callback:
                    callback(f"{prefix}{line}" if prefix else line)
    except Exception:
        pass  # Stream closed or error, just exit
    finally:
        with contextlib.suppress(Exception):
            stream.close()


def _run_claude(
    prompt: str,
    timeout_seconds: int | None = None,
    allowed_tools: list[str] | None = None,
    cwd: str | None = None,
    model: str | None = None,
    output_callback: OutputCallback | None = None,
) -> str:
    """Run a prompt through Claude Code CLI.

    Uses Popen with polling to allow for graceful shutdown on Ctrl-C.
    Optionally streams output to a callback function in real-time.

    Args:
        prompt: The prompt to send to Claude.
        timeout_seconds: Optional timeout in seconds.
        allowed_tools: Optional list of MCP tool names to pre-authorize.
        cwd: Optional working directory for the subprocess.
        model: Optional model identifier (e.g., "claude-opus-4-5-20251101").
        output_callback: Optional callback function called with each line of output.
            Use this to stream output to a log file in real-time.

    Returns:
        Claude's response text.

    Raises:
        subprocess.TimeoutExpired: If the command times out.
        subprocess.CalledProcessError: If the command fails.
        ClaudeProcessInterruptedError: If shutdown was requested during execution.
    """
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
    ]

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

    # Collect output from both streams
    stdout_lines: list[str] = []  # Will contain extracted text deltas
    stderr_lines: list[str] = []
    result_holder: list[str] = []  # Will contain the final result

    # Start reader threads for streaming output
    # stdout uses JSON parsing to extract text deltas and final result
    stdout_thread = threading.Thread(
        target=_stream_reader,
        args=(process.stdout, stdout_lines, output_callback, ""),
        kwargs={"parse_json": True, "result_holder": result_holder},
        daemon=True,
    )
    # stderr is read as raw lines
    stderr_thread = threading.Thread(
        target=_stream_reader,
        args=(process.stderr, stderr_lines, output_callback, "[stderr] "),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

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
                # Wait for reader threads to finish
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                raise ClaudeProcessInterruptedError(
                    "Claude subprocess interrupted by shutdown request"
                )

            # Check if process completed
            return_code = process.poll()
            if return_code is not None:
                # Wait for reader threads to finish collecting output
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)

                stderr = "".join(stderr_lines)

                if return_code != 0:
                    # For errors, include streamed text in output
                    stdout = "".join(stdout_lines)
                    raise subprocess.CalledProcessError(return_code, cmd, stdout, stderr)

                # Return the final result from stream-json, or fall back to joined text
                if result_holder:
                    return result_holder[-1].strip()
                return "".join(stdout_lines).strip()

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
                    # Wait for reader threads to finish
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)
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
        # Wait for reader threads
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
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

    def __init__(
        self,
        base_workdir: Path | None = None,
        log_base_dir: Path | None = None,
    ) -> None:
        """Initialize the agent client.

        Args:
            base_workdir: Base directory for creating agent working directories.
                         If None, agents run in the current directory.
            log_base_dir: Base directory for streaming log files.
                         If None, no streaming logs are created.
        """
        self.base_workdir = base_workdir
        self.log_base_dir = log_base_dir

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
        orchestration_name: str | None = None,
    ) -> str:
        """Run a Claude agent with the given prompt and tools via Claude Code CLI.

        Args:
            prompt: The prompt to send to the agent.
            tools: List of tool names the agent can use (e.g., ["jira", "github"]).
            context: Optional context dict (e.g., GitHub repo info).
            timeout_seconds: Optional timeout in seconds.
            issue_key: Optional issue key for creating a unique working directory.
            model: Optional model identifier. If None, uses the CLI's default model.
            orchestration_name: Optional orchestration name for streaming log files.

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

        # Determine if we should use streaming logs
        use_streaming = (
            self.log_base_dir is not None
            and issue_key is not None
            and orchestration_name is not None
        )

        if use_streaming:
            return self._run_with_streaming_log(
                full_prompt=full_prompt,
                timeout_seconds=timeout_seconds,
                allowed_tools=allowed_tools if allowed_tools else None,
                workdir=workdir,
                model=model,
                issue_key=issue_key,  # type: ignore[arg-type]
                orchestration_name=orchestration_name,  # type: ignore[arg-type]
            )
        else:
            return self._run_without_streaming(
                full_prompt=full_prompt,
                timeout_seconds=timeout_seconds,
                allowed_tools=allowed_tools if allowed_tools else None,
                workdir=workdir,
                model=model,
            )

    def _run_without_streaming(
        self,
        full_prompt: str,
        timeout_seconds: int | None,
        allowed_tools: list[str] | None,
        workdir: str | None,
        model: str | None,
    ) -> str:
        """Run agent without streaming logs."""
        try:
            response = _run_claude(
                full_prompt,
                timeout_seconds=timeout_seconds,
                allowed_tools=allowed_tools,
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

    def _run_with_streaming_log(
        self,
        full_prompt: str,
        timeout_seconds: int | None,
        allowed_tools: list[str] | None,
        workdir: str | None,
        model: str | None,
        issue_key: str,
        orchestration_name: str,
    ) -> str:
        """Run agent with streaming logs to a file."""
        assert self.log_base_dir is not None  # Caller ensures this

        with StreamingLogWriter(
            base_dir=self.log_base_dir,
            orchestration_name=orchestration_name,
            issue_key=issue_key,
            prompt=full_prompt,
        ) as log_writer:
            status = "ERROR"
            try:
                response = _run_claude(
                    full_prompt,
                    timeout_seconds=timeout_seconds,
                    allowed_tools=allowed_tools,
                    cwd=workdir,
                    model=model,
                    output_callback=log_writer.write,
                )
                status = "COMPLETED"
                logger.info(f"Agent execution completed, response length: {len(response)}")
                log_writer.finalize(status)
                return response

            except ClaudeProcessInterruptedError:
                status = "INTERRUPTED"
                log_writer.finalize(status)
                raise  # Let shutdown interruption propagate
            except subprocess.TimeoutExpired as e:
                status = "TIMEOUT"
                log_writer.finalize(status)
                raise AgentTimeoutError(
                    f"Agent execution timed out after {timeout_seconds}s"
                ) from e
            except subprocess.CalledProcessError as e:
                status = "FAILED"
                log_writer.write(f"\n[Error] Process exited with code {e.returncode}\n")
                if e.stderr:
                    log_writer.write(f"[stderr] {e.stderr}\n")
                log_writer.finalize(status)
                raise AgentClientError(f"Agent execution failed: {e.stderr}") from e
            except Exception as e:
                log_writer.write(f"\n[Error] {e}\n")
                log_writer.finalize(status)
                raise AgentClientError(f"Agent execution failed: {e}") from e
