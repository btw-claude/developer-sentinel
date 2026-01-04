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
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinel.executor import AgentClient, AgentClientError, AgentTimeoutError
from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

logger = get_logger(__name__)


def _extract_text_from_stream_json(line: str) -> tuple[str | None, str | None]:
    """Extract text content from a stream-json line.

    Parses stream-json format and extracts human-readable text.

    Args:
        line: A JSON line from stream-json output.

    Returns:
        A tuple of (text_content, final_result):
        - text_content: Extracted text to display, or None
        - final_result: The complete result if this is a result message, or None
    """
    line = line.strip()
    if not line:
        return None, None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None, None

    msg_type = data.get("type")

    # Extract text from streaming deltas (real-time output)
    if msg_type == "stream_event":
        event = data.get("event", {})
        if event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                return delta.get("text"), None

    # Extract text from assistant messages
    if msg_type == "assistant":
        message = data.get("message", {})
        content = message.get("content", [])
        texts = []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        if texts:
            return "".join(texts), None

    # Capture final result
    if msg_type == "result":
        return None, data.get("result", "")

    return None, None


def _stream_json_to_text_writer(
    stream: Any,
    output_file: Any,
    result_holder: list[str],
) -> None:
    """Read stream-json from a stream and write extracted text to a file.

    This function runs in a separate thread. It parses JSON and extracts
    human-readable text, writing it to the output file with OS buffering
    (no explicit flush on every write).

    Args:
        stream: The stdout stream to read from.
        output_file: An open file handle to write text to.
        result_holder: List to store the final result.
    """
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break

            text_content, final_result = _extract_text_from_stream_json(line)

            if text_content:
                output_file.write(text_content)
                output_file.flush()

            if final_result is not None:
                result_holder.append(final_result)

    except Exception:
        pass  # Stream closed or error
    finally:
        with contextlib.suppress(Exception):
            stream.close()


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
) -> None:
    """Read lines from a stream into a list.

    This function runs in a separate thread to avoid blocking.

    Args:
        stream: The stream to read from (stdout or stderr).
        output_lines: List to accumulate output lines.
    """
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            output_lines.append(line)
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
    stdout_file: Path | None = None,
) -> str:
    """Run a prompt through Claude Code CLI.

    Uses Popen with polling to allow for graceful shutdown on Ctrl-C.

    Args:
        prompt: The prompt to send to Claude.
        timeout_seconds: Optional timeout in seconds.
        allowed_tools: Optional list of MCP tool names to pre-authorize.
        cwd: Optional working directory for the subprocess.
        model: Optional model identifier (e.g., "claude-opus-4-5-20251101").
        stdout_file: Optional path to file for streaming text output. When provided,
            stream-json is parsed and human-readable text is written to this file
            in real-time. Use `tail -f` on the file for real-time viewing.

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
        "--verbose",
        "--dangerously-skip-permissions",  # Bypass permission prompts that block output
    ]

    # Add model if specified
    if model:
        cmd.extend(["--model", model])

    # Add allowed tools if specified
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])

    cmd.extend(["-p", prompt])

    logger.debug(f"Running claude command with prompt length {len(prompt)}, cwd={cwd}")

    # Use file output mode if stdout_file is provided (streaming text to file)
    if stdout_file is not None:
        return _run_claude_with_file_output(
            cmd=cmd,
            stdout_file=stdout_file,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
        )

    # Otherwise capture output directly via pipe
    return _run_claude_with_pipe(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
    )


def _run_claude_with_file_output(
    cmd: list[str],
    stdout_file: Path,
    timeout_seconds: int | None,
    cwd: str | None,
) -> str:
    """Run Claude with streaming text output to a file.

    Reads stream-json from Claude, extracts human-readable text, and writes
    it to the file in real-time with OS buffering (no explicit flush per write).

    Args:
        cmd: The command to run.
        stdout_file: Path to file for text output (will be appended to).
        timeout_seconds: Optional timeout in seconds.
        cwd: Optional working directory.

    Returns:
        Claude's response text.

    Raises:
        subprocess.TimeoutExpired: If the command times out.
        subprocess.CalledProcessError: If the command fails.
        ClaudeProcessInterruptedError: If shutdown was requested.
    """
    start_time = time.time()
    poll_interval = 0.5

    # Open file for text output (append mode to preserve any header)
    with open(stdout_file, "a", encoding="utf-8") as f_out:
        # Use pipe for stdout so we can parse stream-json and extract text
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=True,
        )

        # Collect stderr in a thread
        stderr_lines: list[str] = []
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stderr, stderr_lines),
            daemon=True,
        )

        # Stream stdout through JSON parser to extract text and write to file
        result_holder: list[str] = []
        stdout_thread = threading.Thread(
            target=_stream_json_to_text_writer,
            args=(process.stdout, f_out, result_holder),
            daemon=True,
        )

        stderr_thread.start()
        stdout_thread.start()

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
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)
                    raise ClaudeProcessInterruptedError(
                        "Claude subprocess interrupted by shutdown request"
                    )

                # Check if process completed
                return_code = process.poll()
                if return_code is not None:
                    stdout_thread.join(timeout=5)
                    stderr_thread.join(timeout=5)
                    stderr = "".join(stderr_lines)

                    if return_code != 0:
                        raise subprocess.CalledProcessError(
                            return_code, cmd, "", stderr
                        )

                    # Return captured result
                    break

                # Check for timeout
                if timeout_seconds is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout_seconds:
                        logger.warning(
                            f"Claude subprocess timed out after {timeout_seconds}s"
                        )
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        stdout_thread.join(timeout=1)
                        stderr_thread.join(timeout=1)
                        raise subprocess.TimeoutExpired(cmd, timeout_seconds)

                time.sleep(poll_interval)

        except ClaudeProcessInterruptedError:
            raise
        except subprocess.TimeoutExpired:
            raise
        except subprocess.CalledProcessError:
            raise
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            raise

    # Return the result captured from stream-json
    if result_holder:
        return result_holder[-1].strip()
    return ""


def _stream_json_reader(
    stream: Any,
    result_holder: list[str],
) -> None:
    """Read stream-json from a stream and capture the result.

    This function runs in a separate thread. It parses JSON and captures
    the final result.

    Args:
        stream: The stdout stream to read from.
        result_holder: List to store the final result.
    """
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break

            _, final_result = _extract_text_from_stream_json(line)

            if final_result is not None:
                result_holder.append(final_result)

    except Exception:
        pass  # Stream closed or error
    finally:
        with contextlib.suppress(Exception):
            stream.close()


def _run_claude_with_pipe(
    cmd: list[str],
    timeout_seconds: int | None,
    cwd: str | None,
) -> str:
    """Run Claude with output captured via pipe.

    Args:
        cmd: The command to run.
        timeout_seconds: Optional timeout in seconds.
        cwd: Optional working directory.

    Returns:
        Claude's response text.

    Raises:
        subprocess.TimeoutExpired: If the command times out.
        subprocess.CalledProcessError: If the command fails.
        ClaudeProcessInterruptedError: If shutdown was requested.
    """
    start_time = time.time()
    poll_interval = 0.5

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=True,
    )

    # Collect stderr and capture result from stdout
    stderr_lines: list[str] = []
    result_holder: list[str] = []

    # Start reader threads
    stdout_thread = threading.Thread(
        target=_stream_json_reader,
        args=(process.stdout, result_holder),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_reader,
        args=(process.stderr, stderr_lines),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        while True:
            if _shutdown_event.is_set():
                logger.info("Shutdown requested, terminating Claude subprocess")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Claude subprocess did not terminate, killing")
                    process.kill()
                    process.wait()
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                raise ClaudeProcessInterruptedError(
                    "Claude subprocess interrupted by shutdown request"
                )

            return_code = process.poll()
            if return_code is not None:
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)

                stderr = "".join(stderr_lines)

                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd, "", stderr)

                # Return result captured from stream-json
                if result_holder:
                    return result_holder[-1].strip()
                return ""

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
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)
                    raise subprocess.TimeoutExpired(cmd, timeout_seconds)

            time.sleep(poll_interval)

    except ClaudeProcessInterruptedError:
        raise
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
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
        """Run agent with output piped directly to a log file.

        Uses direct file piping for best performance - the OS handles buffering
        instead of Python reading/writing every line.
        """
        assert self.log_base_dir is not None  # Caller ensures this

        # Create log directory and file path
        start_time = datetime.now()
        log_dir = self.log_base_dir / orchestration_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{start_time.strftime('%Y%m%d_%H%M%S')}.log"

        # Write header to log file
        separator = "=" * 80
        header = f"""{separator}
AGENT EXECUTION LOG
{separator}

Issue Key:      {issue_key}
Orchestration:  {orchestration_name}
Start Time:     {start_time.isoformat()}

{separator}
PROMPT
{separator}

{full_prompt}

{separator}
AGENT OUTPUT
{separator}

"""
        log_path.write_text(header, encoding="utf-8")
        logger.info(f"Streaming log started: {log_path}")

        status = "ERROR"
        try:
            response = _run_claude(
                full_prompt,
                timeout_seconds=timeout_seconds,
                allowed_tools=allowed_tools,
                cwd=workdir,
                model=model,
                stdout_file=log_path,
            )
            status = "COMPLETED"
            logger.info(f"Agent execution completed, response length: {len(response)}")
            return response

        except ClaudeProcessInterruptedError:
            status = "INTERRUPTED"
            raise
        except subprocess.TimeoutExpired as e:
            status = "TIMEOUT"
            raise AgentTimeoutError(
                f"Agent execution timed out after {timeout_seconds}s"
            ) from e
        except subprocess.CalledProcessError as e:
            status = "FAILED"
            # Append error info to log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[Error] Process exited with code {e.returncode}\n")
                if e.stderr:
                    f.write(f"[stderr] {e.stderr}\n")
            raise AgentClientError(f"Agent execution failed: {e.stderr}") from e
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[Error] {e}\n")
            raise AgentClientError(f"Agent execution failed: {e}") from e
        finally:
            # Always write footer with final status
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            footer = f"""
{separator}
EXECUTION SUMMARY
{separator}

Status:         {status}
End Time:       {end_time.isoformat()}
Duration:       {duration:.2f}s

{separator}
END OF LOG
{separator}
"""
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(footer)
            logger.info(f"Streaming log completed: {log_path}")
