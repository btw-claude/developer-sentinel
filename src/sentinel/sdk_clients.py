"""Claude Agent SDK client implementations.

These clients use the Claude Agent SDK for Jira and agent operations.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from sentinel.config import Config
from sentinel.executor import AgentClient, AgentClientError, AgentRunResult, AgentTimeoutError
from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

logger = get_logger(__name__)

# Module-level shutdown event for interrupting async operations
_shutdown_event: asyncio.Event | None = None
_shutdown_lock = threading.Lock()


def get_shutdown_event() -> asyncio.Event:
    """Get or create the shutdown event for async operations."""
    global _shutdown_event
    with _shutdown_lock:
        if _shutdown_event is None:
            _shutdown_event = asyncio.Event()
        return _shutdown_event


def request_shutdown() -> None:
    """Request shutdown of any running Claude agent operations."""
    global _shutdown_event
    logger.debug("Shutdown requested for Claude agent operations")
    with _shutdown_lock:
        if _shutdown_event is None:
            _shutdown_event = asyncio.Event()
        _shutdown_event.set()


def reset_shutdown() -> None:
    """Reset the shutdown flag. Used for testing."""
    global _shutdown_event
    with _shutdown_lock:
        _shutdown_event = None


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    global _shutdown_event
    with _shutdown_lock:
        return _shutdown_event is not None and _shutdown_event.is_set()


class ClaudeProcessInterruptedError(Exception):
    """Raised when a Claude agent operation is interrupted by shutdown request."""

    pass


async def _run_query(
    prompt: str,
    model: str | None = None,
    cwd: str | None = None,
) -> str:
    """Run a query using the Claude Agent SDK.

    Args:
        prompt: The prompt to send.
        model: Optional model identifier.
        cwd: Optional working directory.

    Returns:
        The response text.

    Raises:
        ClaudeProcessInterruptedError: If shutdown was requested.
    """
    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model=model,
        cwd=cwd,
        setting_sources=["project", "user"],  # Load skills from project and ~/.claude/skills
    )

    shutdown_event = get_shutdown_event()
    response_text = ""

    async for message in query(prompt=prompt, options=options):
        if shutdown_event.is_set():
            raise ClaudeProcessInterruptedError(
                "Claude agent interrupted by shutdown request"
            )
        if hasattr(message, "text"):
            response_text = message.text
        elif hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    response_text = block.text

    return response_text


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


class ClaudeSdkAgentClient(AgentClient):
    """Agent client that uses Claude Agent SDK."""

    def __init__(
        self,
        config: Config,
        base_workdir: Path | None = None,
        log_base_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.base_workdir = base_workdir
        self.log_base_dir = log_base_dir

    def _create_workdir(self, issue_key: str) -> Path:
        if self.base_workdir is None:
            raise AgentClientError("base_workdir not configured")
        self.base_workdir.mkdir(parents=True, exist_ok=True)
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
    ) -> AgentRunResult:
        """Run a Claude agent with the given prompt and tools."""
        workdir = None
        if self.base_workdir is not None and issue_key is not None:
            workdir = self._create_workdir(issue_key)

        # Build full prompt with context section
        full_prompt = prompt
        if context:
            full_prompt += "\n\nContext:\n" + "".join(f"- {k}: {v}\n" for k, v in context.items())

        use_streaming = self.log_base_dir and issue_key and orchestration_name

        if use_streaming:
            response = asyncio.run(self._run_with_log(full_prompt, tools, timeout_seconds, workdir, model, issue_key, orchestration_name))  # type: ignore
        else:
            response = asyncio.run(self._run_simple(full_prompt, tools, timeout_seconds, workdir, model))
        return AgentRunResult(response=response, workdir=workdir)

    async def _run_simple(
        self, prompt: str, tools: list[str], timeout: int | None, workdir: Path | None, model: str | None
    ) -> str:
        try:
            coro = _run_query(prompt, model, str(workdir) if workdir else None)
            response = await asyncio.wait_for(coro, timeout=timeout) if timeout else await coro
            logger.info(f"Agent execution completed, response length: {len(response)}")
            return response
        except ClaudeProcessInterruptedError:
            raise
        except asyncio.TimeoutError:
            raise AgentTimeoutError(f"Agent execution timed out after {timeout}s")
        except Exception as e:
            raise AgentClientError(f"Agent execution failed: {e}") from e

    async def _run_with_log(
        self, prompt: str, tools: list[str], timeout: int | None, workdir: Path | None, model: str | None, issue_key: str, orch_name: str
    ) -> str:
        assert self.log_base_dir is not None
        start_time = datetime.now()
        log_dir = self.log_base_dir / orch_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{start_time.strftime('%Y%m%d_%H%M%S')}.log"

        sep = "=" * 80
        log_path.write_text(f"{sep}\nAGENT EXECUTION LOG\n{sep}\n\nIssue Key:      {issue_key}\nOrchestration:  {orch_name}\nStart Time:     {start_time.isoformat()}\n\n{sep}\nPROMPT\n{sep}\n\n{prompt}\n\n{sep}\nAGENT OUTPUT\n{sep}\n\n", encoding="utf-8")
        logger.info(f"Streaming log started: {log_path}")

        status = "ERROR"
        try:
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model=model,
                cwd=str(workdir) if workdir else None,
                setting_sources=["project", "user"],  # Load skills from project and ~/.claude/skills
            )
            shutdown_event = get_shutdown_event()

            async def run_streaming() -> str:
                response_text = ""
                with open(log_path, "a", encoding="utf-8") as f:
                    async for message in query(prompt=prompt, options=options):
                        if shutdown_event.is_set():
                            raise ClaudeProcessInterruptedError("Interrupted by shutdown")
                        text = ""
                        if hasattr(message, "text"):
                            text = response_text = message.text
                        elif hasattr(message, "content"):
                            for block in message.content:
                                if hasattr(block, "text"):
                                    text = response_text = block.text
                        if text:
                            f.write(text)
                            f.flush()
                return response_text

            response = await asyncio.wait_for(run_streaming(), timeout=timeout) if timeout else await run_streaming()
            status = "COMPLETED"
            logger.info(f"Agent execution completed, response length: {len(response)}")
            return response
        except ClaudeProcessInterruptedError:
            status = "INTERRUPTED"
            raise
        except asyncio.TimeoutError:
            status = "TIMEOUT"
            raise AgentTimeoutError(f"Agent execution timed out after {timeout}s")
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[Error] {e}\n")
            raise AgentClientError(f"Agent execution failed: {e}") from e
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{sep}\nEXECUTION SUMMARY\n{sep}\n\nStatus:         {status}\nEnd Time:       {end_time.isoformat()}\nDuration:       {duration:.2f}s\n\n{sep}\nEND OF LOG\n{sep}\n")
            logger.info(f"Streaming log completed: {log_path}")
