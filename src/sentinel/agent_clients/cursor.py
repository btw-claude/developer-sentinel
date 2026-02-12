"""Cursor CLI agent client implementation.

This module provides the Cursor CLI client for running agents via subprocess,
supporting different modes (agent, plan, ask) and model selection.

The run_agent method is async to conform to the AgentClient interface,
using asyncio.to_thread() for subprocess operations.
"""

from __future__ import annotations

import asyncio
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any

from sentinel.agent_clients.base import (
    AgentClient,
    AgentClientError,
    AgentRunResult,
    AgentTimeoutError,
    AgentType,
)
from sentinel.config import Config
from sentinel.logging import get_logger

logger = get_logger(__name__)


class CursorMode(StrEnum):
    """Cursor CLI operation modes."""

    AGENT = "agent"
    PLAN = "plan"
    ASK = "ask"

    @classmethod
    def from_string(cls, value: str) -> CursorMode:
        """Convert string to CursorMode, with validation.

        Args:
            value: String value to convert (case-insensitive).

        Returns:
            The corresponding CursorMode.

        Raises:
            ValueError: If the value is not a valid mode.
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_modes = [m.value for m in cls]
            msg = f"Invalid Cursor mode: '{value}'. Valid modes: {valid_modes}"
            raise ValueError(msg) from None


class CursorAgentClient(AgentClient):
    """Agent client that uses Cursor CLI via subprocess.

    This client runs the Cursor CLI tool (`cursor` or `agent` command) to execute
    agent tasks. It supports:
    - Multiple operation modes: agent, plan, ask
    - Model selection via --model flag
    - Timeout handling via subprocess timeout
    - Working directory creation similar to ClaudeSdkAgentClient

    Configuration is provided via the Config dataclass:
    - cursor_path: Path to the Cursor CLI executable
    - cursor_default_model: Default model to use
    - cursor_default_mode: Default operation mode (agent, plan, ask)

    Example:
        config = Config(cursor_path="/usr/local/bin/cursor", cursor_default_mode="agent")
        client = CursorAgentClient(config)
        result = client.run_agent("Write a hello world function")
    """

    def __init__(
        self,
        config: Config,
        base_workdir: Path | None = None,
        log_base_dir: Path | None = None,
    ) -> None:
        """Initialize the Cursor agent client.

        Args:
            config: Configuration object with Cursor settings.
            base_workdir: Base directory for creating agent working directories.
            log_base_dir: Base directory for agent execution logs (reserved for future use).
        """
        self.config = config
        self.base_workdir = base_workdir
        self.log_base_dir = log_base_dir

        # Validate and store cursor path
        self._cursor_path = config.cursor.path or "cursor"
        self._default_model = config.cursor.default_model or None

        # Defensive validation for cursor_default_mode
        if not config.cursor.default_mode or not config.cursor.default_mode.strip():
            raise AgentClientError(
                "cursor_default_mode is not set. Please configure SENTINEL_CURSOR_DEFAULT_MODE "
                "with a valid mode: agent, plan, or ask"
            )
        self._default_mode = CursorMode.from_string(config.cursor.default_mode)

        logger.debug(
            "Initialized CursorAgentClient with path=%s, model=%s, mode=%s",
            self._cursor_path,
            self._default_model,
            self._default_mode.value,
        )

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return AgentType.CURSOR

    def _build_command(
        self,
        prompt: str,
        model: str | None = None,
        mode: CursorMode | None = None,
    ) -> list[str]:
        """Build the Cursor CLI command.

        Args:
            prompt: The prompt to send to the agent.
            model: Model identifier (overrides default if provided).
            mode: Operation mode (overrides default if provided).

        Returns:
            List of command arguments for subprocess.
        """
        effective_mode = mode if mode is not None else self._default_mode
        effective_model = model if model is not None else self._default_model

        cmd = [
            self._cursor_path,
            "-p",
            prompt,
            "--output-format",
            "text",
            f"--mode={effective_mode.value}",
        ]

        if effective_model:
            cmd.extend(["--model", effective_model])

        return cmd

    async def run_agent(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        timeout_seconds: int | float | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
        agent_teams: bool = False,
        attempt: int = 1,
        mode: CursorMode | str | None = None,
    ) -> AgentRunResult:
        """Run a Cursor agent with the given prompt.

        This is an async method that uses asyncio.to_thread() to run the
        subprocess without blocking the event loop.

        Args:
            prompt: The prompt to send to the agent.
            context: Optional context dict to append to prompt.
            timeout_seconds: Optional timeout in seconds. If None, no timeout is applied.
            issue_key: Optional issue key for creating a unique working directory.
            model: Optional model identifier. If None, uses the client's default model.
            orchestration_name: Optional orchestration name (reserved for future logging).
            branch: Optional branch name (reserved for future implementation).
            create_branch: If True and branch doesn't exist, create it (reserved for future).
            base_branch: Base branch to create new branches from (reserved for future).
            agent_teams: Whether to enable Claude Code's experimental Agent Teams feature
                (ignored for Cursor client).
            attempt: Attempt number (1-based) from the executor retry loop.
                Used for unique log filenames and workdir names across retries (DS-960).
            mode: Optional operation mode (agent, plan, ask). Can be a CursorMode enum or string.
                  If None, uses the client's default mode.

        Returns:
            AgentRunResult containing the agent's response text and optional working directory path.

        Raises:
            AgentClientError: If the agent execution fails (non-zero exit code).
            AgentTimeoutError: If the agent execution times out.
        """
        workdir = None
        if self.base_workdir is not None and issue_key is not None:
            workdir = self._create_workdir(issue_key, attempt=attempt)

        # Build full prompt with sanitized context section (DS-675).
        full_prompt = self._build_prompt_with_context(prompt, context)

        # Convert string mode to CursorMode if provided
        effective_mode: CursorMode | None = None
        if mode is not None:
            effective_mode = mode if isinstance(mode, CursorMode) else CursorMode.from_string(mode)

        # Build the command
        cmd = self._build_command(full_prompt, model=model, mode=effective_mode)

        # Use subprocess_timeout from config as fallback when timeout_seconds is not provided
        # This ensures consistency with other subprocess.run() calls in the codebase
        effective_timeout = timeout_seconds
        if effective_timeout is None and self.config.execution.subprocess_timeout > 0:
            effective_timeout = int(self.config.execution.subprocess_timeout)

        logger.info(
            "Running Cursor CLI: mode=%s, model=%s, timeout=%ss",
            (effective_mode or self._default_mode).value,
            model or self._default_model or 'default',
            effective_timeout,
        )
        logger.debug("Cursor command: %s...", ' '.join(cmd[:3]))

        def _run_subprocess() -> subprocess.CompletedProcess[str]:
            """Run subprocess in a thread to avoid blocking the event loop."""
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=str(workdir) if workdir else None,
            )

        try:
            result = await asyncio.to_thread(_run_subprocess)

            if result.returncode != 0:
                error_msg = result.stderr.strip() or f"Exit code: {result.returncode}"
                logger.error("Cursor CLI failed: %s", error_msg)
                raise AgentClientError(f"Cursor CLI execution failed: {error_msg}")

            response = result.stdout.strip()
            logger.info("Cursor execution completed, response length: %s", len(response))

            return AgentRunResult(response=response, workdir=workdir)

        except subprocess.TimeoutExpired:
            logger.debug(
                "Cursor subprocess timeout: timeout=%ss, workdir=%s, mode=%s",
                effective_timeout,
                workdir,
                (effective_mode or self._default_mode).value,
            )
            logger.error("Cursor CLI timed out after %ss", effective_timeout)
            msg = f"Cursor agent execution timed out after {effective_timeout}s"
            raise AgentTimeoutError(msg) from None
        except FileNotFoundError:
            logger.error("Cursor CLI not found at path: %s", self._cursor_path)
            msg = f"Cursor CLI executable not found: {self._cursor_path}"
            raise AgentClientError(msg) from None
        except OSError as e:
            logger.error("Failed to execute Cursor CLI: %s", e)
            raise AgentClientError(f"Failed to execute Cursor CLI: {e}") from e
