"""Cursor CLI agent client implementation.

This module provides the Cursor CLI client for running agents via subprocess,
supporting different modes (agent, plan, ask) and model selection.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from enum import Enum
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


class CursorMode(str, Enum):
    """Cursor CLI operation modes."""

    AGENT = "agent"
    PLAN = "plan"
    ASK = "ask"

    @classmethod
    def from_string(cls, value: str) -> "CursorMode":
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
            raise ValueError(
                f"Invalid Cursor mode: '{value}'. Valid modes: {valid_modes}"
            )


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
        result = client.run_agent("Write a hello world function", tools=[])
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
        self._cursor_path = config.cursor_path or "cursor"
        self._default_model = config.cursor_default_model or None

        # Defensive validation for cursor_default_mode
        if not config.cursor_default_mode or not config.cursor_default_mode.strip():
            raise AgentClientError(
                "cursor_default_mode is not set. Please configure SENTINEL_CURSOR_DEFAULT_MODE "
                "with a valid mode: agent, plan, or ask"
            )
        self._default_mode = CursorMode.from_string(config.cursor_default_mode)

        logger.debug(
            f"Initialized CursorAgentClient with path={self._cursor_path}, "
            f"model={self._default_model}, mode={self._default_mode.value}"
        )

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return "cursor"

    def _create_workdir(self, issue_key: str) -> Path:
        """Create a unique working directory for an agent run.

        Args:
            issue_key: The issue key to include in the directory name.

        Returns:
            Path to the created working directory.

        Raises:
            AgentClientError: If base_workdir is not configured.
        """
        if self.base_workdir is None:
            raise AgentClientError("base_workdir not configured")

        self.base_workdir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = self.base_workdir / f"{issue_key}_{timestamp}"
        workdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created agent working directory: {workdir}")
        return workdir

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

    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        mode: CursorMode | str | None = None,
    ) -> AgentRunResult:
        """Run a Cursor agent with the given prompt.

        Args:
            prompt: The prompt to send to the agent.
            tools: List of tool names (not used by Cursor CLI, reserved for interface compatibility).
            context: Optional context dict to append to prompt.
            timeout_seconds: Optional timeout in seconds. If None, no timeout is applied.
            issue_key: Optional issue key for creating a unique working directory.
            model: Optional model identifier. If None, uses the client's default model.
            orchestration_name: Optional orchestration name (reserved for future logging).
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
            workdir = self._create_workdir(issue_key)

        # Build full prompt with context section
        full_prompt = prompt
        if context:
            full_prompt += "\n\nContext:\n" + "".join(
                f"- {k}: {v}\n" for k, v in context.items()
            )

        # Convert string mode to CursorMode if provided
        effective_mode: CursorMode | None = None
        if mode is not None:
            effective_mode = mode if isinstance(mode, CursorMode) else CursorMode.from_string(mode)

        # Build the command
        cmd = self._build_command(full_prompt, model=model, mode=effective_mode)

        logger.info(
            f"Running Cursor CLI: mode={(effective_mode or self._default_mode).value}, "
            f"model={model or self._default_model or 'default'}, "
            f"timeout={timeout_seconds}s"
        )
        logger.debug(f"Cursor command: {' '.join(cmd[:3])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(workdir) if workdir else None,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or f"Exit code: {result.returncode}"
                logger.error(f"Cursor CLI failed: {error_msg}")
                raise AgentClientError(f"Cursor CLI execution failed: {error_msg}")

            response = result.stdout.strip()
            logger.info(f"Cursor execution completed, response length: {len(response)}")

            return AgentRunResult(response=response, workdir=workdir)

        except subprocess.TimeoutExpired:
            logger.error(f"Cursor CLI timed out after {timeout_seconds}s")
            raise AgentTimeoutError(
                f"Cursor agent execution timed out after {timeout_seconds}s"
            )
        except FileNotFoundError:
            logger.error(f"Cursor CLI not found at path: {self._cursor_path}")
            raise AgentClientError(
                f"Cursor CLI executable not found: {self._cursor_path}"
            )
        except OSError as e:
            logger.error(f"Failed to execute Cursor CLI: {e}")
            raise AgentClientError(f"Failed to execute Cursor CLI: {e}") from e
