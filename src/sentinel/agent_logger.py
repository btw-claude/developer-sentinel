"""Agent execution logging to orchestration-specific directories.

Logs agent executions to structured directories:
    {base_dir}/{orchestration_name}/{datetime}.log

Each log file contains:
- Execution metadata (issue key, orchestration, timestamps)
- Agent prompt
- Agent response
- Execution status
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from sentinel.executor import ExecutionStatus

logger = get_logger(__name__)


class AgentLogger:
    """Writes agent execution logs to orchestration-specific directories."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize the agent logger.

        Args:
            base_dir: Base directory for logs (e.g., ./logs).
        """
        self.base_dir = base_dir

    def _get_log_path(self, orchestration_name: str, timestamp: datetime) -> Path:
        """Get the log file path for an execution.

        Args:
            orchestration_name: Name of the orchestration.
            timestamp: Execution timestamp.

        Returns:
            Path to the log file.
        """
        # Create orchestration-specific directory
        orch_dir = self.base_dir / orchestration_name
        orch_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: YYYYMMDD_HHMMSS.log
        filename = timestamp.strftime("%Y%m%d_%H%M%S") + ".log"
        return orch_dir / filename

    def log_execution(
        self,
        issue_key: str,
        orchestration_name: str,
        prompt: str,
        response: str,
        status: ExecutionStatus,
        attempts: int,
        start_time: datetime,
        end_time: datetime,
    ) -> Path:
        """Write an agent execution log.

        Args:
            issue_key: The Jira issue key.
            orchestration_name: Name of the orchestration.
            prompt: The prompt sent to the agent.
            response: The agent's response.
            status: The execution status.
            attempts: Number of attempts made.
            start_time: When execution started.
            end_time: When execution ended.

        Returns:
            Path to the written log file.
        """
        log_path = self._get_log_path(orchestration_name, start_time)
        duration = (end_time - start_time).total_seconds()
        separator = "=" * 80

        log_content = f"""{separator}
AGENT EXECUTION LOG
{separator}

Issue Key:      {issue_key}
Orchestration:  {orchestration_name}
Status:         {status.value.upper()}
Attempts:       {attempts}
Start Time:     {start_time.isoformat()}
End Time:       {end_time.isoformat()}
Duration:       {duration:.2f}s

{separator}
PROMPT
{separator}

{prompt}

{separator}
RESPONSE
{separator}

{response}

{separator}
END OF LOG
{separator}
"""

        log_path.write_text(log_content)
        logger.info(f"Agent execution log written to {log_path}")
        return log_path
