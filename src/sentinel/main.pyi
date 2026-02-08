"""Type stubs for main.py to improve IDE autocomplete.

See DS-402 for background on why this stub file was created.
GitHubIssueWithRepo stubs moved to poll_coordinator.pyi in DS-748
after the class was refactored from main.py to poll_coordinator.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from sentinel.agent_clients.factory import AgentClientFactory
from sentinel.agent_logger import AgentLogger
from sentinel.config import Config
from sentinel.executor import AgentExecutor, ExecutionResult
from sentinel.github_poller import GitHubPoller
from sentinel.github_rest_client import GitHubTagClient
from sentinel.orchestration import Orchestration
from sentinel.poller import JiraPoller
from sentinel.router import Router
from sentinel.tag_manager import JiraTagClient, TagManager

class AttemptCountEntry:
    count: int
    last_access: float
    def __init__(self, count: int, last_access: float) -> None: ...

class RunningStepInfo:
    issue_key: str
    orchestration_name: str
    attempt_number: int
    started_at: datetime
    issue_url: str
    def __init__(
        self,
        issue_key: str,
        orchestration_name: str,
        attempt_number: int,
        started_at: datetime,
        issue_url: str,
    ) -> None: ...

class QueuedIssueInfo:
    issue_key: str
    orchestration_name: str
    queued_at: datetime
    def __init__(
        self,
        issue_key: str,
        orchestration_name: str,
        queued_at: datetime,
    ) -> None: ...

class DashboardServer:
    def __init__(self, host: str, port: int) -> None: ...
    def start(self, app: Any) -> None: ...
    def shutdown(self) -> None: ...

class Sentinel:
    config: Config
    orchestrations: list[Orchestration]
    router: Router
    executor: AgentExecutor
    tag_manager: TagManager

    def __init__(
        self,
        config: Config,
        orchestrations: list[Orchestration],
        tag_client: JiraTagClient,
        agent_factory: AgentClientFactory,
        jira_poller: JiraPoller,
        agent_logger: AgentLogger | None = ...,
        router: Router | None = ...,
        github_poller: GitHubPoller | None = ...,
        github_tag_client: GitHubTagClient | None = ...,
    ) -> None: ...
    def request_shutdown(self) -> None: ...
    def get_hot_reload_metrics(self) -> dict[str, int]: ...
    def get_running_steps(self) -> list[RunningStepInfo]: ...
    def get_issue_queue(self) -> list[QueuedIssueInfo]: ...
    def get_start_time(self) -> datetime: ...
    def get_last_jira_poll(self) -> datetime | None: ...
    def get_last_github_poll(self) -> datetime | None: ...
    def get_active_versions(self) -> list[dict[str, Any]]: ...
    def get_pending_removal_versions(self) -> list[dict[str, Any]]: ...
    def get_execution_state(self) -> dict[str, Any]: ...
    def is_shutdown_requested(self) -> bool: ...
    def get_per_orch_count(self, orchestration_name: str) -> int: ...
    def get_all_per_orch_counts(self) -> dict[str, int]: ...
    def run_once(self) -> tuple[list[ExecutionResult], int]: ...
    def run_once_and_wait(self) -> list[ExecutionResult]: ...
    def run(self) -> None: ...

def parse_args(args: list[str] | None = ...) -> argparse.Namespace: ...
def main(args: list[str] | None = ...) -> int: ...
