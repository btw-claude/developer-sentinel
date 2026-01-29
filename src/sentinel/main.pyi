"""Type stubs for main.py to improve IDE autocomplete for GitHubIssueWithRepo.

This stub file provides explicit type hints for properties delegated via __getattr__
in GitHubIssueWithRepo, enabling IDE autocomplete while maintaining DRY runtime code.

See DS-402 for background on why this stub file was created.
"""

from __future__ import annotations

import argparse
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinel.agent_clients.factory import AgentClientFactory
from sentinel.agent_logger import AgentLogger
from sentinel.config import Config
from sentinel.executor import AgentClient, AgentExecutor, ExecutionResult
from sentinel.github_poller import GitHubClient, GitHubIssue, GitHubIssueProtocol, GitHubTagClient
from sentinel.logging import OrchestrationLogManager
from sentinel.orchestration import Orchestration, OrchestrationVersion
from sentinel.poller import JiraClient
from sentinel.router import Router, RoutingResult
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

class GitHubIssueWithRepo:
    """Wrapper for GitHubIssue that provides full key with repo context.

    Type stub provides explicit type hints for properties delegated via __getattr__
    to enable IDE autocomplete while maintaining DRY runtime code.
    """

    _issue: GitHubIssue
    _repo: str

    def __init__(self, issue: GitHubIssue, repo: str) -> None: ...

    # Explicit property override
    @property
    def key(self) -> str: ...

    # Properties delegated to GitHubIssue via __getattr__
    # These type hints enable IDE autocomplete for the delegated properties
    @property
    def number(self) -> int: ...
    @property
    def title(self) -> str: ...
    @property
    def body(self) -> str: ...
    @property
    def state(self) -> str: ...
    @property
    def author(self) -> str: ...
    @property
    def assignees(self) -> list[str]: ...
    @property
    def labels(self) -> list[str]: ...
    @property
    def is_pull_request(self) -> bool: ...
    @property
    def head_ref(self) -> str: ...
    @property
    def base_ref(self) -> str: ...
    @property
    def draft(self) -> bool: ...
    @property
    def repo_url(self) -> str: ...
    @property
    def parent_issue_number(self) -> int | None: ...

    def __getattr__(self, name: str) -> Any: ...

def extract_repo_from_url(url: str) -> str | None: ...

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
        jira_client: JiraClient,
        tag_client: JiraTagClient,
        agent_factory: AgentClientFactory | AgentClient | None = ...,
        agent_logger: AgentLogger | None = ...,
        github_client: GitHubClient | None = ...,
        github_tag_client: GitHubTagClient | None = ...,
        *,
        agent_client: AgentClient | None = ...,
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
