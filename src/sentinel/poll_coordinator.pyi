"""Type stubs for poll_coordinator.py.

Provides explicit type hints for GitHubIssueWithRepo forwarding properties
to enable IDE autocomplete and mypy static analysis (DS-748).

Migrated from main.pyi after GitHubIssueWithRepo was refactored from
main.py to poll_coordinator.py.
"""

from __future__ import annotations

import re
from typing import Any

from sentinel.config import Config
from sentinel.github_poller import GitHubIssue, GitHubIssueProtocol, GitHubPoller
from sentinel.orchestration import Orchestration
from sentinel.poller import JiraIssue, JiraPoller
from sentinel.router import Router, RoutingResult

GITHUB_ISSUE_PR_URL_PATTERN: re.Pattern[str]

def extract_repo_from_url(url: str) -> str | None: ...

class GitHubIssueWithRepo:
    """Wrapper for GitHubIssue that provides full key with repo context.

    Type stub provides explicit type hints for forwarding properties
    to enable IDE autocomplete and mypy static analysis (DS-748).
    """

    _issue: GitHubIssue
    _repo: str

    def __init__(self, issue: GitHubIssue, repo: str) -> None: ...

    @property
    def key(self) -> str: ...
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

class PollingResult:
    issues_found: int
    submitted_count: int
    queued_count: int
    def __init__(
        self,
        issues_found: int = ...,
        submitted_count: int = ...,
        queued_count: int = ...,
    ) -> None: ...

class PollCoordinator:
    def __init__(
        self,
        config: Config,
        jira_poller: JiraPoller | None = ...,
        github_poller: GitHubPoller | None = ...,
    ) -> None: ...
    def create_cycle_dedup_set(self) -> set[tuple[str, str]]: ...
    def check_and_mark_submitted(
        self,
        submitted_pairs: set[tuple[str, str]],
        issue_key: str,
        orchestration_name: str,
    ) -> bool: ...
    def poll_jira_triggers(
        self,
        orchestrations: list[Orchestration],
        router: Router,
        shutdown_requested: bool = ...,
        log_callback: Any | None = ...,
    ) -> tuple[list[RoutingResult], int]: ...
    def poll_github_triggers(
        self,
        orchestrations: list[Orchestration],
        router: Router,
        shutdown_requested: bool = ...,
        log_callback: Any | None = ...,
    ) -> tuple[list[RoutingResult], int]: ...
    def construct_issue_url(
        self,
        issue: JiraIssue | GitHubIssueProtocol,
        orchestration: Orchestration,
    ) -> str: ...
    def group_orchestrations_by_source(
        self,
        orchestrations: list[Orchestration],
    ) -> tuple[list[Orchestration], list[Orchestration]]: ...
