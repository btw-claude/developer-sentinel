"""Orchestration configuration schema and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class TriggerConfig:
    """Configuration for what triggers an orchestration."""

    source: Literal["jira"] = "jira"
    project: str = ""
    jql_filter: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class GitHubContext:
    """GitHub repository context for agent operations."""

    host: str = "github.com"
    org: str = ""
    repo: str = ""


@dataclass
class AgentConfig:
    """Configuration for the Claude agent.

    Attributes:
        prompt: The prompt template for the agent.
        tools: List of tool names the agent can use.
        github: Optional GitHub repository context.
        timeout_seconds: Optional timeout in seconds for agent execution.
            If None, no timeout is applied.
    """

    prompt: str = ""
    tools: list[str] = field(default_factory=list)
    github: GitHubContext | None = None
    timeout_seconds: int | None = None


@dataclass
class RetryConfig:
    """Configuration for retry logic based on agent response patterns.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        success_patterns: Patterns indicating success (substring or "regex:..." prefix).
        failure_patterns: Patterns indicating failure (substring or "regex:..." prefix).
        default_status: Status to use when no patterns match ("success" or "failure").
            Defaults to "success" for backwards compatibility.
    """

    max_attempts: int = 3
    success_patterns: list[str] = field(
        default_factory=lambda: ["SUCCESS", "completed successfully"]
    )
    failure_patterns: list[str] = field(default_factory=lambda: ["FAILURE", "failed", "error"])
    default_status: Literal["success", "failure"] = "success"


@dataclass
class OnCompleteConfig:
    """Actions to take after successful orchestration completion."""

    remove_tag: str = ""
    add_tag: str = ""


@dataclass
class OnFailureConfig:
    """Actions to take after failed orchestration (all retries exhausted)."""

    add_tag: str = ""


@dataclass
class Orchestration:
    """A single orchestration configuration."""

    name: str
    trigger: TriggerConfig
    agent: AgentConfig
    retry: RetryConfig = field(default_factory=RetryConfig)
    on_complete: OnCompleteConfig = field(default_factory=OnCompleteConfig)
    on_failure: OnFailureConfig = field(default_factory=OnFailureConfig)


class OrchestrationError(Exception):
    """Raised when orchestration configuration is invalid."""

    pass


def _parse_trigger(data: dict[str, Any]) -> TriggerConfig:
    """Parse trigger configuration from dict."""
    return TriggerConfig(
        source=data.get("source", "jira"),
        project=data.get("project", ""),
        jql_filter=data.get("jql_filter", ""),
        tags=data.get("tags", []),
    )


def _parse_github_context(data: dict[str, Any] | None) -> GitHubContext | None:
    """Parse GitHub context from dict."""
    if not data:
        return None
    return GitHubContext(
        host=data.get("host", "github.com"),
        org=data.get("org", ""),
        repo=data.get("repo", ""),
    )


def _parse_agent(data: dict[str, Any]) -> AgentConfig:
    """Parse agent configuration from dict."""
    timeout = data.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise OrchestrationError(f"Invalid timeout_seconds '{timeout}': must be a positive integer")

    return AgentConfig(
        prompt=data.get("prompt", ""),
        tools=data.get("tools", []),
        github=_parse_github_context(data.get("github")),
        timeout_seconds=timeout,
    )


def _parse_retry(data: dict[str, Any] | None) -> RetryConfig:
    """Parse retry configuration from dict."""
    if not data:
        return RetryConfig()

    # Validate default_status if provided
    default_status = data.get("default_status", "success")
    if default_status not in ("success", "failure"):
        raise OrchestrationError(
            f"Invalid default_status '{default_status}': must be 'success' or 'failure'"
        )

    return RetryConfig(
        max_attempts=data.get("max_attempts", 3),
        success_patterns=data.get("success_patterns", ["SUCCESS", "completed successfully"]),
        failure_patterns=data.get("failure_patterns", ["FAILURE", "failed", "error"]),
        default_status=default_status,
    )


def _parse_on_complete(data: dict[str, Any] | None) -> OnCompleteConfig:
    """Parse on_complete configuration from dict."""
    if not data:
        return OnCompleteConfig()
    return OnCompleteConfig(
        remove_tag=data.get("remove_tag", ""),
        add_tag=data.get("add_tag", ""),
    )


def _parse_on_failure(data: dict[str, Any] | None) -> OnFailureConfig:
    """Parse on_failure configuration from dict."""
    if not data:
        return OnFailureConfig()
    return OnFailureConfig(
        add_tag=data.get("add_tag", ""),
    )


def _parse_orchestration(data: dict[str, Any]) -> Orchestration:
    """Parse a single orchestration from dict."""
    name = data.get("name")
    if not name:
        raise OrchestrationError("Orchestration must have a 'name' field")

    trigger_data = data.get("trigger")
    if not trigger_data:
        raise OrchestrationError(f"Orchestration '{name}' must have a 'trigger' field")

    agent_data = data.get("agent")
    if not agent_data:
        raise OrchestrationError(f"Orchestration '{name}' must have an 'agent' field")

    return Orchestration(
        name=name,
        trigger=_parse_trigger(trigger_data),
        agent=_parse_agent(agent_data),
        retry=_parse_retry(data.get("retry")),
        on_complete=_parse_on_complete(data.get("on_complete")),
        on_failure=_parse_on_failure(data.get("on_failure")),
    )


def load_orchestration_file(file_path: Path) -> list[Orchestration]:
    """Load orchestrations from a single YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        List of Orchestration objects.

    Raises:
        OrchestrationError: If the file is invalid.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise OrchestrationError(f"Invalid YAML in {file_path}: {e}") from e
    except FileNotFoundError:
        raise OrchestrationError(f"Orchestration file not found: {file_path}") from None

    if not data:
        return []

    orchestrations_data = data.get("orchestrations", [])
    if not isinstance(orchestrations_data, list):
        raise OrchestrationError(f"'orchestrations' must be a list in {file_path}")

    return [_parse_orchestration(orch) for orch in orchestrations_data]


def load_orchestrations(directory: Path) -> list[Orchestration]:
    """Load all orchestrations from a directory.

    Loads all .yaml and .yml files from the given directory.

    Args:
        directory: Path to the orchestrations directory.

    Returns:
        List of all Orchestration objects from all files.

    Raises:
        OrchestrationError: If any file is invalid.
    """
    if not directory.exists():
        return []

    if not directory.is_dir():
        raise OrchestrationError(f"Orchestrations path is not a directory: {directory}")

    orchestrations: list[Orchestration] = []

    for file_path in sorted(directory.iterdir()):
        if file_path.suffix in (".yaml", ".yml"):
            orchestrations.extend(load_orchestration_file(file_path))

    return orchestrations
