"""Pydantic request/response models for dashboard API endpoints.

Extracted from routes.py (DS-755, DS-756) for better maintainability as the
module grew past ~1750 lines. Models are grouped by feature:

- Toggle models: ToggleRequest, ToggleResponse, BulkToggleRequest, BulkToggleResponse
- Edit models: TriggerEditRequest, GitHubContextEditRequest, AgentEditRequest,
  RetryEditRequest, OutcomeEditRequest, LifecycleEditRequest, OrchestrationEditRequest,
  OrchestrationEditResponse
- Delete models: DeleteResponse
- Create models: OrchestrationCreateRequest, OrchestrationCreateResponse
- Detail models (DS-754): GitHubContextDetailResponse, TriggerDetailResponse,
  AgentDetailResponse, RetryDetailResponse, OutcomeDetailResponse,
  LifecycleDetailResponse, OrchestrationDetailResponse
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from sentinel.types import AgentTypeLiteral, CursorModeLiteral, TriggerSourceLiteral

# NOTE: Update this list when adding new models to this module.
__all__: list[str] = [
    # Toggle models
    "ToggleRequest",
    "ToggleResponse",
    "BulkToggleRequest",
    "BulkToggleResponse",
    # Edit models
    "FileTriggerEditRequest",
    "TriggerEditRequest",
    "GitHubContextEditRequest",
    "AgentEditRequest",
    "RetryEditRequest",
    "OutcomeEditRequest",
    "LifecycleEditRequest",
    "OrchestrationEditRequest",
    "OrchestrationEditResponse",
    "FileTriggerEditResponse",
    # Delete models
    "DeleteResponse",
    # Create models
    "OrchestrationCreateRequest",
    "OrchestrationCreateResponse",
    # Detail models (DS-754)
    "GitHubContextDetailResponse",
    "TriggerDetailResponse",
    "AgentDetailResponse",
    "RetryDetailResponse",
    "OutcomeDetailResponse",
    "LifecycleDetailResponse",
    "OrchestrationDetailResponse",
]


# Request/Response models for orchestration toggle endpoints
class ToggleRequest(BaseModel):
    """Request model for single orchestration toggle."""

    enabled: bool


class ToggleResponse(BaseModel):
    """Response model for single orchestration toggle."""

    success: bool
    enabled: bool
    name: str


class BulkToggleRequest(BaseModel):
    """Request model for bulk orchestration toggle."""

    source: Literal["jira", "github"]
    identifier: str
    enabled: bool


class BulkToggleResponse(BaseModel):
    """Response model for bulk orchestration toggle."""

    success: bool
    toggled_count: int


# Pydantic models for orchestration edit endpoints (DS-727)
class FileTriggerEditRequest(BaseModel):
    """Request model for editing file-level trigger configuration.

    File-level trigger contains project-scoping fields that apply to all
    steps in the file (DS-896).
    """
    source: TriggerSourceLiteral | None = None
    project: str | None = None
    project_number: int | None = None
    project_scope: Literal["org", "user"] | None = None
    project_owner: str | None = None


class TriggerEditRequest(BaseModel):
    """Request model for editing step-level trigger configuration.

    After DS-896, file-level fields (source, project, project_number,
    project_scope, project_owner) are managed via FileTriggerEditRequest.
    Step-level trigger only contains filter fields.
    """
    jql_filter: str | None = None
    tags: list[str] | None = None
    project_filter: str | None = None
    labels: list[str] | None = None


class GitHubContextEditRequest(BaseModel):
    """Request model for editing GitHub context configuration."""

    host: str | None = None
    org: str | None = None
    repo: str | None = None
    branch: str | None = None
    create_branch: bool | None = None
    base_branch: str | None = None


class AgentEditRequest(BaseModel):
    """Request model for editing agent configuration."""

    prompt: str | None = None
    github: GitHubContextEditRequest | None = None
    timeout_seconds: int | float | None = None
    model: str | None = None
    agent_type: AgentTypeLiteral | None = None
    cursor_mode: CursorModeLiteral | None = None
    agent_teams: bool | None = None
    strict_template_variables: bool | None = None


class RetryEditRequest(BaseModel):
    """Request model for editing retry configuration."""

    max_attempts: int | None = None
    success_patterns: list[str] | None = None
    failure_patterns: list[str] | None = None
    default_status: Literal["success", "failure"] | None = None
    default_outcome: str | None = None


class OutcomeEditRequest(BaseModel):
    """Request model for editing outcome configuration."""

    name: str | None = None
    patterns: list[str] | None = None
    add_tag: str | None = None


class LifecycleEditRequest(BaseModel):
    """Request model for editing lifecycle configuration."""

    on_start_add_tag: str | None = None
    on_complete_remove_tag: str | None = None
    on_complete_add_tag: str | None = None
    on_failure_add_tag: str | None = None


class OrchestrationEditRequest(BaseModel):
    """Request model for editing an orchestration configuration.

    All fields are optional. Only provided fields will be updated.
    The 'name' field is read-only and cannot be edited.
    """

    enabled: bool | None = None
    max_concurrent: int | None = None
    trigger: TriggerEditRequest | None = None
    agent: AgentEditRequest | None = None
    retry: RetryEditRequest | None = None
    outcomes: list[OutcomeEditRequest] | None = None
    lifecycle: LifecycleEditRequest | None = None


class OrchestrationEditResponse(BaseModel):
    """Response model for orchestration edit."""

    success: bool
    name: str
    errors: list[str] = []


class FileTriggerEditResponse(BaseModel):
    """Response model for file-level trigger edit."""

    success: bool
    errors: list[str] = []


class DeleteResponse(BaseModel):
    """Response model for orchestration deletion."""

    success: bool
    name: str


class OrchestrationCreateRequest(BaseModel):
    """Request model for creating a new orchestration."""

    name: str
    target_file: str
    enabled: bool | None = None
    max_concurrent: int | None = None
    file_trigger: FileTriggerEditRequest | None = None  # DS-896
    trigger: TriggerEditRequest | None = None
    agent: AgentEditRequest | None = None
    retry: RetryEditRequest | None = None
    outcomes: list[OutcomeEditRequest] | None = None
    lifecycle: LifecycleEditRequest | None = None


class OrchestrationCreateResponse(BaseModel):
    """Response model for orchestration creation."""

    success: bool
    name: str
    errors: list[str] = []


# Pydantic response models for orchestration detail endpoint (DS-754)
# These use from_attributes=True to enable direct conversion from frozen
# dataclass DTOs via model_validate(), eliminating manual field-by-field
# construction and keeping the models in sync with domain objects automatically.


class GitHubContextDetailResponse(BaseModel):
    """Response model for GitHub context in orchestration detail."""

    model_config = ConfigDict(from_attributes=True)

    host: str
    org: str
    repo: str
    branch: str
    create_branch: bool
    base_branch: str


class TriggerDetailResponse(BaseModel):
    """Response model for trigger configuration in orchestration detail."""

    model_config = ConfigDict(from_attributes=True)

    source: str
    project: str
    jql_filter: str
    tags: list[str]
    project_number: int | None
    project_scope: Literal["org", "user"]
    project_owner: str
    project_filter: str
    labels: list[str]


class AgentDetailResponse(BaseModel):
    """Response model for agent configuration in orchestration detail."""

    model_config = ConfigDict(from_attributes=True)

    prompt: str
    github: GitHubContextDetailResponse | None
    timeout_seconds: int | float | None
    model: str | None
    agent_type: str | None
    cursor_mode: str | None
    strict_template_variables: bool


class RetryDetailResponse(BaseModel):
    """Response model for retry configuration in orchestration detail."""

    model_config = ConfigDict(from_attributes=True)

    max_attempts: int
    success_patterns: list[str]
    failure_patterns: list[str]
    default_status: str
    default_outcome: str


class OutcomeDetailResponse(BaseModel):
    """Response model for an outcome in orchestration detail."""

    model_config = ConfigDict(from_attributes=True)

    name: str
    patterns: list[str]
    add_tag: str


class LifecycleDetailResponse(BaseModel):
    """Response model for lifecycle configuration in orchestration detail."""

    model_config = ConfigDict(from_attributes=True)

    on_start_add_tag: str
    on_complete_remove_tag: str
    on_complete_add_tag: str
    on_failure_add_tag: str


class OrchestrationDetailResponse(BaseModel):
    """Response model for full orchestration detail.

    Uses from_attributes=True to enable direct conversion from
    OrchestrationDetailInfo frozen dataclass via model_validate().
    """

    model_config = ConfigDict(from_attributes=True)

    name: str
    enabled: bool
    max_concurrent: int | None
    source_file: str
    trigger: TriggerDetailResponse
    agent: AgentDetailResponse
    retry: RetryDetailResponse
    outcomes: list[OutcomeDetailResponse]
    lifecycle: LifecycleDetailResponse
