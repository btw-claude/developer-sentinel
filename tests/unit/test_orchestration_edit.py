"""Tests for orchestration edit models and validation (DS-727).

Tests for the Pydantic models, _build_yaml_updates helper,
and _validate_orchestration_updates validation function.
"""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.routes import (
    AgentEditRequest,
    GitHubContextEditRequest,
    LifecycleEditRequest,
    OrchestrationEditRequest,
    OrchestrationEditResponse,
    OutcomeEditRequest,
    RetryEditRequest,
    TriggerEditRequest,
    _build_yaml_updates,
    _deep_merge_dicts,
    _validate_orchestration_updates,
)


class TestBuildYamlUpdates:
    """Tests for _build_yaml_updates helper function."""

    def test_empty_request_returns_empty_dict(self) -> None:
        """Should return empty dict when no fields are set."""
        request = OrchestrationEditRequest()
        updates = _build_yaml_updates(request)
        assert updates == {}

    def test_enabled_field(self) -> None:
        """Should include enabled field when set."""
        request = OrchestrationEditRequest(enabled=False)
        updates = _build_yaml_updates(request)
        assert updates == {"enabled": False}

    def test_max_concurrent_field(self) -> None:
        """Should include max_concurrent field when set."""
        request = OrchestrationEditRequest(max_concurrent=5)
        updates = _build_yaml_updates(request)
        assert updates == {"max_concurrent": 5}

    def test_trigger_fields(self) -> None:
        """Should build trigger dict from TriggerEditRequest."""
        request = OrchestrationEditRequest(
            trigger=TriggerEditRequest(
                source="jira",
                project="NEW-PROJ",
                tags=["tag1", "tag2"],
            )
        )
        updates = _build_yaml_updates(request)
        assert updates == {
            "trigger": {
                "source": "jira",
                "project": "NEW-PROJ",
                "tags": ["tag1", "tag2"],
            }
        }

    def test_trigger_partial_update(self) -> None:
        """Should only include set trigger fields."""
        request = OrchestrationEditRequest(
            trigger=TriggerEditRequest(project="NEW-PROJ")
        )
        updates = _build_yaml_updates(request)
        assert updates == {"trigger": {"project": "NEW-PROJ"}}

    def test_agent_fields(self) -> None:
        """Should build agent dict from AgentEditRequest."""
        request = OrchestrationEditRequest(
            agent=AgentEditRequest(
                prompt="New prompt",
                model="claude-sonnet-4-20250514",
                timeout_seconds=600,
            )
        )
        updates = _build_yaml_updates(request)
        assert updates == {
            "agent": {
                "prompt": "New prompt",
                "model": "claude-sonnet-4-20250514",
                "timeout_seconds": 600,
            }
        }

    def test_agent_with_github(self) -> None:
        """Should build nested github dict in agent."""
        request = OrchestrationEditRequest(
            agent=AgentEditRequest(
                github=GitHubContextEditRequest(
                    host="github.com",
                    org="my-org",
                    repo="my-repo",
                )
            )
        )
        updates = _build_yaml_updates(request)
        assert updates == {
            "agent": {
                "github": {
                    "host": "github.com",
                    "org": "my-org",
                    "repo": "my-repo",
                }
            }
        }

    def test_retry_fields(self) -> None:
        """Should build retry dict from RetryEditRequest."""
        request = OrchestrationEditRequest(
            retry=RetryEditRequest(
                max_attempts=5,
                default_status="failure",
            )
        )
        updates = _build_yaml_updates(request)
        assert updates == {
            "retry": {
                "max_attempts": 5,
                "default_status": "failure",
            }
        }

    def test_outcomes_fields(self) -> None:
        """Should build outcomes list from OutcomeEditRequest list."""
        request = OrchestrationEditRequest(
            outcomes=[
                OutcomeEditRequest(
                    name="approved",
                    patterns=["APPROVED"],
                    add_tag="approved",
                ),
                OutcomeEditRequest(
                    name="changes-requested",
                    patterns=["CHANGES_REQUESTED"],
                ),
            ]
        )
        updates = _build_yaml_updates(request)
        assert "outcomes" in updates
        assert len(updates["outcomes"]) == 2
        assert updates["outcomes"][0] == {
            "name": "approved",
            "patterns": ["APPROVED"],
            "add_tag": "approved",
        }

    def test_lifecycle_fields(self) -> None:
        """Should build lifecycle-related dicts."""
        request = OrchestrationEditRequest(
            lifecycle=LifecycleEditRequest(
                on_start_add_tag="in-progress",
                on_complete_remove_tag="in-progress",
                on_complete_add_tag="done",
                on_failure_add_tag="failed",
            )
        )
        updates = _build_yaml_updates(request)
        assert updates["on_start"] == {"add_tag": "in-progress"}
        assert updates["on_complete"] == {"remove_tag": "in-progress", "add_tag": "done"}
        assert updates["on_failure"] == {"add_tag": "failed"}

    def test_agent_teams_field(self) -> None:
        """Should include agent_teams when set."""
        request = OrchestrationEditRequest(
            agent=AgentEditRequest(agent_teams=True)
        )
        updates = _build_yaml_updates(request)
        assert updates == {"agent": {"agent_teams": True}}

    def test_agent_strict_template_variables(self) -> None:
        """Should include strict_template_variables when set."""
        request = OrchestrationEditRequest(
            agent=AgentEditRequest(strict_template_variables=True)
        )
        updates = _build_yaml_updates(request)
        assert updates == {"agent": {"strict_template_variables": True}}

    def test_empty_trigger_omitted(self) -> None:
        """Should not include trigger key when all trigger fields are None."""
        request = OrchestrationEditRequest(trigger=TriggerEditRequest())
        updates = _build_yaml_updates(request)
        assert "trigger" not in updates

    def test_empty_agent_omitted(self) -> None:
        """Should not include agent key when all agent fields are None."""
        request = OrchestrationEditRequest(agent=AgentEditRequest())
        updates = _build_yaml_updates(request)
        assert "agent" not in updates

    def test_empty_retry_omitted(self) -> None:
        """Should not include retry key when all retry fields are None."""
        request = OrchestrationEditRequest(retry=RetryEditRequest())
        updates = _build_yaml_updates(request)
        assert "retry" not in updates


class TestDeepMergeDicts:
    """Tests for _deep_merge_dicts helper function."""

    def test_simple_merge(self) -> None:
        """Should merge simple key-value pairs."""
        base = {"a": 1, "b": 2}
        updates = {"b": 3, "c": 4}
        result = _deep_merge_dicts(base, updates)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Should deep-merge nested dicts."""
        base = {"trigger": {"source": "jira", "project": "OLD"}}
        updates = {"trigger": {"project": "NEW"}}
        result = _deep_merge_dicts(base, updates)
        assert result == {"trigger": {"source": "jira", "project": "NEW"}}

    def test_list_replacement(self) -> None:
        """Should replace lists entirely (not merge)."""
        base = {"tags": ["old-tag"]}
        updates = {"tags": ["new-tag"]}
        result = _deep_merge_dicts(base, updates)
        assert result == {"tags": ["new-tag"]}

    def test_does_not_modify_original(self) -> None:
        """Should not modify the original base dict."""
        base = {"a": 1, "nested": {"b": 2}}
        updates = {"nested": {"b": 3}}
        _deep_merge_dicts(base, updates)
        assert base == {"a": 1, "nested": {"b": 2}}


class TestValidateOrchestrationUpdates:
    """Tests for _validate_orchestration_updates function."""

    @staticmethod
    def _make_valid_orch_data() -> dict[str, Any]:
        """Create a valid orchestration data dict for testing."""
        return {
            "name": "test-orch",
            "enabled": True,
            "trigger": {
                "source": "jira",
                "project": "TEST",
            },
            "agent": {
                "prompt": "Test prompt",
            },
        }

    def test_valid_update_returns_empty_errors(self) -> None:
        """Should return empty error list for valid updates."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch", current_data, {"enabled": False}
        )
        assert errors == []

    def test_valid_trigger_update(self) -> None:
        """Should validate trigger updates successfully."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"trigger": {"project": "NEW-PROJ"}},
        )
        assert errors == []

    def test_invalid_agent_type_returns_error(self) -> None:
        """Should return error for invalid agent_type."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"agent": {"agent_type": "invalid-agent"}},
        )
        assert len(errors) == 1
        assert "agent_type" in errors[0]

    def test_invalid_cursor_mode_combo_returns_error(self) -> None:
        """Should return error for cursor_mode with non-cursor agent_type."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"agent": {"agent_type": "claude", "cursor_mode": "agent"}},
        )
        assert len(errors) == 1
        assert "cursor_mode" in errors[0]

    def test_invalid_trigger_source_returns_error(self) -> None:
        """Should return error for invalid trigger source."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"trigger": {"source": "gitlab"}},
        )
        assert len(errors) == 1
        assert "source" in errors[0].lower() or "trigger" in errors[0].lower()

    def test_valid_prompt_update(self) -> None:
        """Should validate prompt updates successfully."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"agent": {"prompt": "New prompt text"}},
        )
        assert errors == []

    def test_preserves_name_in_merged_data(self) -> None:
        """Should preserve the orchestration name during validation."""
        current_data = self._make_valid_orch_data()
        # Even if updates don't include name, validation should work
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"enabled": False},
        )
        assert errors == []


class TestOrchestrationEditResponse:
    """Tests for OrchestrationEditResponse model."""

    def test_success_response(self) -> None:
        """Should create success response with no errors."""
        response = OrchestrationEditResponse(success=True, name="test-orch")
        assert response.success is True
        assert response.name == "test-orch"
        assert response.errors == []

    def test_error_response(self) -> None:
        """Should create error response with error list."""
        response = OrchestrationEditResponse(
            success=False,
            name="test-orch",
            errors=["Invalid agent_type"],
        )
        assert response.success is False
        assert len(response.errors) == 1


class TestPydanticModelValidation:
    """Tests for Pydantic model field validation."""

    def test_trigger_source_literal(self) -> None:
        """Should accept valid trigger sources."""
        trigger = TriggerEditRequest(source="jira")
        assert trigger.source == "jira"

        trigger = TriggerEditRequest(source="github")
        assert trigger.source == "github"

    def test_agent_type_literal(self) -> None:
        """Should accept valid agent types."""
        agent = AgentEditRequest(agent_type="claude")
        assert agent.agent_type == "claude"

        agent = AgentEditRequest(agent_type="cursor")
        assert agent.agent_type == "cursor"

    def test_cursor_mode_literal(self) -> None:
        """Should accept valid cursor modes."""
        agent = AgentEditRequest(cursor_mode="agent")
        assert agent.cursor_mode == "agent"

    def test_default_status_literal(self) -> None:
        """Should accept valid default_status values."""
        retry = RetryEditRequest(default_status="success")
        assert retry.default_status == "success"

        retry = RetryEditRequest(default_status="failure")
        assert retry.default_status == "failure"

    def test_all_fields_optional(self) -> None:
        """All fields in OrchestrationEditRequest should be optional."""
        request = OrchestrationEditRequest()
        assert request.enabled is None
        assert request.max_concurrent is None
        assert request.trigger is None
        assert request.agent is None
        assert request.retry is None
        assert request.outcomes is None
        assert request.lifecycle is None


class TestOrchestrationCreateModels:
    """Tests for OrchestrationCreateRequest and OrchestrationCreateResponse models (DS-729)."""

    def test_create_request_required_fields(self) -> None:
        """Should require name and target_file fields."""
        from sentinel.dashboard.routes import OrchestrationCreateRequest
        request = OrchestrationCreateRequest(
            name="test-orch",
            target_file="test.yaml",
        )
        assert request.name == "test-orch"
        assert request.target_file == "test.yaml"

    def test_create_request_optional_fields(self) -> None:
        """Should have all optional configuration fields default to None."""
        from sentinel.dashboard.routes import OrchestrationCreateRequest
        request = OrchestrationCreateRequest(
            name="test-orch",
            target_file="test.yaml",
        )
        assert request.enabled is None
        assert request.max_concurrent is None
        assert request.trigger is None
        assert request.agent is None
        assert request.retry is None
        assert request.outcomes is None
        assert request.lifecycle is None

    def test_create_request_with_all_fields(self) -> None:
        """Should accept all configuration fields."""
        from sentinel.dashboard.routes import OrchestrationCreateRequest
        request = OrchestrationCreateRequest(
            name="test-orch",
            target_file="test.yaml",
            enabled=True,
            max_concurrent=3,
            trigger=TriggerEditRequest(source="jira", project="TEST"),
            agent=AgentEditRequest(prompt="Test prompt"),
            retry=RetryEditRequest(max_attempts=3),
        )
        assert request.name == "test-orch"
        assert request.enabled is True
        assert request.max_concurrent == 3
        assert request.trigger is not None
        assert request.agent is not None

    def test_create_response_success(self) -> None:
        """Should create success response with no errors."""
        from sentinel.dashboard.routes import OrchestrationCreateResponse
        response = OrchestrationCreateResponse(success=True, name="test-orch")
        assert response.success is True
        assert response.name == "test-orch"
        assert response.errors == []

    def test_create_response_with_errors(self) -> None:
        """Should create error response with error list."""
        from sentinel.dashboard.routes import OrchestrationCreateResponse
        response = OrchestrationCreateResponse(
            success=False,
            name="test-orch",
            errors=["Name already exists"],
        )
        assert response.success is False
        assert len(response.errors) == 1
