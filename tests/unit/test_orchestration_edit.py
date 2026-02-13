"""Tests for orchestration edit models and validation (DS-727).

Tests for the Pydantic models, _build_yaml_updates helper,
and _validate_orchestration_updates validation function.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sentinel.config import Config, DashboardConfig, ExecutionConfig
from sentinel.dashboard.models import (
    AgentEditRequest,
    FileGitHubEditRequest,
    FileTriggerEditRequest,
    GitHubContextEditRequest,
    LifecycleEditRequest,
    OrchestrationEditRequest,
    OrchestrationEditResponse,
    OutcomeEditRequest,
    RetryEditRequest,
    TriggerEditRequest,
)
from sentinel.dashboard.state import OrchestrationInfo, SentinelStateAccessor
from sentinel.orchestration_edit import (
    _DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
    _build_file_github_updates,
    _build_file_level_updates,
    _build_file_trigger_updates,
    _build_yaml_updates,
    _deep_merge_dicts,
    _extract_error_tokens,
    _get_dedup_threshold,
    _is_semantically_duplicate,
    _validate_orchestration_updates,
)
from tests.conftest import create_test_app
from tests.helpers import get_csrf_token


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
        """Should build trigger dict from TriggerEditRequest (step-level only)."""
        request = OrchestrationEditRequest(
            trigger=TriggerEditRequest(
                tags=["tag1", "tag2"],
                jql_filter="priority = High",
            )
        )
        updates = _build_yaml_updates(request)
        assert updates == {
            "trigger": {
                "tags": ["tag1", "tag2"],
                "jql_filter": "priority = High",
            }
        }

    def test_trigger_partial_update(self) -> None:
        """Should only include set trigger fields."""
        request = OrchestrationEditRequest(
            trigger=TriggerEditRequest(tags=["new-tag"])
        )
        updates = _build_yaml_updates(request)
        assert updates == {"trigger": {"tags": ["new-tag"]}}

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


class TestBuildFileLevelUpdates:
    """Tests for _build_file_level_updates generic helper (DS-1081)."""

    def test_file_trigger_request(self) -> None:
        """Should serialize FileTriggerEditRequest with non-None fields only."""
        request = FileTriggerEditRequest(source="jira", project="TEST")
        result = _build_file_level_updates(request)
        assert result == {"source": "jira", "project": "TEST"}

    def test_file_github_request(self) -> None:
        """Should serialize FileGitHubEditRequest with non-None fields only."""
        request = FileGitHubEditRequest(host="github.com", org="my-org", repo="my-repo")
        result = _build_file_level_updates(request)
        assert result == {"host": "github.com", "org": "my-org", "repo": "my-repo"}

    def test_empty_request_returns_empty_dict(self) -> None:
        """Should return empty dict when all fields are None."""
        request = FileGitHubEditRequest()
        result = _build_file_level_updates(request)
        assert result == {}

    def test_backwards_compatible_trigger_wrapper(self) -> None:
        """_build_file_trigger_updates should delegate to _build_file_level_updates."""
        request = FileTriggerEditRequest(source="github", project_owner="my-org")
        result = _build_file_trigger_updates(request)
        assert result == {"source": "github", "project_owner": "my-org"}

    def test_backwards_compatible_github_wrapper(self) -> None:
        """_build_file_github_updates should delegate to _build_file_level_updates."""
        request = FileGitHubEditRequest(branch="main", create_branch=True)
        result = _build_file_github_updates(request)
        assert result == {"branch": "main", "create_branch": True}

    def test_partial_file_github_fields(self) -> None:
        """Should only include non-None fields for partial updates."""
        request = FileGitHubEditRequest(org="updated-org")
        result = _build_file_level_updates(request)
        assert result == {"org": "updated-org"}

    def test_all_file_github_fields(self) -> None:
        """Should include all fields when all are set."""
        request = FileGitHubEditRequest(
            host="github.com",
            org="my-org",
            repo="my-repo",
            branch="main",
            create_branch=True,
            base_branch="develop",
        )
        result = _build_file_level_updates(request)
        assert result == {
            "host": "github.com",
            "org": "my-org",
            "repo": "my-repo",
            "branch": "main",
            "create_branch": True,
            "base_branch": "develop",
        }


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

    def test_multi_error_validation_trigger_and_agent(self) -> None:
        """Should collect errors from both trigger AND agent sections (DS-733)."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {
                "trigger": {"source": "gitlab"},
                "agent": {"agent_type": "invalid-agent"},
            },
        )
        # Should have exactly 2 errors - one for trigger, one for agent
        assert len(errors) == 2
        # Check that we have both trigger and agent errors
        error_text = " ".join(errors).lower()
        assert "trigger" in error_text or "source" in error_text
        assert "agent" in error_text

    def test_multi_error_validation_collects_all_errors(self) -> None:
        """Should collect errors from multiple sections independently (DS-733)."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {
                "trigger": {"source": "invalid-source"},
                "agent": {"agent_type": "invalid-type"},
                "max_concurrent": -1,
            },
        )
        # Should have multiple errors
        assert len(errors) == 3

    def test_intra_agent_multi_error(self) -> None:
        """Should collect multiple errors within the agent section (DS-734)."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {
                "agent": {
                    "agent_type": "invalid-type",
                    "timeout_seconds": -5,
                },
            },
        )
        # Both agent_type and timeout_seconds errors should be reported
        assert len(errors) == 2
        error_text = " ".join(errors).lower()
        assert "agent_type" in error_text
        assert "timeout_seconds" in error_text

    def test_intra_trigger_multi_error_github(self) -> None:
        """Should collect multiple errors within the trigger section (DS-734)."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {
                "trigger": {
                    "source": "github",
                    "project_number": -1,
                    "project_scope": "invalid-scope",
                    "project_owner": "",
                },
            },
        )
        # project_number, project_scope, and project_owner errors all reported
        assert len(errors) == 3
        error_text = " ".join(errors).lower()
        assert "project_number" in error_text
        assert "project_scope" in error_text
        assert "project_owner" in error_text

    def test_intra_and_cross_section_errors_combined(self) -> None:
        """Should collect errors from within and across sections (DS-734)."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {
                "trigger": {"source": "invalid-source"},
                "agent": {
                    "agent_type": "invalid-type",
                    "timeout_seconds": -1,
                },
                "max_concurrent": 0,
            },
        )
        # Cross-section: trigger error, max_concurrent error
        # Intra-section: two agent errors (agent_type + timeout_seconds)
        assert len(errors) == 4
        error_text = " ".join(errors).lower()
        assert "trigger" in error_text or "source" in error_text
        assert "agent_type" in error_text
        assert "timeout_seconds" in error_text
        assert "max_concurrent" in error_text

    def test_intra_agent_model_and_agent_type_errors(self) -> None:
        """Should report both model and agent_type errors simultaneously (DS-734)."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {
                "agent": {
                    "agent_type": "nonexistent",
                    "model": 12345,
                },
            },
        )
        assert len(errors) == 2
        error_text = " ".join(errors).lower()
        assert "agent_type" in error_text
        assert "model" in error_text

    def test_max_concurrent_boolean_true_rejected(self) -> None:
        """Should reject boolean True for max_concurrent (DS-762).

        isinstance(True, int) returns True in Python, so without an explicit
        boolean guard, True would pass the int check. This test verifies that
        the explicit boolean guard catches this case.
        """
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"max_concurrent": True},
        )
        assert len(errors) == 1
        assert any("max_concurrent" in e for e in errors)

    def test_max_concurrent_boolean_false_rejected(self) -> None:
        """Should reject boolean False for max_concurrent (DS-762).

        isinstance(False, int) returns True in Python, so without an explicit
        boolean guard, False would pass the int check (and then fail on < 1).
        This test verifies the explicit boolean guard catches False as well.
        """
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"max_concurrent": False},
        )
        assert len(errors) == 1
        assert any("max_concurrent" in e for e in errors)

    def test_cross_section_validation_runs_with_section_errors(self) -> None:
        """Should run full validation even when section-level errors exist (DS-762).

        Previously, _parse_orchestration() only ran when no section errors were
        found. This meant cross-section validation issues could be missed when
        section-level errors were present. Now full validation always runs with
        deduplication.
        """
        current_data = self._make_valid_orch_data()
        # Introduce a section-level error (invalid trigger source)
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"trigger": {"source": "gitlab"}},
        )
        # Should have exactly one error from the trigger section
        assert len(errors) == 1
        assert any("trigger" in e.lower() or "source" in e.lower() for e in errors)

    def test_cross_section_errors_deduplicated(self) -> None:
        """Should not duplicate errors from section and full validation (DS-762).

        When the same error is caught by both a section-specific validator and
        the full _parse_orchestration() call, it should only appear once.
        """
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"agent": {"agent_type": "invalid-agent"}},
        )
        # Errors should be unique (no exact duplicates)
        assert len(errors) == len(set(errors))

    def test_max_concurrent_valid_integer_accepted(self) -> None:
        """Should accept valid positive integer for max_concurrent (DS-762)."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"max_concurrent": 5},
        )
        assert errors == []


class TestExtractErrorTokens:
    """Tests for _extract_error_tokens helper function (DS-772)."""

    def test_strips_section_prefix(self) -> None:
        """Should strip section prefixes like 'Agent error: '."""
        tokens = _extract_error_tokens("Agent error: Invalid agent_type 'bad'")
        assert "agent" not in tokens  # "agent" from prefix should be stripped
        assert "invalid" in tokens
        assert "agent_type" in tokens

    def test_strips_trigger_prefix(self) -> None:
        """Should strip 'Trigger error: ' prefix."""
        tokens = _extract_error_tokens("Trigger error: Invalid source 'gitlab'")
        assert "invalid" in tokens
        assert "source" in tokens
        assert "gitlab" in tokens

    def test_strips_orchestration_name_qualifier(self) -> None:
        """Should strip 'Orchestration 'name' has ' qualifier."""
        tokens = _extract_error_tokens(
            "Orchestration 'test-orch' has invalid 'max_concurrent' value"
        )
        assert "invalid" in tokens
        assert "max_concurrent" in tokens
        # The orchestration name itself should be stripped
        assert "test" not in tokens or "orch" not in tokens

    def test_removes_noise_words(self) -> None:
        """Should remove noise words like 'must', 'be', 'a'."""
        tokens = _extract_error_tokens("value must be a positive integer")
        assert "must" not in tokens
        assert "be" not in tokens
        assert "positive" in tokens
        assert "integer" in tokens

    def test_removes_short_tokens(self) -> None:
        """Should remove single-character tokens."""
        tokens = _extract_error_tokens("a b c invalid agent_type")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "invalid" in tokens

    def test_empty_string(self) -> None:
        """Should return empty set for empty string."""
        tokens = _extract_error_tokens("")
        assert tokens == set()

    def test_normalizes_to_lowercase(self) -> None:
        """Should normalize tokens to lowercase."""
        tokens = _extract_error_tokens("Invalid AGENT_TYPE 'Claude'")
        assert "invalid" in tokens
        assert "agent_type" in tokens
        assert "claude" in tokens


class TestIsSemanticDuplicate:
    """Tests for _is_semantically_duplicate function (DS-772)."""

    def test_exact_substring_match_fast_path(self) -> None:
        """Should detect exact substring matches (original DS-762 behavior)."""
        existing = ["Agent error: Invalid agent_type 'bad': must be 'claude', 'codex', 'cursor'"]
        new_error = "Invalid agent_type 'bad': must be 'claude', 'codex', 'cursor'"
        assert _is_semantically_duplicate(new_error, existing) is True

    def test_semantically_equivalent_different_wording(self) -> None:
        """Should detect semantically equivalent errors with different wording."""
        existing = ["Invalid max_concurrent: must be positive integer, got True"]
        new_error = "Orchestration 'test-orch' has invalid 'max_concurrent' value: must be a positive integer"
        assert _is_semantically_duplicate(new_error, existing) is True

    def test_prefixed_vs_unprefixed(self) -> None:
        """Should detect prefix-stripped variants as duplicates."""
        existing = ["Trigger error: Invalid trigger source 'gitlab': must be 'github', 'jira'"]
        new_error = "Invalid trigger source 'gitlab': must be 'github', 'jira'"
        assert _is_semantically_duplicate(new_error, existing) is True

    def test_unrelated_errors_not_duplicate(self) -> None:
        """Should not flag unrelated errors as duplicates."""
        existing = ["Agent error: Invalid agent_type 'bad': must be 'claude', 'codex', 'cursor'"]
        new_error = "Invalid trigger source 'gitlab': must be 'github', 'jira'"
        assert _is_semantically_duplicate(new_error, existing) is False

    def test_empty_existing_list(self) -> None:
        """Should return False when no existing errors."""
        assert _is_semantically_duplicate("some error", []) is False

    def test_empty_new_error_tokens(self) -> None:
        """Should return True for empty new error via substring fast path.

        An empty string is a substring of any string in Python, so the
        fast-path substring check returns True.  Token extraction is not
        reached.  This edge case is acceptable because empty error messages
        never appear in practice.
        """
        assert _is_semantically_duplicate("", ["existing error"]) is True

    def test_different_field_errors_not_duplicate(self) -> None:
        """Should not treat errors about different fields as duplicates."""
        existing = ["Agent error: Invalid timeout_seconds '-5': must be a positive number"]
        new_error = "Invalid agent_type 'bad': must be 'claude', 'codex', 'cursor'"
        assert _is_semantically_duplicate(new_error, existing) is False

    def test_custom_threshold(self) -> None:
        """Should respect custom threshold parameter."""
        # Use messages where the overlap is partial (not all tokens shared)
        # so that a high threshold rejects them.
        existing = ["Retry error: Invalid default_status 'bad': must be 'success' or 'failure'"]
        new_error = "Invalid agent_type 'bad': must be 'claude', 'codex', 'cursor'"
        # With a very high threshold, partially-overlapping errors should not match
        assert _is_semantically_duplicate(new_error, existing, threshold=1.0) is False
        # With the default threshold, they should still not match (different fields)
        assert _is_semantically_duplicate(new_error, existing, threshold=0.6) is False

    def test_custom_threshold_high_overlap(self) -> None:
        """Should detect duplicates when token overlap meets threshold."""
        existing = ["Agent error: Invalid agent_type 'bad'"]
        new_error = "Invalid agent_type 'bad': extra details here"
        # After prefix stripping, existing tokens are {invalid, agent_type, bad}
        # new tokens are {invalid, agent_type, bad, extra, details, here}
        # overlap = 3, min_size = 3, ratio = 1.0 — matches even at threshold=1.0
        assert _is_semantically_duplicate(new_error, existing, threshold=1.0) is True
        assert _is_semantically_duplicate(new_error, existing, threshold=0.6) is True


class TestSemanticDeduplicationIntegration:
    """Integration tests for semantic deduplication in _validate_orchestration_updates (DS-772)."""

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

    def test_max_concurrent_boolean_deduplicated_semantically(self) -> None:
        """Should deduplicate max_concurrent errors with different wording (DS-772).

        The section-level validator says 'Invalid max_concurrent: must be
        positive integer, got True' while _parse_orchestration says
        'Orchestration 'test-orch' has invalid 'max_concurrent' value: must
        be a positive integer'. These should be deduplicated.
        """
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"max_concurrent": True},
        )
        # Should have exactly 1 error, not 2 (deduplicated)
        max_concurrent_errors = [e for e in errors if "max_concurrent" in e.lower()]
        assert len(max_concurrent_errors) == 1

    def test_agent_type_error_still_deduplicated(self) -> None:
        """Should still deduplicate agent_type errors (regression for DS-762)."""
        current_data = self._make_valid_orch_data()
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {"agent": {"agent_type": "invalid-agent"}},
        )
        # Errors should be unique (no duplicates)
        assert len(errors) == len(set(errors))

    def test_novel_cross_section_error_still_reported(self) -> None:
        """Should still report genuinely novel errors from full validation."""
        current_data = self._make_valid_orch_data()
        # Remove agent section entirely — section-level won't catch missing
        # agent, but _parse_orchestration will
        del current_data["agent"]
        errors = _validate_orchestration_updates(
            "test-orch",
            current_data,
            {},
        )
        # Should have at least one error about missing agent
        assert any("agent" in e.lower() for e in errors)


class TestGetDedupThreshold:
    """Tests for _get_dedup_threshold and env var configurability (DS-773)."""

    def test_returns_default_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return compiled-in default when SENTINEL_DEDUP_THRESHOLD is not set."""
        monkeypatch.delenv("SENTINEL_DEDUP_THRESHOLD", raising=False)
        assert _get_dedup_threshold() == _DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD

    def test_returns_default_when_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return default when SENTINEL_DEDUP_THRESHOLD is empty string."""
        monkeypatch.setenv("SENTINEL_DEDUP_THRESHOLD", "")
        assert _get_dedup_threshold() == _DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD

    def test_returns_default_when_env_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return default when SENTINEL_DEDUP_THRESHOLD is whitespace."""
        monkeypatch.setenv("SENTINEL_DEDUP_THRESHOLD", "   ")
        assert _get_dedup_threshold() == _DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD

    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            ("0.8", 0.8),     # valid float
            ("0.5", 0.5),     # mid-range passthrough (DS-793)
            ("1.5", 1.0),     # clamped above 1.0
            ("-0.3", 0.0),    # clamped negative to 0.0
            ("0.0", 0.0),     # boundary: lower end passes through (DS-774)
            ("1.0", 1.0),     # boundary: upper end passes through (DS-774)
        ],
        ids=[
            "valid_float",
            "midrange_passthrough",
            "clamps_above_one",
            "clamps_negative_to_zero",
            "boundary_zero",
            "boundary_one",
        ],
    )
    def test_parses_and_clamps_env_value(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str, expected: float,
    ) -> None:
        """Should parse SENTINEL_DEDUP_THRESHOLD and clamp to [0.0, 1.0]."""
        monkeypatch.setenv("SENTINEL_DEDUP_THRESHOLD", env_value)
        assert _get_dedup_threshold() == expected

    def test_returns_default_for_non_numeric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return default when env var is not a valid number."""
        monkeypatch.setenv("SENTINEL_DEDUP_THRESHOLD", "not-a-number")
        assert _get_dedup_threshold() == _DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD

    def test_logs_debug_on_invalid_env_var(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should emit a debug log when env var is set but non-numeric (DS-774).

        This aids operator troubleshooting when deduplication behaviour does not
        match expectations due to a misconfigured SENTINEL_DEDUP_THRESHOLD.
        """
        monkeypatch.setenv("SENTINEL_DEDUP_THRESHOLD", "not-a-number")
        with caplog.at_level(logging.DEBUG, logger="sentinel.orchestration_edit"):
            result = _get_dedup_threshold()
        assert result == _DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.DEBUG
        assert "not-a-number" in record.message
        assert "falling back" in record.message.lower()

    def test_env_var_affects_deduplication(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use env-var threshold when no explicit threshold is passed.

        With threshold=1.0, partially overlapping errors should NOT match
        unless every token in the shorter set is present in the longer set.
        """
        monkeypatch.setenv("SENTINEL_DEDUP_THRESHOLD", "1.0")
        # These messages share some tokens (invalid, bad) but differ in the
        # domain-specific tokens (agent_type vs trigger/source/gitlab), so
        # at threshold=1.0 the overlap ratio < 1.0 and they should not match.
        existing = ["Retry error: Invalid default_status 'bad': must be 'success' or 'failure'"]
        new_error = "Invalid agent_type 'bad': must be 'claude', 'codex', 'cursor'"
        assert _is_semantically_duplicate(new_error, existing) is False

        # Now lower the threshold via env var so the same pair DOES match.
        monkeypatch.setenv("SENTINEL_DEDUP_THRESHOLD", "0.1")
        assert _is_semantically_duplicate(new_error, existing) is True

    def test_explicit_threshold_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit threshold parameter should take precedence over env var."""
        monkeypatch.setenv("SENTINEL_DEDUP_THRESHOLD", "1.0")
        existing = ["Agent error: Invalid agent_type 'bad'"]
        new_error = "Invalid agent_type 'bad': extra details here"
        # Env says 1.0 but explicit kwarg says 0.6 — should match
        assert _is_semantically_duplicate(new_error, existing, threshold=0.6) is True


class TestCollectTriggerErrors:
    """Tests for _collect_trigger_errors function (DS-734)."""

    def test_valid_jira_trigger(self) -> None:
        """Should return empty list for valid Jira trigger."""
        from sentinel.orchestration import _collect_trigger_errors
        errors = _collect_trigger_errors({"source": "jira", "project": "TEST"})
        assert errors == []

    def test_invalid_source(self) -> None:
        """Should report invalid source."""
        from sentinel.orchestration import _collect_trigger_errors
        errors = _collect_trigger_errors({"source": "gitlab"})
        assert len(errors) == 1
        assert "source" in errors[0].lower()

    def test_github_multiple_errors(self) -> None:
        """Should collect multiple GitHub trigger errors at once."""
        from sentinel.orchestration import _collect_trigger_errors
        errors = _collect_trigger_errors({
            "source": "github",
            "project_number": -1,
            "project_scope": "invalid",
            "project_owner": "",
        })
        assert len(errors) == 3

    def test_invalid_tags_and_labels(self) -> None:
        """Should report invalid tags and labels."""
        from sentinel.orchestration import _collect_trigger_errors
        errors = _collect_trigger_errors({
            "source": "jira",
            "tags": "not-a-list",
            "labels": "also-not-a-list",
        })
        assert len(errors) == 2


class TestCollectAgentErrors:
    """Tests for _collect_agent_errors function (DS-734)."""

    def test_valid_agent(self) -> None:
        """Should return empty list for valid agent config."""
        from sentinel.orchestration import _collect_agent_errors
        errors = _collect_agent_errors({"prompt": "Test", "agent_type": "claude"})
        assert errors == []

    def test_multiple_field_errors(self) -> None:
        """Should collect errors from multiple agent fields."""
        from sentinel.orchestration import _collect_agent_errors
        errors = _collect_agent_errors({
            "agent_type": "invalid",
            "timeout_seconds": -10,
            "model": 999,
        })
        assert len(errors) == 3
        error_text = " ".join(errors).lower()
        assert "agent_type" in error_text
        assert "timeout_seconds" in error_text
        assert "model" in error_text

    def test_cursor_mode_with_invalid_agent_type(self) -> None:
        """Should report cursor_mode incompatibility with claude agent_type."""
        from sentinel.orchestration import _collect_agent_errors
        errors = _collect_agent_errors({
            "agent_type": "claude",
            "cursor_mode": "agent",
        })
        assert len(errors) == 1
        assert "cursor_mode" in errors[0].lower()

    def test_invalid_cursor_mode_value(self) -> None:
        """Should report invalid cursor_mode value."""
        from sentinel.orchestration import _collect_agent_errors
        errors = _collect_agent_errors({
            "agent_type": "cursor",
            "cursor_mode": "invalid-mode",
        })
        assert len(errors) == 1
        assert "cursor_mode" in errors[0].lower()

    def test_invalid_strict_template_variables(self) -> None:
        """Should report invalid strict_template_variables."""
        from sentinel.orchestration import _collect_agent_errors
        errors = _collect_agent_errors({
            "strict_template_variables": "not-a-bool",
        })
        assert len(errors) == 1
        assert "strict_template_variables" in errors[0].lower()


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
        """Should accept valid trigger sources via FileTriggerEditRequest (DS-896)."""
        from sentinel.dashboard.models import FileTriggerEditRequest

        trigger = FileTriggerEditRequest(source="jira")
        assert trigger.source == "jira"

        trigger = FileTriggerEditRequest(source="github")
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
        from sentinel.dashboard.models import OrchestrationCreateRequest
        request = OrchestrationCreateRequest(
            name="test-orch",
            target_file="test.yaml",
        )
        assert request.name == "test-orch"
        assert request.target_file == "test.yaml"

    def test_create_request_optional_fields(self) -> None:
        """Should have all optional configuration fields default to None."""
        from sentinel.dashboard.models import OrchestrationCreateRequest
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
        from sentinel.dashboard.models import OrchestrationCreateRequest
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
        from sentinel.dashboard.models import OrchestrationCreateResponse
        response = OrchestrationCreateResponse(success=True, name="test-orch")
        assert response.success is True
        assert response.name == "test-orch"
        assert response.errors == []

    def test_create_response_with_errors(self) -> None:
        """Should create error response with error list."""
        from sentinel.dashboard.models import OrchestrationCreateResponse
        response = OrchestrationCreateResponse(
            success=False,
            name="test-orch",
            errors=["Name already exists"],
        )
        assert response.success is False
        assert len(response.errors) == 1


@pytest.fixture
def temp_logs_dir() -> Generator[Path]:
    """Fixture that provides a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class MockSentinel:
    """Mock Sentinel instance for testing dashboard state accessor."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._shutdown_requested = False

    @property
    def orchestrations(self) -> list[Any]:
        return []

    def get_hot_reload_metrics(self) -> dict[str, int]:
        return {
            "orchestrations_loaded_total": 0,
            "orchestrations_unloaded_total": 0,
            "orchestrations_reloaded_total": 0,
        }

    def get_running_steps(self) -> list[Any]:
        return []

    def get_issue_queue(self) -> list[Any]:
        return []

    def get_start_time(self) -> Any:
        from datetime import datetime
        return datetime.now()

    def get_last_jira_poll(self) -> Any:
        return None

    def get_last_github_poll(self) -> Any:
        return None

    def get_active_versions(self) -> list[Any]:
        return []

    def get_pending_removal_versions(self) -> list[Any]:
        return []

    def get_execution_state(self) -> Any:
        from sentinel.dashboard.state import ExecutionStateSnapshot
        return ExecutionStateSnapshot(active_count=0)

    def get_completed_executions(self) -> list[Any]:
        return []

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested


class MockSentinelWithOrchestrations(MockSentinel):
    """Mock Sentinel with configurable orchestrations for edit endpoint tests."""

    def __init__(self, config: Config, orchestrations_data: list[OrchestrationInfo]) -> None:
        super().__init__(config)
        self._orchestrations_data = orchestrations_data

    @property
    def orchestrations(self) -> list[Any]:
        """Return mock orchestrations for state accessor."""
        return []


class MockStateAccessorWithOrchestrations(SentinelStateAccessor):
    """Mock state accessor that returns configured orchestration info."""

    def __init__(self, sentinel: Any, orchestrations_data: list[OrchestrationInfo]) -> None:
        self._sentinel = sentinel
        self._orchestrations_data = orchestrations_data

    def get_state(self) -> Any:
        """Return a mock state with the configured orchestrations."""
        from sentinel.dashboard.state import DashboardState

        return DashboardState(
            poll_interval=60,
            max_concurrent_executions=5,
            max_issues_per_poll=10,
            orchestrations=self._orchestrations_data,
        )


class TestEditOrchestrationEndpoint:
    """Integration tests for PUT /api/orchestrations/{name} endpoint (DS-733)."""

    def test_successful_edit(self, temp_logs_dir: Path) -> None:
        """Test successful orchestration edit returns 200."""
        # Create a temporary YAML file with a valid orchestration
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Original prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Original prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Get CSRF token (DS-926)
            csrf_token = get_csrf_token(client)

            response = client.put(
                "/api/orchestrations/test-orch",
                json={"agent": {"prompt": "Updated prompt"}},
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["name"] == "test-orch"
            assert data["errors"] == []

            # Verify file was updated
            updated_content = orch_file.read_text()
            assert "Updated prompt" in updated_content

    def test_validation_error(self, temp_logs_dir: Path) -> None:
        """Test validation error returns 422 with error details."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Get CSRF token (DS-926)
            csrf_token = get_csrf_token(client)

            response = client.put(
                "/api/orchestrations/test-orch",
                json={"agent": {"agent_type": "invalid-agent"}},
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 422
            detail = response.json()["detail"]
            # Detail should be a list of errors
            assert isinstance(detail, list)
            assert len(detail) == 1
            # Extract error text (handle both string list and dict list formats)
            if isinstance(detail[0], str):
                error_text = " ".join(detail).lower()
            else:
                # FastAPI may wrap as ValidationError format with 'msg' field
                error_text = " ".join(str(item) for item in detail).lower()
            assert "agent" in error_text

    def test_multi_error_validation(self, temp_logs_dir: Path) -> None:
        """Test multiple validation errors are returned together (DS-733).

        After DS-896, TriggerEditRequest no longer has 'source' field (moved to
        FileTriggerEditRequest). This test now uses max_concurrent and agent
        timeout_seconds to generate multiple orchestration validation errors.
        """
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Get CSRF token (DS-926)
            csrf_token = get_csrf_token(client)

            # Send updates with errors in multiple sections (orchestration-level validation)
            response = client.put(
                "/api/orchestrations/test-orch",
                json={
                    "max_concurrent": -1,
                    "agent": {"timeout_seconds": -5},
                },
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 422
            detail = response.json()["detail"]
            # Should return multiple errors
            assert isinstance(detail, list)
            assert len(detail) == 2
            # Extract error text (handle both string list and dict list formats)
            if isinstance(detail[0], str):
                error_text = " ".join(detail).lower()
            else:
                # FastAPI may wrap as ValidationError format
                error_text = " ".join(str(item) for item in detail).lower()
            assert "max_concurrent" in error_text
            assert "timeout_seconds" in error_text

    def test_intra_section_multi_error_via_api(self, temp_logs_dir: Path) -> None:
        """Test intra-section multi-error reporting via PUT endpoint (DS-734).

        Uses values that pass Pydantic model validation (valid Literal types)
        but fail orchestration-level validation (invalid combinations and
        out-of-range integers). This ensures we test the multi-error
        collectors in orchestration_edit, not Pydantic schema validation.
        """
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Get CSRF token (DS-926)
            csrf_token = get_csrf_token(client)

            # cursor_mode="agent" with agent_type="claude" is invalid combo,
            # timeout_seconds=-5 is out-of-range — both pass Pydantic but
            # fail orchestration validation, testing intra-section collection.
            response = client.put(
                "/api/orchestrations/test-orch",
                json={
                    "agent": {
                        "agent_type": "claude",
                        "cursor_mode": "agent",
                        "timeout_seconds": -5,
                    },
                },
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 422
            detail = response.json()["detail"]
            # Should return multiple errors from the same section
            assert isinstance(detail, list)
            assert len(detail) == 2
            if isinstance(detail[0], str):
                error_text = " ".join(detail).lower()
            else:
                error_text = " ".join(str(item) for item in detail).lower()
            assert "cursor_mode" in error_text
            assert "timeout_seconds" in error_text

    def test_not_found(self, temp_logs_dir: Path) -> None:
        """Test 404 when orchestration doesn't exist."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        # No orchestrations configured
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Get CSRF token (DS-926)
            csrf_token = get_csrf_token(client)

            response = client.put(
                "/api/orchestrations/nonexistent",
                json={"enabled": False},
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_rate_limiting(self, temp_logs_dir: Path) -> None:
        """Test rate limiting with two rapid edits returns 429."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        # Use a short cooldown for testing
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
            dashboard=DashboardConfig(toggle_cooldown_seconds=2.0),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # Get CSRF token (DS-926)
            csrf_token = get_csrf_token(client)

            # First edit should succeed
            response1 = client.put(
                "/api/orchestrations/test-orch",
                json={"enabled": False},
                headers={"X-CSRF-Token": csrf_token},
            )
            assert response1.status_code == 200

            # Get fresh CSRF token for second request (DS-926)
            csrf_resp2 = client.get("/api/csrf-token")
            csrf_token2 = csrf_resp2.json()["csrf_token"]

            # Immediate second edit should be rate limited
            response2 = client.put(
                "/api/orchestrations/test-orch",
                json={"enabled": True},
                headers={"X-CSRF-Token": csrf_token2},
            )
            assert response2.status_code == 429
            assert "Rate limit exceeded" in response2.json()["detail"]

    def test_empty_request(self, temp_logs_dir: Path) -> None:
        """Test empty request returns 200 with no changes."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Get CSRF token (DS-926)
            csrf_token = get_csrf_token(client)

            response = client.put(
                "/api/orchestrations/test-orch",
                json={},
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["name"] == "test-orch"
