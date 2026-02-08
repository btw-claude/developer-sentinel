"""Orchestration edit helpers for YAML updates and validation.

This module provides functions for building YAML updates from Pydantic models
and validating orchestration updates. Extracted from dashboard.routes (DS-733).

Functions:
    _build_yaml_updates: Convert OrchestrationEditRequest to YAML-compatible dict
    _validate_orchestration_updates: Validate merged orchestration data
    _deep_merge_dicts: Deep merge two dictionaries
"""

from __future__ import annotations

from typing import Any

from sentinel.dashboard.models import OrchestrationEditRequest
from sentinel.orchestration import (
    OrchestrationError,
    _parse_agent,
    _parse_on_complete,
    _parse_on_failure,
    _parse_on_start,
    _parse_orchestration,
    _parse_outcomes,
    _parse_retry,
    _parse_trigger,
)


def _build_yaml_updates(request: OrchestrationEditRequest) -> dict[str, Any]:
    """Convert an OrchestrationEditRequest to a YAML-compatible update dict.

    Uses Pydantic v2 model_dump(exclude_none=True) for top-level and nested
    models to avoid manual is-not-None checks. The lifecycle section requires
    special post-processing to map flat Pydantic fields to the nested YAML
    structure (on_start, on_complete, on_failure).

    Args:
        request: The edit request containing fields to update.

    Returns:
        A dictionary mirroring the YAML structure with only the
        fields that should be updated.
    """
    # Use model_dump to get all non-None fields automatically
    raw = request.model_dump(exclude_none=True)

    updates: dict[str, Any] = {}

    # Copy top-level scalar fields directly
    for key in ("enabled", "max_concurrent"):
        if key in raw:
            updates[key] = raw[key]

    # Nested models: model_dump already excludes None recursively
    if "trigger" in raw and raw["trigger"]:
        updates["trigger"] = raw["trigger"]

    if "agent" in raw and raw["agent"]:
        updates["agent"] = raw["agent"]

    if "retry" in raw and raw["retry"]:
        updates["retry"] = raw["retry"]

    if "outcomes" in raw:
        updates["outcomes"] = raw["outcomes"]

    # Lifecycle requires special handling: flat Pydantic fields map to nested YAML
    if "lifecycle" in raw and raw["lifecycle"]:
        lc = raw["lifecycle"]
        if "on_start_add_tag" in lc:
            updates.setdefault("on_start", {})["add_tag"] = lc["on_start_add_tag"]
        if "on_complete_remove_tag" in lc:
            updates.setdefault("on_complete", {})["remove_tag"] = lc["on_complete_remove_tag"]
        if "on_complete_add_tag" in lc:
            updates.setdefault("on_complete", {})["add_tag"] = lc["on_complete_add_tag"]
        if "on_failure_add_tag" in lc:
            updates.setdefault("on_failure", {})["add_tag"] = lc["on_failure_add_tag"]

    return updates


def _validate_orchestration_updates(
    orch_name: str,
    current_data: dict[str, Any],
    updates: dict[str, Any],
) -> list[str]:
    """Validate orchestration updates by checking each section independently.

    Merges the updates into a copy of the current orchestration data and
    validates each section (trigger, agent, retry, outcomes, lifecycle) separately.
    This allows collecting multiple validation errors in a single request instead
    of failing on the first error.

    Note: Disabled orchestrations (enabled: false) are not in the active state
    and will return 404 when attempting to edit. To edit a disabled orchestration,
    first re-enable it via the YAML file directly.

    Args:
        orch_name: The name of the orchestration being updated.
        current_data: The current orchestration data as a plain dict.
        updates: The updates to merge.

    Returns:
        A list of validation error messages. Empty list means validation passed.
        Multiple errors may be returned if multiple sections have validation issues.
    """
    # Deep merge updates into a copy of current data
    merged = _deep_merge_dicts(current_data, updates)

    # Ensure name is preserved in merged data
    merged["name"] = orch_name

    # Collect errors from each section independently
    errors: list[str] = []

    # Validate top-level fields
    if "enabled" in merged and not isinstance(merged["enabled"], bool):
        errors.append(f"Invalid enabled value: must be boolean, got {type(merged['enabled'])}")

    if "max_concurrent" in merged:
        max_conc = merged["max_concurrent"]
        if max_conc is not None and (not isinstance(max_conc, int) or max_conc < 1):
            errors.append(f"Invalid max_concurrent: must be positive integer, got {max_conc}")

    # Validate trigger section
    if "trigger" in merged:
        try:
            _parse_trigger(merged["trigger"])
        except OrchestrationError as e:
            errors.append(f"Trigger error: {e}")

    # Validate agent section
    if "agent" in merged:
        try:
            _parse_agent(merged["agent"])
        except OrchestrationError as e:
            errors.append(f"Agent error: {e}")

    # Validate retry section
    if "retry" in merged:
        try:
            _parse_retry(merged.get("retry"))
        except OrchestrationError as e:
            errors.append(f"Retry error: {e}")

    # Validate outcomes section
    if "outcomes" in merged:
        try:
            _parse_outcomes(merged.get("outcomes"))
        except OrchestrationError as e:
            errors.append(f"Outcomes error: {e}")

    # Validate on_start section
    if "on_start" in merged:
        try:
            _parse_on_start(merged.get("on_start"))
        except OrchestrationError as e:
            errors.append(f"on_start error: {e}")

    # Validate on_complete section
    if "on_complete" in merged:
        try:
            _parse_on_complete(merged.get("on_complete"))
        except OrchestrationError as e:
            errors.append(f"on_complete error: {e}")

    # Validate on_failure section
    if "on_failure" in merged:
        try:
            _parse_on_failure(merged.get("on_failure"))
        except OrchestrationError as e:
            errors.append(f"on_failure error: {e}")

    # If no errors so far, run full validation as safety net
    if not errors:
        try:
            _parse_orchestration(merged)
        except OrchestrationError as e:
            errors.append(str(e))

    return errors


def _deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep merge updates into a copy of base dict.

    Creates a new dict with base values overridden by updates.
    Nested dicts are recursively merged. Lists are replaced entirely,
    not appended or element-wise merged.

    Args:
        base: The base dictionary to merge into.
        updates: The updates to apply.

    Returns:
        A new dictionary with merged values.
    """
    result = dict(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
