"""Orchestration edit helpers for YAML updates and validation.

This module provides functions for building YAML updates from Pydantic models
and validating orchestration updates. Extracted from dashboard.routes (DS-733).

Validation uses section-specific error collectors (DS-734) that gather all
validation errors within a section before returning, so the API can report
every issue in a single response instead of one error at a time.

Cross-section validation (DS-762): Full _parse_orchestration() validation
always runs regardless of section-level errors, with deduplication to avoid
reporting the same error twice. The max_concurrent check includes an explicit
boolean guard since isinstance(True, int) returns True in Python.

Semantic error deduplication (DS-772): Error deduplication uses semantic
matching via normalized token overlap rather than simple substring matching.
This identifies conceptually equivalent errors even when the error message
wording differs between the section-specific validator and the full
_parse_orchestration() validator (e.g., different prefixes, varied phrasing,
or orchestration-name-qualified messages).

Functions:
    _build_yaml_updates: Convert OrchestrationEditRequest to YAML-compatible dict
    _validate_orchestration_updates: Validate merged orchestration data
    _deep_merge_dicts: Deep merge two dictionaries
    _is_semantically_duplicate: Check if an error is semantically equivalent to
        any existing error using normalized token overlap
    _extract_error_tokens: Extract meaningful tokens from an error message for
        semantic comparison
"""

from __future__ import annotations

import re
from typing import Any

from sentinel.dashboard.models import OrchestrationEditRequest
from sentinel.orchestration import (
    OrchestrationError,
    _collect_agent_errors,
    _collect_trigger_errors,
    _parse_on_complete,
    _parse_on_failure,
    _parse_on_start,
    _parse_orchestration,
    _parse_outcomes,
    _parse_retry,
)

# Threshold for token overlap ratio to consider two errors semantically
# equivalent.  A value of 0.6 means that if 60% or more of the meaningful
# tokens in the shorter message also appear in the longer message, the two
# errors are considered duplicates.  This is intentionally conservative
# enough to avoid false positives while still catching common reformulations
# (e.g., prefixed vs unprefixed, orchestration-name-qualified variants).
_SEMANTIC_SIMILARITY_THRESHOLD = 0.6

# Section-prefix pattern: strips prefixes like "Agent error: ", "Trigger error: "
# that are added by section-level validation in _validate_orchestration_updates.
_SECTION_PREFIX_RE = re.compile(
    r"^(?:Agent|Trigger|Retry|Outcomes|on_start|on_complete|on_failure)\s+error:\s*",
    re.IGNORECASE,
)

# Orchestration-name qualifier pattern: strips prefixes like
# "Orchestration 'my-orch' has " that _parse_orchestration adds.
_ORCH_NAME_RE = re.compile(
    r"^Orchestration\s+'[^']+'\s+(?:has\s+)?",
    re.IGNORECASE,
)

# Noise words that carry no semantic meaning for error comparison.
_NOISE_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "got", "must", "should", "value",
})


def _extract_error_tokens(message: str) -> set[str]:
    """Extract meaningful, normalised tokens from an error message (DS-772).

    Processing steps:
    1. Strip common section prefixes ("Agent error: ", etc.)
    2. Strip orchestration-name qualifiers ("Orchestration 'name' has ")
    3. Lowercase and split on non-alphanumeric boundaries
    4. Remove noise words and very short tokens (length <= 1)

    The resulting token set captures the semantic content of the error so that
    two messages describing the same validation failure will share a high
    proportion of tokens regardless of superficial wording differences.

    Args:
        message: The raw error message string.

    Returns:
        A set of normalised, meaningful tokens.
    """
    # Strip section prefix (e.g., "Agent error: ")
    stripped = _SECTION_PREFIX_RE.sub("", message)

    # Strip orchestration-name qualifier (e.g., "Orchestration 'test-orch' has ")
    stripped = _ORCH_NAME_RE.sub("", stripped)

    # Lowercase and tokenize on non-alphanumeric boundaries
    tokens = set(re.findall(r"[a-z0-9_]+", stripped.lower()))

    # Remove noise words and very short tokens
    tokens -= _NOISE_WORDS
    tokens = {t for t in tokens if len(t) > 1}

    return tokens


def _is_semantically_duplicate(
    new_error: str,
    existing_errors: list[str],
    threshold: float = _SEMANTIC_SIMILARITY_THRESHOLD,
) -> bool:
    """Check whether *new_error* is semantically equivalent to any existing error (DS-772).

    Two errors are considered semantically equivalent when the token overlap
    ratio (intersection / min-set-size) meets or exceeds *threshold*.  Using
    the **minimum** set size as the denominator means that the shorter message
    only needs to be well-represented inside the longer one, which is the
    correct behaviour when one message is a prefix-stripped variant of the other.

    The function also retains the original substring check from DS-762 as a
    fast path so that exact substring matches are still caught without the
    overhead of tokenisation.

    Args:
        new_error: The candidate error message to check.
        existing_errors: Already-collected error messages.
        threshold: Minimum overlap ratio (0.0–1.0) to treat as duplicate.

    Returns:
        ``True`` if *new_error* is a semantic duplicate of any entry in
        *existing_errors*.
    """
    # Fast path: exact substring match (original DS-762 behaviour)
    if any(new_error in existing for existing in existing_errors):
        return True

    new_tokens = _extract_error_tokens(new_error)
    if not new_tokens:
        return False

    for existing in existing_errors:
        existing_tokens = _extract_error_tokens(existing)
        if not existing_tokens:
            continue

        overlap = len(new_tokens & existing_tokens)
        min_size = min(len(new_tokens), len(existing_tokens))

        if overlap / min_size >= threshold:
            return True

    return False


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
        if max_conc is not None and (
            isinstance(max_conc, bool) or not isinstance(max_conc, int) or max_conc < 1
        ):
            errors.append(f"Invalid max_concurrent: must be positive integer, got {max_conc}")

    # Validate trigger section — use collector to report all errors (DS-734)
    if "trigger" in merged:
        trigger_errors = _collect_trigger_errors(merged["trigger"])
        errors.extend(f"Trigger error: {e}" for e in trigger_errors)

    # Validate agent section — use collector to report all errors (DS-734)
    if "agent" in merged:
        agent_errors = _collect_agent_errors(merged["agent"])
        errors.extend(f"Agent error: {e}" for e in agent_errors)

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

    # Always run full validation as safety net to catch cross-section issues
    # (e.g., interactions between trigger and agent config) that section-specific
    # validators may miss. Deduplicate against already-collected errors using
    # semantic matching (DS-772) so that conceptually equivalent errors are
    # suppressed even when the wording differs between section-specific and
    # full-validation error messages.
    try:
        _parse_orchestration(merged)
    except OrchestrationError as e:
        full_error = str(e)
        if not _is_semantically_duplicate(full_error, errors):
            errors.append(full_error)

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
