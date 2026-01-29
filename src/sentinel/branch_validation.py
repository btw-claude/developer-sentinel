"""Shared branch name validation utilities.

This module provides a core ValidationResult-based branch name validator that can be
used by both config.py (for environment variable validation) and orchestration.py
(for orchestration file validation). This consolidates the duplicate _validate_branch_name
functions that previously existed in both modules.

The core validator returns a ValidationResult, and wrapper functions provide
context-specific behaviors (e.g., logging warnings and returning defaults for config.py).
"""

from __future__ import annotations

import re
from typing import NamedTuple


# Pre-compiled regex pattern for invalid branch characters
# Branch name invalid characters: space, ~, ^, :, ?, *, [, ], \, @{
_BRANCH_INVALID_CHARS = re.compile(r"[ ~^:?*\[\]\\]|@\{")


class BranchValidationResult(NamedTuple):
    """Result of a branch name validation operation.

    Attributes:
        is_valid: True if the branch name is valid, False otherwise.
        error_message: Description of the validation failure, empty if valid.
    """

    is_valid: bool
    error_message: str

    @classmethod
    def success(cls) -> "BranchValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True, error_message="")

    @classmethod
    def failure(cls, message: str) -> "BranchValidationResult":
        """Create a failed validation result with the given error message."""
        return cls(is_valid=False, error_message=message)


def validate_branch_name_core(
    branch: str,
    *,
    allow_empty: bool = False,
    allow_template_variables: bool = False,
) -> BranchValidationResult:
    """Core branch name validation following git branch naming rules.

    Git branch naming rules enforced:
    - Cannot start with a hyphen (-) or period (.)
    - Cannot end with a period (.), forward slash (/), or .lock
    - Cannot contain: space, ~, ^, :, ?, *, [, ], \\, @{
    - Cannot contain consecutive periods (..) or forward slashes (//)
    - Cannot be empty (unless allow_empty=True)

    Args:
        branch: The branch name to validate.
        allow_empty: If True, empty strings return success (for optional fields).
        allow_template_variables: If True, template variables like {jira_issue_key}
            are preserved and only the static parts are validated. This is useful
            for validating branch patterns before runtime substitution.

    Returns:
        BranchValidationResult indicating success or failure with error message.
    """
    # Handle empty input
    if not branch:
        if allow_empty:
            return BranchValidationResult.success()
        return BranchValidationResult.failure("branch name cannot be empty")

    # Strip whitespace for non-empty strings
    branch = branch.strip()
    if not branch:
        if allow_empty:
            return BranchValidationResult.success()
        return BranchValidationResult.failure("branch name cannot be empty")

    # For branch patterns with template variables, validate the static parts
    # by checking each segment between template variables
    # e.g., "feature/{jira_issue_key}/test" -> check "feature/", "/test"
    if allow_template_variables:
        static_parts = re.split(r"\{[^}]+\}", branch)
    else:
        static_parts = [branch]

    # Check for invalid starting characters
    if branch.startswith("-") or branch.startswith("."):
        return BranchValidationResult.failure(
            "branch name cannot start with '-' or '.'"
        )

    # Check for invalid ending characters
    if branch.endswith(".") or branch.endswith("/"):
        return BranchValidationResult.failure(
            "branch name cannot end with '.' or '/'"
        )

    # Check for .lock ending (git disallows this)
    if branch.endswith(".lock"):
        return BranchValidationResult.failure(
            "branch name cannot end with '.lock'"
        )

    # Check for invalid characters in static parts
    for part in static_parts:
        if _BRANCH_INVALID_CHARS.search(part):
            return BranchValidationResult.failure(
                "branch name contains invalid characters "
                "(space, ~, ^, :, ?, *, [, ], \\, or @{)"
            )

    # Check for consecutive periods or slashes
    if ".." in branch or "//" in branch:
        return BranchValidationResult.failure(
            "branch name cannot contain consecutive periods (..) or slashes (//)"
        )

    return BranchValidationResult.success()


def validate_runtime_branch_name(branch: str) -> BranchValidationResult:
    """Validate a fully-resolved branch name at runtime.

    This function validates branch names after template variable substitution
    has occurred. It applies all git branch naming rules without allowing
    template variables.

    This should be called in the executor after expanding branch patterns
    with actual values from the issue context.

    Args:
        branch: The fully-resolved branch name to validate.

    Returns:
        BranchValidationResult indicating success or failure with error message.
    """
    return validate_branch_name_core(branch, allow_empty=False, allow_template_variables=False)
