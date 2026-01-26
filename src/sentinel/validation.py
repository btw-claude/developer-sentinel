"""Validation helper functions for environment variable parsing.

This module provides reusable validation functions for parsing and validating
environment variables with proper error handling and bounds checking.

DS-282: Consolidate validation helpers from routes.py into a dedicated module.
DS-285: Add __all__ to explicitly define the public API.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Explicitly define public API (DS-285)
# This makes it clear which symbols are intended for external use
__all__ = [
    # Bounds constants for rate limiting configuration
    "MIN_TOGGLE_COOLDOWN",
    "MAX_TOGGLE_COOLDOWN",
    "MIN_CACHE_TTL",
    "MAX_CACHE_TTL",
    "MIN_CACHE_MAXSIZE",
    "MAX_CACHE_MAXSIZE",
    # Validation functions
    "validate_positive_float",
    "validate_strictly_positive_float",
    "validate_positive_int",
]


# Bounds constants for rate limiting configuration (DS-278, DS-282)
# These bounds help catch configuration errors and prevent resource exhaustion
MIN_TOGGLE_COOLDOWN: float = 0.0  # Allow 0 to effectively disable cooldown
MAX_TOGGLE_COOLDOWN: float = 86400.0  # 24 hours is a reasonable upper bound
MIN_CACHE_TTL: int = 1  # At least 1 second
MAX_CACHE_TTL: int = 604800  # 1 week
MIN_CACHE_MAXSIZE: int = 1  # At least 1 entry
MAX_CACHE_MAXSIZE: int = 1000000  # 1 million entries


def validate_positive_float(
    env_var: str, value_str: str, default: float, min_val: float, max_val: float
) -> float:
    """Validate and parse a non-negative float environment variable (DS-278, DS-282).

    This function accepts values >= 0 (non-negative). Use validate_strictly_positive_float
    for values that must be > 0.

    Args:
        env_var: Name of the environment variable (for error messages).
        value_str: String value to parse.
        default: Default value if parsing fails.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Validated float value, or default if validation fails.
    """
    try:
        value = float(value_str)
        if value < 0:
            logger.error(
                "%s must be non-negative, got %s. Using default: %s",
                env_var, value, default
            )
            return default
        if value < min_val:
            logger.warning(
                "%s value %s is below minimum %s. Using minimum value.",
                env_var, value, min_val
            )
            return min_val
        if value > max_val:
            logger.warning(
                "%s value %s exceeds maximum %s. Using maximum value.",
                env_var, value, max_val
            )
            return max_val
        return value
    except ValueError:
        logger.error(
            "%s must be a valid float, got '%s'. Using default: %s",
            env_var, value_str, default
        )
        return default


def validate_strictly_positive_float(
    env_var: str, value_str: str, default: float, min_val: float, max_val: float
) -> float:
    """Validate and parse a strictly positive float environment variable (DS-282).

    This function requires values > 0 (strictly positive), paralleling the
    validate_positive_int function behavior. Use validate_positive_float for
    values that can be >= 0.

    Args:
        env_var: Name of the environment variable (for error messages).
        value_str: String value to parse.
        default: Default value if parsing fails.
        min_val: Minimum allowed value (must be > 0).
        max_val: Maximum allowed value.

    Returns:
        Validated float value, or default if validation fails.
    """
    try:
        value = float(value_str)
        if value <= 0:
            logger.error(
                "%s must be positive (greater than 0), got %s. Using default: %s",
                env_var, value, default
            )
            return default
        if value < min_val:
            logger.warning(
                "%s value %s is below minimum %s. Using minimum value.",
                env_var, value, min_val
            )
            return min_val
        if value > max_val:
            logger.warning(
                "%s value %s exceeds maximum %s. Using maximum value.",
                env_var, value, max_val
            )
            return max_val
        return value
    except ValueError:
        logger.error(
            "%s must be a valid float, got '%s'. Using default: %s",
            env_var, value_str, default
        )
        return default


def validate_positive_int(
    env_var: str, value_str: str, default: int, min_val: int, max_val: int
) -> int:
    """Validate and parse a positive integer environment variable (DS-278, DS-282).

    This function requires values > 0 (strictly positive).

    Args:
        env_var: Name of the environment variable (for error messages).
        value_str: String value to parse.
        default: Default value if parsing fails.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Validated integer value, or default if validation fails.
    """
    try:
        value = int(value_str)
        if value <= 0:
            logger.error(
                "%s must be positive (greater than 0), got %s. Using default: %s",
                env_var, value, default
            )
            return default
        if value < min_val:
            logger.warning(
                "%s value %s is below minimum %s. Using minimum value.",
                env_var, value, min_val
            )
            return min_val
        if value > max_val:
            logger.warning(
                "%s value %s exceeds maximum %s. Using maximum value.",
                env_var, value, max_val
            )
            return max_val
        return value
    except ValueError:
        logger.error(
            "%s must be a valid integer, got '%s'. Using default: %s",
            env_var, value_str, default
        )
        return default
