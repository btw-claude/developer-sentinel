"""Shared utility for resolving the 'steps' vs 'orchestrations' YAML key.

Both ``orchestration.py`` (loader) and ``yaml_writer.py`` (editor) need to
resolve which top-level key holds the list of orchestration steps.  The new
format uses ``"steps"`` while the legacy format uses ``"orchestrations"``.

This module provides a single ``resolve_steps_key`` function so that the
resolution logic (including the falsy-empty-list guard from DS-899) is
defined in exactly one place.

The function uses a generic ``VT`` type variable (DS-901) so that callers
passing a ``dict[str, Any]`` get ``list[Any] | None`` while callers passing
a ``CommentedMap`` (whose ``MutableMapping`` value type is
``Any``) also get the correct inferred return type without needing a
``type: ignore`` comment.

DS-902 modernised the ``VT`` declaration from the legacy
``TypeVar("VT")`` call to the Python 3.12+ PEP 695 type parameter
syntax (``def resolve_steps_key[VT](...)``) which is more concise,
avoids the redundant string name, and eliminates the ``typing`` import.

See Also:
    DS-896: Moved trigger to file level, renamed key to ``steps``.
    DS-899: Fixed empty-list edge case with explicit key membership check.
    DS-900: Extracted shared utility to keep the logic DRY.
    DS-901: Made the return type generic to eliminate ``type: ignore``.
    DS-902: Adopted Python 3.12+ PEP 695 type parameter syntax for ``VT``.
"""

from __future__ import annotations


def resolve_steps_key[VT](data: dict[str, VT]) -> VT | None:
    """Resolve the steps/orchestrations list from parsed YAML data.

    Checks for the ``"steps"`` key first (new format introduced in DS-896),
    then falls back to the ``"orchestrations"`` key (legacy format).

    Uses an explicit ``in`` membership test rather than the ``or`` operator
    so that an empty list ``[]`` under ``"steps"`` is returned correctly
    instead of falling through to ``"orchestrations"`` (DS-899).

    The function is generic over the dict's value type (DS-901): when called
    with a plain ``dict[str, Any]`` the return type is ``Any | None``, and
    when called with a ``CommentedMap`` (which satisfies
    ``MutableMapping[str, Any]``) the return type flows through correctly so
    that callers no longer need a ``# type: ignore[return-value]`` comment.

    Args:
        data: The top-level parsed YAML data as a dict (or dict-like
            object such as ``ruamel.yaml.comments.CommentedMap``).

    Returns:
        The value associated with ``"steps"`` or ``"orchestrations"``,
        or ``None`` if neither key is present.

    Examples:
        >>> resolve_steps_key({"steps": [{"name": "a"}]})
        [{'name': 'a'}]

        >>> resolve_steps_key({"steps": []})
        []

        >>> resolve_steps_key({"orchestrations": [{"name": "b"}]})
        [{'name': 'b'}]

        >>> resolve_steps_key({"other_key": "value"}) is None
        True
    """
    if "steps" in data:
        return data.get("steps")
    return data.get("orchestrations")
