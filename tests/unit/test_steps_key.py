"""Tests for the shared steps/orchestrations key resolution utility (DS-900, DS-901).

This module tests ``sentinel.steps_key.resolve_steps_key()`` which is the
single source of truth for resolving the ``"steps"`` vs ``"orchestrations"``
YAML key.

DS-901 added tests verifying that the generic ``_VT`` type variable
allows ``resolve_steps_key()`` to work with ``CommentedMap`` input
(returning the original ``CommentedSeq`` instances) without requiring
``type: ignore`` annotations at call sites.
"""

from __future__ import annotations

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from sentinel.steps_key import resolve_steps_key


class TestResolveStepsKey:
    """Tests for resolve_steps_key function."""

    def test_returns_steps_when_steps_key_present(self) -> None:
        """Should return the steps list when 'steps' key exists."""
        data = {"steps": [{"name": "a"}]}
        result = resolve_steps_key(data)
        assert result == [{"name": "a"}]

    def test_returns_empty_list_when_steps_is_empty(self) -> None:
        """Should return empty list when 'steps' key exists with empty list.

        Verifies the DS-899 fix: an empty list [] is falsy in Python,
        so using ``or`` would incorrectly fall through to 'orchestrations'.
        """
        data = {"steps": []}
        result = resolve_steps_key(data)
        assert result is not None
        assert result == []

    def test_empty_steps_does_not_fallthrough_to_orchestrations(self) -> None:
        """Should not fall through to 'orchestrations' when 'steps' is empty.

        If both keys exist and 'steps' is empty, the function must return
        the empty steps list, not the orchestrations list.
        """
        data = {
            "steps": [],
            "orchestrations": [{"name": "should-not-be-returned"}],
        }
        result = resolve_steps_key(data)
        assert result is not None
        assert result == []

    def test_falls_back_to_orchestrations_when_no_steps(self) -> None:
        """Should fall back to 'orchestrations' when 'steps' key is absent."""
        data = {"orchestrations": [{"name": "b"}]}
        result = resolve_steps_key(data)
        assert result == [{"name": "b"}]

    def test_returns_none_when_neither_key_present(self) -> None:
        """Should return None when neither 'steps' nor 'orchestrations' exists."""
        data = {"other_key": "value"}
        result = resolve_steps_key(data)
        assert result is None

    def test_returns_none_for_empty_dict(self) -> None:
        """Should return None for an empty dict."""
        result = resolve_steps_key({})
        assert result is None

    def test_steps_takes_precedence_over_orchestrations(self) -> None:
        """Should return 'steps' when both keys are present with data."""
        data = {
            "steps": [{"name": "step-item"}],
            "orchestrations": [{"name": "orch-item"}],
        }
        result = resolve_steps_key(data)
        assert result == [{"name": "step-item"}]

    def test_returns_none_value_under_steps_key(self) -> None:
        """Should return None when 'steps' key exists but value is None."""
        data = {"steps": None}
        result = resolve_steps_key(data)
        assert result is None

    def test_returns_none_value_under_orchestrations_key(self) -> None:
        """Should return None when 'orchestrations' key exists but value is None."""
        data = {"orchestrations": None}
        result = resolve_steps_key(data)
        assert result is None


class TestResolveStepsKeyGenericType:
    """Tests for generic return type with CommentedMap input (DS-901).

    When ``resolve_steps_key()`` is called with a ``CommentedMap`` (as
    ``yaml_writer.py`` does), the generic ``_VT`` type variable ensures the
    return type preserves the original container type (``CommentedSeq``)
    without requiring a ``type: ignore`` comment.
    """

    def test_returns_commented_seq_from_commented_map_steps(self) -> None:
        """Should return CommentedSeq when CommentedMap has 'steps' key."""
        data = CommentedMap()
        steps = CommentedSeq([{"name": "a"}])
        data["steps"] = steps
        result = resolve_steps_key(data)
        assert result is steps

    def test_returns_commented_seq_from_commented_map_orchestrations(self) -> None:
        """Should return CommentedSeq when CommentedMap has 'orchestrations' key."""
        data = CommentedMap()
        orchestrations = CommentedSeq([{"name": "b"}])
        data["orchestrations"] = orchestrations
        result = resolve_steps_key(data)
        assert result is orchestrations

    def test_returns_none_from_commented_map_no_matching_key(self) -> None:
        """Should return None when CommentedMap has neither key."""
        data = CommentedMap()
        data["other"] = "value"
        result = resolve_steps_key(data)
        assert result is None

    def test_commented_map_steps_takes_precedence(self) -> None:
        """Should return 'steps' CommentedSeq when both keys exist in CommentedMap."""
        data = CommentedMap()
        steps = CommentedSeq([{"name": "step"}])
        orchestrations = CommentedSeq([{"name": "orch"}])
        data["steps"] = steps
        data["orchestrations"] = orchestrations
        result = resolve_steps_key(data)
        assert result is steps

    def test_commented_map_empty_steps_does_not_fallthrough(self) -> None:
        """Should return empty CommentedSeq from CommentedMap, not fall through."""
        data = CommentedMap()
        empty_steps = CommentedSeq()
        data["steps"] = empty_steps
        data["orchestrations"] = CommentedSeq([{"name": "should-not-return"}])
        result = resolve_steps_key(data)
        assert result is empty_steps
