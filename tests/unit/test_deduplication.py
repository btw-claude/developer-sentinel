"""Tests for the deduplication module.

Tests for the DeduplicationManager and CycleTracker classes.
"""

from __future__ import annotations

from sentinel.deduplication import DeduplicationManager


class TestDeduplicationManager:
    """Tests for DeduplicationManager class."""

    def test_create_cycle_set_returns_empty_set(self) -> None:
        """create_cycle_set returns a new empty set."""
        manager = DeduplicationManager()
        result = manager.create_cycle_set()
        assert result == set()
        assert isinstance(result, set)

    def test_create_cycle_set_returns_new_set_each_time(self) -> None:
        """create_cycle_set returns a new set each time."""
        manager = DeduplicationManager()
        set1 = manager.create_cycle_set()
        set2 = manager.create_cycle_set()
        assert set1 is not set2

    def test_check_and_mark_returns_true_for_new_pair(self) -> None:
        """check_and_mark returns True for a new pair."""
        manager = DeduplicationManager()
        submitted_pairs: set[tuple[str, str]] = set()

        result = manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-1")

        assert result is True
        assert ("PROJ-123", "orch-1") in submitted_pairs

    def test_check_and_mark_returns_false_for_duplicate(self) -> None:
        """check_and_mark returns False for a duplicate pair."""
        manager = DeduplicationManager()
        submitted_pairs: set[tuple[str, str]] = set()

        # First call should succeed
        manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-1")

        # Second call with same pair should return False
        result = manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-1")

        assert result is False

    def test_check_and_mark_allows_different_pairs(self) -> None:
        """check_and_mark allows different pairs."""
        manager = DeduplicationManager()
        submitted_pairs: set[tuple[str, str]] = set()

        # Different issue keys
        assert manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-1") is True
        assert manager.check_and_mark(submitted_pairs, "PROJ-456", "orch-1") is True

        # Different orchestration names
        assert manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-2") is True

        # Same pairs should now be duplicates
        assert manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-1") is False
        assert manager.check_and_mark(submitted_pairs, "PROJ-456", "orch-1") is False
        assert manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-2") is False

    def test_is_duplicate_returns_false_for_new_pair(self) -> None:
        """is_duplicate returns False for a new pair."""
        manager = DeduplicationManager()
        submitted_pairs: set[tuple[str, str]] = set()

        result = manager.is_duplicate(submitted_pairs, "PROJ-123", "orch-1")

        assert result is False
        # Should not add to set
        assert ("PROJ-123", "orch-1") not in submitted_pairs

    def test_is_duplicate_returns_true_for_existing_pair(self) -> None:
        """is_duplicate returns True for an existing pair."""
        manager = DeduplicationManager()
        submitted_pairs: set[tuple[str, str]] = {("PROJ-123", "orch-1")}

        result = manager.is_duplicate(submitted_pairs, "PROJ-123", "orch-1")

        assert result is True

    def test_mark_submitted_adds_to_set(self) -> None:
        """mark_submitted adds the pair to the set."""
        manager = DeduplicationManager()
        submitted_pairs: set[tuple[str, str]] = set()

        manager.mark_submitted(submitted_pairs, "PROJ-123", "orch-1")

        assert ("PROJ-123", "orch-1") in submitted_pairs

    def test_mark_submitted_is_idempotent(self) -> None:
        """mark_submitted can be called multiple times."""
        manager = DeduplicationManager()
        submitted_pairs: set[tuple[str, str]] = set()

        manager.mark_submitted(submitted_pairs, "PROJ-123", "orch-1")
        manager.mark_submitted(submitted_pairs, "PROJ-123", "orch-1")

        assert len(submitted_pairs) == 1
        assert ("PROJ-123", "orch-1") in submitted_pairs

    def test_reset_cycle_clears_internal_state(self) -> None:
        """reset_cycle clears the internal current cycle pairs."""
        manager = DeduplicationManager()

        # Use a cycle tracker to populate internal state
        with manager.cycle_tracker() as tracker:
            tracker.should_submit("PROJ-123", "orch-1")

        # Reset should clear the internal state
        manager.reset_cycle()

        # Verify internal state is cleared
        assert len(manager._current_cycle_pairs) == 0

    def test_works_with_github_style_keys(self) -> None:
        """check_and_mark works with GitHub-style issue keys."""
        manager = DeduplicationManager()
        submitted_pairs: set[tuple[str, str]] = set()

        assert manager.check_and_mark(submitted_pairs, "org/repo#123", "orch-1") is True
        assert manager.check_and_mark(submitted_pairs, "org/repo#123", "orch-1") is False
        assert manager.check_and_mark(submitted_pairs, "org/repo#456", "orch-1") is True


class TestCycleTracker:
    """Tests for CycleTracker class."""

    def test_should_submit_returns_true_for_new_pair(self) -> None:
        """should_submit returns True for a new pair."""
        manager = DeduplicationManager()

        with manager.cycle_tracker() as tracker:
            result = tracker.should_submit("PROJ-123", "orch-1")
            assert result is True

    def test_should_submit_returns_false_for_duplicate(self) -> None:
        """should_submit returns False for a duplicate pair."""
        manager = DeduplicationManager()

        with manager.cycle_tracker() as tracker:
            tracker.should_submit("PROJ-123", "orch-1")
            result = tracker.should_submit("PROJ-123", "orch-1")
            assert result is False

    def test_is_duplicate_returns_false_for_new_pair(self) -> None:
        """is_duplicate returns False for a new pair."""
        manager = DeduplicationManager()

        with manager.cycle_tracker() as tracker:
            result = tracker.is_duplicate("PROJ-123", "orch-1")
            assert result is False

    def test_is_duplicate_returns_true_for_submitted_pair(self) -> None:
        """is_duplicate returns True after pair is submitted."""
        manager = DeduplicationManager()

        with manager.cycle_tracker() as tracker:
            tracker.should_submit("PROJ-123", "orch-1")
            result = tracker.is_duplicate("PROJ-123", "orch-1")
            assert result is True

    def test_submitted_count_tracks_submissions(self) -> None:
        """submitted_count returns the number of unique submissions."""
        manager = DeduplicationManager()

        with manager.cycle_tracker() as tracker:
            assert tracker.submitted_count == 0

            tracker.should_submit("PROJ-123", "orch-1")
            assert tracker.submitted_count == 1

            tracker.should_submit("PROJ-456", "orch-1")
            assert tracker.submitted_count == 2

            # Duplicate should not increment
            tracker.should_submit("PROJ-123", "orch-1")
            assert tracker.submitted_count == 2

    def test_get_submitted_pairs_returns_copy(self) -> None:
        """get_submitted_pairs returns a copy of submitted pairs."""
        manager = DeduplicationManager()

        with manager.cycle_tracker() as tracker:
            tracker.should_submit("PROJ-123", "orch-1")

            pairs = tracker.get_submitted_pairs()
            assert pairs == {("PROJ-123", "orch-1")}

            # Modifying the copy shouldn't affect the tracker
            pairs.add(("PROJ-456", "orch-2"))
            assert tracker.submitted_count == 1

    def test_context_manager_merges_pairs_to_global_set(self) -> None:
        """cycle_tracker context manager merges pairs to global set."""
        manager = DeduplicationManager()

        with manager.cycle_tracker() as tracker:
            tracker.should_submit("PROJ-123", "orch-1")
            tracker.should_submit("PROJ-456", "orch-2")

        # After context exits, pairs should be in global set
        assert ("PROJ-123", "orch-1") in manager._current_cycle_pairs
        assert ("PROJ-456", "orch-2") in manager._current_cycle_pairs

    def test_multiple_trackers_accumulate_in_global_set(self) -> None:
        """Multiple trackers accumulate their pairs in the global set."""
        manager = DeduplicationManager()

        with manager.cycle_tracker() as tracker1:
            tracker1.should_submit("PROJ-123", "orch-1")

        with manager.cycle_tracker() as tracker2:
            tracker2.should_submit("PROJ-456", "orch-2")

        # Both trackers' pairs should be in global set
        assert ("PROJ-123", "orch-1") in manager._current_cycle_pairs
        assert ("PROJ-456", "orch-2") in manager._current_cycle_pairs


class TestDeduplicationIntegration:
    """Integration tests for deduplication behavior."""

    def test_typical_polling_cycle_workflow(self) -> None:
        """Test typical workflow in a polling cycle."""
        manager = DeduplicationManager()

        # Start of cycle - create fresh set
        submitted_pairs = manager.create_cycle_set()

        # Simulate routing results - same issue matches multiple orchestrations
        assert manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-1") is True
        assert manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-2") is True

        # Simulate overlapping triggers - same issue/orch pair from different polls
        assert manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-1") is False

        # Different issue is fine
        assert manager.check_and_mark(submitted_pairs, "PROJ-456", "orch-1") is True

    def test_mixed_jira_and_github_keys(self) -> None:
        """Test deduplication works with mixed Jira and GitHub keys."""
        manager = DeduplicationManager()
        submitted_pairs = manager.create_cycle_set()

        # Jira-style key
        assert manager.check_and_mark(submitted_pairs, "PROJ-123", "orch-1") is True

        # GitHub-style key
        assert manager.check_and_mark(submitted_pairs, "org/repo#456", "orch-1") is True

        # Same orchestration name doesn't cause collision across sources
        assert ("PROJ-123", "orch-1") in submitted_pairs
        assert ("org/repo#456", "orch-1") in submitted_pairs
        assert len(submitted_pairs) == 2
