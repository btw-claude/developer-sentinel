"""Tests for mock classes in tests/mocks.py.

These tests verify the behavior of our mock implementations, particularly
the observer pattern in MockTagClient.
"""

import gc

from tests.mocks import MockJiraClient, MockJiraPoller, MockTagClient


class TestMockTagClientObserver:
    """Tests for the MockTagClient observer pattern."""

    def test_register_observer_receives_notifications(self) -> None:
        """Registered observers should receive label removal notifications."""
        tag_client = MockTagClient()
        notifications: list[tuple[str, str]] = []

        def observer(issue_key: str, label: str) -> None:
            notifications.append((issue_key, label))

        tag_client.register_observer(observer)
        tag_client.remove_label("PROJ-123", "needs-review")

        assert notifications == [("PROJ-123", "needs-review")]

    def test_unregister_observer_stops_notifications(self) -> None:
        """Unregistered observers should not receive further notifications."""
        tag_client = MockTagClient()
        notifications: list[tuple[str, str]] = []

        def observer(issue_key: str, label: str) -> None:
            notifications.append((issue_key, label))

        tag_client.register_observer(observer)
        tag_client.remove_label("PROJ-123", "label1")

        # Unregister and verify no more notifications
        result = tag_client.unregister_observer(observer)
        assert result is True

        tag_client.remove_label("PROJ-456", "label2")
        assert notifications == [("PROJ-123", "label1")]

    def test_unregister_observer_returns_false_for_unknown_observer(self) -> None:
        """Unregistering an unknown observer should return False."""
        tag_client = MockTagClient()

        def observer(issue_key: str, label: str) -> None:
            pass

        result = tag_client.unregister_observer(observer)
        assert result is False

    def test_multiple_observers_all_notified(self) -> None:
        """Multiple registered observers should all receive notifications."""
        tag_client = MockTagClient()
        notifications1: list[tuple[str, str]] = []
        notifications2: list[tuple[str, str]] = []

        def observer1(issue_key: str, label: str) -> None:
            notifications1.append((issue_key, label))

        def observer2(issue_key: str, label: str) -> None:
            notifications2.append((issue_key, label))

        tag_client.register_observer(observer1)
        tag_client.register_observer(observer2)
        tag_client.remove_label("PROJ-123", "needs-review")

        assert notifications1 == [("PROJ-123", "needs-review")]
        assert notifications2 == [("PROJ-123", "needs-review")]

    def test_unregister_one_observer_keeps_others(self) -> None:
        """Unregistering one observer should not affect other observers."""
        tag_client = MockTagClient()
        notifications1: list[tuple[str, str]] = []
        notifications2: list[tuple[str, str]] = []

        def observer1(issue_key: str, label: str) -> None:
            notifications1.append((issue_key, label))

        def observer2(issue_key: str, label: str) -> None:
            notifications2.append((issue_key, label))

        tag_client.register_observer(observer1)
        tag_client.register_observer(observer2)

        # Unregister first observer
        tag_client.unregister_observer(observer1)
        tag_client.remove_label("PROJ-123", "needs-review")

        # Only second observer should be notified
        assert notifications1 == []
        assert notifications2 == [("PROJ-123", "needs-review")]


class TestMockTagClientWeakReferences:
    """Tests for MockTagClient weak reference behavior."""

    def test_bound_method_observer_is_cleaned_up_on_garbage_collection(self) -> None:
        """Bound method observers should be cleaned up when their object is GC'd."""
        tag_client = MockTagClient()
        notifications: list[tuple[str, str]] = []

        class Observer:
            def on_label_removed(self, issue_key: str, label: str) -> None:
                notifications.append((issue_key, label))

        # Create observer and register bound method
        observer = Observer()
        tag_client.register_observer(observer.on_label_removed)

        # Verify observer works
        tag_client.remove_label("PROJ-123", "label1")
        assert notifications == [("PROJ-123", "label1")]

        # Delete observer and force garbage collection
        del observer
        gc.collect()

        # Observer should be cleaned up - no new notifications
        notifications.clear()
        tag_client.remove_label("PROJ-456", "label2")
        assert notifications == []

    def test_multiple_bound_method_observers_one_gc_keeps_other(self) -> None:
        """When one bound method observer is GC'd, others should still work."""
        tag_client = MockTagClient()
        notifications1: list[tuple[str, str]] = []
        notifications2: list[tuple[str, str]] = []

        class Observer1:
            def on_label_removed(self, issue_key: str, label: str) -> None:
                notifications1.append((issue_key, label))

        class Observer2:
            def on_label_removed(self, issue_key: str, label: str) -> None:
                notifications2.append((issue_key, label))

        observer1 = Observer1()
        observer2 = Observer2()
        tag_client.register_observer(observer1.on_label_removed)
        tag_client.register_observer(observer2.on_label_removed)

        # Verify both work
        tag_client.remove_label("PROJ-123", "label1")
        assert notifications1 == [("PROJ-123", "label1")]
        assert notifications2 == [("PROJ-123", "label1")]

        # Delete first observer
        del observer1
        gc.collect()

        # Only second observer should receive notifications
        notifications1.clear()
        notifications2.clear()
        tag_client.remove_label("PROJ-456", "label2")
        assert notifications1 == []
        assert notifications2 == [("PROJ-456", "label2")]

        # Keep observer2 alive for the test duration
        assert observer2 is not None

    def test_regular_function_observer_works(self) -> None:
        """Regular function observers should continue to work."""
        tag_client = MockTagClient()
        notifications: list[tuple[str, str]] = []

        def observer(issue_key: str, label: str) -> None:
            notifications.append((issue_key, label))

        tag_client.register_observer(observer)
        tag_client.remove_label("PROJ-123", "needs-review")

        assert notifications == [("PROJ-123", "needs-review")]

    def test_cleanup_dead_observers_is_called(self) -> None:
        """Dead observers should be cleaned up when remove_label is called."""
        tag_client = MockTagClient()

        class Observer:
            def on_label_removed(self, issue_key: str, label: str) -> None:
                pass

        observer = Observer()
        tag_client.register_observer(observer.on_label_removed)

        # Should have 1 observer
        assert len(tag_client._observers) == 1

        # Delete observer and trigger cleanup via remove_label
        del observer
        gc.collect()
        tag_client.remove_label("PROJ-123", "label")

        # Dead observer should be cleaned up
        assert len(tag_client._observers) == 0


class TestMockJiraClientObserverIntegration:
    """Tests for MockJiraClient integration with MockTagClient observer pattern."""

    def test_jira_client_filters_issues_after_label_removal(self) -> None:
        """MockJiraClient should filter issues after labels are removed via tag_client."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "PROJ-123", "fields": {"summary": "Test issue"}},
                {"key": "PROJ-456", "fields": {"summary": "Other issue"}},
            ],
            tag_client=tag_client,
        )

        # Both issues returned initially
        results = jira_client.search_issues("labels = trigger")
        assert len(results) == 2

        # After label removal, issue is filtered
        tag_client.remove_label("PROJ-123", "trigger")
        results = jira_client.search_issues("labels = trigger")
        assert len(results) == 1
        assert results[0]["key"] == "PROJ-456"


class TestMockJiraPollerObserverIntegration:
    """Tests for MockJiraPoller integration with MockTagClient observer pattern."""

    def test_jira_poller_filters_issues_after_label_removal(self) -> None:
        """MockJiraPoller should filter issues after labels are removed via tag_client."""
        from sentinel.orchestration import TriggerConfig
        from sentinel.poller import JiraIssue

        tag_client = MockTagClient()
        issues = [
            JiraIssue(
                key="PROJ-123",
                summary="Test issue",
                description="",
                labels=["trigger"],
                status="Open",
            ),
            JiraIssue(
                key="PROJ-456",
                summary="Other issue",
                description="",
                labels=["trigger"],
                status="Open",
            ),
        ]
        poller = MockJiraPoller(issues=issues, tag_client=tag_client)
        trigger = TriggerConfig(source="jira", tags=["trigger"])

        # Both issues returned initially
        results = poller.poll(trigger)
        assert len(results) == 2

        # After label removal, issue is filtered
        tag_client.remove_label("PROJ-123", "trigger")
        results = poller.poll(trigger)
        assert len(results) == 1
        assert results[0].key == "PROJ-456"
