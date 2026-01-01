"""Tests for post-processing tag manager module."""

from sentinel.executor import ExecutionResult, ExecutionStatus
from sentinel.orchestration import (
    AgentConfig,
    OnCompleteConfig,
    OnFailureConfig,
    Orchestration,
    TriggerConfig,
)
from sentinel.tag_manager import (
    JiraTagClient,
    JiraTagClientError,
    TagManager,
    TagUpdateResult,
)


class MockJiraTagClient(JiraTagClient):
    """Mock Jira tag client for testing."""

    def __init__(self) -> None:
        self.labels: dict[str, list[str]] = {}
        self.add_calls: list[tuple[str, str]] = []
        self.remove_calls: list[tuple[str, str]] = []
        self.should_fail_add = False
        self.should_fail_remove = False

    def add_label(self, issue_key: str, label: str) -> None:
        self.add_calls.append((issue_key, label))
        if self.should_fail_add:
            raise JiraTagClientError("Mock add error")
        if issue_key not in self.labels:
            self.labels[issue_key] = []
        if label not in self.labels[issue_key]:
            self.labels[issue_key].append(label)

    def remove_label(self, issue_key: str, label: str) -> None:
        self.remove_calls.append((issue_key, label))
        if self.should_fail_remove:
            raise JiraTagClientError("Mock remove error")
        if issue_key in self.labels and label in self.labels[issue_key]:
            self.labels[issue_key].remove(label)


def make_result(
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    issue_key: str = "TEST-1",
    orchestration_name: str = "test-orch",
) -> ExecutionResult:
    """Helper to create an ExecutionResult for testing."""
    return ExecutionResult(
        status=status,
        response="Test response",
        attempts=1,
        issue_key=issue_key,
        orchestration_name=orchestration_name,
    )


def make_orchestration(
    name: str = "test-orch",
    remove_tag: str = "",
    add_tag: str = "",
    failure_tag: str = "",
) -> Orchestration:
    """Helper to create an Orchestration for testing."""
    return Orchestration(
        name=name,
        trigger=TriggerConfig(),
        agent=AgentConfig(prompt="Test prompt"),
        on_complete=OnCompleteConfig(remove_tag=remove_tag, add_tag=add_tag),
        on_failure=OnFailureConfig(add_tag=failure_tag),
    )


class TestTagUpdateResult:
    """Tests for TagUpdateResult dataclass."""

    def test_success_when_no_errors(self) -> None:
        result = TagUpdateResult(
            issue_key="TEST-1",
            added_tags=["processed"],
            removed_tags=["needs-review"],
            errors=[],
        )
        assert result.success is True

    def test_not_success_when_errors(self) -> None:
        result = TagUpdateResult(
            issue_key="TEST-1",
            added_tags=[],
            removed_tags=[],
            errors=["Failed to add tag"],
        )
        assert result.success is False


class TestTagManagerUpdateTagsSuccess:
    """Tests for TagManager.update_tags on successful execution."""

    def test_removes_tag_on_success(self) -> None:
        client = MockJiraTagClient()
        client.labels["TEST-1"] = ["needs-review", "bug"]
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-1")
        orch = make_orchestration(remove_tag="needs-review")

        update_result = manager.update_tags(result, orch)

        assert update_result.success is True
        assert "needs-review" in update_result.removed_tags
        assert ("TEST-1", "needs-review") in client.remove_calls
        assert "needs-review" not in client.labels["TEST-1"]

    def test_adds_tag_on_success(self) -> None:
        client = MockJiraTagClient()
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-1")
        orch = make_orchestration(add_tag="processed")

        update_result = manager.update_tags(result, orch)

        assert update_result.success is True
        assert "processed" in update_result.added_tags
        assert ("TEST-1", "processed") in client.add_calls
        assert "processed" in client.labels["TEST-1"]

    def test_removes_and_adds_tags_on_success(self) -> None:
        client = MockJiraTagClient()
        client.labels["TEST-1"] = ["needs-review"]
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-1")
        orch = make_orchestration(remove_tag="needs-review", add_tag="reviewed")

        update_result = manager.update_tags(result, orch)

        assert update_result.success is True
        assert "needs-review" in update_result.removed_tags
        assert "reviewed" in update_result.added_tags
        assert "needs-review" not in client.labels["TEST-1"]
        assert "reviewed" in client.labels["TEST-1"]

    def test_no_changes_when_no_tags_configured(self) -> None:
        client = MockJiraTagClient()
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.SUCCESS)
        orch = make_orchestration()  # No tags configured

        update_result = manager.update_tags(result, orch)

        assert update_result.success is True
        assert update_result.added_tags == []
        assert update_result.removed_tags == []
        assert len(client.add_calls) == 0
        assert len(client.remove_calls) == 0


class TestTagManagerUpdateTagsFailure:
    """Tests for TagManager.update_tags on failed execution."""

    def test_adds_failure_tag_on_failure(self) -> None:
        client = MockJiraTagClient()
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.FAILURE, issue_key="TEST-1")
        orch = make_orchestration(failure_tag="agent-failed")

        update_result = manager.update_tags(result, orch)

        assert update_result.success is True
        assert "agent-failed" in update_result.added_tags
        assert ("TEST-1", "agent-failed") in client.add_calls

    def test_adds_failure_tag_on_error(self) -> None:
        client = MockJiraTagClient()
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.ERROR, issue_key="TEST-1")
        orch = make_orchestration(failure_tag="agent-error")

        update_result = manager.update_tags(result, orch)

        assert update_result.success is True
        assert "agent-error" in update_result.added_tags

    def test_keeps_trigger_tag_on_failure(self) -> None:
        client = MockJiraTagClient()
        client.labels["TEST-1"] = ["needs-review", "bug"]
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.FAILURE, issue_key="TEST-1")
        # Even with remove_tag configured, it should NOT be removed on failure
        orch = make_orchestration(remove_tag="needs-review", failure_tag="agent-failed")

        update_result = manager.update_tags(result, orch)

        assert update_result.success is True
        # Should NOT have removed the trigger tag
        assert "needs-review" not in update_result.removed_tags
        assert "needs-review" in client.labels["TEST-1"]
        # Should have added failure tag
        assert "agent-failed" in update_result.added_tags

    def test_no_changes_when_no_failure_tag_configured(self) -> None:
        client = MockJiraTagClient()
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.FAILURE)
        orch = make_orchestration()  # No failure tag configured

        update_result = manager.update_tags(result, orch)

        assert update_result.success is True
        assert update_result.added_tags == []
        assert update_result.removed_tags == []


class TestTagManagerErrors:
    """Tests for TagManager error handling."""

    def test_records_error_when_remove_fails(self) -> None:
        client = MockJiraTagClient()
        client.should_fail_remove = True
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-1")
        orch = make_orchestration(remove_tag="needs-review")

        update_result = manager.update_tags(result, orch)

        assert update_result.success is False
        assert len(update_result.errors) == 1
        assert "needs-review" in update_result.errors[0]
        assert update_result.removed_tags == []

    def test_records_error_when_add_fails(self) -> None:
        client = MockJiraTagClient()
        client.should_fail_add = True
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-1")
        orch = make_orchestration(add_tag="processed")

        update_result = manager.update_tags(result, orch)

        assert update_result.success is False
        assert len(update_result.errors) == 1
        assert "processed" in update_result.errors[0]
        assert update_result.added_tags == []

    def test_continues_after_remove_error(self) -> None:
        client = MockJiraTagClient()
        client.should_fail_remove = True
        manager = TagManager(client)

        result = make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-1")
        orch = make_orchestration(remove_tag="needs-review", add_tag="processed")

        update_result = manager.update_tags(result, orch)

        # Should still have tried to add tag even after remove failed
        assert ("TEST-1", "processed") in client.add_calls
        assert "processed" in update_result.added_tags
        # But should record the error
        assert len(update_result.errors) == 1


class TestTagManagerBatch:
    """Tests for TagManager.update_tags_batch."""

    def test_processes_all_results(self) -> None:
        client = MockJiraTagClient()
        manager = TagManager(client)

        results = [
            (
                make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-1"),
                make_orchestration(add_tag="processed"),
            ),
            (
                make_result(status=ExecutionStatus.FAILURE, issue_key="TEST-2"),
                make_orchestration(failure_tag="agent-failed"),
            ),
            (
                make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-3"),
                make_orchestration(remove_tag="needs-review"),
            ),
        ]

        update_results = manager.update_tags_batch(results)

        assert len(update_results) == 3
        assert update_results[0].issue_key == "TEST-1"
        assert update_results[1].issue_key == "TEST-2"
        assert update_results[2].issue_key == "TEST-3"

    def test_empty_batch_returns_empty(self) -> None:
        client = MockJiraTagClient()
        manager = TagManager(client)

        update_results = manager.update_tags_batch([])

        assert len(update_results) == 0

    def test_counts_successes_and_failures(self) -> None:
        client = MockJiraTagClient()
        client.should_fail_add = True
        manager = TagManager(client)

        results = [
            (
                make_result(status=ExecutionStatus.SUCCESS, issue_key="TEST-1"),
                make_orchestration(add_tag="processed"),  # Will fail
            ),
            (
                make_result(status=ExecutionStatus.FAILURE, issue_key="TEST-2"),
                make_orchestration(),  # No tag to add, will succeed
            ),
        ]

        update_results = manager.update_tags_batch(results)

        assert update_results[0].success is False
        assert update_results[1].success is True
