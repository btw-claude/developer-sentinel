"""Dedicated tests for poll_coordinator.py to increase coverage above 85%.

This file covers the gaps not addressed by test_poll_coordinator_exceptions.py,
which focuses on exception handler ordering. The tests here cover:

- extract_repo_from_url() module-level function
- GitHubIssueWithRepo wrapper class (key, forwarding properties, __getattr__)
- PollingResult dataclass
- GroupedOrchestrations named tuple
- PollCoordinator.__init__
- create_cycle_dedup_set()
- check_and_mark_submitted()
- poll_jira_triggers() happy path, no-poller, shutdown, log_callback, dedup
- poll_github_triggers() happy path, no-poller, shutdown, log_callback, dedup,
  _add_repo_context_from_urls with valid/invalid URLs
- construct_issue_url() for Jira and GitHub sources
- group_orchestrations_by_source() grouping logic

Related: DS-929
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from sentinel.config import Config
from sentinel.orchestration import GitHubContext, Orchestration
from sentinel.poll_coordinator import (
    GitHubIssueWithRepo,
    GroupedOrchestrations,
    PollCoordinator,
    PollingResult,
    extract_repo_from_url,
)
from tests.helpers import make_config, make_github_issue, make_issue, make_orchestration
from tests.mocks import MockGitHubPoller, MockJiraPoller

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_poll_coordinator(
    jira_poller: MockJiraPoller | None = None,
    github_poller: MockGitHubPoller | None = None,
    config: Config | None = None,
) -> PollCoordinator:
    """Create a PollCoordinator with sensible defaults for testing."""
    if config is None:
        config = make_config()
    return PollCoordinator(
        config=config,
        jira_poller=jira_poller,
        github_poller=github_poller,
    )


# ===========================================================================
# TestExtractRepoFromUrl
# ===========================================================================


class TestExtractRepoFromUrl:
    """Tests for the extract_repo_from_url() module-level function."""

    def test_valid_issue_url(self) -> None:
        """A standard GitHub issue URL should return 'owner/repo'."""
        result = extract_repo_from_url("https://github.com/myorg/myrepo/issues/42")
        assert result == "myorg/myrepo"

    def test_valid_pr_url(self) -> None:
        """A standard GitHub pull request URL should return 'owner/repo'."""
        result = extract_repo_from_url("https://github.com/acme/widget/pull/99")
        assert result == "acme/widget"

    def test_empty_string_returns_none(self) -> None:
        """An empty string should return None."""
        result = extract_repo_from_url("")
        assert result is None

    def test_invalid_url_returns_none(self) -> None:
        """A URL that does not match the GitHub issue/PR pattern returns None."""
        result = extract_repo_from_url("https://example.com/not-github")
        assert result is None

    def test_non_matching_github_url_returns_none(self) -> None:
        """A GitHub URL without /issues/ or /pull/ segment returns None."""
        result = extract_repo_from_url("https://github.com/org/repo/tree/main")
        assert result is None

    def test_http_url(self) -> None:
        """HTTP (non-HTTPS) URLs should also be matched."""
        result = extract_repo_from_url("http://github.com/org/repo/issues/1")
        assert result == "org/repo"

    def test_enterprise_github_url(self) -> None:
        """Enterprise GitHub URLs with custom hosts should be matched."""
        result = extract_repo_from_url("https://github.mycorp.com/team/project/pull/55")
        assert result == "team/project"


# ===========================================================================
# TestGitHubIssueWithRepo
# ===========================================================================


class TestGitHubIssueWithRepo:
    """Tests for the GitHubIssueWithRepo wrapper class."""

    def test_key_returns_repo_and_number(self) -> None:
        """The key property should return 'org/repo#number'."""
        issue = make_github_issue(number=123)
        wrapper = GitHubIssueWithRepo(issue, "org/repo")
        assert wrapper.key == "org/repo#123"

    # NOTE: Individual forwarding tests for number, title, body, state,
    # author, is_pull_request, head_ref, base_ref, and draft have been
    # removed because the parametrized test_property_forwarded (below)
    # already covers all of these properties.  See DS-948.

    def test_assignees_forwarded(self) -> None:
        """The assignees property should forward to the wrapped issue."""
        issue = make_github_issue(assignees=["alice", "bob"])
        wrapper = GitHubIssueWithRepo(issue, "org/repo")
        assert wrapper.assignees == ["alice", "bob"]

    def test_labels_forwarded(self) -> None:
        """The labels property should forward to the wrapped issue."""
        issue = make_github_issue(labels=["bug", "critical"])
        wrapper = GitHubIssueWithRepo(issue, "org/repo")
        assert wrapper.labels == ["bug", "critical"]

    def test_repo_url_forwarded(self) -> None:
        """The repo_url property should forward to the wrapped issue."""
        url = "https://github.com/org/repo/issues/1"
        issue = make_github_issue(repo_url=url)
        wrapper = GitHubIssueWithRepo(issue, "org/repo")
        assert wrapper.repo_url == url

    def test_parent_issue_number_forwarded(self) -> None:
        """The parent_issue_number property should forward to the wrapped issue."""
        issue = make_github_issue(parent_issue_number=10)
        wrapper = GitHubIssueWithRepo(issue, "org/repo")
        assert wrapper.parent_issue_number == 10

    def test_parent_issue_number_none(self) -> None:
        """Parent issue number should be None when not set."""
        issue = make_github_issue(parent_issue_number=None)
        wrapper = GitHubIssueWithRepo(issue, "org/repo")
        assert wrapper.parent_issue_number is None

    def test_getattr_fallback(self) -> None:
        """__getattr__ should delegate to the wrapped issue for unknown attributes."""
        issue = make_github_issue(number=7)
        # GitHubIssue is a dataclass, so it has __dataclass_fields__
        wrapper = GitHubIssueWithRepo(issue, "org/repo")
        # Access an attribute that exists on the underlying GitHubIssue dataclass
        # but is not an explicit forwarding property on GitHubIssueWithRepo.
        # The 'key' property on GitHubIssue returns "#7", but our wrapper overrides it.
        # Instead, test __getattr__ by accessing a dunder attribute from the dataclass.
        assert hasattr(wrapper, "__dataclass_fields__")

    @pytest.mark.parametrize(
        "prop,value",
        [
            ("number", 42),
            ("title", "My title"),
            ("body", "Issue body text"),
            ("state", "closed"),
            ("author", "alice"),
            ("is_pull_request", True),
            ("head_ref", "feature-branch"),
            ("base_ref", "main"),
            ("draft", True),
        ],
    )
    def test_property_forwarded(self, prop: str, value: object) -> None:
        """Each forwarding property should delegate to the wrapped issue."""
        kwargs = {prop: value}
        issue = make_github_issue(**kwargs)
        wrapper = GitHubIssueWithRepo(issue, "org/repo")
        assert getattr(wrapper, prop) == value


# ===========================================================================
# TestPollingResult
# ===========================================================================


class TestPollingResult:
    """Tests for the PollingResult dataclass."""

    def test_default_values(self) -> None:
        """Default constructor should set all counters to zero."""
        result = PollingResult()
        assert result.issues_found == 0
        assert result.submitted_count == 0
        assert result.queued_count == 0

    def test_custom_values(self) -> None:
        """Constructor should accept custom values for all fields."""
        result = PollingResult(issues_found=5, submitted_count=3, queued_count=2)
        assert result.issues_found == 5
        assert result.submitted_count == 3
        assert result.queued_count == 2


# ===========================================================================
# TestGroupedOrchestrations
# ===========================================================================


class TestGroupedOrchestrations:
    """Tests for the GroupedOrchestrations named tuple."""

    def test_constructor(self) -> None:
        """Named tuple should hold jira and github lists."""
        jira_list: list[Orchestration] = [make_orchestration(name="j1")]
        github_list: list[Orchestration] = [
            make_orchestration(name="g1", source="github", project_number=1, project_owner="org")
        ]
        grouped = GroupedOrchestrations(jira=jira_list, github=github_list)
        assert grouped.jira is jira_list
        assert grouped.github is github_list

    def test_named_field_access(self) -> None:
        """Fields should be accessible by name and by index."""
        jira_list: list[Orchestration] = []
        github_list: list[Orchestration] = []
        grouped = GroupedOrchestrations(jira=jira_list, github=github_list)
        # Named access
        assert grouped.jira == []
        assert grouped.github == []
        # Positional access
        assert grouped[0] == []
        assert grouped[1] == []


# ===========================================================================
# TestPollCoordinatorInit
# ===========================================================================


class TestPollCoordinatorInit:
    """Tests for PollCoordinator.__init__."""

    def test_init_with_all_args(self) -> None:
        """PollCoordinator should store config and both pollers."""
        config = make_config()
        jira_poller = MockJiraPoller()
        github_poller = MockGitHubPoller()
        coordinator = PollCoordinator(
            config=config,
            jira_poller=jira_poller,
            github_poller=github_poller,
        )
        assert coordinator._config is config
        assert coordinator._jira_poller is jira_poller
        assert coordinator._github_poller is github_poller

    def test_init_with_defaults(self) -> None:
        """PollCoordinator should accept None for both pollers."""
        config = make_config()
        coordinator = PollCoordinator(config=config)
        assert coordinator._config is config
        assert coordinator._jira_poller is None
        assert coordinator._github_poller is None


# ===========================================================================
# TestCreateCycleDedupSet
# ===========================================================================


class TestCreateCycleDedupSet:
    """Tests for PollCoordinator.create_cycle_dedup_set()."""

    def test_returns_empty_set(self) -> None:
        """create_cycle_dedup_set should return a fresh empty set."""
        coordinator = _make_poll_coordinator()
        result = coordinator.create_cycle_dedup_set()
        assert isinstance(result, set)
        assert len(result) == 0


# ===========================================================================
# TestCheckAndMarkSubmitted
# ===========================================================================


class TestCheckAndMarkSubmitted:
    """Tests for PollCoordinator.check_and_mark_submitted()."""

    def test_first_submission_returns_true(self) -> None:
        """First time a pair is seen, check_and_mark_submitted should return True."""
        coordinator = _make_poll_coordinator()
        submitted = coordinator.create_cycle_dedup_set()
        result = coordinator.check_and_mark_submitted(submitted, "TEST-1", "orch-a")
        assert result is True

    def test_duplicate_returns_false(self) -> None:
        """Second time the same pair is seen, it should return False."""
        coordinator = _make_poll_coordinator()
        submitted = coordinator.create_cycle_dedup_set()
        coordinator.check_and_mark_submitted(submitted, "TEST-1", "orch-a")
        result = coordinator.check_and_mark_submitted(submitted, "TEST-1", "orch-a")
        assert result is False

    def test_different_pairs_both_return_true(self) -> None:
        """Different issue/orchestration pairs should each return True on first check."""
        coordinator = _make_poll_coordinator()
        submitted = coordinator.create_cycle_dedup_set()
        assert coordinator.check_and_mark_submitted(submitted, "TEST-1", "orch-a") is True
        assert coordinator.check_and_mark_submitted(submitted, "TEST-2", "orch-b") is True
        # Same issue, different orch
        assert coordinator.check_and_mark_submitted(submitted, "TEST-1", "orch-b") is True
        # Same orch, different issue
        assert coordinator.check_and_mark_submitted(submitted, "TEST-2", "orch-a") is True


# ===========================================================================
# TestPollJiraTriggers
# ===========================================================================


class TestPollJiraTriggers:
    """Tests for PollCoordinator.poll_jira_triggers()."""

    def test_no_jira_poller_returns_empty(self) -> None:
        """When no jira_poller is configured, should return empty results."""
        coordinator = _make_poll_coordinator(jira_poller=None)
        orch = make_orchestration(name="jira-orch", project="PROJ", tags=["review"])
        router = MagicMock()

        results, issues_found, error_count = coordinator.poll_jira_triggers([orch], router)

        assert results == []
        assert issues_found == 0
        assert error_count == 0
        router.route_matched_only.assert_not_called()

    def test_shutdown_requested_returns_early(self) -> None:
        """When shutdown_requested is True, polling should stop immediately."""
        jira_poller = MockJiraPoller(issues=[make_issue(key="TEST-1")])
        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orch = make_orchestration(name="jira-orch", project="PROJ", tags=["review"])
        router = MagicMock()

        results, issues_found, error_count = coordinator.poll_jira_triggers(
            [orch], router, shutdown_requested=True
        )

        assert results == []
        assert issues_found == 0
        assert error_count == 0
        # Poller should not have been called because shutdown was immediate
        assert len(jira_poller.poll_calls) == 0

    def test_successful_poll_with_routing(self) -> None:
        """Successful polling should fetch issues and route them."""
        issues = [make_issue(key="TEST-1", labels=["review"])]
        jira_poller = MockJiraPoller(issues=issues)
        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orch = make_orchestration(name="jira-orch", project="PROJ", tags=["review"])

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])

        results, issues_found, error_count = coordinator.poll_jira_triggers([orch], router)

        assert issues_found == 1
        assert error_count == 0
        assert len(jira_poller.poll_calls) == 1
        router.route_matched_only.assert_called_once_with(issues)

    def test_log_callback_called(self) -> None:
        """Log callback should be invoked for polling start and result."""
        issues = [make_issue(key="TEST-1")]
        jira_poller = MockJiraPoller(issues=issues)
        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orch = make_orchestration(name="jira-orch", project="PROJ", tags=["review"])

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])
        log_cb = MagicMock()

        coordinator.poll_jira_triggers([orch], router, log_callback=log_cb)

        # Should be called at least twice: once for "Polling Jira..." and once for "Found N..."
        assert log_cb.call_count >= 2
        # First call: polling start
        first_call_args = log_cb.call_args_list[0]
        assert first_call_args[0][0] == "jira-orch"
        assert first_call_args[0][1] == logging.INFO
        assert "Polling Jira" in first_call_args[0][2]
        # Second call: issues found
        second_call_args = log_cb.call_args_list[1]
        assert second_call_args[0][0] == "jira-orch"
        assert "Found 1 Jira issues" in second_call_args[0][2]

    def test_duplicate_trigger_dedup(self) -> None:
        """Two orchestrations with the same trigger key should only poll once."""
        issues = [make_issue(key="TEST-1")]
        jira_poller = MockJiraPoller(issues=issues)
        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        # Two orchestrations with identical project+tags produce the same trigger_key
        orch1 = make_orchestration(name="orch-a", project="PROJ", tags=["review"])
        orch2 = make_orchestration(name="orch-b", project="PROJ", tags=["review"])

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])

        coordinator.poll_jira_triggers([orch1, orch2], router)

        # Only one poll call because the trigger_key is the same
        assert len(jira_poller.poll_calls) == 1

    def test_different_triggers_poll_separately(self) -> None:
        """Orchestrations with different trigger keys should each be polled."""
        jira_poller = MockJiraPoller(issues=[])
        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orch1 = make_orchestration(name="orch-a", project="PROJ", tags=["review"])
        orch2 = make_orchestration(name="orch-b", project="PROJ", tags=["deploy"])

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])

        coordinator.poll_jira_triggers([orch1, orch2], router)

        # Two poll calls because the tags differ
        assert len(jira_poller.poll_calls) == 2


# ===========================================================================
# TestPollGitHubTriggers
# ===========================================================================


class TestPollGitHubTriggers:
    """Tests for PollCoordinator.poll_github_triggers()."""

    def test_no_github_poller_returns_empty(self) -> None:
        """When no github_poller is configured, should return empty results."""
        coordinator = _make_poll_coordinator(github_poller=None)
        orch = make_orchestration(
            name="gh-orch", source="github", project_number=1, project_owner="org"
        )
        router = MagicMock()

        results, issues_found, error_count = coordinator.poll_github_triggers([orch], router)

        assert results == []
        assert issues_found == 0
        assert error_count == 0

    def test_shutdown_requested_returns_early(self) -> None:
        """When shutdown_requested is True, polling should stop immediately."""
        gh_issues = [make_github_issue(number=1)]
        github_poller = MockGitHubPoller(issues=gh_issues)
        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orch = make_orchestration(
            name="gh-orch", source="github", project_number=1, project_owner="org"
        )
        router = MagicMock()

        results, issues_found, error_count = coordinator.poll_github_triggers(
            [orch], router, shutdown_requested=True
        )

        assert results == []
        assert issues_found == 0
        assert error_count == 0
        assert len(github_poller.poll_calls) == 0

    def test_successful_poll_with_routing_and_repo_context(self) -> None:
        """Successful polling should wrap issues with repo context and route them."""
        gh_issues = [
            make_github_issue(
                number=42,
                repo_url="https://github.com/myorg/myrepo/issues/42",
            )
        ]
        github_poller = MockGitHubPoller(issues=gh_issues)
        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orch = make_orchestration(
            name="gh-orch", source="github", project_number=1, project_owner="org"
        )

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])

        results, issues_found, error_count = coordinator.poll_github_triggers([orch], router)

        assert issues_found == 1
        assert error_count == 0
        assert len(github_poller.poll_calls) == 1
        # Verify route_matched_only was called with wrapped issues
        router.route_matched_only.assert_called_once()
        routed_issues = router.route_matched_only.call_args[0][0]
        assert len(routed_issues) == 1
        assert isinstance(routed_issues[0], GitHubIssueWithRepo)
        assert routed_issues[0].key == "myorg/myrepo#42"

    def test_log_callback_called(self) -> None:
        """Log callback should be invoked for polling start and result."""
        gh_issues = [make_github_issue(number=1)]
        github_poller = MockGitHubPoller(issues=gh_issues)
        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orch = make_orchestration(
            name="gh-orch", source="github", project_number=1, project_owner="org"
        )

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])
        log_cb = MagicMock()

        coordinator.poll_github_triggers([orch], router, log_callback=log_cb)

        assert log_cb.call_count >= 2
        first_call_args = log_cb.call_args_list[0]
        assert first_call_args[0][0] == "gh-orch"
        assert first_call_args[0][1] == logging.INFO
        assert "Polling GitHub" in first_call_args[0][2]
        second_call_args = log_cb.call_args_list[1]
        assert "Found 1 GitHub issues/PRs" in second_call_args[0][2]

    def test_duplicate_trigger_dedup(self) -> None:
        """Two orchestrations with the same GitHub trigger key should only poll once."""
        github_poller = MockGitHubPoller(issues=[])
        coordinator = _make_poll_coordinator(github_poller=github_poller)
        # Identical project_number + project_owner + no filter/labels = same key
        orch1 = make_orchestration(
            name="orch-a", source="github", project_number=1, project_owner="org"
        )
        orch2 = make_orchestration(
            name="orch-b", source="github", project_number=1, project_owner="org"
        )

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])

        coordinator.poll_github_triggers([orch1, orch2], router)

        assert len(github_poller.poll_calls) == 1

    def test_add_repo_context_invalid_url_logs_warning(self) -> None:
        """Issues with invalid repo URLs should be logged as warnings and excluded."""
        gh_issues = [
            make_github_issue(number=99, repo_url="not-a-valid-url"),
        ]
        github_poller = MockGitHubPoller(issues=gh_issues)
        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orch = make_orchestration(
            name="gh-orch", source="github", project_number=1, project_owner="org"
        )

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])

        with patch("sentinel.poll_coordinator.logger") as mock_logger:
            coordinator.poll_github_triggers([orch], router)
            mock_logger.warning.assert_called_once()
            warning_args = mock_logger.warning.call_args[0]
            assert "Could not extract repo" in warning_args[0]
            assert warning_args[1] == 99

    def test_add_repo_context_mixed_valid_and_invalid(self) -> None:
        """Valid URLs produce wrapped issues; invalid URLs are skipped."""
        gh_issues = [
            make_github_issue(
                number=1,
                repo_url="https://github.com/org/repo/issues/1",
            ),
            make_github_issue(
                number=2,
                repo_url="invalid-url",
            ),
            make_github_issue(
                number=3,
                repo_url="https://github.com/org/other/pull/3",
            ),
        ]
        github_poller = MockGitHubPoller(issues=gh_issues)
        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orch = make_orchestration(
            name="gh-orch", source="github", project_number=1, project_owner="org"
        )

        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])

        coordinator.poll_github_triggers([orch], router)

        routed_issues = router.route_matched_only.call_args[0][0]
        # Only 2 of 3 issues have valid URLs
        assert len(routed_issues) == 2
        assert routed_issues[0].key == "org/repo#1"
        assert routed_issues[1].key == "org/other#3"


# ===========================================================================
# TestConstructIssueUrl
# ===========================================================================


class TestConstructIssueUrl:
    """Tests for PollCoordinator.construct_issue_url()."""

    def test_jira_with_base_url(self) -> None:
        """Jira source with a configured base_url should produce a browse URL."""
        config = make_config(jira_base_url="https://jira.example.com")
        coordinator = _make_poll_coordinator(config=config)
        issue = make_issue(key="PROJ-123")
        orch = make_orchestration(name="jira-orch", project="PROJ", tags=["review"])

        url = coordinator.construct_issue_url(issue, orch)

        assert url == "https://jira.example.com/browse/PROJ-123"

    def test_jira_with_trailing_slash_base_url(self) -> None:
        """Trailing slashes in base_url should be stripped."""
        config = make_config(jira_base_url="https://jira.example.com/")
        coordinator = _make_poll_coordinator(config=config)
        issue = make_issue(key="PROJ-456")
        orch = make_orchestration(name="jira-orch", project="PROJ", tags=["review"])

        url = coordinator.construct_issue_url(issue, orch)

        assert url == "https://jira.example.com/browse/PROJ-456"

    def test_jira_without_base_url(self) -> None:
        """Jira source without a base_url should return empty string."""
        config = make_config(jira_base_url="")
        coordinator = _make_poll_coordinator(config=config)
        issue = make_issue(key="PROJ-789")
        orch = make_orchestration(name="jira-orch", project="PROJ", tags=["review"])

        url = coordinator.construct_issue_url(issue, orch)

        assert url == ""

    def test_github_with_repo_url(self) -> None:
        """GitHub source with repo_url on the issue should return the repo_url."""
        coordinator = _make_poll_coordinator()
        gh_issue = make_github_issue(
            number=42,
            repo_url="https://github.com/org/repo/issues/42",
        )
        wrapper = GitHubIssueWithRepo(gh_issue, "org/repo")
        orch = make_orchestration(
            name="gh-orch", source="github", project_number=1, project_owner="org"
        )

        url = coordinator.construct_issue_url(wrapper, orch)

        assert url == "https://github.com/org/repo/issues/42"

    def test_github_without_repo_url_but_with_github_context(self) -> None:
        """GitHub source without repo_url should fall back to agent.github context."""
        coordinator = _make_poll_coordinator()
        gh_issue = make_github_issue(number=55, repo_url="")
        wrapper = GitHubIssueWithRepo(gh_issue, "myorg/myrepo")
        orch = make_orchestration(
            name="gh-orch",
            source="github",
            project_number=1,
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo"),
        )

        url = coordinator.construct_issue_url(wrapper, orch)

        assert url == "https://github.com/myorg/myrepo/issues/55"

    def test_github_with_no_url_info(self) -> None:
        """GitHub source with no repo_url and no github context returns empty string."""
        coordinator = _make_poll_coordinator()
        gh_issue = make_github_issue(number=10, repo_url="")
        wrapper = GitHubIssueWithRepo(gh_issue, "org/repo")
        # Orchestration without github context
        orch = make_orchestration(
            name="gh-orch",
            source="github",
            project_number=1,
            project_owner="org",
        )

        url = coordinator.construct_issue_url(wrapper, orch)

        assert url == ""

    def test_github_issue_key_with_hash(self) -> None:
        """GitHub issue key containing '#' should extract the number correctly."""
        coordinator = _make_poll_coordinator()
        gh_issue = make_github_issue(number=77, repo_url="")
        wrapper = GitHubIssueWithRepo(gh_issue, "org/repo")
        # wrapper.key == "org/repo#77"
        orch = make_orchestration(
            name="gh-orch",
            source="github",
            project_number=1,
            github=GitHubContext(host="github.com", org="org", repo="repo"),
        )

        url = coordinator.construct_issue_url(wrapper, orch)

        assert url == "https://github.com/org/repo/issues/77"

    def test_github_context_with_custom_host(self) -> None:
        """GitHub context with a custom host should use that host in the URL."""
        coordinator = _make_poll_coordinator()
        gh_issue = make_github_issue(number=5, repo_url="")
        wrapper = GitHubIssueWithRepo(gh_issue, "myorg/myrepo")
        orch = make_orchestration(
            name="gh-orch",
            source="github",
            project_number=1,
            github=GitHubContext(host="github.mycorp.com", org="myorg", repo="myrepo"),
        )

        url = coordinator.construct_issue_url(wrapper, orch)

        assert url == "https://github.mycorp.com/myorg/myrepo/issues/5"

    def test_unknown_trigger_source_returns_empty(self) -> None:
        """An unknown trigger source should return an empty string."""
        coordinator = _make_poll_coordinator()
        issue = make_issue(key="TEST-1")
        # Create orchestration with a non-standard source by directly manipulating trigger
        orch = make_orchestration(name="custom-orch")
        orch.trigger.source = "unknown"

        url = coordinator.construct_issue_url(issue, orch)

        assert url == ""


# ===========================================================================
# TestGroupOrchestrationsBySource
# ===========================================================================


class TestGroupOrchestrationsBySource:
    """Tests for PollCoordinator.group_orchestrations_by_source()."""

    def test_mixed_sources(self) -> None:
        """Mixed Jira and GitHub orchestrations should be correctly grouped."""
        coordinator = _make_poll_coordinator()
        jira_orch = make_orchestration(name="jira-orch", project="PROJ", tags=["review"])
        github_orch = make_orchestration(
            name="gh-orch", source="github", project_number=1, project_owner="org"
        )

        result = coordinator.group_orchestrations_by_source([jira_orch, github_orch])

        assert isinstance(result, GroupedOrchestrations)
        assert len(result.jira) == 1
        assert len(result.github) == 1
        assert result.jira[0].name == "jira-orch"
        assert result.github[0].name == "gh-orch"

    def test_all_jira(self) -> None:
        """All Jira orchestrations should end up in the jira list."""
        coordinator = _make_poll_coordinator()
        orch1 = make_orchestration(name="orch-a", project="A")
        orch2 = make_orchestration(name="orch-b", project="B")

        result = coordinator.group_orchestrations_by_source([orch1, orch2])

        assert len(result.jira) == 2
        assert len(result.github) == 0

    def test_all_github(self) -> None:
        """All GitHub orchestrations should end up in the github list."""
        coordinator = _make_poll_coordinator()
        orch1 = make_orchestration(
            name="gh-a", source="github", project_number=1, project_owner="org"
        )
        orch2 = make_orchestration(
            name="gh-b", source="github", project_number=2, project_owner="org"
        )

        result = coordinator.group_orchestrations_by_source([orch1, orch2])

        assert len(result.jira) == 0
        assert len(result.github) == 2

    def test_empty_list(self) -> None:
        """An empty orchestrations list should produce empty groups."""
        coordinator = _make_poll_coordinator()

        result = coordinator.group_orchestrations_by_source([])

        assert len(result.jira) == 0
        assert len(result.github) == 0

    def test_unknown_source_grouped_as_jira(self) -> None:
        """Orchestrations with non-github sources default to the jira bucket."""
        coordinator = _make_poll_coordinator()
        orch = make_orchestration(name="mystery")
        # Manually set a non-standard source
        orch.trigger.source = "something_else"

        result = coordinator.group_orchestrations_by_source([orch])

        # The implementation uses: if source == GITHUB -> github, else -> jira
        assert len(result.jira) == 1
        assert len(result.github) == 0
