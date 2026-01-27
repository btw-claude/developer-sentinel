"""Tests for tag-based router module."""

from typing import Literal

from sentinel.github_poller import GitHubIssue
from sentinel.orchestration import AgentConfig, Orchestration, TriggerConfig
from sentinel.poller import JiraIssue
from sentinel.router import AnyIssue, Router, RoutingResult


def make_issue(
    key: str = "TEST-1",
    labels: list[str] | None = None,
    summary: str = "Test issue",
) -> JiraIssue:
    """Helper to create a JiraIssue for testing."""
    return JiraIssue(
        key=key,
        summary=summary,
        labels=labels or [],
    )


def make_github_issue(
    number: int = 1,
    labels: list[str] | None = None,
    title: str = "Test GitHub issue",
) -> GitHubIssue:
    """Helper to create a GitHubIssue for testing."""
    return GitHubIssue(
        number=number,
        title=title,
        labels=labels or [],
    )


def make_orchestration(
    name: str = "test-orch",
    source: Literal["jira", "github"] = "jira",
    project: str = "",
    repo: str = "",
    tags: list[str] | None = None,
    labels: list[str] | None = None,
) -> Orchestration:
    """Helper to create an Orchestration for testing."""
    return Orchestration(
        name=name,
        trigger=TriggerConfig(
            source=source,
            project=project,
            repo=repo,
            tags=tags or [],
            labels=labels or [],
        ),
        agent=AgentConfig(prompt="Test prompt"),
    )


class TestRoutingResult:
    """Tests for RoutingResult dataclass."""

    def test_matched_with_orchestrations(self) -> None:
        issue = make_issue()
        orch = make_orchestration()
        result = RoutingResult(issue=issue, orchestrations=[orch])
        assert result.matched is True

    def test_not_matched_without_orchestrations(self) -> None:
        issue = make_issue()
        result = RoutingResult(issue=issue, orchestrations=[])
        assert result.matched is False


class TestRouterMatchesTrigger:
    """Tests for Router._matches_trigger."""

    def test_matches_with_no_filters(self) -> None:
        router = Router([])
        issue = make_issue(labels=["anything"])
        orch = make_orchestration()  # No project or tags
        assert router._matches_trigger(issue, orch) is True

    def test_matches_project(self) -> None:
        router = Router([])
        issue = make_issue(key="PROJ-123")
        orch = make_orchestration(project="PROJ")
        assert router._matches_trigger(issue, orch) is True

    def test_matches_project_case_insensitive(self) -> None:
        router = Router([])
        issue = make_issue(key="proj-123")
        orch = make_orchestration(project="PROJ")
        assert router._matches_trigger(issue, orch) is True

    def test_rejects_wrong_project(self) -> None:
        router = Router([])
        issue = make_issue(key="OTHER-123")
        orch = make_orchestration(project="PROJ")
        assert router._matches_trigger(issue, orch) is False

    def test_matches_single_tag(self) -> None:
        router = Router([])
        issue = make_issue(labels=["needs-review", "bug"])
        orch = make_orchestration(tags=["needs-review"])
        assert router._matches_trigger(issue, orch) is True

    def test_matches_multiple_tags(self) -> None:
        router = Router([])
        issue = make_issue(labels=["needs-review", "urgent", "bug"])
        orch = make_orchestration(tags=["needs-review", "urgent"])
        assert router._matches_trigger(issue, orch) is True

    def test_matches_tags_case_insensitive(self) -> None:
        router = Router([])
        issue = make_issue(labels=["Needs-Review", "URGENT"])
        orch = make_orchestration(tags=["needs-review", "urgent"])
        assert router._matches_trigger(issue, orch) is True

    def test_rejects_missing_tag(self) -> None:
        router = Router([])
        issue = make_issue(labels=["needs-review"])
        orch = make_orchestration(tags=["needs-review", "urgent"])
        assert router._matches_trigger(issue, orch) is False

    def test_rejects_no_matching_tags(self) -> None:
        router = Router([])
        issue = make_issue(labels=["bug", "feature"])
        orch = make_orchestration(tags=["needs-review"])
        assert router._matches_trigger(issue, orch) is False

    def test_matches_project_and_tags(self) -> None:
        router = Router([])
        issue = make_issue(key="PROJ-123", labels=["needs-review"])
        orch = make_orchestration(project="PROJ", tags=["needs-review"])
        assert router._matches_trigger(issue, orch) is True

    def test_rejects_wrong_project_with_correct_tags(self) -> None:
        router = Router([])
        issue = make_issue(key="OTHER-123", labels=["needs-review"])
        orch = make_orchestration(project="PROJ", tags=["needs-review"])
        assert router._matches_trigger(issue, orch) is False

    def test_rejects_correct_project_with_missing_tags(self) -> None:
        router = Router([])
        issue = make_issue(key="PROJ-123", labels=["bug"])
        orch = make_orchestration(project="PROJ", tags=["needs-review"])
        assert router._matches_trigger(issue, orch) is False


class TestRouterRoute:
    """Tests for Router.route."""

    def test_routes_to_matching_orchestration(self) -> None:
        orch = make_orchestration(name="review", tags=["needs-review"])
        router = Router([orch])
        issue = make_issue(labels=["needs-review"])

        result = router.route(issue)

        assert result.matched is True
        assert len(result.orchestrations) == 1
        assert result.orchestrations[0].name == "review"

    def test_routes_to_multiple_orchestrations(self) -> None:
        orch1 = make_orchestration(name="review", tags=["needs-review"])
        orch2 = make_orchestration(name="docs", tags=["needs-review"])
        router = Router([orch1, orch2])
        issue = make_issue(labels=["needs-review"])

        result = router.route(issue)

        assert result.matched is True
        assert len(result.orchestrations) == 2

    def test_routes_to_specific_orchestrations(self) -> None:
        orch1 = make_orchestration(name="review", tags=["needs-review"])
        orch2 = make_orchestration(name="docs", tags=["needs-docs"])
        router = Router([orch1, orch2])
        issue = make_issue(labels=["needs-review"])

        result = router.route(issue)

        assert result.matched is True
        assert len(result.orchestrations) == 1
        assert result.orchestrations[0].name == "review"

    def test_no_match_returns_empty(self) -> None:
        orch = make_orchestration(name="review", tags=["needs-review"])
        router = Router([orch])
        issue = make_issue(labels=["bug"])

        result = router.route(issue)

        assert result.matched is False
        assert len(result.orchestrations) == 0

    def test_empty_router_returns_no_matches(self) -> None:
        router = Router([])
        issue = make_issue(labels=["anything"])

        result = router.route(issue)

        assert result.matched is False


class TestRouterRouteAll:
    """Tests for Router.route_all."""

    def test_routes_all_issues(self) -> None:
        orch = make_orchestration(name="review", tags=["needs-review"])
        router = Router([orch])
        issues = [
            make_issue(key="TEST-1", labels=["needs-review"]),
            make_issue(key="TEST-2", labels=["bug"]),
            make_issue(key="TEST-3", labels=["needs-review"]),
        ]

        results = router.route_all(issues)

        assert len(results) == 3
        assert results[0].matched is True
        assert results[1].matched is False
        assert results[2].matched is True

    def test_empty_issues_returns_empty(self) -> None:
        orch = make_orchestration(name="review", tags=["needs-review"])
        router = Router([orch])

        results = router.route_all([])

        assert len(results) == 0


class TestRouterRouteMatchedOnly:
    """Tests for Router.route_matched_only."""

    def test_returns_only_matched(self) -> None:
        orch = make_orchestration(name="review", tags=["needs-review"])
        router = Router([orch])
        issues = [
            make_issue(key="TEST-1", labels=["needs-review"]),
            make_issue(key="TEST-2", labels=["bug"]),
            make_issue(key="TEST-3", labels=["needs-review"]),
        ]

        results = router.route_matched_only(issues)

        assert len(results) == 2
        assert all(r.matched for r in results)
        assert results[0].issue.key == "TEST-1"
        assert results[1].issue.key == "TEST-3"

    def test_returns_empty_when_no_matches(self) -> None:
        orch = make_orchestration(name="review", tags=["needs-review"])
        router = Router([orch])
        issues = [
            make_issue(key="TEST-1", labels=["bug"]),
            make_issue(key="TEST-2", labels=["feature"]),
        ]

        results = router.route_matched_only(issues)

        assert len(results) == 0


class TestGitHubIssueRouting:
    """Tests for routing GitHub issues."""

    def test_github_issue_matches_github_trigger(self) -> None:
        router = Router([])
        issue = make_github_issue(labels=["needs-review"])
        orch = make_orchestration(source="github", tags=["needs-review"])
        assert router._matches_trigger(issue, orch) is True

    def test_github_issue_rejects_jira_trigger(self) -> None:
        router = Router([])
        issue = make_github_issue(labels=["needs-review"])
        orch = make_orchestration(source="jira", tags=["needs-review"])
        assert router._matches_trigger(issue, orch) is False

    def test_jira_issue_rejects_github_trigger(self) -> None:
        router = Router([])
        issue = make_issue(labels=["needs-review"])
        orch = make_orchestration(source="github", tags=["needs-review"])
        assert router._matches_trigger(issue, orch) is False

    def test_github_issue_matches_tags_case_insensitive(self) -> None:
        router = Router([])
        issue = make_github_issue(labels=["Needs-Review", "URGENT"])
        orch = make_orchestration(source="github", tags=["needs-review", "urgent"])
        assert router._matches_trigger(issue, orch) is True

    def test_github_issue_rejects_missing_tag(self) -> None:
        router = Router([])
        issue = make_github_issue(labels=["needs-review"])
        orch = make_orchestration(source="github", tags=["needs-review", "urgent"])
        assert router._matches_trigger(issue, orch) is False

    def test_routes_github_issue_to_github_orchestration(self) -> None:
        orch = make_orchestration(name="github-review", source="github", tags=["needs-review"])
        router = Router([orch])
        issue = make_github_issue(labels=["needs-review"])

        result = router.route(issue)

        assert result.matched is True
        assert len(result.orchestrations) == 1
        assert result.orchestrations[0].name == "github-review"

    def test_routes_to_correct_source_orchestration(self) -> None:
        jira_orch = make_orchestration(name="jira-review", source="jira", tags=["needs-review"])
        github_orch = make_orchestration(name="github-review", source="github", tags=["needs-review"])
        router = Router([jira_orch, github_orch])

        jira_issue = make_issue(labels=["needs-review"])
        github_issue = make_github_issue(labels=["needs-review"])

        jira_result = router.route(jira_issue)
        github_result = router.route(github_issue)

        assert jira_result.matched is True
        assert len(jira_result.orchestrations) == 1
        assert jira_result.orchestrations[0].name == "jira-review"

        assert github_result.matched is True
        assert len(github_result.orchestrations) == 1
        assert github_result.orchestrations[0].name == "github-review"

    def test_routes_mixed_issues(self) -> None:
        jira_orch = make_orchestration(name="jira-review", source="jira", tags=["needs-review"])
        github_orch = make_orchestration(name="github-review", source="github", tags=["needs-review"])
        router = Router([jira_orch, github_orch])

        issues: list[AnyIssue] = [
            make_issue(key="PROJ-1", labels=["needs-review"]),
            make_github_issue(number=1, labels=["needs-review"]),
            make_issue(key="PROJ-2", labels=["bug"]),
            make_github_issue(number=2, labels=["bug"]),
        ]

        results = router.route_matched_only(issues)

        assert len(results) == 2
        # First match should be the Jira issue
        assert isinstance(results[0].issue, JiraIssue)
        assert results[0].orchestrations[0].name == "jira-review"
        # Second match should be the GitHub issue
        assert isinstance(results[1].issue, GitHubIssue)
        assert results[1].orchestrations[0].name == "github-review"


class TestGitHubLabelsFieldRouting:
    """Tests for routing GitHub issues using the labels field (DS-333).

    These tests verify the labels field routing functionality:
    - Test GitHub issue matches with correct labels
    - Test GitHub issue rejected with missing labels
    - Test case-insensitive label matching
    """

    def test_github_issue_matches_with_correct_labels(self) -> None:
        """GitHub issue should match orchestration with matching labels."""
        router = Router([])
        issue = make_github_issue(labels=["bug", "urgent"])
        orch = make_orchestration(source="github", labels=["bug", "urgent"])
        assert router._matches_trigger(issue, orch) is True

    def test_github_issue_rejected_with_missing_labels(self) -> None:
        """GitHub issue should be rejected if it doesn't have all required labels."""
        router = Router([])
        issue = make_github_issue(labels=["bug"])  # Missing "urgent"
        orch = make_orchestration(source="github", labels=["bug", "urgent"])
        assert router._matches_trigger(issue, orch) is False

    def test_github_issue_labels_case_insensitive_matching(self) -> None:
        """GitHub issue label matching should be case-insensitive."""
        router = Router([])
        issue = make_github_issue(labels=["BUG", "Urgent"])
        orch = make_orchestration(source="github", labels=["bug", "URGENT"])
        assert router._matches_trigger(issue, orch) is True

    def test_github_issue_matches_empty_labels_filter(self) -> None:
        """GitHub issue should match when no labels filter is specified."""
        router = Router([])
        issue = make_github_issue(labels=["anything"])
        orch = make_orchestration(source="github", labels=[])
        assert router._matches_trigger(issue, orch) is True

    def test_github_issue_matches_with_extra_labels(self) -> None:
        """GitHub issue should match if it has all required labels plus extras."""
        router = Router([])
        issue = make_github_issue(labels=["bug", "urgent", "extra-label"])
        orch = make_orchestration(source="github", labels=["bug", "urgent"])
        assert router._matches_trigger(issue, orch) is True

    def test_github_issue_no_labels_rejected_with_labels_filter(self) -> None:
        """GitHub issue with no labels should be rejected when labels filter is set."""
        router = Router([])
        issue = make_github_issue(labels=[])
        orch = make_orchestration(source="github", labels=["bug"])
        assert router._matches_trigger(issue, orch) is False

    def test_routes_github_issue_with_labels_to_matching_orchestration(self) -> None:
        """GitHub issue should be routed to orchestration with matching labels."""
        orch = make_orchestration(
            name="bug-triage",
            source="github",
            labels=["bug", "needs-triage"],
        )
        router = Router([orch])
        issue = make_github_issue(labels=["bug", "needs-triage", "high-priority"])

        result = router.route(issue)

        assert result.matched is True
        assert len(result.orchestrations) == 1
        assert result.orchestrations[0].name == "bug-triage"

    def test_github_labels_and_tags_can_be_used_together(self) -> None:
        """Test that both labels (new) and tags (deprecated) fields are checked.

        Note: The 'tags' field is DEPRECATED and will be removed in a future release.
        New code should use 'labels' exclusively. This test ensures backward
        compatibility during the deprecation period where both fields may be used
        together. When 'tags' is removed, this test should be updated to only
        test 'labels' functionality.
        """
        router = Router([])
        issue = make_github_issue(labels=["bug", "urgent", "team-a"])
        orch = make_orchestration(
            source="github",
            tags=["team-a"],  # Deprecated tags field - will be removed in future release
            labels=["bug"],   # New labels field - preferred approach
        )
        # Both must match
        assert router._matches_trigger(issue, orch) is True

    def test_github_labels_rejected_when_tags_missing(self) -> None:
        """Issue should be rejected if labels match but tags don't."""
        router = Router([])
        issue = make_github_issue(labels=["bug", "urgent"])
        orch = make_orchestration(
            source="github",
            tags=["team-a"],  # Issue doesn't have this
            labels=["bug"],   # Issue has this
        )
        assert router._matches_trigger(issue, orch) is False

    def test_routes_to_multiple_orchestrations_with_labels(self) -> None:
        """Issue should route to all matching orchestrations based on labels."""
        orch1 = make_orchestration(
            name="bug-handler",
            source="github",
            labels=["bug"],
        )
        orch2 = make_orchestration(
            name="urgent-handler",
            source="github",
            labels=["urgent"],
        )
        orch3 = make_orchestration(
            name="feature-handler",
            source="github",
            labels=["feature"],
        )
        router = Router([orch1, orch2, orch3])
        issue = make_github_issue(labels=["bug", "urgent"])

        result = router.route(issue)

        assert result.matched is True, (
            "Issue with ['bug', 'urgent'] labels should match at least one orchestration"
        )
        assert len(result.orchestrations) == 2, (
            f"Expected 2 matching orchestrations (bug-handler and urgent-handler), "
            f"but got {len(result.orchestrations)}: {[o.name for o in result.orchestrations]}"
        )
        names = [o.name for o in result.orchestrations]
        assert "bug-handler" in names, (
            f"bug-handler should match issue with 'bug' label, but matched: {names}"
        )
        assert "urgent-handler" in names, (
            f"urgent-handler should match issue with 'urgent' label, but matched: {names}"
        )
        assert "feature-handler" not in names, (
            f"feature-handler should NOT match issue without 'feature' label, but it did"
        )
