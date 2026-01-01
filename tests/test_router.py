"""Tests for tag-based router module."""

from sentinel.orchestration import AgentConfig, Orchestration, TriggerConfig
from sentinel.poller import JiraIssue
from sentinel.router import Router, RoutingResult


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


def make_orchestration(
    name: str = "test-orch",
    project: str = "",
    tags: list[str] | None = None,
) -> Orchestration:
    """Helper to create an Orchestration for testing."""
    return Orchestration(
        name=name,
        trigger=TriggerConfig(
            project=project,
            tags=tags or [],
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
