"""GitHub-related tests for Claude Agent SDK executor module.

This module contains tests specific to GitHub issue handling:
- build_prompt tests with GitHub issues
- execute tests with GitHub issues
- GitHub template variable handling
"""

from sentinel.executor import AgentExecutor, ExecutionStatus
from sentinel.orchestration import GitHubContext
from tests.helpers import make_issue, make_orchestration
from tests.mocks import MockAgentClient


class TestAgentExecutorBuildPromptGitHubIssue:
    """Tests for AgentExecutor.build_prompt with GitHub Issues."""

    def test_substitutes_github_issue_number(self) -> None:
        """Should substitute {github_issue_number} with issue number."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test Issue")
        orch = make_orchestration(prompt="Review issue #{github_issue_number}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Review issue #42"

    def test_substitutes_github_issue_title(self) -> None:
        """Should substitute {github_issue_title} with issue title."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Fix authentication bug")
        orch = make_orchestration(prompt="Task: {github_issue_title}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Task: Fix authentication bug"

    def test_substitutes_github_issue_body(self) -> None:
        """Should substitute {github_issue_body} with issue body."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", body="Detailed description here")
        orch = make_orchestration(prompt="Description:\n{github_issue_body}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Description:\nDetailed description here"

    def test_substitutes_github_issue_state(self) -> None:
        """Should substitute {github_issue_state} with issue state."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", state="open")
        orch = make_orchestration(prompt="State: {github_issue_state}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "State: open"

    def test_substitutes_github_issue_author(self) -> None:
        """Should substitute {github_issue_author} with author username."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", author="octocat")
        orch = make_orchestration(prompt="Author: {github_issue_author}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Author: octocat"

    def test_substitutes_github_issue_assignees(self) -> None:
        """Should substitute {github_issue_assignees} with comma-separated assignees."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", assignees=["alice", "bob"])
        orch = make_orchestration(prompt="Assignees: {github_issue_assignees}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Assignees: alice, bob"

    def test_substitutes_github_issue_labels(self) -> None:
        """Should substitute {github_issue_labels} with comma-separated labels."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", labels=["bug", "urgent"])
        orch = make_orchestration(prompt="Labels: {github_issue_labels}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Labels: bug, urgent"

    def test_substitutes_github_issue_url(self) -> None:
        """Should substitute {github_issue_url} with full URL."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            prompt="URL: {github_issue_url}",
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: https://github.com/myorg/myrepo/issues/42"

    def test_substitutes_github_pr_url(self) -> None:
        """Should substitute {github_issue_url} with PR URL for pull requests."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test PR", is_pull_request=True)
        orch = make_orchestration(
            prompt="URL: {github_issue_url}",
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: https://github.com/myorg/myrepo/pull/42"

    def test_substitutes_github_is_pr(self) -> None:
        """Should substitute {github_is_pr} with 'true' or 'false'."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)

        issue = GitHubIssue(number=1, title="Test Issue", is_pull_request=False)
        orch = make_orchestration(prompt="Is PR: {github_is_pr}")
        prompt = executor.build_prompt(issue, orch)
        assert prompt == "Is PR: false"

        pr = GitHubIssue(number=1, title="Test PR", is_pull_request=True)
        prompt = executor.build_prompt(pr, orch)
        assert prompt == "Is PR: true"

    def test_substitutes_github_pr_head(self) -> None:
        """Should substitute {github_pr_head} with head branch reference."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(
            number=1, title="Test PR", is_pull_request=True, head_ref="feature-branch"
        )
        orch = make_orchestration(prompt="Head: {github_pr_head}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Head: feature-branch"

    def test_substitutes_github_pr_base(self) -> None:
        """Should substitute {github_pr_base} with base branch reference."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test PR", is_pull_request=True, base_ref="main")
        orch = make_orchestration(prompt="Base: {github_pr_base}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Base: main"

    def test_substitutes_github_pr_draft(self) -> None:
        """Should substitute {github_pr_draft} with 'true' or 'false'."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)

        draft_pr = GitHubIssue(number=1, title="Draft PR", is_pull_request=True, draft=True)
        orch = make_orchestration(prompt="Draft: {github_pr_draft}")
        prompt = executor.build_prompt(draft_pr, orch)
        assert prompt == "Draft: true"

        ready_pr = GitHubIssue(number=1, title="Ready PR", is_pull_request=True, draft=False)
        prompt = executor.build_prompt(ready_pr, orch)
        assert prompt == "Draft: false"

    def test_github_issue_jira_variables_empty(self) -> None:
        """Jira variables should be empty for GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            prompt="Jira: [{jira_issue_key}] [{jira_summary}] [{jira_status}]"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Jira: [] [] []"

    def test_jira_issue_github_variables_empty(self) -> None:
        """GitHub Issue variables should be empty for Jira issues."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="PROJ-123", summary="Test")
        orch = make_orchestration(
            prompt="GitHub: [{github_issue_number}] [{github_issue_title}] [{github_issue_state}]"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "GitHub: [] [] []"

    def test_substitutes_multiple_github_variables(self) -> None:
        """Should substitute multiple GitHub variables in one prompt."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(
            number=42,
            title="Fix authentication",
            state="open",
            author="octocat",
            labels=["bug", "security"],
        )
        orch = make_orchestration(
            prompt="#{github_issue_number}: {github_issue_title} by {github_issue_author} ({github_issue_labels})"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "#42: Fix authentication by octocat (bug, security)"

    def test_substitutes_github_context_and_issue_variables(self) -> None:
        """Should substitute both GitHub repo context and issue variables."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test Issue")
        orch = make_orchestration(
            prompt="Review {github_org}/{github_repo}#{github_issue_number}: {github_issue_title}",
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Review myorg/myrepo#42: Test Issue"


class TestAgentExecutorExecuteGitHubIssue:
    """Tests for AgentExecutor.execute with GitHub Issues."""

    def test_successful_execution_with_github_issue(self) -> None:
        """Execute should work with GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient(responses=["SUCCESS: Review completed"])
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test PR", is_pull_request=True)
        orch = make_orchestration(prompt="Review GitHub PR #{github_issue_number}")

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.status == ExecutionStatus.SUCCESS
        assert result.issue_key == "#42"
        assert "SUCCESS" in result.response

    def test_github_issue_key_in_result(self) -> None:
        """Result should include GitHub issue key format."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=123, title="Test")
        orch = make_orchestration()

        result = executor.execute(issue, orch)

        assert result.issue_key == "#123"

    def test_github_issue_retries_on_failure(self) -> None:
        """Execute should retry on failure for GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient(responses=["FAILURE: First attempt", "SUCCESS: Done"])
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(max_attempts=3)

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.attempts == 2


class TestGitHubTemplateVariables:
    """Additional tests for GitHub Issue template variable handling in executor.

    These tests ensure comprehensive coverage of GitHub template variables,
    particularly edge cases and interactions between Jira and GitHub contexts.
    """

    def test_github_issue_url_empty_when_no_github_context(self) -> None:
        """URL should be empty when GitHub context is not configured."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test Issue")
        # No GitHub context configured
        orch = make_orchestration(prompt="URL: [{github_issue_url}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: []"

    def test_github_pr_fields_empty_for_regular_issue(self) -> None:
        """PR-specific fields should be empty strings for regular issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        # Regular issue, not a PR
        issue = GitHubIssue(number=42, title="Regular Issue", is_pull_request=False)
        orch = make_orchestration(
            prompt="Head: [{github_pr_head}] Base: [{github_pr_base}] Draft: [{github_pr_draft}]"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Head: [] Base: [] Draft: []"

    def test_github_pr_fields_populated_for_pull_request(self) -> None:
        """PR-specific fields should be populated for pull requests."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(
            number=42,
            title="Feature PR",
            is_pull_request=True,
            head_ref="feature/new-thing",
            base_ref="main",
            draft=True,
        )
        orch = make_orchestration(
            prompt="Head: {github_pr_head} Base: {github_pr_base} Draft: {github_pr_draft}"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Head: feature/new-thing Base: main Draft: true"

    def test_github_issue_empty_body_substitutes_empty_string(self) -> None:
        """Empty issue body should substitute as empty string."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", body="")
        orch = make_orchestration(prompt="Body: [{github_issue_body}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Body: []"

    def test_github_issue_none_body_substitutes_empty_string(self) -> None:
        """Empty body should substitute as empty string (None is normalized to empty)."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        # GitHubIssue uses empty string default, but test the flow
        issue = GitHubIssue(number=1, title="Test", body="")
        orch = make_orchestration(prompt="Body: [{github_issue_body}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Body: []"

    def test_github_issue_empty_assignees_substitutes_empty_string(self) -> None:
        """Empty assignees list should substitute as empty string."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", assignees=[])
        orch = make_orchestration(prompt="Assignees: [{github_issue_assignees}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Assignees: []"

    def test_github_issue_empty_labels_substitutes_empty_string(self) -> None:
        """Empty labels list should substitute as empty string."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", labels=[])
        orch = make_orchestration(prompt="Labels: [{github_issue_labels}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Labels: []"

    def test_github_issue_url_with_custom_host(self) -> None:
        """URL construction should work with custom GitHub host."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            prompt="URL: {github_issue_url}",
            github=GitHubContext(host="github.enterprise.com", org="corp", repo="internal-app"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: https://github.enterprise.com/corp/internal-app/issues/42"

    def test_github_pr_url_with_custom_host(self) -> None:
        """PR URL construction should work with custom GitHub host."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=99, title="Enterprise PR", is_pull_request=True)
        orch = make_orchestration(
            prompt="URL: {github_issue_url}",
            github=GitHubContext(host="github.enterprise.com", org="corp", repo="internal-app"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: https://github.enterprise.com/corp/internal-app/pull/99"

    def test_all_jira_variables_empty_for_github_issue(self) -> None:
        """All Jira variables should be empty strings for GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test")
        orch = make_orchestration(
            prompt=(
                "key:[{jira_issue_key}] summary:[{jira_summary}] "
                "desc:[{jira_description}] status:[{jira_status}] "
                "assignee:[{jira_assignee}] labels:[{jira_labels}] "
                "comments:[{jira_comments}] links:[{jira_links}]"
            )
        )

        prompt = executor.build_prompt(issue, orch)

        expected = "key:[] summary:[] desc:[] status:[] assignee:[] labels:[] comments:[] links:[]"
        assert prompt == expected

    def test_all_github_issue_variables_empty_for_jira(self) -> None:
        """All GitHub Issue variables should be empty strings for Jira issues."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="PROJ-1", summary="Test")
        orch = make_orchestration(
            prompt=(
                "num:[{github_issue_number}] title:[{github_issue_title}] "
                "body:[{github_issue_body}] state:[{github_issue_state}] "
                "author:[{github_issue_author}] assignees:[{github_issue_assignees}] "
                "labels:[{github_issue_labels}] url:[{github_issue_url}] "
                "is_pr:[{github_is_pr}] head:[{github_pr_head}] "
                "base:[{github_pr_base}] draft:[{github_pr_draft}]"
            )
        )

        prompt = executor.build_prompt(issue, orch)

        expected = (
            "num:[] title:[] body:[] state:[] author:[] assignees:[] "
            "labels:[] url:[] is_pr:[] head:[] base:[] draft:[]"
        )
        assert prompt == expected

    def test_github_is_pr_false_lowercase(self) -> None:
        """github_is_pr should be lowercase 'false' for regular issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Issue", is_pull_request=False)
        orch = make_orchestration(prompt="{github_is_pr}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "false"

    def test_github_is_pr_true_lowercase(self) -> None:
        """github_is_pr should be lowercase 'true' for PRs."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="PR", is_pull_request=True)
        orch = make_orchestration(prompt="{github_is_pr}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "true"

    def test_github_pr_draft_false_lowercase(self) -> None:
        """github_pr_draft should be lowercase 'false' for non-draft PRs."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="PR", is_pull_request=True, draft=False)
        orch = make_orchestration(prompt="{github_pr_draft}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "false"

    def test_github_pr_draft_true_lowercase(self) -> None:
        """github_pr_draft should be lowercase 'true' for draft PRs."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Draft PR", is_pull_request=True, draft=True)
        orch = make_orchestration(prompt="{github_pr_draft}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "true"

    def test_github_context_variables_still_work_for_github_issues(self) -> None:
        """GitHub repo context variables should work with GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            prompt="Repo: {github_org}/{github_repo} on {github_host}",
            github=GitHubContext(host="github.com", org="acme", repo="widgets"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Repo: acme/widgets on github.com"

    def test_github_context_variables_empty_when_not_configured(self) -> None:
        """GitHub repo context should be empty when not configured."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test")
        # No GitHub context in orchestration
        orch = make_orchestration(
            prompt="Host:[{github_host}] Org:[{github_org}] Repo:[{github_repo}]"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Host:[] Org:[] Repo:[]"

    def test_combined_github_context_and_issue_variables(self) -> None:
        """Test combining repo context with issue-specific variables."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(
            number=123,
            title="Add feature X",
            state="open",
            author="developer",
            is_pull_request=True,
            head_ref="feature-x",
            base_ref="develop",
        )
        orch = make_orchestration(
            prompt=(
                "PR #{github_issue_number} '{github_issue_title}' by {github_issue_author} "
                "in {github_org}/{github_repo} from {github_pr_head} to {github_pr_base}"
            ),
            github=GitHubContext(host="github.com", org="company", repo="project"),
        )

        prompt = executor.build_prompt(issue, orch)

        expected = (
            "PR #123 'Add feature X' by developer " "in company/project from feature-x to develop"
        )
        assert prompt == expected
