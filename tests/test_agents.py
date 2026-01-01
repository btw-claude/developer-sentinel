"""Tests for agent tools package."""

from __future__ import annotations

from typing import Any

import pytest

from sentinel.agents import (
    ConfluenceToolClient,
    GitHubToolClient,
    JiraToolClient,
    ParameterType,
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    get_confluence_tools,
    get_github_tools,
    get_jira_tools,
    get_tool_schemas_for_orchestration,
    get_tools_for_orchestration,
)

# =============================================================================
# Test Fixtures - Mock Clients
# =============================================================================


class MockJiraClient(JiraToolClient):
    """Mock Jira client for testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_issue(self, issue_key: str) -> dict[str, Any]:
        self.calls.append(("get_issue", {"issue_key": issue_key}))
        return {"key": issue_key, "summary": "Test issue"}

    def update_issue(
        self,
        issue_key: str,
        summary: str | None = None,
        description: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "update_issue",
                {
                    "issue_key": issue_key,
                    "summary": summary,
                    "description": description,
                    "priority": priority,
                },
            )
        )
        return {"key": issue_key, "updated": True}

    def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        self.calls.append(("add_comment", {"issue_key": issue_key, "body": body}))
        return {"id": "12345", "body": body}

    def transition_issue(self, issue_key: str, transition_id: str) -> dict[str, Any]:
        self.calls.append(
            ("transition_issue", {"issue_key": issue_key, "transition_id": transition_id})
        )
        return {"success": True}

    def add_label(self, issue_key: str, label: str) -> dict[str, Any]:
        self.calls.append(("add_label", {"issue_key": issue_key, "label": label}))
        return {"success": True}

    def remove_label(self, issue_key: str, label: str) -> dict[str, Any]:
        self.calls.append(("remove_label", {"issue_key": issue_key, "label": label}))
        return {"success": True}

    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        self.calls.append(("search_issues", {"jql": jql, "max_results": max_results}))
        return [{"key": "TEST-1", "summary": "Test"}]

    def get_transitions(self, issue_key: str) -> list[dict[str, Any]]:
        self.calls.append(("get_transitions", {"issue_key": issue_key}))
        return [{"id": "21", "name": "Done"}]


class MockConfluenceClient(ConfluenceToolClient):
    """Mock Confluence client for testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_page(self, page_id: str) -> dict[str, Any]:
        self.calls.append(("get_page", {"page_id": page_id}))
        return {"id": page_id, "title": "Test Page"}

    def create_page(
        self,
        space_key: str,
        title: str,
        body: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "create_page",
                {"space_key": space_key, "title": title, "body": body, "parent_id": parent_id},
            )
        )
        return {"id": "123", "title": title}

    def update_page(
        self,
        page_id: str,
        title: str,
        body: str,
        version: int,
    ) -> dict[str, Any]:
        self.calls.append(
            ("update_page", {"page_id": page_id, "title": title, "body": body, "version": version})
        )
        return {"id": page_id, "version": version + 1}

    def search_content(self, cql: str, limit: int = 25) -> list[dict[str, Any]]:
        self.calls.append(("search_content", {"cql": cql, "limit": limit}))
        return [{"id": "1", "title": "Result"}]

    def add_comment(self, page_id: str, body: str) -> dict[str, Any]:
        self.calls.append(("add_comment", {"page_id": page_id, "body": body}))
        return {"id": "456", "body": body}

    def get_page_by_title(self, space_key: str, title: str) -> dict[str, Any] | None:
        self.calls.append(("get_page_by_title", {"space_key": space_key, "title": title}))
        if title == "Not Found":
            return None
        return {"id": "123", "title": title}


class MockGitHubClient(GitHubToolClient):
    """Mock GitHub client for testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_issue(self, issue_number: int) -> dict[str, Any]:
        self.calls.append(("get_issue", {"issue_number": issue_number}))
        return {"number": issue_number, "title": "Test Issue"}

    def get_pull_request(self, pr_number: int) -> dict[str, Any]:
        self.calls.append(("get_pull_request", {"pr_number": pr_number}))
        return {"number": pr_number, "title": "Test PR"}

    def list_pr_files(self, pr_number: int) -> list[dict[str, Any]]:
        self.calls.append(("list_pr_files", {"pr_number": pr_number}))
        return [{"filename": "test.py", "status": "modified"}]

    def get_file_contents(self, path: str, ref: str | None = None) -> dict[str, Any]:
        self.calls.append(("get_file_contents", {"path": path, "ref": ref}))
        return {"path": path, "content": "test content"}

    def create_issue_comment(self, issue_number: int, body: str) -> dict[str, Any]:
        self.calls.append(("create_issue_comment", {"issue_number": issue_number, "body": body}))
        return {"id": 789, "body": body}

    def create_pr_review(
        self,
        pr_number: int,
        body: str,
        event: str,
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "create_pr_review",
                {"pr_number": pr_number, "body": body, "event": event, "comments": comments},
            )
        )
        return {"id": 101, "state": event}

    def list_pr_comments(self, pr_number: int) -> list[dict[str, Any]]:
        self.calls.append(("list_pr_comments", {"pr_number": pr_number}))
        return [{"id": 1, "body": "Comment"}]


# =============================================================================
# Base Classes Tests
# =============================================================================


class TestToolParameter:
    """Tests for ToolParameter."""

    def test_to_schema_basic(self) -> None:
        param = ToolParameter(
            name="test",
            description="A test parameter",
            type=ParameterType.STRING,
        )
        schema = param.to_schema()

        assert schema["type"] == "string"
        assert schema["description"] == "A test parameter"

    def test_to_schema_with_enum(self) -> None:
        param = ToolParameter(
            name="status",
            description="Status value",
            type=ParameterType.STRING,
            enum=["open", "closed"],
        )
        schema = param.to_schema()

        assert schema["enum"] == ["open", "closed"]

    def test_to_schema_with_array(self) -> None:
        param = ToolParameter(
            name="tags",
            description="List of tags",
            type=ParameterType.ARRAY,
            items_type=ParameterType.STRING,
        )
        schema = param.to_schema()

        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_to_schema_with_default(self) -> None:
        param = ToolParameter(
            name="limit",
            description="Max results",
            type=ParameterType.INTEGER,
            default=50,
        )
        schema = param.to_schema()

        assert schema["default"] == 50


class TestToolSchema:
    """Tests for ToolSchema."""

    def test_to_function_schema_minimal(self) -> None:
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
        )
        func_schema = schema.to_function_schema()

        assert func_schema["name"] == "test_tool"
        assert func_schema["description"] == "A test tool"
        assert func_schema["input_schema"]["type"] == "object"
        assert func_schema["input_schema"]["properties"] == {}
        assert "required" not in func_schema["input_schema"]

    def test_to_function_schema_with_params(self) -> None:
        schema = ToolSchema(
            name="get_issue",
            description="Get an issue",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="Issue key",
                    type=ParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="expand",
                    description="Fields to expand",
                    type=ParameterType.STRING,
                    required=False,
                ),
            ],
        )
        func_schema = schema.to_function_schema()

        assert "issue_key" in func_schema["input_schema"]["properties"]
        assert "expand" in func_schema["input_schema"]["properties"]
        assert func_schema["input_schema"]["required"] == ["issue_key"]


class TestToolResult:
    """Tests for ToolResult."""

    def test_ok_result(self) -> None:
        result = ToolResult.ok({"key": "TEST-1"})

        assert result.success is True
        assert result.data == {"key": "TEST-1"}
        assert result.error is None

    def test_fail_result(self) -> None:
        result = ToolResult.fail("Something went wrong", "ERROR_CODE")

        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"
        assert result.error_code == "ERROR_CODE"

    def test_to_response_success(self) -> None:
        result = ToolResult.ok({"id": 123})
        response = result.to_response()

        assert response == {"success": True, "data": {"id": 123}}

    def test_to_response_failure(self) -> None:
        result = ToolResult.fail("Error", "ERR")
        response = result.to_response()

        assert response == {"success": False, "error": "Error", "error_code": "ERR"}


# =============================================================================
# Jira Tools Tests
# =============================================================================


class TestJiraTools:
    """Tests for Jira tools."""

    @pytest.fixture
    def client(self) -> MockJiraClient:
        return MockJiraClient()

    @pytest.fixture
    def tools(self, client: MockJiraClient) -> list[Tool]:
        return get_jira_tools(client)

    def test_get_jira_tools_returns_all_tools(self, tools: list[Tool]) -> None:
        tool_names = [t.name for t in tools]
        assert "jira_get_issue" in tool_names
        assert "jira_update_issue" in tool_names
        assert "jira_add_comment" in tool_names
        assert "jira_transition_issue" in tool_names
        assert "jira_add_label" in tool_names
        assert "jira_remove_label" in tool_names
        assert "jira_search_issues" in tool_names
        assert "jira_get_transitions" in tool_names

    def test_all_tools_have_jira_category(self, tools: list[Tool]) -> None:
        for tool in tools:
            assert tool.category == "jira"

    def test_get_issue_tool_executes(self, client: MockJiraClient) -> None:
        tools = get_jira_tools(client)
        get_issue = next(t for t in tools if t.name == "jira_get_issue")

        result = get_issue.execute(issue_key="TEST-123")

        assert result.success
        assert result.data["key"] == "TEST-123"
        assert client.calls[0] == ("get_issue", {"issue_key": "TEST-123"})

    def test_add_comment_tool_executes(self, client: MockJiraClient) -> None:
        tools = get_jira_tools(client)
        add_comment = next(t for t in tools if t.name == "jira_add_comment")

        result = add_comment.execute(issue_key="TEST-1", body="Test comment")

        assert result.success
        assert client.calls[0][0] == "add_comment"

    def test_search_issues_tool_executes(self, client: MockJiraClient) -> None:
        tools = get_jira_tools(client)
        search = next(t for t in tools if t.name == "jira_search_issues")

        result = search.execute(jql="project = TEST", max_results=10)

        assert result.success
        assert result.data["count"] == 1

    def test_tool_validates_required_params(self, client: MockJiraClient) -> None:
        tools = get_jira_tools(client)
        get_issue = next(t for t in tools if t.name == "jira_get_issue")

        result = get_issue.execute()  # Missing issue_key

        assert not result.success
        assert result.error_code == "MISSING_PARAMETER"


# =============================================================================
# Confluence Tools Tests
# =============================================================================


class TestConfluenceTools:
    """Tests for Confluence tools."""

    @pytest.fixture
    def client(self) -> MockConfluenceClient:
        return MockConfluenceClient()

    @pytest.fixture
    def tools(self, client: MockConfluenceClient) -> list[Tool]:
        return get_confluence_tools(client)

    def test_get_confluence_tools_returns_all_tools(self, tools: list[Tool]) -> None:
        tool_names = [t.name for t in tools]
        assert "confluence_get_page" in tool_names
        assert "confluence_get_page_by_title" in tool_names
        assert "confluence_create_page" in tool_names
        assert "confluence_update_page" in tool_names
        assert "confluence_search_content" in tool_names
        assert "confluence_add_comment" in tool_names

    def test_all_tools_have_confluence_category(self, tools: list[Tool]) -> None:
        for tool in tools:
            assert tool.category == "confluence"

    def test_get_page_tool_executes(self, client: MockConfluenceClient) -> None:
        tools = get_confluence_tools(client)
        get_page = next(t for t in tools if t.name == "confluence_get_page")

        result = get_page.execute(page_id="123")

        assert result.success
        assert result.data["id"] == "123"

    def test_get_page_by_title_found(self, client: MockConfluenceClient) -> None:
        tools = get_confluence_tools(client)
        get_by_title = next(t for t in tools if t.name == "confluence_get_page_by_title")

        result = get_by_title.execute(space_key="DEV", title="Test Page")

        assert result.success
        assert result.data["found"] is True

    def test_get_page_by_title_not_found(self, client: MockConfluenceClient) -> None:
        tools = get_confluence_tools(client)
        get_by_title = next(t for t in tools if t.name == "confluence_get_page_by_title")

        result = get_by_title.execute(space_key="DEV", title="Not Found")

        assert result.success
        assert result.data["found"] is False

    def test_create_page_tool_executes(self, client: MockConfluenceClient) -> None:
        tools = get_confluence_tools(client)
        create_page = next(t for t in tools if t.name == "confluence_create_page")

        result = create_page.execute(
            space_key="DEV",
            title="New Page",
            body="<p>Content</p>",
        )

        assert result.success
        assert result.data["title"] == "New Page"


# =============================================================================
# GitHub Tools Tests
# =============================================================================


class TestGitHubTools:
    """Tests for GitHub tools."""

    @pytest.fixture
    def client(self) -> MockGitHubClient:
        return MockGitHubClient()

    @pytest.fixture
    def tools(self, client: MockGitHubClient) -> list[Tool]:
        return get_github_tools(client)

    def test_get_github_tools_returns_all_tools(self, tools: list[Tool]) -> None:
        tool_names = [t.name for t in tools]
        assert "github_get_issue" in tool_names
        assert "github_get_pull_request" in tool_names
        assert "github_list_pr_files" in tool_names
        assert "github_get_file_contents" in tool_names
        assert "github_create_comment" in tool_names
        assert "github_create_pr_review" in tool_names
        assert "github_list_pr_comments" in tool_names

    def test_all_tools_have_github_category(self, tools: list[Tool]) -> None:
        for tool in tools:
            assert tool.category == "github"

    def test_get_issue_tool_executes(self, client: MockGitHubClient) -> None:
        tools = get_github_tools(client)
        get_issue = next(t for t in tools if t.name == "github_get_issue")

        result = get_issue.execute(issue_number=42)

        assert result.success
        assert result.data["number"] == 42

    def test_get_pr_tool_executes(self, client: MockGitHubClient) -> None:
        tools = get_github_tools(client)
        get_pr = next(t for t in tools if t.name == "github_get_pull_request")

        result = get_pr.execute(pr_number=100)

        assert result.success
        assert result.data["number"] == 100

    def test_create_review_validates_event(self, client: MockGitHubClient) -> None:
        tools = get_github_tools(client)
        create_review = next(t for t in tools if t.name == "github_create_pr_review")

        result = create_review.execute(
            pr_number=1,
            body="LGTM",
            event="INVALID_EVENT",
        )

        assert not result.success
        assert result.error_code == "INVALID_ENUM"


# =============================================================================
# Registry Tests
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_jira_tools(MockJiraClient())
        registry.register_confluence_tools(MockConfluenceClient())
        registry.register_github_tools(MockGitHubClient())
        return registry

    def test_registry_has_all_categories(self, registry: ToolRegistry) -> None:
        assert "jira" in registry.categories
        assert "confluence" in registry.categories
        assert "github" in registry.categories

    def test_get_tools_by_category(self, registry: ToolRegistry) -> None:
        jira_tools = registry.get_tools_by_category("jira")

        assert len(jira_tools) == 8
        assert all(t.category == "jira" for t in jira_tools)

    def test_get_tool_by_name(self, registry: ToolRegistry) -> None:
        tool = registry.get_tool("jira_get_issue")

        assert tool is not None
        assert tool.name == "jira_get_issue"

    def test_get_tool_not_found(self, registry: ToolRegistry) -> None:
        tool = registry.get_tool("nonexistent")

        assert tool is None

    def test_get_tools_for_categories(self, registry: ToolRegistry) -> None:
        tools = registry.get_tools_for_categories(["jira", "github"])

        categories = {t.category for t in tools}
        assert categories == {"jira", "github"}

    def test_get_all_schemas(self, registry: ToolRegistry) -> None:
        schemas = registry.get_all_schemas()

        assert len(schemas) == registry.tool_count
        assert all("name" in s for s in schemas)
        assert all("input_schema" in s for s in schemas)

    def test_duplicate_registration_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registry.register_jira_tools(MockJiraClient())


class TestToolRegistryFunctions:
    """Tests for registry helper functions."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_jira_tools(MockJiraClient())
        registry.register_confluence_tools(MockConfluenceClient())
        registry.register_github_tools(MockGitHubClient())
        return registry

    def test_get_tools_for_orchestration(self, registry: ToolRegistry) -> None:
        tools = get_tools_for_orchestration(registry, ["jira", "github"])

        tool_categories = {t.category for t in tools}
        assert tool_categories == {"jira", "github"}
        assert "confluence" not in tool_categories

    def test_get_tool_schemas_for_orchestration(self, registry: ToolRegistry) -> None:
        schemas = get_tool_schemas_for_orchestration(registry, ["confluence"])

        assert len(schemas) == 6  # All Confluence tools
        assert all("confluence" in s["name"] for s in schemas)
