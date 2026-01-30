"""Tests for the exception handling utilities."""

from __future__ import annotations

from typing import Any

import pytest

from sentinel.agents.base import Tool, ToolParameter, ToolResult, ToolSchema
from sentinel.agents.exceptions import (
    ConfigurationError,
    ValidationError,
    _extract_context,
    handle_tool_exceptions,
)


class MockToolSchema(ToolSchema):
    """Mock tool schema for testing."""

    def __init__(self, name: str = "mock_tool") -> None:
        super().__init__(
            name=name,
            description="A mock tool for testing",
            category="test",
            parameters=[
                ToolParameter(
                    name="issue_key",
                    description="The issue key",
                    required=True,
                ),
            ],
        )


class BaseMockTool(Tool):
    """Base mock tool class for testing."""

    def __init__(self) -> None:
        self._schema = MockToolSchema()

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


class TestExtractContext:
    """Tests for _extract_context helper function."""

    def test_extracts_issue_key(self) -> None:
        kwargs = {"issue_key": "TEST-123"}
        assert _extract_context(kwargs) == "TEST-123"

    def test_extracts_page_id(self) -> None:
        kwargs = {"page_id": "12345"}
        assert _extract_context(kwargs) == "12345"

    def test_extracts_pr_number_with_hash(self) -> None:
        kwargs = {"pr_number": 42}
        assert _extract_context(kwargs) == "#42"

    def test_extracts_issue_number_with_hash(self) -> None:
        kwargs = {"issue_number": 100}
        assert _extract_context(kwargs) == "#100"

    def test_extracts_path(self) -> None:
        kwargs = {"path": "src/main.py"}
        assert _extract_context(kwargs) == "src/main.py"

    def test_extracts_cql(self) -> None:
        kwargs = {"cql": "space = DEV"}
        assert _extract_context(kwargs) == "space = DEV"

    def test_extracts_jql(self) -> None:
        kwargs = {"jql": "project = TEST"}
        assert _extract_context(kwargs) == "project = TEST"

    def test_returns_empty_for_no_context(self) -> None:
        kwargs = {"unknown_param": "value"}
        assert _extract_context(kwargs) == ""

    def test_priority_order(self) -> None:
        # issue_key has higher priority than page_id
        kwargs = {"issue_key": "TEST-1", "page_id": "123"}
        assert _extract_context(kwargs) == "TEST-1"


class TestHandleToolExceptionsDecorator:
    """Tests for handle_tool_exceptions decorator."""

    def test_successful_execution(self) -> None:
        """Test that successful execution works without modification."""

        class SuccessTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="TEST_ERROR",
                error_message_template="Failed: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult.ok({"data": "success"})

        tool = SuccessTool()
        result = tool.execute(issue_key="TEST-1")

        assert result.success is True
        assert result.data == {"data": "success"}

    def test_handles_oserror(self) -> None:
        """Test that OSError is caught and converted to ToolResult.fail()."""

        class OSErrorTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="TEST_ERROR",
                error_message_template="Failed for {context}: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise OSError("Connection refused")

        tool = OSErrorTool()
        result = tool.execute(issue_key="TEST-1")

        assert result.success is False
        assert result.error_code == "TEST_ERROR"
        assert "TEST-1" in result.error
        assert "Connection refused" in result.error

    def test_handles_timeout_error(self) -> None:
        """Test that TimeoutError is caught and converted to ToolResult.fail()."""

        class TimeoutTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="TIMEOUT_ERROR",
                error_message_template="Timeout for {context}: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise TimeoutError("Request timed out")

        tool = TimeoutTool()
        result = tool.execute(issue_key="PROJ-123")

        assert result.success is False
        assert result.error_code == "TIMEOUT_ERROR"
        assert "PROJ-123" in result.error
        assert "timed out" in result.error

    def test_handles_key_error(self) -> None:
        """Test that KeyError is caught and converted to ToolResult.fail()."""

        class KeyErrorTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="DATA_ERROR",
                error_message_template="Data error: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise KeyError("missing_key")

        tool = KeyErrorTool()
        result = tool.execute(issue_key="TEST-1")

        assert result.success is False
        assert result.error_code == "DATA_ERROR"
        assert "missing_key" in result.error

    def test_typeerror_propagates(self) -> None:
        """Test that TypeError propagates (indicates programming bug).

        TypeError should NOT be caught because it typically indicates a
        programming bug (wrong argument types) rather than a data or
        configuration error. Let TypeError propagate so bugs are visible.
        """

        class TypeErrorTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="DATA_ERROR",
                error_message_template="Type error: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise TypeError("Expected string")

        tool = TypeErrorTool()
        with pytest.raises(TypeError, match="Expected string"):
            tool.execute(issue_key="TEST-1")

    def test_handles_configuration_error(self) -> None:
        """Test that ConfigurationError is caught and converted to ToolResult.fail()."""

        class ConfigErrorTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="CONFIG_ERROR",
                error_message_template="Configuration error: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise ConfigurationError("Missing required field 'api_key'")

        tool = ConfigErrorTool()
        result = tool.execute(issue_key="TEST-1")

        assert result.success is False
        assert result.error_code == "CONFIG_ERROR"
        assert "Missing required field 'api_key'" in result.error

    def test_handles_validation_error(self) -> None:
        """Test that ValidationError is caught and converted to ToolResult.fail()."""

        class ValidationErrorTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="VALIDATION_ERROR",
                error_message_template="Validation error: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise ValidationError("Issue key must match pattern 'PROJ-\\d+'")

        tool = ValidationErrorTool()
        result = tool.execute(issue_key="TEST-1")

        assert result.success is False
        assert result.error_code == "VALIDATION_ERROR"
        assert "Issue key must match pattern" in result.error

    def test_handles_value_error(self) -> None:
        """Test that ValueError is caught and converted to ToolResult.fail()."""

        class ValueErrorTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="DATA_ERROR",
                error_message_template="Value error: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise ValueError("Invalid value")

        tool = ValueErrorTool()
        result = tool.execute(issue_key="TEST-1")

        assert result.success is False
        assert result.error_code == "DATA_ERROR"
        assert "Invalid value" in result.error

    def test_handles_runtime_error(self) -> None:
        """Test that RuntimeError is caught and converted to ToolResult.fail()."""

        class RuntimeErrorTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="RUNTIME_ERROR",
                error_message_template="Runtime error: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise RuntimeError("Unexpected condition")

        tool = RuntimeErrorTool()
        result = tool.execute(issue_key="TEST-1")

        assert result.success is False
        assert result.error_code == "RUNTIME_ERROR"
        assert "Unexpected condition" in result.error

    def test_validates_params_before_execution(self) -> None:
        """Test that parameter validation runs before the decorated function."""

        class ValidationTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="TEST_ERROR",
                error_message_template="Failed: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                # Should never reach here due to validation
                return ToolResult.ok({"data": "success"})

        tool = ValidationTool()
        result = tool.execute()  # Missing required issue_key

        assert result.success is False
        assert result.error_code == "MISSING_PARAMETER"

    def test_context_extraction_in_error_messages(self) -> None:
        """Test that context is properly extracted for different parameter types."""
        from sentinel.agents.base import ParameterType

        class PRTool(Tool):
            def __init__(self) -> None:
                self._schema = ToolSchema(
                    name="pr_tool",
                    description="PR tool",
                    category="github",
                    parameters=[
                        ToolParameter(
                            name="pr_number",
                            description="PR number",
                            type=ParameterType.INTEGER,
                            required=True,
                        ),
                    ],
                )

            @property
            def schema(self) -> ToolSchema:
                return self._schema

            @handle_tool_exceptions(
                error_code="GITHUB_ERROR",
                error_message_template="Failed for PR {context}: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise OSError("API error")

        tool = PRTool()
        result = tool.execute(pr_number=42)

        assert result.success is False
        assert "#42" in result.error  # PR number should be prefixed with #

    def test_preserves_function_metadata(self) -> None:
        """Test that the decorator preserves function metadata."""

        class MetadataTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="TEST_ERROR",
                error_message_template="Failed: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                """Execute the tool."""
                return ToolResult.ok({})

        tool = MetadataTool()
        assert tool.execute.__name__ == "execute"
        assert "Execute the tool" in (tool.execute.__doc__ or "")

    def test_unhandled_exceptions_propagate(self) -> None:
        """Test that unhandled exception types still propagate."""

        class CustomError(Exception):
            pass

        class UnhandledExceptionTool(BaseMockTool):
            @handle_tool_exceptions(
                error_code="TEST_ERROR",
                error_message_template="Failed: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                raise CustomError("Custom error")

        tool = UnhandledExceptionTool()
        with pytest.raises(CustomError):
            tool.execute(issue_key="TEST-1")


class TestDecoratorIntegration:
    """Integration tests to verify the decorator works with actual tool patterns."""

    def test_jira_like_tool_success(self) -> None:
        """Test a Jira-like tool pattern with successful execution."""

        class MockClient:
            def get_issue(self, issue_key: str) -> dict[str, Any]:
                return {"key": issue_key, "summary": "Test issue"}

        class GetIssueTool(Tool):
            def __init__(self, client: MockClient) -> None:
                self.client = client
                self._schema = ToolSchema(
                    name="jira_get_issue",
                    description="Get a Jira issue",
                    category="jira",
                    parameters=[
                        ToolParameter(
                            name="issue_key",
                            description="Issue key",
                            required=True,
                        ),
                    ],
                )

            @property
            def schema(self) -> ToolSchema:
                return self._schema

            @handle_tool_exceptions(
                error_code="JIRA_ERROR",
                error_message_template="Failed to get issue {context}: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                issue_key = kwargs["issue_key"]
                result = self.client.get_issue(issue_key)
                return ToolResult.ok(result)

        client = MockClient()
        tool = GetIssueTool(client)
        result = tool.execute(issue_key="TEST-123")

        assert result.success is True
        assert result.data["key"] == "TEST-123"

    def test_jira_like_tool_failure(self) -> None:
        """Test a Jira-like tool pattern with API failure."""

        class MockClient:
            def get_issue(self, issue_key: str) -> dict[str, Any]:
                raise OSError("API connection failed")

        class GetIssueTool(Tool):
            def __init__(self, client: MockClient) -> None:
                self.client = client
                self._schema = ToolSchema(
                    name="jira_get_issue",
                    description="Get a Jira issue",
                    category="jira",
                    parameters=[
                        ToolParameter(
                            name="issue_key",
                            description="Issue key",
                            required=True,
                        ),
                    ],
                )

            @property
            def schema(self) -> ToolSchema:
                return self._schema

            @handle_tool_exceptions(
                error_code="JIRA_ERROR",
                error_message_template="Failed to get issue {context}: {error}",
            )
            def execute(self, **kwargs: Any) -> ToolResult:
                issue_key = kwargs["issue_key"]
                result = self.client.get_issue(issue_key)
                return ToolResult.ok(result)

        client = MockClient()
        tool = GetIssueTool(client)
        result = tool.execute(issue_key="TEST-123")

        assert result.success is False
        assert result.error_code == "JIRA_ERROR"
        assert "TEST-123" in result.error
        assert "API connection failed" in result.error


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_configuration_error_is_exception(self) -> None:
        """Test that ConfigurationError is a proper exception."""
        error = ConfigurationError("Missing required field 'api_key'")
        assert isinstance(error, Exception)
        assert str(error) == "Missing required field 'api_key'"

    def test_configuration_error_can_be_raised_and_caught(self) -> None:
        """Test that ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError, match="Invalid config"):
            raise ConfigurationError("Invalid config file format")

    def test_validation_error_is_exception(self) -> None:
        """Test that ValidationError is a proper exception."""
        error = ValidationError("Issue key must match pattern 'PROJ-\\d+'")
        assert isinstance(error, Exception)
        assert "Issue key must match pattern" in str(error)

    def test_validation_error_can_be_raised_and_caught(self) -> None:
        """Test that ValidationError can be raised and caught."""
        with pytest.raises(ValidationError, match="Invalid value"):
            raise ValidationError("Invalid value for field 'priority'")

    def test_configuration_error_not_caught_by_value_error(self) -> None:
        """Test that ConfigurationError is not caught by ValueError handler."""
        try:
            raise ConfigurationError("Config error")
        except ValueError:
            pytest.fail("ConfigurationError should not be caught by ValueError")
        except ConfigurationError:
            pass  # Expected

    def test_validation_error_not_caught_by_key_error(self) -> None:
        """Test that ValidationError is not caught by KeyError handler."""
        try:
            raise ValidationError("Validation error")
        except KeyError:
            pytest.fail("ValidationError should not be caught by KeyError")
        except ValidationError:
            pass  # Expected
