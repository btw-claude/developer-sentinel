"""Base classes for agent tool definitions.

This module provides the foundational classes for defining tools that Claude agents
can use. Tools are defined with clear schemas that describe their parameters and
expected behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterType(Enum):
    """Types of parameters that tools can accept."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Definition of a single tool parameter.

    Attributes:
        name: Parameter name (used in function calls).
        description: Human-readable description of the parameter.
        type: The parameter type.
        required: Whether the parameter is required.
        enum: Optional list of allowed values.
        items_type: Type of array items (if type is ARRAY).
        default: Default value if not provided.
    """

    name: str
    description: str
    type: ParameterType = ParameterType.STRING
    required: bool = True
    enum: list[str] | None = None
    items_type: ParameterType | None = None
    default: Any = None

    def to_schema(self) -> dict[str, Any]:
        """Convert parameter to JSON Schema format.

        Returns:
            JSON Schema representation of the parameter.
        """
        schema: dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        if self.type == ParameterType.ARRAY and self.items_type:
            schema["items"] = {"type": self.items_type.value}

        if self.default is not None:
            schema["default"] = self.default

        return schema


@dataclass
class ToolSchema:
    """Schema definition for a tool.

    Attributes:
        name: Unique tool name (used for invocation).
        description: Human-readable description of what the tool does.
        parameters: List of parameter definitions.
        category: Tool category (e.g., "jira", "confluence", "github").
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    category: str = ""

    def to_function_schema(self) -> dict[str, Any]:
        """Convert to Claude function calling schema format.

        Returns:
            Schema in Claude's expected function format.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
            },
        }

        if required:
            schema["input_schema"]["required"] = required

        return schema


@dataclass
class ToolResult:
    """Result of executing a tool.

    Attributes:
        success: Whether the tool execution succeeded.
        data: The result data (if successful).
        error: Error message (if failed).
        error_code: Optional error code for categorization.
    """

    success: bool
    data: Any = None
    error: str | None = None
    error_code: str | None = None

    def to_response(self) -> dict[str, Any]:
        """Convert to a response format for the agent.

        Returns:
            Dict representation suitable for agent consumption.
        """
        if self.success:
            return {
                "success": True,
                "data": self.data,
            }
        return {
            "success": False,
            "error": self.error,
            "error_code": self.error_code,
        }

    @classmethod
    def ok(cls, data: Any) -> ToolResult:
        """Create a successful result.

        Args:
            data: The result data.

        Returns:
            ToolResult indicating success.
        """
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str, error_code: str | None = None) -> ToolResult:
        """Create a failed result.

        Args:
            error: Error message.
            error_code: Optional error code.

        Returns:
            ToolResult indicating failure.
        """
        return cls(success=False, error=error, error_code=error_code)


class Tool(ABC):
    """Abstract base class for tool implementations.

    Tools must implement the schema property and execute method.
    """

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the tool's schema definition.

        Returns:
            ToolSchema describing the tool.
        """
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool parameters as defined in the schema.

        Returns:
            ToolResult with the execution outcome.
        """
        pass

    @property
    def name(self) -> str:
        """Return the tool's name."""
        return self.schema.name

    @property
    def category(self) -> str:
        """Return the tool's category."""
        return self.schema.category

    def to_function_schema(self) -> dict[str, Any]:
        """Return the tool's schema in function calling format."""
        return self.schema.to_function_schema()

    def validate_params(self, **kwargs: Any) -> ToolResult | None:
        """Validate parameters against the schema.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            ToolResult with error if validation fails, None if valid.
        """
        for param in self.schema.parameters:
            if param.required and param.name not in kwargs:
                return ToolResult.fail(
                    f"Missing required parameter: {param.name}",
                    error_code="MISSING_PARAMETER",
                )

            if param.name in kwargs:
                value = kwargs[param.name]

                # Type validation for common types
                if param.type == ParameterType.STRING and not isinstance(value, str):
                    return ToolResult.fail(
                        f"Parameter '{param.name}' must be a string",
                        error_code="INVALID_TYPE",
                    )
                if param.type == ParameterType.INTEGER and not isinstance(value, int):
                    return ToolResult.fail(
                        f"Parameter '{param.name}' must be an integer",
                        error_code="INVALID_TYPE",
                    )
                if param.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return ToolResult.fail(
                        f"Parameter '{param.name}' must be a boolean",
                        error_code="INVALID_TYPE",
                    )
                if param.type == ParameterType.ARRAY and not isinstance(value, list):
                    return ToolResult.fail(
                        f"Parameter '{param.name}' must be an array",
                        error_code="INVALID_TYPE",
                    )

                # Enum validation
                if param.enum and value not in param.enum:
                    return ToolResult.fail(
                        f"Parameter '{param.name}' must be one of: {param.enum}",
                        error_code="INVALID_ENUM",
                    )

        return None
