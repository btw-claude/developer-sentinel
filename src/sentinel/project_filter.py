"""JQL-like query parser for filtering GitHub Project items by field values.

This module provides a JQL-like query language for filtering GitHub Project items
based on their field values. It supports equality, inequality, IN, NOT IN operators,
as well as AND/OR boolean combinations with parentheses for grouping.

Example queries:
    Status = "Ready"
    Status != "Done"
    Status IN ("Ready", "In Progress")
    Status NOT IN ("Done", "Archived")
    Status = "Ready" AND Priority = "High"
    (Status = "Ready" OR Status = "In Progress") AND Priority = "High"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "Operator",
    "BooleanOperator",
    "Condition",
    "FilterExpression",
    "ProjectFilterParser",
    "parse_and_evaluate",
]


class Operator(Enum):
    """Operators supported in filter conditions."""

    EQUALS = "="
    NOT_EQUALS = "!="
    IN = "IN"
    NOT_IN = "NOT IN"


class BooleanOperator(Enum):
    """Boolean operators for combining conditions."""

    AND = "AND"
    OR = "OR"


@dataclass(frozen=True)
class Condition:
    """A single filter condition.

    Represents a comparison between a field and one or more values.

    Attributes:
        field: The field name to compare (e.g., "Status", "Priority").
        operator: The comparison operator.
        values: The value(s) to compare against. Single value for EQUALS/NOT_EQUALS,
            list of values for IN/NOT_IN.
    """

    field: str
    operator: Operator
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate condition arguments."""
        if not self.field:
            raise ValueError("Field name cannot be empty")
        if not self.values:
            raise ValueError("Condition must have at least one value")

    @property
    def value(self) -> str:
        """Return the first value (convenience for single-value operators)."""
        return self.values[0]


@dataclass(frozen=True)
class FilterExpression:
    """An AST node for filter expressions.

    Can be either a single Condition or a combination of expressions
    joined by a boolean operator.

    Attributes:
        condition: A single condition (leaf node), or None if this is a composite.
        left: Left sub-expression for composite expressions.
        right: Right sub-expression for composite expressions.
        bool_operator: The boolean operator joining left and right.
    """

    condition: Condition | None = None
    left: FilterExpression | None = None
    right: FilterExpression | None = None
    bool_operator: BooleanOperator | None = None

    def __post_init__(self) -> None:
        """Validate expression structure."""
        is_leaf = self.condition is not None
        is_composite = (
            self.left is not None
            and self.right is not None
            and self.bool_operator is not None
        )

        if is_leaf and is_composite:
            raise ValueError(
                "Expression cannot have both a condition and sub-expressions"
            )
        if not is_leaf and not is_composite:
            raise ValueError(
                "Expression must have either a condition or "
                "(left, right, bool_operator)"
            )

    @property
    def is_leaf(self) -> bool:
        """Return True if this is a leaf node (single condition)."""
        return self.condition is not None


def parse_and_evaluate(query: str, fields: dict[str, Any]) -> bool:
    """Convenience function to parse a query and evaluate it in one step.

    This function combines parsing and evaluation for common use cases
    where you don't need to reuse the parsed expression.

    Args:
        query: The query string to parse (e.g., 'Status = "Ready"').
        fields: Dictionary mapping field names to their values.

    Returns:
        True if the fields match the query expression, False otherwise.

    Raises:
        ValueError: If the query is invalid or malformed.

    Example:
        >>> result = parse_and_evaluate('Status = "Ready"', {"Status": "Ready"})
        >>> print(result)
        True
    """
    parser = ProjectFilterParser()
    expression = parser.parse(query)
    return parser.evaluate(expression, fields)


class ProjectFilterParser:
    """Parser for JQL-like filter expressions.

    Parses query strings into FilterExpression ASTs and evaluates them
    against field data.

    The grammar supports:
        - Field comparisons: field = "value", field != "value"
        - Set membership: field IN ("v1", "v2"), field NOT IN ("v1", "v2")
        - Boolean combinations: expr AND expr, expr OR expr
        - Grouping: (expr)

    Example:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" AND Priority = "High"')
        result = parser.evaluate(expr, {"Status": "Ready", "Priority": "High"})
    """

    # Token patterns
    _STRING_PATTERN = r'"([^"]*)"'

    def __init__(self) -> None:
        """Initialize the parser."""
        self._tokens: list[str] = []
        self._pos: int = 0

    def parse(self, query: str) -> FilterExpression:
        """Parse a query string into a FilterExpression.

        Args:
            query: The query string to parse.

        Returns:
            The parsed FilterExpression AST.

        Raises:
            ValueError: If the query is invalid or malformed.
        """
        self._tokenize(query)
        self._pos = 0

        if not self._tokens:
            raise ValueError("Empty query")

        expr = self._parse_expression()

        if self._pos < len(self._tokens):
            remaining = " ".join(self._tokens[self._pos :])
            raise ValueError(f"Unexpected tokens after expression: {remaining}")

        return expr

    def evaluate(
        self, expression: FilterExpression, fields: dict[str, Any]
    ) -> bool:
        """Evaluate a FilterExpression against field values.

        Args:
            expression: The FilterExpression to evaluate.
            fields: Dictionary mapping field names to their values.

        Returns:
            True if the fields match the expression, False otherwise.
        """
        if expression.is_leaf:
            assert expression.condition is not None
            return self._evaluate_condition(expression.condition, fields)

        assert expression.left is not None
        assert expression.right is not None
        assert expression.bool_operator is not None

        left_result = self.evaluate(expression.left, fields)
        right_result = self.evaluate(expression.right, fields)

        if expression.bool_operator == BooleanOperator.AND:
            return left_result and right_result
        else:  # OR
            return left_result or right_result

    def _tokenize(self, query: str) -> None:
        """Tokenize the query string.

        Args:
            query: The query string to tokenize.
        """
        self._tokens = []
        query = query.strip()
        i = 0

        while i < len(query):
            # Skip whitespace
            if query[i].isspace():
                i += 1
                continue

            # Match parentheses
            if query[i] in "()":
                self._tokens.append(query[i])
                i += 1
                continue

            # Match operators
            if query[i : i + 2] == "!=":
                self._tokens.append("!=")
                i += 2
                continue
            if query[i] == "=":
                self._tokens.append("=")
                i += 1
                continue
            if query[i] == ",":
                self._tokens.append(",")
                i += 1
                continue

            # Match quoted strings
            if query[i] == '"':
                match = re.match(self._STRING_PATTERN, query[i:])
                if match:
                    # Store the full quoted string as token
                    self._tokens.append(match.group(0))
                    i += len(match.group(0))
                    continue
                else:
                    raise ValueError(f"Unterminated string at position {i}")

            # Match keywords (AND, OR, IN, NOT IN) - must check before identifiers
            upper_rest = query[i:].upper()
            if upper_rest.startswith("NOT IN"):
                # Check it's a complete word (followed by space, ( or end)
                end_pos = i + 6
                if end_pos >= len(query) or not query[end_pos].isalnum():
                    self._tokens.append("NOT IN")
                    i += 6
                    continue
            if upper_rest.startswith("AND"):
                end_pos = i + 3
                if end_pos >= len(query) or not query[end_pos].isalnum():
                    self._tokens.append("AND")
                    i += 3
                    continue
            if upper_rest.startswith("OR"):
                end_pos = i + 2
                if end_pos >= len(query) or not query[end_pos].isalnum():
                    self._tokens.append("OR")
                    i += 2
                    continue
            if upper_rest.startswith("IN"):
                end_pos = i + 2
                if end_pos >= len(query) or not query[end_pos].isalnum():
                    self._tokens.append("IN")
                    i += 2
                    continue

            # Match identifiers (field names) - can contain spaces but must stop
            # before keywords, operators, and special characters
            if query[i].isalpha() or query[i] == "_":
                identifier = self._parse_identifier(query, i)
                self._tokens.append(identifier)
                i += len(identifier)
                continue

            raise ValueError(f"Unexpected character at position {i}: {query[i]!r}")

    def _parse_identifier(self, query: str, start: int) -> str:
        """Parse an identifier (field name) from the query.

        Identifiers can contain letters, digits, underscores, hyphens, and spaces.
        They stop at keywords (AND, OR, IN, NOT IN), operators (=, !=), and
        special characters (parentheses, quotes, commas).

        Args:
            query: The full query string.
            start: Starting position in the query.

        Returns:
            The parsed identifier string (with trailing whitespace stripped).
        """
        i = start
        result = []

        while i < len(query):
            char = query[i]

            # Stop at operators and special characters
            if char in '=!(),"':
                break

            # Check for keywords at word boundaries
            if char.isspace():
                # Look ahead to see if next word is a keyword
                j = i
                while j < len(query) and query[j].isspace():
                    j += 1
                if j < len(query):
                    rest_upper = query[j:].upper()
                    # Check for keywords
                    for keyword, length in [("NOT IN", 6), ("AND", 3), ("OR", 2), ("IN", 2)]:
                        if rest_upper.startswith(keyword):
                            end_pos = j + length
                            # Verify it's a complete word
                            if end_pos >= len(query) or not query[end_pos].isalnum():
                                # Found keyword, stop here
                                return "".join(result).strip()

            # Valid identifier character
            if char.isalnum() or char in "_- " or char.isspace():
                result.append(char)
                i += 1
            else:
                break

        return "".join(result).strip()

    def _parse_expression(self) -> FilterExpression:
        """Parse an expression (handles OR, lowest precedence).

        Returns:
            The parsed FilterExpression.
        """
        left = self._parse_and_expression()

        while self._peek() == "OR":
            self._advance()  # consume OR
            right = self._parse_and_expression()
            left = FilterExpression(
                left=left, right=right, bool_operator=BooleanOperator.OR
            )

        return left

    def _parse_and_expression(self) -> FilterExpression:
        """Parse an AND expression (higher precedence than OR).

        Returns:
            The parsed FilterExpression.
        """
        left = self._parse_primary()

        while self._peek() == "AND":
            self._advance()  # consume AND
            right = self._parse_primary()
            left = FilterExpression(
                left=left, right=right, bool_operator=BooleanOperator.AND
            )

        return left

    def _parse_primary(self) -> FilterExpression:
        """Parse a primary expression (condition or parenthesized expression).

        Returns:
            The parsed FilterExpression.
        """
        if self._peek() == "(":
            self._advance()  # consume (
            expr = self._parse_expression()
            if self._peek() != ")":
                raise ValueError("Expected closing parenthesis")
            self._advance()  # consume )
            return expr

        return self._parse_condition()

    def _parse_condition(self) -> FilterExpression:
        """Parse a single condition.

        Returns:
            FilterExpression containing a Condition.
        """
        # Parse field name
        field = self._advance()
        if field is None:
            raise ValueError("Expected field name")
        if field.startswith('"') or field in ("AND", "OR", "IN", "NOT IN", "(", ")"):
            raise ValueError(f"Expected field name, got: {field}")

        # Parse operator
        op_token = self._advance()
        if op_token == "=":
            operator = Operator.EQUALS
        elif op_token == "!=":
            operator = Operator.NOT_EQUALS
        elif op_token == "IN":
            operator = Operator.IN
        elif op_token == "NOT IN":
            operator = Operator.NOT_IN
        else:
            raise ValueError(f"Expected operator, got: {op_token}")

        # Parse value(s)
        if operator in (Operator.IN, Operator.NOT_IN):
            values = self._parse_value_list()
        else:
            value = self._parse_value()
            values = (value,)

        condition = Condition(field=field, operator=operator, values=values)
        return FilterExpression(condition=condition)

    def _parse_value(self) -> str:
        """Parse a single quoted value.

        Returns:
            The unquoted string value.
        """
        token = self._advance()
        if token is None:
            raise ValueError("Expected value")
        if not token.startswith('"'):
            raise ValueError(f"Expected quoted string, got: {token}")
        # Extract content between quotes
        return token[1:-1]

    def _parse_value_list(self) -> tuple[str, ...]:
        """Parse a parenthesized list of values.

        Returns:
            Tuple of string values.
        """
        if self._peek() != "(":
            raise ValueError("Expected '(' for value list")
        self._advance()  # consume (

        values: list[str] = []
        values.append(self._parse_value())

        while self._peek() == ",":
            self._advance()  # consume ,
            values.append(self._parse_value())

        if self._peek() != ")":
            raise ValueError("Expected ')' to close value list")
        self._advance()  # consume )

        return tuple(values)

    def _peek(self) -> str | None:
        """Peek at the current token without advancing.

        Returns:
            The current token, or None if at end.
        """
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> str | None:
        """Consume and return the current token.

        Returns:
            The current token, or None if at end.
        """
        token = self._peek()
        if token is not None:
            self._pos += 1
        return token

    def _evaluate_condition(
        self, condition: Condition, fields: dict[str, Any]
    ) -> bool:
        """Evaluate a single condition against field values.

        Field matching is case-insensitive for field names.
        Value matching is case-insensitive.

        Args:
            condition: The condition to evaluate.
            fields: Dictionary mapping field names to their values.

        Returns:
            True if the condition matches, False otherwise.
        """
        # Find field value (case-insensitive field name matching)
        field_value: str | None = None
        for fname, fvalue in fields.items():
            if fname.lower() == condition.field.lower():
                field_value = str(fvalue) if fvalue is not None else None
                break

        # Handle missing field as None
        if field_value is None:
            field_value_lower = None
        else:
            field_value_lower = field_value.lower()

        # Case-insensitive value comparison
        condition_values_lower = tuple(v.lower() for v in condition.values)

        if condition.operator == Operator.EQUALS:
            if field_value_lower is None:
                return False
            return field_value_lower == condition_values_lower[0]

        elif condition.operator == Operator.NOT_EQUALS:
            if field_value_lower is None:
                return True  # None != any value
            return field_value_lower != condition_values_lower[0]

        elif condition.operator == Operator.IN:
            if field_value_lower is None:
                return False
            return field_value_lower in condition_values_lower

        elif condition.operator == Operator.NOT_IN:
            if field_value_lower is None:
                return True  # None not in any set
            return field_value_lower not in condition_values_lower

        return False
