"""Tests for JQL-like project filter parser module."""

import pytest

from sentinel.project_filter import (
    BooleanOperator,
    Condition,
    FilterExpression,
    Operator,
    ProjectFilterParser,
    parse_and_evaluate,
)


class TestOperator:
    """Tests for Operator enum."""

    def test_operator_values(self) -> None:
        assert Operator.EQUALS.value == "="
        assert Operator.NOT_EQUALS.value == "!="
        assert Operator.IN.value == "IN"
        assert Operator.NOT_IN.value == "NOT IN"


class TestBooleanOperator:
    """Tests for BooleanOperator enum."""

    def test_boolean_operator_values(self) -> None:
        assert BooleanOperator.AND.value == "AND"
        assert BooleanOperator.OR.value == "OR"


class TestCondition:
    """Tests for Condition dataclass."""

    def test_create_condition(self) -> None:
        condition = Condition(field="Status", operator=Operator.EQUALS, values=("Ready",))
        assert condition.field == "Status"
        assert condition.operator == Operator.EQUALS
        assert condition.values == ("Ready",)
        assert condition.value == "Ready"

    def test_condition_with_multiple_values(self) -> None:
        condition = Condition(field="Status", operator=Operator.IN, values=("Ready", "In Progress"))
        assert condition.values == ("Ready", "In Progress")
        assert condition.value == "Ready"  # First value

    def test_condition_is_frozen(self) -> None:
        condition = Condition(field="Status", operator=Operator.EQUALS, values=("Ready",))
        with pytest.raises(AttributeError):
            condition.field = "Priority"  # type: ignore[misc]

    def test_condition_empty_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Field name cannot be empty"):
            Condition(field="", operator=Operator.EQUALS, values=("Ready",))

    def test_condition_empty_values_raises(self) -> None:
        with pytest.raises(ValueError, match="Condition must have at least one value"):
            Condition(field="Status", operator=Operator.EQUALS, values=())


class TestFilterExpression:
    """Tests for FilterExpression dataclass."""

    def test_leaf_expression(self) -> None:
        condition = Condition(field="Status", operator=Operator.EQUALS, values=("Ready",))
        expr = FilterExpression(condition=condition)
        assert expr.is_leaf is True
        assert expr.condition == condition

    def test_composite_expression(self) -> None:
        cond1 = Condition(field="Status", operator=Operator.EQUALS, values=("Ready",))
        cond2 = Condition(field="Priority", operator=Operator.EQUALS, values=("High",))
        left = FilterExpression(condition=cond1)
        right = FilterExpression(condition=cond2)
        expr = FilterExpression(left=left, right=right, bool_operator=BooleanOperator.AND)
        assert expr.is_leaf is False
        assert expr.left == left
        assert expr.right == right
        assert expr.bool_operator == BooleanOperator.AND

    def test_invalid_expression_both(self) -> None:
        """Expression cannot have both condition and sub-expressions."""
        cond = Condition(field="Status", operator=Operator.EQUALS, values=("Ready",))
        left = FilterExpression(condition=cond)
        with pytest.raises(ValueError, match="cannot have both"):
            FilterExpression(
                condition=cond, left=left, right=left, bool_operator=BooleanOperator.AND
            )

    def test_invalid_expression_neither(self) -> None:
        """Expression must have either condition or sub-expressions."""
        with pytest.raises(ValueError, match="must have either"):
            FilterExpression()

    def test_invalid_expression_partial_composite(self) -> None:
        """Composite expression requires all three: left, right, bool_operator."""
        cond = Condition(field="Status", operator=Operator.EQUALS, values=("Ready",))
        left = FilterExpression(condition=cond)
        with pytest.raises(ValueError, match="must have either"):
            FilterExpression(left=left)  # Missing right and bool_operator

    def test_filter_expression_is_frozen(self) -> None:
        """FilterExpression should be immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        cond = Condition(field="Status", operator=Operator.EQUALS, values=("Ready",))
        expr = FilterExpression(condition=cond)
        with pytest.raises(FrozenInstanceError):
            expr.condition = None  # type: ignore[misc]


class TestProjectFilterParserTokenization:
    """Tests for tokenization."""

    def test_parse_simple_equality(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready"')
        assert expr.is_leaf
        assert expr.condition is not None
        assert expr.condition.field == "Status"
        assert expr.condition.operator == Operator.EQUALS
        assert expr.condition.value == "Ready"

    def test_parse_not_equals(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status != "Done"')
        assert expr.is_leaf
        assert expr.condition is not None
        assert expr.condition.operator == Operator.NOT_EQUALS
        assert expr.condition.value == "Done"

    def test_parse_quoted_string_with_spaces(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "In Progress"')
        assert expr.condition is not None
        assert expr.condition.value == "In Progress"

    def test_parse_field_with_spaces(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Custom Field = "Value"')
        assert expr.condition is not None
        assert expr.condition.field == "Custom Field"

    def test_parse_empty_query_raises(self) -> None:
        parser = ProjectFilterParser()
        with pytest.raises(ValueError, match="Empty query"):
            parser.parse("")

    def test_parse_unterminated_string_raises(self) -> None:
        parser = ProjectFilterParser()
        with pytest.raises(ValueError, match="Unterminated string"):
            parser.parse('Status = "Ready')


class TestProjectFilterParserInOperator:
    """Tests for IN operator parsing."""

    def test_parse_in_single_value(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("Ready")')
        assert expr.condition is not None
        assert expr.condition.operator == Operator.IN
        assert expr.condition.values == ("Ready",)

    def test_parse_in_multiple_values(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("Ready", "In Progress")')
        assert expr.condition is not None
        assert expr.condition.operator == Operator.IN
        assert expr.condition.values == ("Ready", "In Progress")

    def test_parse_in_many_values(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("A", "B", "C", "D")')
        assert expr.condition is not None
        assert expr.condition.values == ("A", "B", "C", "D")

    def test_parse_not_in(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status NOT IN ("Done", "Archived")')
        assert expr.condition is not None
        assert expr.condition.operator == Operator.NOT_IN
        assert expr.condition.values == ("Done", "Archived")


class TestProjectFilterParserBooleanOperators:
    """Tests for AND/OR parsing."""

    def test_parse_and(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" AND Priority = "High"')
        assert not expr.is_leaf
        assert expr.bool_operator == BooleanOperator.AND
        assert expr.left is not None
        assert expr.left.condition is not None
        assert expr.left.condition.field == "Status"
        assert expr.right is not None
        assert expr.right.condition is not None
        assert expr.right.condition.field == "Priority"

    def test_parse_or(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" OR Status = "In Progress"')
        assert not expr.is_leaf
        assert expr.bool_operator == BooleanOperator.OR

    def test_parse_multiple_and(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('A = "1" AND B = "2" AND C = "3"')
        # Should be left-associative: ((A AND B) AND C)
        assert expr.bool_operator == BooleanOperator.AND
        assert expr.left is not None
        assert expr.left.bool_operator == BooleanOperator.AND

    def test_parse_and_or_precedence(self) -> None:
        parser = ProjectFilterParser()
        # AND has higher precedence than OR
        # A OR B AND C should parse as A OR (B AND C)
        expr = parser.parse('A = "1" OR B = "2" AND C = "3"')
        assert expr.bool_operator == BooleanOperator.OR
        assert expr.left is not None
        assert expr.left.condition is not None
        assert expr.left.condition.field == "A"
        assert expr.right is not None
        assert expr.right.bool_operator == BooleanOperator.AND


class TestProjectFilterParserParentheses:
    """Tests for parentheses grouping."""

    def test_parse_parentheses(self) -> None:
        parser = ProjectFilterParser()
        # Parentheses should override precedence
        # (A OR B) AND C
        expr = parser.parse('(Status = "Ready" OR Status = "In Progress") AND Priority = "High"')
        assert expr.bool_operator == BooleanOperator.AND
        assert expr.left is not None
        assert expr.left.bool_operator == BooleanOperator.OR
        assert expr.right is not None
        assert expr.right.condition is not None
        assert expr.right.condition.field == "Priority"

    def test_parse_nested_parentheses(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('((A = "1" OR B = "2") AND C = "3")')
        assert expr.bool_operator == BooleanOperator.AND

    def test_parse_unclosed_parenthesis_raises(self) -> None:
        parser = ProjectFilterParser()
        with pytest.raises(ValueError, match="Expected closing parenthesis"):
            parser.parse('(Status = "Ready"')


class TestProjectFilterParserEdgeCases:
    """Tests for edge cases in parsing."""

    def test_parse_case_insensitive_keywords(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status in ("Ready", "Done")')
        assert expr.condition is not None
        assert expr.condition.operator == Operator.IN

    def test_parse_case_insensitive_and(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('A = "1" and B = "2"')
        assert expr.bool_operator == BooleanOperator.AND

    def test_parse_case_insensitive_or(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('A = "1" or B = "2"')
        assert expr.bool_operator == BooleanOperator.OR

    def test_parse_case_insensitive_not_in(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status not in ("Done")')
        assert expr.condition is not None
        assert expr.condition.operator == Operator.NOT_IN

    def test_parse_unexpected_token_raises(self) -> None:
        parser = ProjectFilterParser()
        with pytest.raises(ValueError, match="Unexpected tokens"):
            parser.parse('Status = "Ready" extra')

    def test_parse_missing_value_raises(self) -> None:
        parser = ProjectFilterParser()
        with pytest.raises(ValueError, match="Expected"):
            parser.parse("Status =")

    def test_parse_invalid_operator_raises(self) -> None:
        parser = ProjectFilterParser()
        with pytest.raises(ValueError, match="Expected operator"):
            parser.parse('Status "Ready"')


class TestProjectFilterParserInvalidSyntaxErrors:
    """Tests for invalid filter syntax error handling.

    Verifies that ProjectFilterParser raises appropriate errors with helpful
    messages for various malformed filter expressions.
    """

    def test_empty_string_raises_with_helpful_message(self) -> None:
        """Test that empty query string raises ValueError with 'Empty query' message."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse("")
        assert "Empty query" in str(exc_info.value)

    def test_whitespace_only_raises_with_helpful_message(self) -> None:
        """Test that whitespace-only query raises ValueError."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse("   ")
        assert "Empty query" in str(exc_info.value)

    def test_unterminated_string_raises_with_position(self) -> None:
        """Test that unterminated string raises error indicating position."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status = "unterminated')
        error_msg = str(exc_info.value)
        assert "Unterminated string" in error_msg
        assert "position" in error_msg.lower()

    def test_missing_operator_raises_with_helpful_message(self) -> None:
        """Test missing operator between field and value raises clear error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status "Ready"')
        assert "Expected operator" in str(exc_info.value)

    def test_missing_value_after_equals_raises(self) -> None:
        """Test missing value after = operator raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse("Status =")
        assert "Expected" in str(exc_info.value)

    def test_missing_value_after_not_equals_raises(self) -> None:
        """Test missing value after != operator raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse("Status !=")
        assert "Expected" in str(exc_info.value)

    def test_unquoted_value_accepted(self) -> None:
        """Test that unquoted string values are accepted."""
        parser = ProjectFilterParser()
        expr = parser.parse("Status = Ready")
        assert expr.condition is not None
        assert expr.condition.field == "Status"
        assert expr.condition.operator == Operator.EQUALS
        assert expr.condition.values == ("Ready",)

    def test_unclosed_parenthesis_raises(self) -> None:
        """Test unclosed parenthesis raises clear error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('(Status = "Ready"')
        assert "closing parenthesis" in str(exc_info.value).lower()

    def test_extra_closing_parenthesis_raises(self) -> None:
        """Test extra closing parenthesis raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status = "Ready")')
        assert "Unexpected" in str(exc_info.value)

    def test_missing_right_operand_for_and_raises(self) -> None:
        """Test missing operand after AND raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status = "Ready" AND')
        assert "Expected" in str(exc_info.value)

    def test_missing_right_operand_for_or_raises(self) -> None:
        """Test missing operand after OR raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status = "Ready" OR')
        assert "Expected" in str(exc_info.value)

    def test_in_without_parentheses_raises(self) -> None:
        """Test IN operator without parenthesized list raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status IN "Ready"')
        assert "(" in str(exc_info.value) or "value list" in str(exc_info.value).lower()

    def test_in_with_unclosed_list_raises(self) -> None:
        """Test IN operator with unclosed value list raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status IN ("Ready", "Done"')
        assert ")" in str(exc_info.value) or "close" in str(exc_info.value).lower()

    def test_in_with_empty_list_raises(self) -> None:
        """Test IN operator with empty list raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse("Status IN ()")
        assert "Expected" in str(exc_info.value)

    def test_unexpected_token_after_complete_expression_raises(self) -> None:
        """Test trailing tokens after valid expression raises error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status = "Ready" garbage')
        error_msg = str(exc_info.value)
        assert "Unexpected" in error_msg
        assert "garbage" in error_msg

    def test_double_operator_raises(self) -> None:
        """Test double operators raise error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('Status = = "Ready"')
        # Should fail because = is not a valid field name
        assert "Expected" in str(exc_info.value) or "Unexpected" in str(exc_info.value)

    def test_special_character_in_wrong_place_raises(self) -> None:
        """Test special characters in wrong places raise errors."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('@Status = "Ready"')
        assert "Unexpected character" in str(exc_info.value)

    def test_nested_unclosed_parentheses_raises(self) -> None:
        """Test deeply nested unclosed parentheses raise error."""
        parser = ProjectFilterParser()
        with pytest.raises(ValueError) as exc_info:
            parser.parse('((Status = "Ready")')
        assert "closing parenthesis" in str(exc_info.value).lower()


class TestProjectFilterParserEvaluateEquals:
    """Tests for evaluating EQUALS conditions."""

    def test_evaluate_equals_match(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready"')
        assert parser.evaluate(expr, {"Status": "Ready"}) is True

    def test_evaluate_equals_no_match(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready"')
        assert parser.evaluate(expr, {"Status": "Done"}) is False

    def test_evaluate_equals_case_insensitive(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "ready"')
        assert parser.evaluate(expr, {"Status": "READY"}) is True

    def test_evaluate_equals_field_case_insensitive(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('STATUS = "Ready"')
        assert parser.evaluate(expr, {"status": "Ready"}) is True

    def test_evaluate_equals_missing_field(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready"')
        assert parser.evaluate(expr, {"Priority": "High"}) is False

    def test_evaluate_equals_none_value(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready"')
        assert parser.evaluate(expr, {"Status": None}) is False


class TestProjectFilterParserEvaluateNotEquals:
    """Tests for evaluating NOT_EQUALS conditions."""

    def test_evaluate_not_equals_match(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status != "Done"')
        assert parser.evaluate(expr, {"Status": "Ready"}) is True

    def test_evaluate_not_equals_no_match(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status != "Done"')
        assert parser.evaluate(expr, {"Status": "Done"}) is False

    def test_evaluate_not_equals_case_insensitive(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status != "done"')
        assert parser.evaluate(expr, {"Status": "DONE"}) is False

    def test_evaluate_not_equals_missing_field(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status != "Done"')
        # Missing field means None, which is != "Done"
        assert parser.evaluate(expr, {"Priority": "High"}) is True

    def test_evaluate_not_equals_none_value(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status != "Done"')
        # None != "Done" is True
        assert parser.evaluate(expr, {"Status": None}) is True


class TestProjectFilterParserEvaluateIn:
    """Tests for evaluating IN conditions."""

    def test_evaluate_in_match(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("Ready", "In Progress")')
        assert parser.evaluate(expr, {"Status": "Ready"}) is True

    def test_evaluate_in_match_second_value(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("Ready", "In Progress")')
        assert parser.evaluate(expr, {"Status": "In Progress"}) is True

    def test_evaluate_in_no_match(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("Ready", "In Progress")')
        assert parser.evaluate(expr, {"Status": "Done"}) is False

    def test_evaluate_in_case_insensitive(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("ready", "in progress")')
        assert parser.evaluate(expr, {"Status": "READY"}) is True

    def test_evaluate_in_missing_field(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("Ready", "In Progress")')
        assert parser.evaluate(expr, {"Priority": "High"}) is False


class TestProjectFilterParserEvaluateNotIn:
    """Tests for evaluating NOT IN conditions."""

    def test_evaluate_not_in_match(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status NOT IN ("Done", "Archived")')
        assert parser.evaluate(expr, {"Status": "Ready"}) is True

    def test_evaluate_not_in_no_match(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status NOT IN ("Done", "Archived")')
        assert parser.evaluate(expr, {"Status": "Done"}) is False

    def test_evaluate_not_in_case_insensitive(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status NOT IN ("done", "archived")')
        assert parser.evaluate(expr, {"Status": "DONE"}) is False

    def test_evaluate_not_in_missing_field(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status NOT IN ("Done", "Archived")')
        # Missing field means None, which is NOT IN the set
        assert parser.evaluate(expr, {"Priority": "High"}) is True


class TestProjectFilterParserEvaluateBoolean:
    """Tests for evaluating AND/OR combinations."""

    def test_evaluate_and_both_true(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" AND Priority = "High"')
        assert parser.evaluate(expr, {"Status": "Ready", "Priority": "High"}) is True

    def test_evaluate_and_one_false(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" AND Priority = "High"')
        assert parser.evaluate(expr, {"Status": "Ready", "Priority": "Low"}) is False

    def test_evaluate_and_both_false(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" AND Priority = "High"')
        assert parser.evaluate(expr, {"Status": "Done", "Priority": "Low"}) is False

    def test_evaluate_or_both_true(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" OR Status = "In Progress"')
        # Can't have two Status values, but both conditions could be checked
        assert parser.evaluate(expr, {"Status": "Ready"}) is True

    def test_evaluate_or_one_true(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" OR Priority = "High"')
        assert parser.evaluate(expr, {"Status": "Done", "Priority": "High"}) is True

    def test_evaluate_or_both_false(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status = "Ready" OR Status = "In Progress"')
        assert parser.evaluate(expr, {"Status": "Done"}) is False

    def test_evaluate_complex_and_or(self) -> None:
        parser = ProjectFilterParser()
        # (Status = "Ready" OR Status = "In Progress") AND Priority = "High"
        expr = parser.parse('(Status = "Ready" OR Status = "In Progress") AND Priority = "High"')
        assert parser.evaluate(expr, {"Status": "Ready", "Priority": "High"}) is True
        assert parser.evaluate(expr, {"Status": "In Progress", "Priority": "High"}) is True
        assert parser.evaluate(expr, {"Status": "Ready", "Priority": "Low"}) is False
        assert parser.evaluate(expr, {"Status": "Done", "Priority": "High"}) is False


class TestProjectFilterParserEvaluateComplexQueries:
    """Tests for complex query scenarios."""

    def test_evaluate_multiple_and(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('A = "1" AND B = "2" AND C = "3"')
        assert parser.evaluate(expr, {"A": "1", "B": "2", "C": "3"}) is True
        assert parser.evaluate(expr, {"A": "1", "B": "2", "C": "4"}) is False

    def test_evaluate_multiple_or(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('A = "1" OR B = "2" OR C = "3"')
        assert parser.evaluate(expr, {"A": "1"}) is True
        assert parser.evaluate(expr, {"B": "2"}) is True
        assert parser.evaluate(expr, {"C": "3"}) is True
        assert parser.evaluate(expr, {"A": "x", "B": "y", "C": "z"}) is False

    def test_evaluate_nested_parentheses(self) -> None:
        parser = ProjectFilterParser()
        # ((A OR B) AND C)
        expr = parser.parse('((A = "1" OR B = "2") AND C = "3")')
        assert parser.evaluate(expr, {"A": "1", "C": "3"}) is True
        assert parser.evaluate(expr, {"B": "2", "C": "3"}) is True
        assert parser.evaluate(expr, {"A": "1", "C": "x"}) is False

    def test_evaluate_in_with_and(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status IN ("Ready", "In Progress") AND Priority = "High"')
        assert parser.evaluate(expr, {"Status": "Ready", "Priority": "High"}) is True
        assert parser.evaluate(expr, {"Status": "Done", "Priority": "High"}) is False

    def test_evaluate_not_in_with_or(self) -> None:
        parser = ProjectFilterParser()
        expr = parser.parse('Status NOT IN ("Done", "Archived") OR Override = "true"')
        assert parser.evaluate(expr, {"Status": "Ready"}) is True
        assert parser.evaluate(expr, {"Status": "Done", "Override": "true"}) is True
        assert parser.evaluate(expr, {"Status": "Done", "Override": "false"}) is False


class TestProjectFilterParserIntegration:
    """Integration tests for real-world scenarios."""

    def test_github_project_filtering(self) -> None:
        """Test filtering GitHub Project items by Status and Priority."""
        parser = ProjectFilterParser()

        # Filter for items ready for work
        expr = parser.parse('Status IN ("Ready", "In Progress") AND Priority IN ("High", "Urgent")')

        # Sample project items
        items = [
            {"Status": "Ready", "Priority": "High", "Title": "Fix bug"},
            {"Status": "In Progress", "Priority": "Urgent", "Title": "Deploy"},
            {"Status": "Done", "Priority": "High", "Title": "Completed task"},
            {"Status": "Ready", "Priority": "Low", "Title": "Nice to have"},
        ]

        matches = [item for item in items if parser.evaluate(expr, item)]
        assert len(matches) == 2
        assert matches[0]["Title"] == "Fix bug"
        assert matches[1]["Title"] == "Deploy"

    def test_exclude_done_items(self) -> None:
        """Test excluding completed items."""
        parser = ProjectFilterParser()
        expr = parser.parse('Status NOT IN ("Done", "Archived", "Cancelled")')

        items = [
            {"Status": "Ready", "Title": "Item 1"},
            {"Status": "Done", "Title": "Item 2"},
            {"Status": "In Progress", "Title": "Item 3"},
            {"Status": "Archived", "Title": "Item 4"},
        ]

        matches = [item for item in items if parser.evaluate(expr, item)]
        assert len(matches) == 2
        assert matches[0]["Title"] == "Item 1"
        assert matches[1]["Title"] == "Item 3"

    def test_complex_sprint_filter(self) -> None:
        """Test complex filtering for sprint planning."""
        parser = ProjectFilterParser()
        expr = parser.parse(
            '(Status = "Ready" OR Status = "Blocked") AND '
            'Priority != "Low" AND '
            'Sprint = "Sprint 5"'
        )

        items = [
            {"Status": "Ready", "Priority": "High", "Sprint": "Sprint 5"},
            {"Status": "Blocked", "Priority": "Medium", "Sprint": "Sprint 5"},
            {"Status": "Ready", "Priority": "Low", "Sprint": "Sprint 5"},
            {"Status": "Ready", "Priority": "High", "Sprint": "Sprint 4"},
            {"Status": "Done", "Priority": "High", "Sprint": "Sprint 5"},
        ]

        matches = [item for item in items if parser.evaluate(expr, item)]
        assert len(matches) == 2
        assert matches[0]["Status"] == "Ready"
        assert matches[1]["Status"] == "Blocked"


class TestParseAndEvaluate:
    """Tests for the parse_and_evaluate convenience function."""

    def test_parse_and_evaluate_simple_match(self) -> None:
        """Test simple matching query."""
        result = parse_and_evaluate('Status = "Ready"', {"Status": "Ready"})
        assert result is True

    def test_parse_and_evaluate_simple_no_match(self) -> None:
        """Test simple non-matching query."""
        result = parse_and_evaluate('Status = "Ready"', {"Status": "Done"})
        assert result is False

    def test_parse_and_evaluate_complex_query(self) -> None:
        """Test complex query with AND operator."""
        result = parse_and_evaluate(
            'Status = "Ready" AND Priority = "High"',
            {"Status": "Ready", "Priority": "High"},
        )
        assert result is True

    def test_parse_and_evaluate_in_operator(self) -> None:
        """Test IN operator."""
        result = parse_and_evaluate(
            'Status IN ("Ready", "In Progress")',
            {"Status": "In Progress"},
        )
        assert result is True

    def test_parse_and_evaluate_invalid_query(self) -> None:
        """Test that invalid queries raise ValueError."""
        with pytest.raises(ValueError):
            parse_and_evaluate("", {"Status": "Ready"})

    def test_parse_and_evaluate_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        result = parse_and_evaluate('Status = "ready"', {"Status": "READY"})
        assert result is True
