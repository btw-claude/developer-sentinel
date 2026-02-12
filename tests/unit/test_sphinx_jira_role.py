"""Tests for the custom Sphinx :jira: role extension.

Validates that the sentinel.sphinx_jira_role extension correctly:

1. Validates Jira issue key format (PROJECT-123 pattern)
2. Generates hyperlink nodes with the correct URL structure
3. Integrates with Sphinx configuration for the Jira base URL
4. Reports warnings for malformed issue keys

This test module was created as part of DS-1016 to accompany the custom
Sphinx role implementation.  The role was chosen over intersphinx because
Jira does not publish a Sphinx inventory (``objects.inv``), making
intersphinx unsuitable for Jira issue linking.

Usage:
    pytest tests/unit/test_sphinx_jira_role.py -v
"""

from __future__ import annotations

import pytest

from sentinel.sphinx_jira_role import (
    _DEFAULT_JIRA_BASE_URL,
    _JIRA_ISSUE_KEY_PATTERN,
    _validate_issue_key,
)


class TestJiraIssueKeyValidation:
    """Tests for Jira issue key format validation."""

    @pytest.mark.parametrize(
        "issue_key",
        [
            "DS-1",
            "DS-123",
            "DS-1016",
            "DS-99999",
            "PROJ-1",
            "ABC-42",
            "MYPROJECT-1",
            "A-1",
            "AB-12345",
            "ABCDEFGHIJ-1",
        ],
        ids=[
            "single-digit",
            "three-digit",
            "four-digit",
            "five-digit",
            "different-project",
            "short-project",
            "long-project",
            "single-letter-project",
            "two-letter-project",
            "max-length-project",
        ],
    )
    def test_valid_issue_keys(self, issue_key: str) -> None:
        """Verify valid Jira issue keys are accepted."""
        assert _validate_issue_key(issue_key) is True

    @pytest.mark.parametrize(
        ("issue_key", "reason"),
        [
            ("ds-123", "lowercase project key"),
            ("123-DS", "digits before hyphen"),
            ("DS123", "missing hyphen"),
            ("DS-", "missing number after hyphen"),
            ("-123", "missing project key before hyphen"),
            ("", "empty string"),
            ("DS-0123abc", "letters after number"),
            ("DS_123", "underscore instead of hyphen"),
            ("ABCDEFGHIJK-1", "project key exceeds 10 characters"),
            ("DS -123", "space in key"),
            ("ds-abc", "non-numeric issue number"),
            ("A1B2C3-1", "alphanumeric project key"),
            ("D2-42", "digit in project key"),
        ],
        ids=[
            "lowercase",
            "reversed-format",
            "no-hyphen",
            "missing-number",
            "missing-project",
            "empty",
            "trailing-letters",
            "underscore-separator",
            "project-too-long",
            "space-in-key",
            "non-numeric-number",
            "alphanumeric-project-key",
            "digit-in-project-key",
        ],
    )
    def test_invalid_issue_keys(self, issue_key: str, reason: str) -> None:
        """Verify invalid Jira issue keys are rejected.

        Each parametrised case exercises a distinct malformed pattern to
        ensure the regex rejects all common typos and formatting errors.
        The stricter all-alpha regex (DS-1019) also rejects alphanumeric
        project keys like A1B2C3-1, which are not valid Jira keys.
        """
        assert _validate_issue_key(issue_key) is False, (
            f"Expected {issue_key!r} to be invalid ({reason})"
        )


class TestJiraIssueKeyPattern:
    """Tests for the compiled regex pattern directly."""

    def test_pattern_anchored_start(self) -> None:
        """Verify the pattern is anchored at the start of the string."""
        # Should not match if there's a prefix
        assert _JIRA_ISSUE_KEY_PATTERN.match("xDS-123") is None

    def test_pattern_anchored_end(self) -> None:
        """Verify the pattern is anchored at the end of the string."""
        # match() only checks from the start, but $ anchor prevents partial
        assert _JIRA_ISSUE_KEY_PATTERN.match("DS-123x") is None

    def test_pattern_captures_full_key(self) -> None:
        """Verify the pattern matches the entire issue key."""
        match = _JIRA_ISSUE_KEY_PATTERN.match("DS-1016")
        assert match is not None
        assert match.group(0) == "DS-1016"

    def test_pattern_rejects_alphanumeric_project_key(self) -> None:
        """Verify the pattern rejects alphanumeric project keys (DS-1019).

        Real Jira project keys are all-alpha uppercase.  The regex was
        tightened from ``[A-Z][A-Z0-9]{0,9}`` to ``[A-Z]{1,10}`` to
        reject keys like ``A1B2C3-1``.
        """
        assert _JIRA_ISSUE_KEY_PATTERN.match("A1B2C3-1") is None
        assert _JIRA_ISSUE_KEY_PATTERN.match("D2-42") is None


class TestDefaultConfiguration:
    """Tests for default configuration values."""

    def test_default_base_url_is_placeholder(self) -> None:
        """Verify the default Jira base URL is a placeholder.

        The default URL should clearly indicate it needs configuration,
        not point to a real Jira instance.
        """
        assert "example.com" in _DEFAULT_JIRA_BASE_URL

    def test_default_base_url_uses_https(self) -> None:
        """Verify the default Jira base URL uses HTTPS."""
        assert _DEFAULT_JIRA_BASE_URL.startswith("https://")


class TestSphinxExtensionSetup:
    """Tests for the Sphinx extension setup function."""

    def _make_mock_app(self) -> object:
        """Create a minimal mock Sphinx app with config and connect support.

        Returns:
            A mock app object suitable for passing to ``setup()``.
        """

        class MockApp:
            def __init__(self) -> None:
                self.config_values: list[tuple[str, str, str]] = []
                self.connections: list[tuple[str, object]] = []

            def add_config_value(self, name: str, default: str, rebuild: str) -> None:
                self.config_values.append((name, default, rebuild))

            def connect(self, event: str, callback: object) -> None:
                self.connections.append((event, callback))

        return MockApp()

    def test_setup_returns_metadata(self) -> None:
        """Verify setup() returns the required extension metadata dict.

        Sphinx expects the metadata dict to contain ``version``,
        ``parallel_read_safe``, and ``parallel_write_safe`` keys.
        """
        # Ensure sphinx is available; skip gracefully if not installed.
        pytest.importorskip(
            "sphinx",
            reason="sphinx is required for extension setup tests",
        )

        from sentinel.sphinx_jira_role import setup

        app = self._make_mock_app()
        result = setup(app)

        assert isinstance(result, dict)
        assert "version" in result
        assert result["parallel_read_safe"] is True
        assert result["parallel_write_safe"] is True

    def test_setup_registers_config_value(self) -> None:
        """Verify setup() registers the jira_base_url config value."""
        pytest.importorskip(
            "sphinx",
            reason="sphinx is required for extension setup tests",
        )

        from sentinel.sphinx_jira_role import setup

        app = self._make_mock_app()
        setup(app)

        # Verify jira_base_url was registered
        config_names = [name for name, _, _ in app.config_values]  # type: ignore[union-attr]
        assert "jira_base_url" in config_names

    def test_setup_connects_config_inited_callback(self) -> None:
        """Verify setup() registers a config-inited callback (DS-1019).

        The callback warns when jira_base_url is still the default placeholder.
        """
        pytest.importorskip(
            "sphinx",
            reason="sphinx is required for extension setup tests",
        )

        from sentinel.sphinx_jira_role import setup

        app = self._make_mock_app()
        setup(app)

        event_names = [event for event, _ in app.connections]  # type: ignore[union-attr]
        assert "config-inited" in event_names


class TestJiraRoleNodeGeneration:
    """Tests for the jira_role function's node generation.

    These tests verify the role function produces correct docutils nodes
    by using lightweight mocks of the Sphinx/docutils infrastructure.
    """

    def _make_inliner(
        self, base_url: str = "https://test.atlassian.net"
    ) -> object:
        """Create a minimal mock Inliner with the required attributes.

        Args:
            base_url: The Jira base URL to configure.

        Returns:
            A mock inliner object suitable for passing to ``jira_role()``.
        """

        class MockConfig:
            def __init__(self, url: str) -> None:
                self.jira_base_url = url

        class MockEnv:
            def __init__(self, url: str) -> None:
                self.config = MockConfig(url)

        class MockSettings:
            def __init__(self, url: str) -> None:
                self.env = MockEnv(url)

        class MockReporter:
            def warning(self, message: str, line: int = 0) -> object:
                class MockMessage:
                    pass
                return MockMessage()

        class MockDocument:
            def __init__(self, url: str) -> None:
                self.settings = MockSettings(url)

        class MockInliner:
            def __init__(self, url: str) -> None:
                self.document = MockDocument(url)
                self.reporter = MockReporter()

            def problematic(
                self, rawtext: str, text: str, msg: object
            ) -> object:
                from docutils import nodes as n

                return n.problematic(rawtext, text)

        return MockInliner(base_url)

    def test_generates_reference_node_for_valid_key(self) -> None:
        """Verify a valid issue key produces a single reference node."""
        from sentinel.sphinx_jira_role import _jira_role_impl

        inliner = self._make_inliner()
        node_list, messages = _jira_role_impl(
            "jira", ":jira:`DS-1016`", "DS-1016", 1, inliner,
        )

        assert len(node_list) == 1
        assert len(messages) == 0

    def test_reference_node_has_correct_url(self) -> None:
        """Verify the generated hyperlink points to the correct Jira URL."""
        from docutils import nodes

        from sentinel.sphinx_jira_role import _jira_role_impl

        inliner = self._make_inliner("https://myteam.atlassian.net")
        node_list, _ = _jira_role_impl(
            "jira", ":jira:`DS-1016`", "DS-1016", 1, inliner,
        )

        ref_node = node_list[0]
        assert isinstance(ref_node, nodes.reference)
        assert ref_node["refuri"] == "https://myteam.atlassian.net/browse/DS-1016"

    def test_reference_node_display_text_is_issue_key(self) -> None:
        """Verify the hyperlink display text matches the issue key."""
        from docutils import nodes

        from sentinel.sphinx_jira_role import _jira_role_impl

        inliner = self._make_inliner()
        node_list, _ = _jira_role_impl(
            "jira", ":jira:`DS-1016`", "DS-1016", 1, inliner,
        )

        ref_node = node_list[0]
        assert isinstance(ref_node, nodes.reference)
        assert ref_node.astext() == "DS-1016"

    def test_trailing_slash_in_base_url_handled(self) -> None:
        """Verify a trailing slash in the base URL does not produce double slashes."""
        from docutils import nodes

        from sentinel.sphinx_jira_role import _jira_role_impl

        inliner = self._make_inliner("https://test.atlassian.net/")
        node_list, _ = _jira_role_impl(
            "jira", ":jira:`DS-123`", "DS-123", 1, inliner,
        )

        ref_node = node_list[0]
        assert isinstance(ref_node, nodes.reference)
        assert ref_node["refuri"] == "https://test.atlassian.net/browse/DS-123"

    def test_invalid_key_returns_warning(self) -> None:
        """Verify an invalid issue key produces a warning message."""
        from sentinel.sphinx_jira_role import _jira_role_impl

        inliner = self._make_inliner()
        node_list, messages = _jira_role_impl(
            "jira", ":jira:`bad-key`", "bad-key", 1, inliner,
        )

        assert len(messages) == 1

    def test_whitespace_stripped_from_issue_key(self) -> None:
        """Verify leading/trailing whitespace in the issue key is stripped."""
        from docutils import nodes

        from sentinel.sphinx_jira_role import _jira_role_impl

        inliner = self._make_inliner()
        node_list, messages = _jira_role_impl(
            "jira", ":jira:` DS-1016 `", " DS-1016 ", 1, inliner,
        )

        assert len(messages) == 0
        assert len(node_list) == 1
        ref_node = node_list[0]
        assert isinstance(ref_node, nodes.reference)
        assert ref_node.astext() == "DS-1016"

    def test_different_project_key(self) -> None:
        """Verify the role works with different Jira project keys."""
        from docutils import nodes

        from sentinel.sphinx_jira_role import _jira_role_impl

        inliner = self._make_inliner("https://company.atlassian.net")
        node_list, _ = _jira_role_impl(
            "jira", ":jira:`PROJ-42`", "PROJ-42", 1, inliner,
        )

        ref_node = node_list[0]
        assert isinstance(ref_node, nodes.reference)
        assert ref_node["refuri"] == "https://company.atlassian.net/browse/PROJ-42"
        assert ref_node.astext() == "PROJ-42"


class TestJiraRolePublicAdapter:
    """Tests for the public jira_role() adapter function (DS-1019).

    These tests exercise the public ``jira_role()`` adapter directly,
    not just the internal ``_jira_role_impl``, to ensure complete
    coverage of the public API surface.
    """

    def _make_inliner(
        self, base_url: str = "https://test.atlassian.net"
    ) -> object:
        """Create a minimal mock Inliner with the required attributes.

        Args:
            base_url: The Jira base URL to configure.

        Returns:
            A mock inliner object suitable for passing to ``jira_role()``.
        """

        class MockConfig:
            def __init__(self, url: str) -> None:
                self.jira_base_url = url

        class MockEnv:
            def __init__(self, url: str) -> None:
                self.config = MockConfig(url)

        class MockSettings:
            def __init__(self, url: str) -> None:
                self.env = MockEnv(url)

        class MockReporter:
            def warning(self, message: str, line: int = 0) -> object:
                class MockMessage:
                    pass
                return MockMessage()

        class MockDocument:
            def __init__(self, url: str) -> None:
                self.settings = MockSettings(url)

        class MockInliner:
            def __init__(self, url: str) -> None:
                self.document = MockDocument(url)
                self.reporter = MockReporter()

            def problematic(
                self, rawtext: str, text: str, msg: object
            ) -> object:
                from docutils import nodes as n

                return n.problematic(rawtext, text)

        return MockInliner(base_url)

    def test_adapter_produces_reference_node(self) -> None:
        """Verify the public adapter produces a reference node for a valid key."""
        from docutils import nodes

        from sentinel.sphinx_jira_role import jira_role

        inliner = self._make_inliner("https://myteam.atlassian.net")
        node_list, messages = jira_role(
            "jira", ":jira:`DS-1016`", "DS-1016", 1, inliner, {}, [],
        )

        assert len(node_list) == 1
        assert len(messages) == 0
        ref_node = node_list[0]
        assert isinstance(ref_node, nodes.reference)
        assert ref_node["refuri"] == "https://myteam.atlassian.net/browse/DS-1016"
        assert ref_node.astext() == "DS-1016"

    def test_adapter_returns_warning_for_invalid_key(self) -> None:
        """Verify the public adapter returns a warning for an invalid key."""
        from sentinel.sphinx_jira_role import jira_role

        inliner = self._make_inliner()
        node_list, messages = jira_role(
            "jira", ":jira:`bad-key`", "bad-key", 1, inliner, {}, [],
        )

        assert len(messages) == 1

    def test_adapter_forwards_options_and_content(self) -> None:
        """Verify the adapter correctly forwards options and content arguments."""
        from docutils import nodes

        from sentinel.sphinx_jira_role import jira_role

        inliner = self._make_inliner()
        options = {"class": "custom-class"}
        content = ["some content"]
        node_list, messages = jira_role(
            "jira", ":jira:`DS-42`", "DS-42", 1, inliner, options, content,
        )

        # Should still produce a valid reference node regardless of options/content
        assert len(node_list) == 1
        assert len(messages) == 0
        ref_node = node_list[0]
        assert isinstance(ref_node, nodes.reference)
        assert ref_node.astext() == "DS-42"


class TestCheckJiraBaseUrl:
    """Tests for the _check_jira_base_url configuration warning (DS-1019).

    Verifies that the Sphinx build-time warning is emitted when the
    jira_base_url config value is still the default placeholder.
    """

    def test_warns_on_default_placeholder_url(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify a warning is logged when jira_base_url contains example.com."""
        import logging

        from sentinel.sphinx_jira_role import _check_jira_base_url

        class MockConfig:
            jira_base_url = "https://jira.example.com"

        with caplog.at_level(logging.WARNING, logger="sentinel.sphinx_jira_role"):
            _check_jira_base_url(None, MockConfig())

        assert len(caplog.records) == 1
        assert "default placeholder" in caplog.records[0].message
        assert "example.com" in caplog.records[0].message

    def test_no_warning_on_configured_url(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify no warning is logged when jira_base_url is properly configured."""
        import logging

        from sentinel.sphinx_jira_role import _check_jira_base_url

        class MockConfig:
            jira_base_url = "https://myteam.atlassian.net"

        with caplog.at_level(logging.WARNING, logger="sentinel.sphinx_jira_role"):
            _check_jira_base_url(None, MockConfig())

        assert len(caplog.records) == 0

    def test_warns_on_custom_example_dot_com_url(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify a warning is emitted for any URL containing example.com."""
        import logging

        from sentinel.sphinx_jira_role import _check_jira_base_url

        class MockConfig:
            jira_base_url = "https://custom.example.com/jira"

        with caplog.at_level(logging.WARNING, logger="sentinel.sphinx_jira_role"):
            _check_jira_base_url(None, MockConfig())

        assert len(caplog.records) == 1

    def test_warns_when_config_missing_attribute(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify a warning is emitted when config has no jira_base_url attribute.

        Falls back to the default placeholder which contains example.com.
        """
        import logging

        from sentinel.sphinx_jira_role import _check_jira_base_url

        class MockConfig:
            pass

        with caplog.at_level(logging.WARNING, logger="sentinel.sphinx_jira_role"):
            _check_jira_base_url(None, MockConfig())

        assert len(caplog.records) == 1
        assert "default placeholder" in caplog.records[0].message
