"""Tests for dashboard startup graceful degradation in app.py.

These tests verify that the start_dashboard function handles failures gracefully
and allows Sentinel to continue operating without the dashboard (DS-515).

Also includes tests for the format_duration Jinja2 filter (DS-529).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel.app import start_dashboard
from sentinel.bootstrap import BootstrapContext
from sentinel.config import Config, DashboardConfig
from sentinel.dashboard.app import format_duration, url_quote


class MockSentinel:
    """Mock Sentinel for testing dashboard startup."""

    def __init__(self, config: Any) -> None:
        """Initialize mock sentinel with config."""
        self.config = config


@pytest.fixture
def mock_bootstrap_context() -> BootstrapContext:
    """Create a mock BootstrapContext for testing."""
    config = Config(
        dashboard=DashboardConfig(
            enabled=True,
            host="127.0.0.1",
            port=8080,
        )
    )
    # Create minimal mock for BootstrapContext
    context = MagicMock(spec=BootstrapContext)
    context.config = config
    return context


@pytest.fixture
def mock_sentinel() -> MockSentinel:
    """Create a mock Sentinel for testing."""
    config = Config()
    return MockSentinel(config)


class TestStartDashboardGracefulDegradation:
    """Tests for start_dashboard graceful degradation behavior."""

    def test_returns_none_when_dashboard_disabled(
        self, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard returns None when dashboard is disabled."""
        config = Config(dashboard=DashboardConfig(enabled=False))
        context = MagicMock(spec=BootstrapContext)
        context.config = config

        result = start_dashboard(context, mock_sentinel)  # type: ignore[arg-type]

        assert result is None

    def test_logs_info_when_dashboard_disabled(
        self, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard logs info when dashboard is disabled."""
        config = Config(dashboard=DashboardConfig(enabled=False))
        context = MagicMock(spec=BootstrapContext)
        context.config = config

        with patch("sentinel.app.logger") as mock_logger:
            start_dashboard(context, mock_sentinel)  # type: ignore[arg-type]

            mock_logger.info.assert_called_once()
            assert "disabled" in mock_logger.info.call_args[0][0].lower()

    def test_returns_none_on_import_error(
        self, mock_bootstrap_context: BootstrapContext, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard returns None on ImportError."""
        import sys

        with patch("sentinel.app.logger") as mock_logger:
            # Remove the dashboard module from sys.modules to force a fresh import
            modules_to_remove = [k for k in sys.modules if k.startswith("sentinel.dashboard")]
            saved_modules = {k: sys.modules[k] for k in modules_to_remove if k in sys.modules}

            try:
                for k in modules_to_remove:
                    if k in sys.modules:
                        del sys.modules[k]

                # Mock the import of sentinel.dashboard to raise ImportError
                original_import = __builtins__["__import__"]

                def mock_import(
                    name: str,
                    globals: Any = None,
                    locals: Any = None,
                    fromlist: Any = (),
                    level: int = 0,
                ) -> Any:
                    if name == "sentinel.dashboard" or (
                        fromlist and "create_app" in fromlist and "sentinel.dashboard" in name
                    ):
                        raise ImportError("Test import error")
                    return original_import(name, globals, locals, fromlist, level)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = start_dashboard(
                        mock_bootstrap_context, mock_sentinel  # type: ignore[arg-type]
                    )

                    assert result is None
                    mock_logger.warning.assert_called_once()
                    warning_msg = mock_logger.warning.call_args[0][0]
                    assert "dependencies not available" in warning_msg
                    assert "continue without dashboard" in warning_msg

            finally:
                # Restore modules
                sys.modules.update(saved_modules)

    def test_returns_none_on_os_error(
        self, mock_bootstrap_context: BootstrapContext, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard returns None on OSError."""
        with patch("sentinel.app.logger") as mock_logger:
            with patch("sentinel.dashboard.create_app") as mock_create_app:
                mock_create_app.return_value = MagicMock()
                with patch(
                    "sentinel.dashboard_server.DashboardServer.start",
                    side_effect=OSError("Address already in use"),
                ):
                    result = start_dashboard(
                        mock_bootstrap_context, mock_sentinel  # type: ignore[arg-type]
                    )

                    assert result is None
                    mock_logger.warning.assert_called_once()
                    warning_msg = mock_logger.warning.call_args[0][0]
                    assert "network/OS error" in warning_msg
                    assert "continue without dashboard" in warning_msg

    def test_returns_none_on_runtime_error(
        self, mock_bootstrap_context: BootstrapContext, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard returns None on RuntimeError."""
        with patch("sentinel.app.logger") as mock_logger:
            with patch("sentinel.dashboard.create_app") as mock_create_app:
                mock_create_app.return_value = MagicMock()
                with patch(
                    "sentinel.dashboard_server.DashboardServer.start",
                    side_effect=RuntimeError("Thread error"),
                ):
                    result = start_dashboard(
                        mock_bootstrap_context, mock_sentinel  # type: ignore[arg-type]
                    )

                    assert result is None
                    mock_logger.warning.assert_called_once()
                    warning_msg = mock_logger.warning.call_args[0][0]
                    assert "runtime error" in warning_msg
                    assert "continue without dashboard" in warning_msg

    def test_returns_none_on_value_error(
        self, mock_bootstrap_context: BootstrapContext, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard returns None on ValueError."""
        with patch("sentinel.app.logger") as mock_logger:
            with patch("sentinel.dashboard.create_app") as mock_create_app:
                mock_create_app.side_effect = ValueError("Invalid configuration")
                result = start_dashboard(
                    mock_bootstrap_context, mock_sentinel  # type: ignore[arg-type]
                )

                assert result is None
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "configuration error" in warning_msg
                assert "continue without dashboard" in warning_msg

    def test_returns_none_on_type_error(
        self, mock_bootstrap_context: BootstrapContext, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard returns None on TypeError."""
        with patch("sentinel.app.logger") as mock_logger:
            with patch("sentinel.dashboard.create_app") as mock_create_app:
                mock_create_app.side_effect = TypeError("Type mismatch")
                result = start_dashboard(
                    mock_bootstrap_context, mock_sentinel  # type: ignore[arg-type]
                )

                assert result is None
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "configuration error" in warning_msg
                assert "continue without dashboard" in warning_msg

    def test_returns_none_on_unexpected_exception(
        self, mock_bootstrap_context: BootstrapContext, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard returns None on unexpected Exception."""
        with patch("sentinel.app.logger") as mock_logger:
            with patch("sentinel.dashboard.create_app") as mock_create_app:
                mock_create_app.side_effect = Exception("Unexpected error")
                result = start_dashboard(
                    mock_bootstrap_context, mock_sentinel  # type: ignore[arg-type]
                )

                assert result is None
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "unexpected error" in warning_msg
                assert "continue without dashboard" in warning_msg

    def test_logs_warning_with_host_and_port_on_error(
        self, mock_bootstrap_context: BootstrapContext, mock_sentinel: MockSentinel
    ) -> None:
        """Test that warning logs include host and port information."""
        with patch("sentinel.app.logger") as mock_logger:
            with patch("sentinel.dashboard.create_app") as mock_create_app:
                mock_create_app.side_effect = OSError("Port in use")
                start_dashboard(
                    mock_bootstrap_context, mock_sentinel  # type: ignore[arg-type]
                )

                # Check extra dict contains host and port
                call_kwargs = mock_logger.warning.call_args[1]
                assert "extra" in call_kwargs
                extra = call_kwargs["extra"]
                assert extra["host"] == "127.0.0.1"
                assert extra["port"] == 8080

    def test_returns_server_on_success(
        self, mock_bootstrap_context: BootstrapContext, mock_sentinel: MockSentinel
    ) -> None:
        """Test that start_dashboard returns DashboardServer on success."""
        with patch("sentinel.app.DashboardServer") as mock_server_class:
            mock_server = MagicMock()
            mock_server_class.return_value = mock_server

            result = start_dashboard(
                mock_bootstrap_context, mock_sentinel  # type: ignore[arg-type]
            )

            assert result is mock_server
            mock_server.start.assert_called_once()


class TestFormatDurationFilter:
    """Tests for the format_duration Jinja2 filter."""

    def test_format_duration_seconds_only(self) -> None:
        """Test formatting durations less than 60 seconds."""
        assert format_duration(45) == "45s"
        assert format_duration(0) == "0s"
        assert format_duration(59) == "59s"

    def test_format_duration_with_minutes(self) -> None:
        """Test formatting durations of 60 seconds or more."""
        assert format_duration(60) == "1m 0s"
        assert format_duration(125) == "2m 5s"
        assert format_duration(3661) == "61m 1s"

    def test_format_duration_float_input(self) -> None:
        """Test that float inputs are handled correctly."""
        assert format_duration(45.7) == "45s"
        assert format_duration(125.9) == "2m 5s"

    def test_format_duration_none_input(self) -> None:
        """Test that None input returns '0s'."""
        assert format_duration(None) == "0s"

    def test_format_duration_negative_input(self) -> None:
        """Test that negative input returns '0s'."""
        assert format_duration(-5) == "0s"
        assert format_duration(-100) == "0s"


class TestUrlQuoteFilter:
    """Tests for the url_quote Jinja2 filter (DS-776).

    Verifies that url_quote uses percent-encoding (%20 for spaces)
    instead of quote_plus encoding (+ for spaces), ensuring correct
    round-tripping with JavaScript's decodeURIComponent().
    """

    def test_spaces_encoded_as_percent_20(self) -> None:
        """Test that spaces are encoded as %20, not +."""
        assert url_quote("my orchestration") == "my%20orchestration"

    def test_plus_sign_encoded(self) -> None:
        """Test that literal + signs are percent-encoded."""
        assert url_quote("hello+world") == "hello%2Bworld"

    def test_simple_string_unchanged(self) -> None:
        """Test that simple alphanumeric strings are not modified."""
        assert url_quote("simple") == "simple"
        assert url_quote("test-orch") == "test-orch"

    def test_special_characters_encoded(self) -> None:
        """Test that special characters are properly percent-encoded."""
        assert url_quote("name with spaces & symbols!") == "name%20with%20spaces%20%26%20symbols%21"

    def test_empty_string(self) -> None:
        """Test that empty string returns empty string."""
        assert url_quote("") == ""

    def test_already_safe_characters(self) -> None:
        """Test that hyphens and dots are encoded since safe=''."""
        # With safe="", even - and . are left as-is by urllib.parse.quote
        # because they are unreserved characters per RFC 3986
        assert url_quote("a-b.c") == "a-b.c"

    def test_unicode_characters(self) -> None:
        """Test that unicode characters are percent-encoded."""
        result = url_quote("caf\u00e9")
        assert "caf" in result
        assert "%" in result  # e with accent should be encoded

    def test_non_string_input_converted(self) -> None:
        """Test that non-string input is converted to string first."""
        assert url_quote(123) == "123"  # type: ignore[arg-type]

    def test_slash_encoded(self) -> None:
        """Test that slashes are encoded since safe=''."""
        assert url_quote("path/to/thing") == "path%2Fto%2Fthing"
