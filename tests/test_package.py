"""Tests for package-level functionality."""


def test_package_imports() -> None:
    """Verify the package imports correctly."""
    from sentinel import __version__

    assert __version__
    assert isinstance(__version__, str)


def test_version_format() -> None:
    """Verify the version follows expected format."""
    from sentinel import __version__

    # Version should be either a proper semver or dev version
    parts = __version__.split(".")
    assert len(parts) >= 2, f"Version should have at least major.minor: {__version__}"
