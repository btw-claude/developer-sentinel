"""Tests for documentation link validation.

This module verifies that all markdown links in documentation files resolve
to valid targets. This helps catch broken links early in CI before they
reach users.

Usage:
    pytest tests/test_docs_link_validation.py -v
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest


class MarkdownLink(NamedTuple):
    """Represents a markdown link found in a file."""

    text: str
    url: str
    file_path: str
    line_number: int


# Pattern to match markdown links: [text](url)
# Excludes URLs starting with http:// or https:// (external links)
RELATIVE_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate from tests/unit/ to project root
    return Path(__file__).parent.parent.parent


def find_markdown_files(root: Path) -> list[Path]:
    """Find all markdown files in the project.

    Args:
        root: Project root directory.

    Returns:
        List of paths to markdown files.
    """
    md_files: list[Path] = []

    # Search in docs directory
    docs_dir = root / "docs"
    if docs_dir.exists():
        md_files.extend(docs_dir.glob("**/*.md"))

    # Include README.md at project root
    readme = root / "README.md"
    if readme.exists():
        md_files.append(readme)

    # Include orchestrations directory markdown (if any)
    orchestrations_dir = root / "orchestrations"
    if orchestrations_dir.exists():
        md_files.extend(orchestrations_dir.glob("**/*.md"))

    return md_files


def extract_links(file_path: Path) -> list[MarkdownLink]:
    """Extract all markdown links from a file.

    Args:
        file_path: Path to the markdown file.

    Returns:
        List of MarkdownLink objects found in the file.
    """
    links: list[MarkdownLink] = []

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            for match in RELATIVE_LINK_PATTERN.finditer(line):
                text = match.group(1)
                url = match.group(2)
                links.append(
                    MarkdownLink(
                        text=text,
                        url=url,
                        file_path=str(file_path),
                        line_number=line_num,
                    )
                )

    return links


def is_external_link(url: str) -> bool:
    """Check if a URL is an external link.

    Args:
        url: The URL to check.

    Returns:
        True if the URL is external (http/https), False otherwise.
    """
    return url.startswith(("http://", "https://", "mailto:"))


def is_anchor_link(url: str) -> bool:
    """Check if a URL is an anchor-only link (e.g., #section).

    Args:
        url: The URL to check.

    Returns:
        True if the URL is an anchor-only link.
    """
    return url.startswith("#")


def resolve_relative_link(base_file: Path, link_url: str) -> Path | None:
    """Resolve a relative link to an absolute file path.

    Args:
        base_file: The file containing the link.
        link_url: The relative URL from the link.

    Returns:
        Resolved absolute path, or None if the link is invalid.
    """
    # Remove anchor from URL (e.g., "../README.md#section" -> "../README.md")
    url_without_anchor = link_url.split("#")[0]

    if not url_without_anchor:
        # Pure anchor link (e.g., "#section") - treated as valid
        return base_file

    # Resolve relative to the file's directory
    base_dir = base_file.parent
    resolved = (base_dir / url_without_anchor).resolve()

    return resolved


def validate_link(base_file: Path, link: MarkdownLink, project_root: Path) -> str | None:
    """Validate that a link target exists.

    Args:
        base_file: The file containing the link.
        link: The MarkdownLink to validate.
        project_root: The project root directory.

    Returns:
        Error message if the link is broken, None if valid.
    """
    url = link.url

    # Skip external links (we don't validate those in this test)
    if is_external_link(url):
        return None

    # Anchor-only links are valid if they reference headings in the same file
    # (we don't validate heading existence, just that it's syntactically correct)
    if is_anchor_link(url):
        return None

    # Resolve the relative link
    resolved_path = resolve_relative_link(base_file, url)

    if resolved_path is None:
        return f"Invalid link syntax: {url}"

    # Check if the target file exists
    if not resolved_path.exists():
        return f"Broken link: '{url}' -> file not found: {resolved_path}"

    return None


class TestDocumentationLinks:
    """Test class for documentation link validation."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return get_project_root()

    @pytest.fixture
    def markdown_files(self, project_root: Path) -> list[Path]:
        """Get all markdown files in the project."""
        return find_markdown_files(project_root)

    def test_markdown_files_exist(self, markdown_files: list[Path]) -> None:
        """Verify that documentation files exist."""
        assert len(markdown_files) > 0, "No markdown files found in project"

    def test_readme_exists(self, project_root: Path) -> None:
        """Verify README.md exists at project root."""
        readme = project_root / "README.md"
        assert readme.exists(), "README.md not found at project root"

    def test_github_migration_docs_exist(self, project_root: Path) -> None:
        """Verify GitHub trigger migration documentation exists."""
        migration_doc = project_root / "docs" / "GITHUB_TRIGGER_MIGRATION.md"
        assert migration_doc.exists(), "GITHUB_TRIGGER_MIGRATION.md not found"

    def test_all_relative_links_resolve(
        self, markdown_files: list[Path], project_root: Path
    ) -> None:
        """Verify all relative links in documentation files resolve to valid targets.

        This test extracts all markdown links from documentation files and
        verifies that relative links point to files that exist.
        """
        broken_links: list[str] = []

        for md_file in markdown_files:
            links = extract_links(md_file)

            for link in links:
                error = validate_link(md_file, link, project_root)
                if error:
                    broken_links.append(
                        f"{md_file.relative_to(project_root)}:{link.line_number}: "
                        f"[{link.text}]({link.url}) - {error}"
                    )

        if broken_links:
            error_report = "\n".join(broken_links)
            pytest.fail(f"Found {len(broken_links)} broken link(s):\n{error_report}")

    def test_github_migration_internal_links(self, project_root: Path) -> None:
        """Verify internal links in GITHUB_TRIGGER_MIGRATION.md are valid."""
        migration_doc = project_root / "docs" / "GITHUB_TRIGGER_MIGRATION.md"
        links = extract_links(migration_doc)

        # Check that expected cross-references exist
        relative_links = [link for link in links if not is_external_link(link.url)]

        # Validate each relative link
        for link in relative_links:
            if is_anchor_link(link.url):
                continue  # Skip anchor validation

            resolved = resolve_relative_link(migration_doc, link.url)
            assert resolved is not None, f"Invalid link: {link.url}"
            assert resolved.exists(), (
                f"Broken link in GITHUB_TRIGGER_MIGRATION.md line {link.line_number}: "
                f"[{link.text}]({link.url}) -> {resolved} does not exist"
            )

    def test_readme_documentation_links(self, project_root: Path) -> None:
        """Verify README.md links to documentation are valid."""
        readme = project_root / "README.md"
        links = extract_links(readme)

        for link in links:
            if is_external_link(link.url) or is_anchor_link(link.url):
                continue

            resolved = resolve_relative_link(readme, link.url)
            assert resolved is not None, f"Invalid link: {link.url}"
            assert resolved.exists(), (
                f"Broken link in README.md line {link.line_number}: "
                f"[{link.text}]({link.url}) -> {resolved} does not exist"
            )


class TestLinkExtractionHelpers:
    """Unit tests for link extraction helper functions."""

    def test_extract_simple_link(self, tmp_path: Path) -> None:
        """Test extracting a simple markdown link."""
        md_file = tmp_path / "test.md"
        md_file.write_text("See [docs](./docs/file.md) for more info.\n")

        links = extract_links(md_file)

        assert len(links) == 1
        assert links[0].text == "docs"
        assert links[0].url == "./docs/file.md"
        assert links[0].line_number == 1

    def test_extract_multiple_links_on_same_line(self, tmp_path: Path) -> None:
        """Test extracting multiple links from the same line."""
        md_file = tmp_path / "test.md"
        md_file.write_text("See [A](a.md) and [B](b.md) for details.\n")

        links = extract_links(md_file)

        assert len(links) == 2
        assert links[0].text == "A"
        assert links[1].text == "B"

    def test_extract_links_from_multiple_lines(self, tmp_path: Path) -> None:
        """Test extracting links from multiple lines."""
        md_file = tmp_path / "test.md"
        md_file.write_text("Line 1\n[link1](a.md)\nLine 3\n[link2](b.md)\n")

        links = extract_links(md_file)

        assert len(links) == 2
        assert links[0].line_number == 2
        assert links[1].line_number == 4

    def test_is_external_link(self) -> None:
        """Test external link detection."""
        assert is_external_link("https://github.com")
        assert is_external_link("http://example.com")
        assert is_external_link("mailto:test@example.com")
        assert not is_external_link("./docs/file.md")
        assert not is_external_link("../README.md")
        assert not is_external_link("#section")

    def test_is_anchor_link(self) -> None:
        """Test anchor link detection."""
        assert is_anchor_link("#section")
        assert is_anchor_link("#heading-with-dashes")
        assert not is_anchor_link("./file.md#section")
        assert not is_anchor_link("https://example.com#anchor")

    def test_resolve_relative_link(self, tmp_path: Path) -> None:
        """Test resolving relative links."""
        # Create directory structure
        docs = tmp_path / "docs"
        docs.mkdir()
        base_file = docs / "guide.md"
        base_file.touch()
        target = tmp_path / "README.md"
        target.touch()

        resolved = resolve_relative_link(base_file, "../README.md")

        assert resolved is not None
        assert resolved == target.resolve()

    def test_resolve_relative_link_with_anchor(self, tmp_path: Path) -> None:
        """Test resolving relative links that include anchors."""
        docs = tmp_path / "docs"
        docs.mkdir()
        base_file = docs / "guide.md"
        base_file.touch()
        target = tmp_path / "README.md"
        target.touch()

        resolved = resolve_relative_link(base_file, "../README.md#section")

        assert resolved is not None
        # Should resolve to README.md, ignoring the anchor
        assert resolved == target.resolve()
