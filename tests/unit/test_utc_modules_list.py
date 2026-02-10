"""Tests for UTC_TIMESTAMPS.md affected modules list accuracy.

Validates that the manually curated list of modules in docs/UTC_TIMESTAMPS.md
stays in sync with the actual source files using ``datetime.now(tz=UTC)``.
This prevents the documented list from becoming stale as the codebase evolves.

Follow-up from DS-886 / DS-885 code review.

Usage:
    pytest tests/unit/test_utc_modules_list.py -v
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate from tests/unit/ to project root
    return Path(__file__).parent.parent.parent


def get_documented_modules(utc_doc: Path) -> set[str]:
    """Extract module paths listed in the 'Affected modules' section.

    Parses the markdown bullet list under the ``### Affected modules`` heading
    and returns the set of ``src/...`` paths found in back-tick spans.

    Args:
        utc_doc: Path to the UTC_TIMESTAMPS.md file.

    Returns:
        Set of module path strings (e.g. ``{"src/sentinel/main.py", ...}``).
    """
    content = utc_doc.read_text(encoding="utf-8")

    # Find the "### Affected modules" section
    section_match = re.search(
        r"### Affected modules\s*\n(.*?)(?=\n###|\n## |\Z)",
        content,
        re.DOTALL,
    )
    if section_match is None:
        return set()

    section_text = section_match.group(1)

    # Extract back-ticked module paths from bullet items
    # Pattern: - `src/sentinel/some/module.py` -- optional description
    return set(re.findall(r"`(src/[^`]+\.py)`", section_text))


def get_actual_modules(project_root: Path) -> set[str]:
    """Find source modules that actually use ``datetime.now(tz=UTC)``.

    Uses ``grep`` to search the ``src/`` directory for Python files containing
    the pattern, matching the command documented in UTC_TIMESTAMPS.md.

    Args:
        project_root: Path to the project root directory.

    Returns:
        Set of module path strings relative to the project root.
    """
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "-l", "datetime.now(tz=UTC)", "src/"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # returncode 1 means no matches; other codes are real errors
        if result.returncode == 1:
            return set()
        msg = f"grep failed with return code {result.returncode}: {result.stderr}"
        raise RuntimeError(msg)

    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


class TestUTCModulesList:
    """Validate the affected modules list in UTC_TIMESTAMPS.md."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return get_project_root()

    @pytest.fixture
    def utc_doc(self, project_root: Path) -> Path:
        """Get the UTC_TIMESTAMPS.md file path."""
        return project_root / "docs" / "UTC_TIMESTAMPS.md"

    def test_utc_timestamps_doc_exists(self, utc_doc: Path) -> None:
        """Verify that docs/UTC_TIMESTAMPS.md exists."""
        assert utc_doc.exists(), (
            "docs/UTC_TIMESTAMPS.md not found. "
            "This file documents the UTC timestamp convention (DS-880)."
        )

    def test_affected_modules_section_exists(self, utc_doc: Path) -> None:
        """Verify the Affected modules section exists in the document."""
        content = utc_doc.read_text(encoding="utf-8")
        assert "### Affected modules" in content, (
            "docs/UTC_TIMESTAMPS.md is missing the '### Affected modules' section."
        )

    def test_documented_modules_match_actual(
        self, project_root: Path, utc_doc: Path
    ) -> None:
        """Verify the documented module list matches the actual source files.

        Compares the manually curated list in UTC_TIMESTAMPS.md against
        a live ``grep`` of the source tree. Fails if any modules are
        missing from or extra in the documentation.
        """
        documented = get_documented_modules(utc_doc)
        actual = get_actual_modules(project_root)

        missing_from_docs = actual - documented
        extra_in_docs = documented - actual

        errors: list[str] = []

        if missing_from_docs:
            module_list = "\n  ".join(sorted(missing_from_docs))
            errors.append(
                f"Modules using datetime.now(tz=UTC) but NOT listed in "
                f"docs/UTC_TIMESTAMPS.md:\n  {module_list}"
            )

        if extra_in_docs:
            module_list = "\n  ".join(sorted(extra_in_docs))
            errors.append(
                f"Modules listed in docs/UTC_TIMESTAMPS.md but NOT using "
                f"datetime.now(tz=UTC):\n  {module_list}"
            )

        if errors:
            error_report = "\n\n".join(errors)
            hint = (
                "\nTo update the list, edit the '### Affected modules' section in "
                "docs/UTC_TIMESTAMPS.md.\n"
                "To find current modules: "
                'grep -r --include="*.py" "datetime.now(tz=UTC)" src/'
            )
            pytest.fail(f"Affected modules list is out of sync:\n\n{error_report}\n{hint}")

    def test_documented_modules_are_not_empty(self, utc_doc: Path) -> None:
        """Verify the documented list is not accidentally empty."""
        documented = get_documented_modules(utc_doc)
        assert len(documented) > 0, (
            "No modules found in the '### Affected modules' section of "
            "docs/UTC_TIMESTAMPS.md. The section may have been accidentally cleared "
            "or the parsing pattern may need updating."
        )


class TestUTCModulesListHelpers:
    """Unit tests for the helper functions used in module list validation."""

    def test_get_documented_modules_extracts_paths(self, tmp_path: Path) -> None:
        """Test extraction of module paths from markdown content."""
        doc = tmp_path / "UTC_TIMESTAMPS.md"
        doc.write_text(
            "# UTC Timestamp Convention\n\n"
            "### Affected modules\n\n"
            "The following source modules use `datetime.now(tz=UTC)`:\n\n"
            "- `src/sentinel/main.py` -- poll timestamps\n"
            "- `src/sentinel/executor.py` -- execution timing\n\n"
            "## History\n"
        )

        result = get_documented_modules(doc)

        assert result == {"src/sentinel/main.py", "src/sentinel/executor.py"}

    def test_get_documented_modules_empty_section(self, tmp_path: Path) -> None:
        """Test extraction when the section has no module entries."""
        doc = tmp_path / "UTC_TIMESTAMPS.md"
        doc.write_text(
            "# UTC Timestamp Convention\n\n"
            "### Affected modules\n\n"
            "No modules yet.\n\n"
            "## History\n"
        )

        result = get_documented_modules(doc)

        assert result == set()

    def test_get_documented_modules_missing_section(self, tmp_path: Path) -> None:
        """Test extraction when the Affected modules section is missing."""
        doc = tmp_path / "UTC_TIMESTAMPS.md"
        doc.write_text(
            "# UTC Timestamp Convention\n\n"
            "## History\n\n"
            "Some history here.\n"
        )

        result = get_documented_modules(doc)

        assert result == set()
