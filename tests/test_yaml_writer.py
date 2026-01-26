"""Tests for YAML writer module.

DS-248: Create YAML writer module for safe orchestration file modification
"""

import os
from pathlib import Path

import pytest

from sentinel.yaml_writer import (
    OrchestrationYamlWriter,
    OrchestrationYamlWriterError,
    _file_lock,
)


class TestOrchestrationYamlWriter:
    """Tests for OrchestrationYamlWriter class."""

    def test_initialization(self) -> None:
        """Writer should initialize with round-trip YAML configuration."""
        writer = OrchestrationYamlWriter()
        assert writer._yaml.preserve_quotes is True

    def test_toggle_orchestration_success(self, tmp_path: Path) -> None:
        """Should successfully toggle an orchestration's enabled status."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "test-orch", False)

        assert result is True
        # Verify the file was updated
        updated_content = file_path.read_text()
        assert "enabled: false" in updated_content

    def test_toggle_orchestration_enable(self, tmp_path: Path) -> None:
        """Should enable a disabled orchestration."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: false
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "test-orch", True)

        assert result is True
        updated_content = file_path.read_text()
        assert "enabled: true" in updated_content

    def test_toggle_orchestration_not_found(self, tmp_path: Path) -> None:
        """Should return False when orchestration is not found."""
        yaml_content = """
orchestrations:
  - name: "other-orch"
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "nonexistent", False)

        assert result is False

    def test_toggle_orchestration_no_orchestrations_key(self, tmp_path: Path) -> None:
        """Should return False when no orchestrations key exists."""
        yaml_content = """
some_other_key: value
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "test-orch", False)

        assert result is False

    def test_toggle_orchestration_file_not_found(self, tmp_path: Path) -> None:
        """Should raise error when file does not exist."""
        writer = OrchestrationYamlWriter()

        with pytest.raises(
            OrchestrationYamlWriterError, match="Orchestration file not found"
        ):
            writer.toggle_orchestration(tmp_path / "nonexistent.yaml", "test", False)

    def test_toggle_orchestration_preserves_formatting(self, tmp_path: Path) -> None:
        """Should preserve YAML formatting and comments."""
        yaml_content = """# File-level comment
orchestrations:
  # Orchestration comment
  - name: "test-orch"
    enabled: true  # inline comment
    trigger:
      source: jira
      project: "TEST"
      tags:
        - "tag1"
        - "tag2"
    agent:
      prompt: |
        Multi-line
        prompt here
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        writer.toggle_orchestration(file_path, "test-orch", False)

        updated_content = file_path.read_text()
        # Comments should be preserved
        assert "# File-level comment" in updated_content
        assert "# Orchestration comment" in updated_content
        # Multiline prompt should be preserved
        assert "Multi-line" in updated_content
        assert "prompt here" in updated_content
        # The enabled value should be changed
        assert "enabled: false" in updated_content

    def test_toggle_orchestration_adds_enabled_field(self, tmp_path: Path) -> None:
        """Should add enabled field if it doesn't exist."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "test-orch", False)

        assert result is True
        updated_content = file_path.read_text()
        assert "enabled: false" in updated_content


class TestToggleAllInFile:
    """Tests for toggle_all_in_file method."""

    def test_toggle_all_in_file_success(self, tmp_path: Path) -> None:
        """Should toggle all orchestrations in a file."""
        yaml_content = """
orchestrations:
  - name: "orch-one"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "First"
  - name: "orch-two"
    enabled: false
    trigger:
      source: jira
    agent:
      prompt: "Second"
  - name: "orch-three"
    trigger:
      source: jira
    agent:
      prompt: "Third"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_all_in_file(file_path, False)

        assert count == 3
        updated_content = file_path.read_text()
        assert updated_content.count("enabled: false") == 3

    def test_toggle_all_in_file_enable(self, tmp_path: Path) -> None:
        """Should enable all orchestrations in a file."""
        yaml_content = """
orchestrations:
  - name: "orch-one"
    enabled: false
    trigger:
      source: jira
    agent:
      prompt: "First"
  - name: "orch-two"
    enabled: false
    trigger:
      source: jira
    agent:
      prompt: "Second"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_all_in_file(file_path, True)

        assert count == 2
        updated_content = file_path.read_text()
        assert updated_content.count("enabled: true") == 2

    def test_toggle_all_in_file_no_orchestrations(self, tmp_path: Path) -> None:
        """Should return 0 when no orchestrations key exists."""
        yaml_content = """
other_key: value
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_all_in_file(file_path, False)

        assert count == 0

    def test_toggle_all_in_file_empty_orchestrations(self, tmp_path: Path) -> None:
        """Should return 0 when orchestrations list is empty."""
        yaml_content = """
orchestrations: []
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_all_in_file(file_path, False)

        assert count == 0


class TestToggleByProject:
    """Tests for toggle_by_project method."""

    def test_toggle_by_project_success(self, tmp_path: Path) -> None:
        """Should toggle orchestrations matching the project."""
        yaml_content = """
orchestrations:
  - name: "proj-a-orch"
    enabled: true
    trigger:
      source: jira
      project: "PROJ-A"
    agent:
      prompt: "First"
  - name: "proj-b-orch"
    enabled: true
    trigger:
      source: jira
      project: "PROJ-B"
    agent:
      prompt: "Second"
  - name: "proj-a-orch-2"
    enabled: true
    trigger:
      source: jira
      project: "PROJ-A"
    agent:
      prompt: "Third"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_by_project(
            {"test": file_path}, "PROJ-A", False
        )

        assert count == 2
        updated_content = file_path.read_text()
        # PROJ-A orchestrations should be disabled
        assert updated_content.count("enabled: false") == 2
        # PROJ-B should still be enabled
        assert "enabled: true" in updated_content

    def test_toggle_by_project_multiple_files(self, tmp_path: Path) -> None:
        """Should toggle orchestrations across multiple files."""
        file1_content = """
orchestrations:
  - name: "file1-orch"
    trigger:
      source: jira
      project: "PROJ"
    agent:
      prompt: "File 1"
"""
        file2_content = """
orchestrations:
  - name: "file2-orch"
    trigger:
      source: jira
      project: "PROJ"
    agent:
      prompt: "File 2"
"""
        file1 = tmp_path / "file1.yaml"
        file2 = tmp_path / "file2.yaml"
        file1.write_text(file1_content)
        file2.write_text(file2_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_by_project(
            {"orch1": file1, "orch2": file2}, "PROJ", False
        )

        assert count == 2
        assert "enabled: false" in file1.read_text()
        assert "enabled: false" in file2.read_text()

    def test_toggle_by_project_no_matches(self, tmp_path: Path) -> None:
        """Should return 0 when no orchestrations match the project."""
        yaml_content = """
orchestrations:
  - name: "other-orch"
    trigger:
      source: jira
      project: "OTHER"
    agent:
      prompt: "Other"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_by_project(
            {"test": file_path}, "NONEXISTENT", False
        )

        assert count == 0

    def test_toggle_by_project_github_trigger_ignored(self, tmp_path: Path) -> None:
        """Should not match GitHub triggers (they don't use project field for Jira)."""
        yaml_content = """
orchestrations:
  - name: "github-orch"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
    agent:
      prompt: "GitHub"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_by_project(
            {"test": file_path}, "my-org", False
        )

        # Should not match because GitHub triggers use project_owner, not project
        assert count == 0

    def test_toggle_by_project_deduplicates_files(self, tmp_path: Path) -> None:
        """Should handle duplicate file paths correctly."""
        yaml_content = """
orchestrations:
  - name: "orch-one"
    trigger:
      source: jira
      project: "PROJ"
    agent:
      prompt: "One"
  - name: "orch-two"
    trigger:
      source: jira
      project: "PROJ"
    agent:
      prompt: "Two"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        # Same file referenced twice with different keys
        count = writer.toggle_by_project(
            {"orch-one": file_path, "orch-two": file_path}, "PROJ", False
        )

        # Should only count 2 (the orchestrations), not process file twice
        assert count == 2


class TestFileLocking:
    """Tests for file locking functionality."""

    def test_file_lock_creates_lock_file(self, tmp_path: Path) -> None:
        """File lock should create a .lock file."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")
        lock_path = file_path.with_suffix(".yaml.lock")

        with _file_lock(file_path):
            assert lock_path.exists()

    def test_file_lock_releases_lock(self, tmp_path: Path) -> None:
        """File lock should be released after context exit."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")

        with _file_lock(file_path):
            pass

        # Should be able to acquire lock again
        with _file_lock(file_path):
            pass


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid YAML."""
        file_path = tmp_path / "invalid.yaml"
        file_path.write_text("{ invalid yaml: [")

        writer = OrchestrationYamlWriter()

        with pytest.raises(OrchestrationYamlWriterError, match="Failed to parse YAML"):
            writer.toggle_orchestration(file_path, "test", False)

    def test_permission_error_reading(self, tmp_path: Path) -> None:
        """Should raise error when file cannot be read."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("orchestrations: []")
        # Make file unreadable
        os.chmod(file_path, 0o000)

        writer = OrchestrationYamlWriter()

        try:
            with pytest.raises(OrchestrationYamlWriterError, match="Permission denied"):
                writer.toggle_orchestration(file_path, "test", False)
        finally:
            # Restore permissions for cleanup
            os.chmod(file_path, 0o644)

    def test_permission_error_writing(self, tmp_path: Path) -> None:
        """Should raise error when file cannot be written."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)
        # Make file read-only
        os.chmod(file_path, 0o444)

        writer = OrchestrationYamlWriter()

        try:
            with pytest.raises(OrchestrationYamlWriterError, match="Permission denied"):
                writer.toggle_orchestration(file_path, "test-orch", False)
        finally:
            # Restore permissions for cleanup
            os.chmod(file_path, 0o644)

    def test_empty_yaml_file(self, tmp_path: Path) -> None:
        """Should handle empty YAML file gracefully."""
        file_path = tmp_path / "empty.yaml"
        file_path.write_text("")

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "test", False)

        assert result is False

    def test_yaml_with_only_null(self, tmp_path: Path) -> None:
        """Should handle YAML file with only null content."""
        file_path = tmp_path / "null.yaml"
        file_path.write_text("null")

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "test", False)

        assert result is False


class TestQuotePreservation:
    """Tests for quote preservation in YAML files."""

    def test_preserves_single_quotes(self, tmp_path: Path) -> None:
        """Should preserve single quotes in YAML."""
        yaml_content = """
orchestrations:
  - name: 'test-orch'
    enabled: true
    trigger:
      source: jira
      project: 'TEST'
    agent:
      prompt: 'Test prompt'
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        writer.toggle_orchestration(file_path, "test-orch", False)

        updated_content = file_path.read_text()
        # Single quotes should be preserved
        assert "'test-orch'" in updated_content
        assert "'TEST'" in updated_content
        assert "'Test prompt'" in updated_content

    def test_preserves_double_quotes(self, tmp_path: Path) -> None:
        """Should preserve double quotes in YAML."""
        yaml_content = '''
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
'''
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        writer.toggle_orchestration(file_path, "test-orch", False)

        updated_content = file_path.read_text()
        # Double quotes should be preserved
        assert '"test-orch"' in updated_content
        assert '"TEST"' in updated_content
        assert '"Test prompt"' in updated_content


class TestMultipleOrchestrations:
    """Tests for handling files with multiple orchestrations."""

    def test_toggle_first_orchestration(self, tmp_path: Path) -> None:
        """Should toggle only the first matching orchestration."""
        yaml_content = """
orchestrations:
  - name: "first-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "First"
  - name: "second-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Second"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "first-orch", False)

        assert result is True
        updated_content = file_path.read_text()
        # First should be disabled
        assert updated_content.count("enabled: false") == 1
        # Second should still be enabled
        assert updated_content.count("enabled: true") == 1

    def test_toggle_last_orchestration(self, tmp_path: Path) -> None:
        """Should toggle only the last matching orchestration."""
        yaml_content = """
orchestrations:
  - name: "first-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "First"
  - name: "second-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Second"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        result = writer.toggle_orchestration(file_path, "second-orch", False)

        assert result is True
        updated_content = file_path.read_text()
        # Second should be disabled
        assert updated_content.count("enabled: false") == 1
        # First should still be enabled
        assert updated_content.count("enabled: true") == 1
