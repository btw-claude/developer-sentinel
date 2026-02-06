"""Tests for YAML writer module.

Create YAML writer module for safe orchestration file modification.
Add timeout and cleanup improvements.
Minor enhancements from code review.
"""

import fcntl
import os
import threading
import time
from pathlib import Path

import pytest

from sentinel.yaml_writer import (
    FileLockTimeoutError,
    OrchestrationYamlWriter,
    OrchestrationYamlWriterError,
    _file_lock,
    cleanup_orphaned_lock_files,
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

        with pytest.raises(OrchestrationYamlWriterError, match="Orchestration file not found"):
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
        count = writer.toggle_by_project({"test": file_path}, "PROJ-A", False)

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
        count = writer.toggle_by_project({"orch1": file1, "orch2": file2}, "PROJ", False)

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
        count = writer.toggle_by_project({"test": file_path}, "NONEXISTENT", False)

        assert count == 0

    def test_toggle_by_project_github_trigger_matched_by_project_owner(
        self, tmp_path: Path
    ) -> None:
        """Should match GitHub triggers by project_owner field."""
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
        count = writer.toggle_by_project({"test": file_path}, "my-org", False)

        # Should match because toggle_by_project also checks project_owner
        assert count == 1

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
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
"""
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


class TestFileLockTimeout:
    """Tests for file lock timeout functionality."""

    def test_file_lock_with_default_timeout(self, tmp_path: Path) -> None:
        """File lock should use default timeout when not specified."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")

        # Should succeed with default timeout
        with _file_lock(file_path):
            pass

    def test_file_lock_with_custom_timeout(self, tmp_path: Path) -> None:
        """File lock should accept custom timeout."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")

        # Should succeed with custom timeout
        with _file_lock(file_path, max_wait_seconds=5.0):
            pass

    def test_file_lock_timeout_raises_error(self, tmp_path: Path) -> None:
        """File lock should raise FileLockTimeoutError when timeout expires."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")
        lock_path = file_path.with_suffix(".yaml.lock")

        # Hold the lock in another context
        with open(lock_path, "w") as lock_file:  # noqa: SIM117
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            try:
                # Attempt to acquire with short timeout
                with (
                    pytest.raises(FileLockTimeoutError, match="Timed out waiting for lock"),
                    _file_lock(file_path, max_wait_seconds=0.2),
                ):
                    pass
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def test_file_lock_zero_timeout_blocks_indefinitely(self, tmp_path: Path) -> None:
        """File lock with max_wait_seconds=0 should block indefinitely."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")

        lock_acquired = threading.Event()
        main_thread_waiting = threading.Event()

        def hold_lock_briefly() -> None:
            lock_path = file_path.with_suffix(".yaml.lock")
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                lock_acquired.set()
                # Wait for main thread to start waiting for lock
                main_thread_waiting.wait(timeout=1.0)
                # Hold lock a bit longer
                time.sleep(0.1)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        # Start thread to hold lock briefly
        thread = threading.Thread(target=hold_lock_briefly)
        thread.start()

        # Wait for lock to be acquired by other thread
        lock_acquired.wait(timeout=1.0)

        # Signal that we're about to wait for the lock
        main_thread_waiting.set()

        # This should block until the other thread releases
        # (using max_wait_seconds=0 for indefinite blocking)
        with _file_lock(file_path, max_wait_seconds=0):
            # We successfully acquired the lock after blocking
            pass

        thread.join()

    def test_writer_with_custom_timeout(self, tmp_path: Path) -> None:
        """Writer should use configured timeout for file locks."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter(lock_timeout_seconds=5.0)
        result = writer.toggle_orchestration(file_path, "test-orch", False)

        assert result is True


class TestFileLockCleanup:
    """Tests for file lock cleanup functionality."""

    def test_file_lock_cleanup_removes_lock_file(self, tmp_path: Path) -> None:
        """File lock should remove lock file when cleanup is enabled."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")
        lock_path = file_path.with_suffix(".yaml.lock")

        with _file_lock(file_path, cleanup_lock_file=True):
            assert lock_path.exists()

        # Lock file should be removed after context exit
        assert not lock_path.exists()

    def test_file_lock_no_cleanup_by_default(self, tmp_path: Path) -> None:
        """File lock should NOT remove lock file by default."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")
        lock_path = file_path.with_suffix(".yaml.lock")

        with _file_lock(file_path, cleanup_lock_file=False):
            assert lock_path.exists()

        # Lock file should remain after context exit
        assert lock_path.exists()

    def test_writer_with_cleanup_enabled(self, tmp_path: Path) -> None:
        """Writer should clean up lock files when configured."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)
        lock_path = file_path.with_suffix(".yaml.lock")

        writer = OrchestrationYamlWriter(cleanup_lock_files=True)
        writer.toggle_orchestration(file_path, "test-orch", False)

        # Lock file should be removed
        assert not lock_path.exists()

    def test_cleanup_orphaned_lock_files(self, tmp_path: Path) -> None:
        """Should remove orphaned lock files older than max age."""
        # Create some lock files
        old_lock = tmp_path / "old.yaml.lock"
        old_lock.write_text("")
        # Set modification time to 2 hours ago
        old_time = time.time() - 7200
        os.utime(old_lock, (old_time, old_time))

        new_lock = tmp_path / "new.yaml.lock"
        new_lock.write_text("")

        # Clean up files older than 1 hour
        removed = cleanup_orphaned_lock_files(tmp_path, max_age_seconds=3600)

        assert removed == 1
        assert not old_lock.exists()
        assert new_lock.exists()

    def test_cleanup_orphaned_lock_files_nested(self, tmp_path: Path) -> None:
        """Should find lock files in nested directories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        old_lock = subdir / "nested.yaml.lock"
        old_lock.write_text("")
        old_time = time.time() - 7200
        os.utime(old_lock, (old_time, old_time))

        removed = cleanup_orphaned_lock_files(tmp_path, max_age_seconds=3600)

        assert removed == 1
        assert not old_lock.exists()


class TestBackupFunctionality:
    """Tests for backup before modification functionality."""

    def test_backup_creates_bak_file(self, tmp_path: Path) -> None:
        """Writer should create .bak file when backups are enabled."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)
        backup_path = file_path.with_suffix(".yaml.bak")

        writer = OrchestrationYamlWriter(create_backups=True)
        writer.toggle_orchestration(file_path, "test-orch", False)

        assert backup_path.exists()
        # Backup should contain original content
        assert "enabled: true" in backup_path.read_text()
        # Original file should be modified
        assert "enabled: false" in file_path.read_text()

    def test_backup_with_custom_suffix(self, tmp_path: Path) -> None:
        """Writer should use custom backup suffix when specified."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)
        backup_path = file_path.with_suffix(".yaml.backup")

        writer = OrchestrationYamlWriter(create_backups=True, backup_suffix=".backup")
        writer.toggle_orchestration(file_path, "test-orch", False)

        assert backup_path.exists()
        assert "enabled: true" in backup_path.read_text()

    def test_backup_with_timestamp(self, tmp_path: Path) -> None:
        """Writer should create timestamped backups when specified."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter(create_backups=True, backup_suffix="timestamp")
        writer.toggle_orchestration(file_path, "test-orch", False)

        # Find timestamped backup (now uses test.TIMESTAMP.bak format)
        backup_files = list(tmp_path.glob("test.*.bak"))
        assert len(backup_files) == 1
        assert "enabled: true" in backup_files[0].read_text()

    def test_no_backup_by_default(self, tmp_path: Path) -> None:
        """Writer should NOT create backups by default."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        writer.toggle_orchestration(file_path, "test-orch", False)

        # No backup files should exist
        backup_files = list(tmp_path.glob("*.bak"))
        assert len(backup_files) == 0

    def test_backup_preserves_metadata(self, tmp_path: Path) -> None:
        """Backup should preserve file metadata using shutil.copy2."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        # Get original modification time
        original_mtime = file_path.stat().st_mtime

        # Wait a bit to ensure times differ
        time.sleep(0.1)

        writer = OrchestrationYamlWriter(create_backups=True)
        writer.toggle_orchestration(file_path, "test-orch", False)

        backup_path = file_path.with_suffix(".yaml.bak")
        # Backup should have the original modification time (from copy2)
        assert backup_path.stat().st_mtime == original_mtime

    def test_backup_on_toggle_all_in_file(self, tmp_path: Path) -> None:
        """Backup should be created for toggle_all_in_file."""
        yaml_content = """
orchestrations:
  - name: "orch-one"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "First"
  - name: "orch-two"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Second"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)
        backup_path = file_path.with_suffix(".yaml.bak")

        writer = OrchestrationYamlWriter(create_backups=True)
        writer.toggle_all_in_file(file_path, False)

        assert backup_path.exists()
        backup_content = backup_path.read_text()
        assert backup_content.count("enabled: true") == 2

    def test_backup_on_toggle_by_project(self, tmp_path: Path) -> None:
        """Backup should be created for toggle_by_project."""
        yaml_content = """
orchestrations:
  - name: "proj-orch"
    enabled: true
    trigger:
      source: jira
      project: "PROJ"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)
        backup_path = file_path.with_suffix(".yaml.bak")

        writer = OrchestrationYamlWriter(create_backups=True)
        writer.toggle_by_project({"test": file_path}, "PROJ", False)

        assert backup_path.exists()
        assert "enabled: true" in backup_path.read_text()


class TestWriterConfiguration:
    """Tests for OrchestrationYamlWriter configuration options."""

    def test_default_configuration(self) -> None:
        """Writer should have sensible defaults."""
        writer = OrchestrationYamlWriter()
        assert writer._lock_timeout_seconds is None  # Uses DEFAULT_LOCK_TIMEOUT_SECONDS
        assert writer._cleanup_lock_files is False
        assert writer._create_backups is False
        assert writer._backup_suffix == ".bak"

    def test_custom_configuration(self) -> None:
        """Writer should accept custom configuration."""
        writer = OrchestrationYamlWriter(
            lock_timeout_seconds=10.0,
            cleanup_lock_files=True,
            create_backups=True,
            backup_suffix=".backup",
        )
        assert writer._lock_timeout_seconds == 10.0
        assert writer._cleanup_lock_files is True
        assert writer._create_backups is True
        assert writer._backup_suffix == ".backup"

    def test_combined_features(self, tmp_path: Path) -> None:
        """Writer should handle all features together."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter(
            lock_timeout_seconds=5.0,
            retry_interval_seconds=0.2,
            cleanup_lock_files=True,
            create_backups=True,
            backup_suffix="timestamp",
        )
        writer.toggle_orchestration(file_path, "test-orch", False)

        # Should have created timestamped backup (uses test.TIMESTAMP.bak format)
        backup_files = list(tmp_path.glob("test.*.bak"))
        assert len(backup_files) == 1

        # Lock file should be cleaned up
        lock_path = file_path.with_suffix(".yaml.lock")
        assert not lock_path.exists()

        # File should be modified
        assert "enabled: false" in file_path.read_text()


class TestCodeReviewEnhancements:
    """Tests for enhancements from code review."""

    def test_configurable_retry_interval(self, tmp_path: Path) -> None:
        """Writer should accept custom retry interval parameter."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        # Create writer with custom retry interval
        writer = OrchestrationYamlWriter(retry_interval_seconds=0.2)
        assert writer._retry_interval_seconds == 0.2

        # Should still work correctly
        result = writer.toggle_orchestration(file_path, "test-orch", False)
        assert result is True
        assert "enabled: false" in file_path.read_text()

    def test_file_lock_with_custom_retry_interval(self, tmp_path: Path) -> None:
        """File lock should use custom retry interval."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: value")

        # Should succeed with custom retry interval
        with _file_lock(file_path, retry_interval_seconds=0.05):
            pass

    def test_backup_with_yml_extension(self, tmp_path: Path) -> None:
        """Backup should work correctly with .yml extension."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yml"
        file_path.write_text(yaml_content)
        backup_path = file_path.with_suffix(".yml.bak")

        writer = OrchestrationYamlWriter(create_backups=True)
        writer.toggle_orchestration(file_path, "test-orch", False)

        assert backup_path.exists()
        assert "enabled: true" in backup_path.read_text()
        assert "enabled: false" in file_path.read_text()

    def test_timestamp_backup_with_yml_extension(self, tmp_path: Path) -> None:
        """Timestamped backup should have consistent naming for .yml files."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "config.yml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter(create_backups=True, backup_suffix="timestamp")
        writer.toggle_orchestration(file_path, "test-orch", False)

        # Find timestamped backup - should be config.TIMESTAMP.bak
        backup_files = list(tmp_path.glob("config.*.bak"))
        assert len(backup_files) == 1
        assert "enabled: true" in backup_files[0].read_text()

    def test_cleanup_orphaned_yml_lock_files(self, tmp_path: Path) -> None:
        """Should clean up orphaned .yml.lock files."""
        # Create old .yml.lock file
        old_yml_lock = tmp_path / "old.yml.lock"
        old_yml_lock.write_text("")
        old_time = time.time() - 7200
        os.utime(old_yml_lock, (old_time, old_time))

        # Create old .yaml.lock file
        old_yaml_lock = tmp_path / "old.yaml.lock"
        old_yaml_lock.write_text("")
        os.utime(old_yaml_lock, (old_time, old_time))

        # Create new lock files (should not be removed)
        new_yml_lock = tmp_path / "new.yml.lock"
        new_yml_lock.write_text("")
        new_yaml_lock = tmp_path / "new.yaml.lock"
        new_yaml_lock.write_text("")

        # Clean up files older than 1 hour
        removed = cleanup_orphaned_lock_files(tmp_path, max_age_seconds=3600)

        assert removed == 2
        assert not old_yml_lock.exists()
        assert not old_yaml_lock.exists()
        assert new_yml_lock.exists()
        assert new_yaml_lock.exists()

    def test_writer_configuration_with_retry_interval(self) -> None:
        """Writer should store retry interval configuration."""
        writer = OrchestrationYamlWriter(retry_interval_seconds=0.5)
        assert writer._retry_interval_seconds == 0.5

    def test_default_retry_interval_is_none(self) -> None:
        """Writer should default retry interval to None (uses constant)."""
        writer = OrchestrationYamlWriter()
        assert writer._retry_interval_seconds is None


class TestLoggingEnhancements:
    """Tests for logging enhancements from code review."""

    def test_toggle_by_project_logs_debug_at_start(self, tmp_path: Path, caplog) -> None:
        """toggle_by_project should log debug message at start."""
        import logging

        caplog.set_level(logging.DEBUG, logger="sentinel.yaml_writer")

        yaml_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "PROJ"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        writer.toggle_by_project({"test": file_path}, "PROJ", False)

        # Check that debug log was emitted
        assert any(
            "toggle_by_project called" in record.message
            and "PROJ" in record.message
            and "enabled=False" in record.message
            for record in caplog.records
            if record.levelno == logging.DEBUG
        )

    def test_toggle_by_project_warns_on_unknown_source_type(self, tmp_path: Path, caplog) -> None:
        """toggle_by_project should log warning for unrecognized source types."""
        import logging

        caplog.set_level(logging.WARNING, logger="sentinel.yaml_writer")

        yaml_content = """
orchestrations:
  - name: "unknown-source-orch"
    enabled: true
    trigger:
      source: gitlab
      project: "PROJ"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_by_project({"test": file_path}, "PROJ", False)

        # The orchestration should not be toggled since source type is unknown
        assert count == 0

        # Check that warning log was emitted for unrecognized source type
        assert any(
            "Unrecognized source type" in record.message
            and "gitlab" in record.message
            and "unknown-source-orch" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_toggle_by_project_no_warning_for_known_source_types(
        self, tmp_path: Path, caplog
    ) -> None:
        """toggle_by_project should not warn for jira and github source types."""
        import logging

        caplog.set_level(logging.WARNING, logger="sentinel.yaml_writer")

        yaml_content = """
orchestrations:
  - name: "jira-orch"
    enabled: true
    trigger:
      source: jira
      project: "PROJ"
    agent:
      prompt: "Jira test"
  - name: "github-orch"
    enabled: true
    trigger:
      source: github
      project_owner: "my-org"
    agent:
      prompt: "GitHub test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        writer.toggle_by_project({"test": file_path}, "PROJ", False)

        # No warnings about unrecognized source types should be emitted
        assert not any(
            "Unrecognized source type" in record.message
            for record in caplog.records
        )

    def test_toggle_by_project_warns_unnamed_orchestration(self, tmp_path: Path, caplog) -> None:
        """toggle_by_project should handle unnamed orchestrations in warning log."""
        import logging

        caplog.set_level(logging.WARNING, logger="sentinel.yaml_writer")

        yaml_content = """
orchestrations:
  - enabled: true
    trigger:
      source: bitbucket
      project: "PROJ"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        writer = OrchestrationYamlWriter()
        count = writer.toggle_by_project({"test": file_path}, "PROJ", False)

        assert count == 0

        # Should still log warning with <unnamed> placeholder
        assert any(
            "Unrecognized source type" in record.message
            and "bitbucket" in record.message
            and "<unnamed>" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        )
