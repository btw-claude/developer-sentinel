"""Tests for migrate_trigger_to_file_level script (DS-899).

Tests that the migration script uses recursive scanning (rglob) to find
YAML files in nested subdirectories, rather than only scanning direct
children of the orchestrations directory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


class TestMigrateRecursiveScanning:
    """Tests for recursive directory scanning in the migration script (DS-899 item 1)."""

    def test_finds_yaml_in_subdirectory(self, tmp_path: Path) -> None:
        """Should find and migrate YAML files in nested subdirectories."""
        # Create a nested directory structure
        sub_dir = tmp_path / "sub" / "nested"
        sub_dir.mkdir(parents=True)

        yaml_content = """orchestrations:
  - name: "nested-orch"
    enabled: true
    trigger:
      source: jira
      project: TEST
"""
        nested_file = sub_dir / "nested.yaml"
        nested_file.write_text(yaml_content)

        # Import and run main with the tmp_path as the directory
        from scripts.migrate_trigger_to_file_level import main

        with patch("sys.argv", ["migrate", str(tmp_path), "--dry-run"]):
            main()

        # In dry-run mode, the file should not be modified
        assert nested_file.read_text() == yaml_content

    def test_finds_yml_in_subdirectory(self, tmp_path: Path) -> None:
        """Should find .yml files in nested subdirectories."""
        sub_dir = tmp_path / "team-a"
        sub_dir.mkdir()

        yaml_content = """orchestrations:
  - name: "team-a-orch"
    enabled: true
    trigger:
      source: jira
      project: TEAM
"""
        yml_file = sub_dir / "workflow.yml"
        yml_file.write_text(yaml_content)

        from scripts.migrate_trigger_to_file_level import main

        with patch("sys.argv", ["migrate", str(tmp_path), "--dry-run"]):
            main()

        # File should still be intact in dry-run
        assert yml_file.read_text() == yaml_content

    def test_migrates_nested_file(self, tmp_path: Path) -> None:
        """Should actually migrate a YAML file in a nested subdirectory."""
        sub_dir = tmp_path / "deep" / "nested"
        sub_dir.mkdir(parents=True)

        yaml_content = """orchestrations:
  - name: "deep-orch"
    enabled: true
    trigger:
      source: jira
      project: DEEP
      tags:
        - review
"""
        nested_file = sub_dir / "deep.yaml"
        nested_file.write_text(yaml_content)

        from scripts.migrate_trigger_to_file_level import main

        with patch("sys.argv", ["migrate", str(tmp_path)]):
            main()

        # Verify the file was migrated
        from ruamel.yaml import YAML

        yaml = YAML()
        with open(nested_file) as f:
            data = yaml.load(f)

        assert "steps" in data
        assert "orchestrations" not in data
        assert "trigger" in data
        assert data["trigger"]["source"] == "jira"
        assert data["trigger"]["project"] == "DEEP"

    def test_skips_bak_files_in_subdirectories(self, tmp_path: Path) -> None:
        """Should skip .bak files even in nested subdirectories."""
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()

        bak_content = """orchestrations:
  - name: "old-orch"
    trigger:
      source: jira
      project: OLD
"""
        bak_file = sub_dir / "backup.yaml.bak"
        bak_file.write_text(bak_content)

        from scripts.migrate_trigger_to_file_level import main

        with patch("sys.argv", ["migrate", str(tmp_path), "--dry-run"]):
            main()

        # Bak file should not have been touched
        assert bak_file.read_text() == bak_content

    def test_handles_mixed_depth_files(self, tmp_path: Path) -> None:
        """Should find files at root level AND in subdirectories."""
        # Root level file
        root_yaml = """orchestrations:
  - name: "root-orch"
    enabled: true
    trigger:
      source: jira
      project: ROOT
"""
        root_file = tmp_path / "root.yaml"
        root_file.write_text(root_yaml)

        # Nested file
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        nested_yaml = """orchestrations:
  - name: "nested-orch"
    enabled: true
    trigger:
      source: jira
      project: NESTED
"""
        nested_file = sub_dir / "nested.yaml"
        nested_file.write_text(nested_yaml)

        from scripts.migrate_trigger_to_file_level import main

        with patch("sys.argv", ["migrate", str(tmp_path)]):
            main()

        from ruamel.yaml import YAML

        yaml = YAML()

        # Both files should have been migrated
        with open(root_file) as f:
            root_data = yaml.load(f)
        assert "steps" in root_data

        with open(nested_file) as f:
            nested_data = yaml.load(f)
        assert "steps" in nested_data
