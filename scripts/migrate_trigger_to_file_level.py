#!/usr/bin/env python3
"""Migrate orchestration YAML files from per-step to file-level trigger format.

This script converts orchestration YAML files from the old format where
project-level trigger fields (source, project, project_number, project_scope,
project_owner) are repeated in every step, to the new format where these
fields are defined once at file level and the key is renamed from
"orchestrations" to "steps".

Usage:
    python scripts/migrate_trigger_to_file_level.py <orchestrations_dir>

Example:
    python scripts/migrate_trigger_to_file_level.py ./orchestrations
    python scripts/migrate_trigger_to_file_level.py ./examples/basic-setup/orchestrations

The script:
- Recursively scans the directory for .yaml and .yml files, skipping symlinks
  to avoid potential hangs from circular symlinks.  On Python 3.13+ this uses
  ``rglob(recurse_symlinks=False)`` for traversal-level filtering; on older
  Pythons it falls back to post-hoc ``is_symlink()`` checks (DS-900, DS-901)
- Extracts project-level fields from the first step
- Validates all steps share the same project-level values
- Creates a file-level trigger block
- Removes project-level fields from each step's trigger
- Renames "orchestrations" key to "steps"
- Preserves comments and formatting via ruamel.yaml round-trip
- Creates .bak backups before modifying files
- Skips files already in new format (have "steps" key)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ruamel.yaml import YAML

# Python 3.13+ added a ``recurse_symlinks`` parameter to ``Path.rglob()``,
# allowing symlink filtering at the directory-traversal level rather than
# post-hoc filtering via ``p.is_symlink()``.  With ``recurse_symlinks=False``
# (the default in 3.13+), ``rglob()`` will not follow symlinks when expanding
# ``**`` wildcards, preventing the walker from descending into symlink cycles.
# We detect support once at import time so that ``_rglob_no_symlinks()`` can
# choose the most robust strategy available.  See DS-901 item 2.
_RGLOB_SUPPORTS_RECURSE_SYMLINKS: bool = sys.version_info >= (3, 13)

# Fields that move from step-level trigger to file-level trigger
FILE_LEVEL_FIELDS = frozenset({
    "source", "project", "project_number", "project_scope", "project_owner",
})


def _rglob_no_symlinks(directory: Path, pattern: str) -> set[Path]:
    """Recursively glob for *pattern* under *directory*, excluding symlinks.

    On Python 3.13+ this uses ``recurse_symlinks=False`` (the default) to
    prevent ``**`` expansion from descending into symlinked directories,
    which avoids hangs from circular directory symlinks at the traversal
    level (DS-901 item 2).  Symlinked *files* that match the pattern are
    still returned by ``rglob()``, so we apply a post-hoc
    ``p.is_symlink()`` filter in all cases.

    On older Pythons the ``recurse_symlinks`` parameter is not available,
    so we rely solely on the post-hoc filter (DS-900).
    """
    if _RGLOB_SUPPORTS_RECURSE_SYMLINKS:
        # Python 3.13+: prevent traversal into symlinked directories, then
        # filter out any remaining symlinked files.
        return {
            p for p in directory.rglob(pattern, recurse_symlinks=False)
            if not p.is_symlink()
        }
    # Python <3.13: post-hoc filter only
    return {p for p in directory.rglob(pattern) if not p.is_symlink()}


def migrate_file(file_path: Path, *, dry_run: bool = False) -> bool:
    """Migrate a single orchestration YAML file to the new format.

    Args:
        file_path: Path to the YAML file.
        dry_run: If True, only report what would change without modifying.

    Returns:
        True if the file was migrated (or would be in dry-run), False if skipped.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    yaml.width = 4096

    with open(file_path, encoding="utf-8") as f:
        data = yaml.load(f)

    if data is None:
        print(f"  SKIP {file_path}: empty file")
        return False

    # Skip if already in new format
    if "steps" in data:
        print(f"  SKIP {file_path}: already uses 'steps' key")
        return False

    # Skip if no orchestrations key
    if "orchestrations" not in data:
        print(f"  SKIP {file_path}: no 'orchestrations' key found")
        return False

    orchestrations = data["orchestrations"]
    if not isinstance(orchestrations, list) or not orchestrations:
        print(f"  SKIP {file_path}: empty or invalid orchestrations list")
        return False

    # Extract file-level fields from the first step
    first_trigger = None
    for step in orchestrations:
        if isinstance(step, dict) and "trigger" in step:
            first_trigger = step["trigger"]
            break

    if not first_trigger or not isinstance(first_trigger, dict):
        print(f"  SKIP {file_path}: no valid trigger found in first step")
        return False

    # Collect file-level fields from first step
    file_trigger = {}
    for field_name in FILE_LEVEL_FIELDS:
        if field_name in first_trigger:
            file_trigger[field_name] = first_trigger[field_name]

    if not file_trigger:
        print(f"  SKIP {file_path}: no file-level trigger fields found")
        return False

    # Validate all steps share the same file-level values
    for step in orchestrations:
        if not isinstance(step, dict) or "trigger" not in step:
            continue
        step_trigger = step["trigger"]
        if not isinstance(step_trigger, dict):
            continue
        for field_name, expected_value in file_trigger.items():
            actual_value = step_trigger.get(field_name)
            if actual_value is not None and actual_value != expected_value:
                print(
                    f"  ERROR {file_path}: step '{step.get('name', '<unnamed>')}' "
                    f"has different {field_name}={actual_value!r} "
                    f"(expected {expected_value!r}). "
                    f"Cannot auto-migrate files with different project-level values."
                )
                return False

    if dry_run:
        print(f"  WOULD MIGRATE {file_path}")
        print(f"    File-level trigger: {dict(file_trigger)}")
        return True

    # Create backup
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copy2(file_path, backup_path)
    print(f"  Backup: {backup_path}")

    # Add file-level trigger block
    from ruamel.yaml.comments import CommentedMap

    # Build new data with trigger at the right position
    new_data = CommentedMap()

    # Copy file-level 'enabled' if present
    if "enabled" in data:
        new_data["enabled"] = data["enabled"]

    # Add file-level trigger
    trigger_map = CommentedMap()
    for field_name in ("source", "project", "project_number", "project_scope", "project_owner"):
        if field_name in file_trigger:
            trigger_map[field_name] = file_trigger[field_name]
    new_data["trigger"] = trigger_map

    # Remove file-level fields from each step's trigger
    for step in orchestrations:
        if not isinstance(step, dict) or "trigger" not in step:
            continue
        step_trigger = step["trigger"]
        if not isinstance(step_trigger, dict):
            continue
        for field_name in FILE_LEVEL_FIELDS:
            if field_name in step_trigger:
                del step_trigger[field_name]
        # Remove empty trigger
        if not step_trigger:
            del step["trigger"]

    # Rename orchestrations to steps
    new_data["steps"] = orchestrations

    # Copy any other top-level keys
    for key in data:
        if key not in ("enabled", "orchestrations", "trigger"):
            new_data[key] = data[key]

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(new_data, f)

    print(f"  MIGRATED {file_path}")
    return True


def main() -> None:
    """Run the migration script."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_trigger_to_file_level.py <orchestrations_dir> [--dry-run]")
        sys.exit(1)

    directory = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        sys.exit(1)

    print(f"Scanning {directory} for orchestration YAML files...")
    if dry_run:
        print("(DRY RUN - no files will be modified)")
    print()

    migrated = 0
    skipped = 0
    errors = 0

    # Use rglob for recursive scanning so nested subdirectories are included.
    # On Python 3.13+ symlinks are excluded at traversal time via
    # recurse_symlinks=False; on older versions they are filtered post-hoc
    # (DS-900, DS-901).
    yaml_files = sorted(
        _rglob_no_symlinks(directory, "*.yaml")
        | _rglob_no_symlinks(directory, "*.yml")
    )
    for file_path in yaml_files:
        if file_path.name.endswith(".bak"):
            continue
        try:
            if migrate_file(file_path, dry_run=dry_run):
                migrated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR {file_path}: {e}")
            errors += 1

    print()
    print(f"Results: {migrated} migrated, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
