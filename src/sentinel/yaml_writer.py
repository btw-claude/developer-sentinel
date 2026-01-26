"""YAML writer module for safe orchestration file modification.

This module provides functionality to safely modify orchestration YAML files
while preserving formatting, comments, and structure using ruamel.yaml's
round-trip editing capabilities.

DS-248: Create YAML writer module for safe orchestration file modification
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from ruamel.yaml.comments import CommentedSeq

logger = get_logger(__name__)


class OrchestrationYamlWriterError(Exception):
    """Raised when orchestration YAML modification fails."""

    pass


@contextmanager
def _file_lock(file_path: Path) -> Iterator[None]:
    """Context manager for file locking to prevent concurrent modifications.

    Uses flock for advisory locking. The lock is released when the context exits.

    Args:
        file_path: Path to the file to lock.

    Yields:
        None

    Raises:
        OrchestrationYamlWriterError: If the file cannot be locked.
    """
    lock_path = file_path.with_suffix(file_path.suffix + ".lock")
    lock_file = None
    try:
        lock_file = open(lock_path, "w")  # noqa: SIM115
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    except OSError as e:
        if lock_file is not None:
            lock_file.close()
        raise OrchestrationYamlWriterError(
            f"Failed to acquire lock for {file_path}: {e}"
        ) from e
    finally:
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


class OrchestrationYamlWriter:
    """Writer for safely modifying orchestration YAML files.

    Uses ruamel.yaml with preserve_quotes=True for round-trip editing,
    maintaining formatting, comments, and structure of the original file.

    Example usage:
        writer = OrchestrationYamlWriter()

        # Toggle a single orchestration
        success = writer.toggle_orchestration(
            file_path=Path("orchestrations/review.yaml"),
            orch_name="code-review",
            enabled=False
        )

        # Toggle all orchestrations in a file
        count = writer.toggle_all_in_file(
            file_path=Path("orchestrations/review.yaml"),
            enabled=True
        )

        # Toggle orchestrations by project
        count = writer.toggle_by_project(
            orch_files={"review": Path("orchestrations/review.yaml")},
            project="PROJ",
            enabled=False
        )
    """

    def __init__(self) -> None:
        """Initialize the YAML writer with round-trip configuration."""
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        # Preserve the original formatting as much as possible
        self._yaml.default_flow_style = False
        self._yaml.width = 4096  # Prevent line wrapping

    def _load_yaml(self, file_path: Path) -> CommentedMap:
        """Load a YAML file with round-trip preservation.

        Args:
            file_path: Path to the YAML file.

        Returns:
            The parsed YAML data as a CommentedMap.

        Raises:
            OrchestrationYamlWriterError: If the file cannot be read or parsed.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                data = self._yaml.load(f)
                if data is None:
                    return CommentedMap()
                # ruamel.yaml.load returns Any, but for our orchestration files
                # we expect a CommentedMap
                return CommentedMap(data) if not isinstance(data, CommentedMap) else data
        except FileNotFoundError:
            raise OrchestrationYamlWriterError(
                f"Orchestration file not found: {file_path}"
            ) from None
        except PermissionError as e:
            raise OrchestrationYamlWriterError(
                f"Permission denied reading {file_path}: {e}"
            ) from e
        except Exception as e:
            raise OrchestrationYamlWriterError(
                f"Failed to parse YAML in {file_path}: {e}"
            ) from e

    def _save_yaml(self, file_path: Path, data: CommentedMap) -> None:
        """Save YAML data to a file with round-trip preservation.

        Args:
            file_path: Path to the YAML file.
            data: The YAML data to save.

        Raises:
            OrchestrationYamlWriterError: If the file cannot be written.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                self._yaml.dump(data, f)
        except PermissionError as e:
            raise OrchestrationYamlWriterError(
                f"Permission denied writing to {file_path}: {e}"
            ) from e
        except OSError as e:
            raise OrchestrationYamlWriterError(
                f"Failed to write to {file_path}: {e}"
            ) from e

    def _find_orchestration_index(
        self, orchestrations: CommentedSeq, orch_name: str
    ) -> int | None:
        """Find the index of an orchestration by name.

        Args:
            orchestrations: The list of orchestration configurations.
            orch_name: The name of the orchestration to find.

        Returns:
            The index of the orchestration, or None if not found.
        """
        for i, orch in enumerate(orchestrations):
            if isinstance(orch, dict) and orch.get("name") == orch_name:
                return i
        return None

    def toggle_orchestration(
        self, file_path: Path, orch_name: str, enabled: bool
    ) -> bool:
        """Toggle the enabled status of a specific orchestration.

        Modifies the orchestration's `enabled` field in place, preserving
        all other formatting and comments in the file.

        Args:
            file_path: Path to the orchestration YAML file.
            orch_name: Name of the orchestration to toggle.
            enabled: The new enabled status (True or False).

        Returns:
            True if the orchestration was found and updated, False if not found.

        Raises:
            OrchestrationYamlWriterError: If there's an error reading, parsing,
                or writing the file.
        """
        with _file_lock(file_path):
            data = self._load_yaml(file_path)

            orchestrations = data.get("orchestrations")
            if orchestrations is None:
                logger.warning(
                    "No orchestrations key found in %s",
                    file_path,
                )
                return False

            idx = self._find_orchestration_index(orchestrations, orch_name)
            if idx is None:
                logger.warning(
                    "Orchestration '%s' not found in %s",
                    orch_name,
                    file_path,
                )
                return False

            orchestrations[idx]["enabled"] = enabled
            self._save_yaml(file_path, data)

            logger.info(
                "Set orchestration '%s' enabled=%s in %s",
                orch_name,
                enabled,
                file_path,
            )
            return True

    def toggle_all_in_file(self, file_path: Path, enabled: bool) -> int:
        """Toggle the enabled status of all orchestrations in a file.

        Sets the `enabled` field for every orchestration in the file,
        preserving all other formatting and comments.

        Args:
            file_path: Path to the orchestration YAML file.
            enabled: The new enabled status (True or False).

        Returns:
            The number of orchestrations that were updated.

        Raises:
            OrchestrationYamlWriterError: If there's an error reading, parsing,
                or writing the file.
        """
        with _file_lock(file_path):
            data = self._load_yaml(file_path)

            orchestrations = data.get("orchestrations")
            if orchestrations is None:
                logger.warning(
                    "No orchestrations key found in %s",
                    file_path,
                )
                return 0

            count = 0
            for orch in orchestrations:
                if isinstance(orch, dict) and "name" in orch:
                    orch["enabled"] = enabled
                    count += 1

            if count > 0:
                self._save_yaml(file_path, data)
                logger.info(
                    "Set enabled=%s for %d orchestration(s) in %s",
                    enabled,
                    count,
                    file_path,
                )

            return count

    def toggle_by_project(
        self, orch_files: dict[str, Path], project: str, enabled: bool
    ) -> int:
        """Toggle orchestrations that match a specific project.

        Searches through the provided orchestration files and toggles the
        `enabled` status for any orchestration whose trigger.project matches
        the specified project.

        Args:
            orch_files: A mapping of orchestration names to their file paths.
                The keys are not used for matching; all files are searched.
            project: The Jira project key to match against trigger.project.
            enabled: The new enabled status (True or False).

        Returns:
            The total number of orchestrations that were updated across all files.

        Raises:
            OrchestrationYamlWriterError: If there's an error reading, parsing,
                or writing any file.
        """
        total_count = 0
        # Get unique file paths
        unique_files = set(orch_files.values())

        for file_path in unique_files:
            with _file_lock(file_path):
                data = self._load_yaml(file_path)

                orchestrations = data.get("orchestrations")
                if orchestrations is None:
                    continue

                file_count = 0
                for orch in orchestrations:
                    if not isinstance(orch, dict):
                        continue

                    trigger = orch.get("trigger")
                    if not isinstance(trigger, dict):
                        continue

                    if trigger.get("project") == project:
                        orch["enabled"] = enabled
                        file_count += 1

                if file_count > 0:
                    self._save_yaml(file_path, data)
                    logger.info(
                        "Set enabled=%s for %d orchestration(s) with project '%s' in %s",
                        enabled,
                        file_count,
                        project,
                        file_path,
                    )

                total_count += file_count

        if total_count > 0:
            logger.info(
                "Total: set enabled=%s for %d orchestration(s) matching project '%s'",
                enabled,
                total_count,
                project,
            )
        else:
            logger.warning(
                "No orchestrations found matching project '%s'",
                project,
            )

        return total_count

    def toggle_by_repo(
        self, orch_files: dict[str, Path], repo: str, enabled: bool
    ) -> int:
        """Toggle orchestrations that match a specific GitHub repository.

        Searches through the provided orchestration files and toggles the
        `enabled` status for any orchestration whose trigger.repo matches
        the specified repository.

        Args:
            orch_files: A mapping of orchestration names to their file paths.
                The keys are not used for matching; all files are searched.
            repo: The GitHub repository identifier to match against trigger.repo
                (e.g., "org/repo-name").
            enabled: The new enabled status (True or False).

        Returns:
            The total number of orchestrations that were updated across all files.

        Raises:
            OrchestrationYamlWriterError: If there's an error reading, parsing,
                or writing any file.
        """
        total_count = 0
        # Get unique file paths
        unique_files = set(orch_files.values())

        for file_path in unique_files:
            with _file_lock(file_path):
                data = self._load_yaml(file_path)

                orchestrations = data.get("orchestrations")
                if orchestrations is None:
                    continue

                file_count = 0
                for orch in orchestrations:
                    if not isinstance(orch, dict):
                        continue

                    trigger = orch.get("trigger")
                    if not isinstance(trigger, dict):
                        continue

                    if trigger.get("repo") == repo:
                        orch["enabled"] = enabled
                        file_count += 1

                if file_count > 0:
                    self._save_yaml(file_path, data)
                    logger.info(
                        "Set enabled=%s for %d orchestration(s) with repo '%s' in %s",
                        enabled,
                        file_count,
                        repo,
                        file_path,
                    )

                total_count += file_count

        if total_count > 0:
            logger.info(
                "Total: set enabled=%s for %d orchestration(s) matching repo '%s'",
                enabled,
                total_count,
                repo,
            )
        else:
            logger.warning(
                "No orchestrations found matching repo '%s'",
                repo,
            )

        return total_count
