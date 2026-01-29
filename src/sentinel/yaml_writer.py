"""YAML writer module for safe orchestration file modification.

This module provides functionality to safely modify orchestration YAML files
while preserving formatting, comments, and structure using ruamel.yaml's
round-trip editing capabilities.
"""

from __future__ import annotations

import fcntl
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.comments import CommentedMap

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from ruamel.yaml.comments import CommentedSeq

logger = get_logger(__name__)

# Default timeout for file lock acquisition (in seconds)
DEFAULT_LOCK_TIMEOUT_SECONDS: float = 30.0

# Default retry interval for lock acquisition (in seconds)
DEFAULT_LOCK_RETRY_INTERVAL: float = 0.1


class OrchestrationYamlWriterError(Exception):
    """Raised when orchestration YAML modification fails."""

    pass


class FileLockTimeoutError(OrchestrationYamlWriterError):
    """Raised when file lock acquisition times out."""

    pass


@contextmanager
def _file_lock(
    file_path: Path,
    max_wait_seconds: float | None = None,
    cleanup_lock_file: bool = False,
    retry_interval_seconds: float | None = None,
) -> Iterator[None]:
    """Context manager for file locking to prevent concurrent modifications.

    Uses flock for advisory locking with optional timeout. The lock is released
    when the context exits. Optionally removes the lock file after release.

    Args:
        file_path: Path to the file to lock.
        max_wait_seconds: Maximum time to wait for lock acquisition in seconds.
            If None, uses DEFAULT_LOCK_TIMEOUT_SECONDS. If 0, blocks indefinitely
            (original behavior).
        cleanup_lock_file: If True, removes the .lock file after releasing the lock.
            Defaults to False for backward compatibility.
        retry_interval_seconds: Time to wait between lock acquisition attempts in
            seconds. If None, uses DEFAULT_LOCK_RETRY_INTERVAL (0.1s). Useful for
            environments with different I/O characteristics.

    Yields:
        None

    Raises:
        FileLockTimeoutError: If the lock cannot be acquired within the timeout.
        OrchestrationYamlWriterError: If the file cannot be locked due to other errors.
    """
    if max_wait_seconds is None:
        max_wait_seconds = DEFAULT_LOCK_TIMEOUT_SECONDS

    if retry_interval_seconds is None:
        retry_interval_seconds = DEFAULT_LOCK_RETRY_INTERVAL

    lock_path = file_path.with_suffix(file_path.suffix + ".lock")
    lock_file = None
    start_time = time.monotonic()

    try:
        lock_file = open(lock_path, "w")  # noqa: SIM115

        if max_wait_seconds == 0:
            # Block indefinitely (original behavior)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        else:
            # Non-blocking with retry loop
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break  # Lock acquired successfully
                except BlockingIOError:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= max_wait_seconds:
                        lock_file.close()
                        lock_file = None
                        raise FileLockTimeoutError(
                            f"Timed out waiting for lock on {file_path} "
                            f"after {max_wait_seconds:.1f} seconds"
                        ) from None
                    # Wait before retrying
                    time.sleep(retry_interval_seconds)
        yield
    except (FileLockTimeoutError, OrchestrationYamlWriterError):
        raise
    except OSError as e:
        if lock_file is not None:
            lock_file.close()
            lock_file = None
        raise OrchestrationYamlWriterError(
            f"Failed to acquire lock for {file_path}: {e}"
        ) from e
    finally:
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            if cleanup_lock_file:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    # Best effort cleanup - don't fail if we can't remove the lock file
                    logger.debug("Could not remove lock file: %s", lock_path)


def cleanup_orphaned_lock_files(directory: Path, max_age_seconds: float = 3600) -> int:
    """Remove orphaned lock files older than the specified age.

    This utility function can be used to clean up .yaml.lock and .yml.lock files
    that may have been left behind due to crashes or other unexpected terminations.

    Args:
        directory: Directory to search for orphaned lock files.
        max_age_seconds: Maximum age of lock files to keep (in seconds).
            Files older than this will be removed. Defaults to 1 hour.

    Returns:
        The number of lock files that were removed.
    """
    removed_count = 0
    current_time = time.time()

    # Handle both .yaml.lock and .yml.lock files
    lock_patterns = ["**/*.yaml.lock", "**/*.yml.lock"]

    try:
        for pattern in lock_patterns:
            for lock_file in directory.glob(pattern):
                try:
                    file_age = current_time - lock_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        lock_file.unlink()
                        removed_count += 1
                        logger.info("Removed orphaned lock file: %s", lock_file)
                except OSError as e:
                    logger.debug("Could not process lock file %s: %s", lock_file, e)
    except OSError as e:
        logger.warning("Error scanning for orphaned lock files in %s: %s", directory, e)

    return removed_count


class OrchestrationYamlWriter:
    """Writer for safely modifying orchestration YAML files.

    Uses ruamel.yaml with preserve_quotes=True for round-trip editing,
    maintaining formatting, comments, and structure of the original file.

    Args:
        lock_timeout_seconds: Maximum time to wait for file lock acquisition.
            Defaults to DEFAULT_LOCK_TIMEOUT_SECONDS (30 seconds).
            Set to 0 to block indefinitely.
        retry_interval_seconds: Time to wait between lock acquisition attempts.
            Defaults to DEFAULT_LOCK_RETRY_INTERVAL (0.1s). Useful for
            environments with different I/O characteristics.
        cleanup_lock_files: If True, removes lock files after releasing locks.
            Defaults to False.
        create_backups: If True, creates backup files before modifications.
            Defaults to False.
        backup_suffix: Suffix for backup files. Defaults to ".bak".
            Can also be "timestamp" to use timestamped backups.

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

        # With backup and lock cleanup enabled
        writer = OrchestrationYamlWriter(
            create_backups=True,
            backup_suffix="timestamp",
            cleanup_lock_files=True,
            lock_timeout_seconds=10.0,
            retry_interval_seconds=0.2
        )
    """

    def __init__(
        self,
        lock_timeout_seconds: float | None = None,
        retry_interval_seconds: float | None = None,
        cleanup_lock_files: bool = False,
        create_backups: bool = False,
        backup_suffix: str = ".bak",
    ) -> None:
        """Initialize the YAML writer with round-trip configuration."""
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        # Preserve the original formatting as much as possible
        self._yaml.default_flow_style = False
        self._yaml.width = 4096  # Prevent line wrapping

        # Lock configuration
        self._lock_timeout_seconds = lock_timeout_seconds
        self._retry_interval_seconds = retry_interval_seconds
        self._cleanup_lock_files = cleanup_lock_files

        # Backup configuration
        self._create_backups = create_backups
        self._backup_suffix = backup_suffix

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
        except YAMLError as e:
            raise OrchestrationYamlWriterError(
                f"Failed to parse YAML in {file_path}: {e}"
            ) from e
        except UnicodeDecodeError as e:
            raise OrchestrationYamlWriterError(
                f"Failed to decode {file_path} as UTF-8: {e}"
            ) from e

    def _create_backup(self, file_path: Path) -> Path | None:
        """Create a backup of the file before modification.

        Args:
            file_path: Path to the file to backup.

        Returns:
            Path to the backup file, or None if backup creation failed.

        Raises:
            OrchestrationYamlWriterError: If backup creation fails critically.
        """
        if not self._create_backups:
            return None

        if not file_path.exists():
            return None

        try:
            if self._backup_suffix == "timestamp":
                # Use UTC timezone for consistent timestamps across environments
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                # Handle both .yaml and .yml extensions consistently
                base_name = file_path.stem
                if file_path.suffix.lower() in (".yaml", ".yml"):
                    backup_path = file_path.parent / f"{base_name}.{timestamp}.bak"
                else:
                    backup_path = file_path.parent / f"{file_path.name}.{timestamp}.bak"
            else:
                backup_path = file_path.with_suffix(file_path.suffix + self._backup_suffix)

            shutil.copy2(file_path, backup_path)
            logger.info("Created backup: %s", backup_path)
            return backup_path
        except OSError as e:
            logger.warning("Failed to create backup of %s: %s", file_path, e)
            raise OrchestrationYamlWriterError(
                f"Failed to create backup of {file_path}: {e}"
            ) from e

    def _save_yaml(self, file_path: Path, data: CommentedMap) -> None:
        """Save YAML data to a file with round-trip preservation.

        Creates a backup of the original file if backups are enabled.

        Args:
            file_path: Path to the YAML file.
            data: The YAML data to save.

        Raises:
            OrchestrationYamlWriterError: If the file cannot be written or
                backup creation fails.
        """
        # Create backup before modification
        self._create_backup(file_path)

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
            FileLockTimeoutError: If the file lock cannot be acquired within
                the configured timeout.
        """
        with _file_lock(
            file_path,
            max_wait_seconds=self._lock_timeout_seconds,
            cleanup_lock_file=self._cleanup_lock_files,
            retry_interval_seconds=self._retry_interval_seconds,
        ):
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
            FileLockTimeoutError: If the file lock cannot be acquired within
                the configured timeout.
        """
        with _file_lock(
            file_path,
            max_wait_seconds=self._lock_timeout_seconds,
            cleanup_lock_file=self._cleanup_lock_files,
            retry_interval_seconds=self._retry_interval_seconds,
        ):
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
            FileLockTimeoutError: If any file lock cannot be acquired within
                the configured timeout.
        """
        logger.debug(
            "toggle_by_project called: project='%s', enabled=%s, files=%d",
            project,
            enabled,
            len(orch_files),
        )
        total_count = 0
        # Get unique file paths
        unique_files = set(orch_files.values())

        for file_path in unique_files:
            with _file_lock(
                file_path,
                max_wait_seconds=self._lock_timeout_seconds,
                cleanup_lock_file=self._cleanup_lock_files,
                retry_interval_seconds=self._retry_interval_seconds,
            ):
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
            FileLockTimeoutError: If any file lock cannot be acquired within
                the configured timeout.
        """
        logger.debug(
            "toggle_by_repo called: repo='%s', enabled=%s, files=%d",
            repo,
            enabled,
            len(orch_files),
        )
        total_count = 0
        # Get unique file paths
        unique_files = set(orch_files.values())

        for file_path in unique_files:
            with _file_lock(
                file_path,
                max_wait_seconds=self._lock_timeout_seconds,
                cleanup_lock_file=self._cleanup_lock_files,
                retry_interval_seconds=self._retry_interval_seconds,
            ):
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
