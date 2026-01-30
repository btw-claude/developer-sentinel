"""Orchestration registry for hot-reload and version tracking.

This module provides the OrchestrationRegistry class which manages:
- Loading orchestrations from files
- Hot-reload detection for new, modified, and deleted files
- Version tracking for active executions
- Pending removal of old versions

This is part of the Sentinel refactoring to split the God Object into focused,
composable components (DS-384).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from sentinel.logging import get_logger
from sentinel.orchestration import (
    Orchestration,
    OrchestrationError,
    OrchestrationVersion,
    load_orchestration_file,
)

if TYPE_CHECKING:
    from sentinel.router import Router

logger = get_logger(__name__)


class OrchestrationRegistry:
    """Manages orchestration loading, hot-reload, and version tracking.

    This class is responsible for:
    - Tracking known orchestration files and their modification times
    - Detecting new, modified, and deleted orchestration files
    - Loading and unloading orchestrations with version tracking
    - Managing pending removal versions until their executions complete
    - Providing observability metrics for hot-reload operations

    Thread Safety:
        All public methods that modify shared state use internal locks.
    """

    def __init__(
        self,
        orchestrations_dir: Path,
        router_factory: Callable[[list[Orchestration]], Router],
    ) -> None:
        """Initialize the orchestration registry.

        Args:
            orchestrations_dir: Path to the directory containing orchestration files.
            router_factory: Factory function to create a Router from orchestrations.
        """
        self._orchestrations_dir = orchestrations_dir
        self._router_factory = router_factory

        # List of active orchestrations
        self._orchestrations: list[Orchestration] = []

        # The router for routing issues to orchestrations
        self._router: Router | None = None

        # Track known orchestration files for hot-reload detection
        # Maps file path to its last known mtime
        self._known_orchestration_files: dict[Path, float] = {}

        # Versioned orchestrations for hot-reload support
        self._active_versions: list[OrchestrationVersion] = []
        self._pending_removal_versions: list[OrchestrationVersion] = []
        self._versions_lock = threading.Lock()

        # Observability counters for hot-reload metrics
        self._orchestrations_loaded_total: int = 0
        self._orchestrations_unloaded_total: int = 0
        self._orchestrations_reloaded_total: int = 0

    @property
    def orchestrations(self) -> list[Orchestration]:
        """Get the list of active orchestrations."""
        return self._orchestrations

    @property
    def router(self) -> Router | None:
        """Get the current router."""
        return self._router

    def init_from_directory(self) -> None:
        """Initialize the registry from the orchestrations directory.

        This scans the orchestrations directory and records all .yaml/.yml files
        with their modification times. This establishes the baseline for detecting
        new and modified files in subsequent poll cycles.
        """
        if not self._orchestrations_dir.exists() or not self._orchestrations_dir.is_dir():
            return

        for file_path in self._orchestrations_dir.iterdir():
            if file_path.suffix in (".yaml", ".yml"):
                try:
                    mtime = file_path.stat().st_mtime
                    self._known_orchestration_files[file_path] = mtime
                except OSError as e:
                    logger.warning(f"Could not stat orchestration file {file_path}: {e}")

        logger.debug(
            f"Initialized with {len(self._known_orchestration_files)} known orchestration files"
        )

    def get_hot_reload_metrics(self) -> dict[str, int]:
        """Get observability metrics for hot-reload operations.

        Returns:
            Dict with hot-reload metric counters:
            - orchestrations_loaded_total: Count of orchestrations loaded from new files
            - orchestrations_unloaded_total: Count of orchestrations unloaded from deleted files
            - orchestrations_reloaded_total: Count of orchestrations reloaded from modified files
        """
        return {
            "orchestrations_loaded_total": self._orchestrations_loaded_total,
            "orchestrations_unloaded_total": self._orchestrations_unloaded_total,
            "orchestrations_reloaded_total": self._orchestrations_reloaded_total,
        }

    def get_active_versions(self) -> list[OrchestrationVersion]:
        """Get the list of active orchestration versions.

        Returns:
            List of active OrchestrationVersion objects.
        """
        with self._versions_lock:
            return list(self._active_versions)

    def get_pending_removal_versions(self) -> list[OrchestrationVersion]:
        """Get the list of versions pending removal.

        Returns:
            List of OrchestrationVersion objects pending removal.
        """
        with self._versions_lock:
            return list(self._pending_removal_versions)

    def get_version_for_orchestration(
        self, orchestration: Orchestration
    ) -> OrchestrationVersion | None:
        """Get the OrchestrationVersion for an orchestration.

        Args:
            orchestration: The orchestration to find.

        Returns:
            The OrchestrationVersion if found, None otherwise.
        """
        with self._versions_lock:
            for version in self._active_versions:
                if version.orchestration is orchestration:
                    return version
            # Also check pending removal versions for in-flight executions
            for version in self._pending_removal_versions:
                if version.orchestration is orchestration:
                    return version
        return None

    def detect_and_load_orchestration_changes(self) -> tuple[int, int]:
        """Detect and load new and modified orchestration files.

        Scans the orchestrations directory for:
        1. New .yaml/.yml files not in the known files dict
        2. Modified files (mtime changed since last check)

        Returns:
            Tuple of (new_orchestrations_count, modified_orchestrations_count).
        """
        if not self._orchestrations_dir.exists() or not self._orchestrations_dir.is_dir():
            return 0, 0

        new_orchestrations_count = 0
        modified_orchestrations_count = 0

        # Scan for new and modified files
        for file_path in sorted(self._orchestrations_dir.iterdir()):
            if file_path.suffix not in (".yaml", ".yml"):
                continue

            try:
                current_mtime = file_path.stat().st_mtime
            except OSError as e:
                logger.warning(f"Could not stat orchestration file {file_path}: {e}")
                continue

            known_mtime = self._known_orchestration_files.get(file_path)

            if known_mtime is None:
                # New file detected
                logger.info(f"Detected new orchestration file: {file_path}")
                loaded = self._load_orchestrations_from_file(
                    file_path, current_mtime, rebuild_router=False
                )
                new_orchestrations_count += loaded
                self._known_orchestration_files[file_path] = current_mtime

            elif current_mtime > known_mtime:
                # Modified file detected
                logger.info(
                    f"Detected modified orchestration file: {file_path} "
                    f"(mtime: {known_mtime} -> {current_mtime})"
                )
                reloaded = self._reload_modified_file(file_path, current_mtime)
                modified_orchestrations_count += reloaded
                self._known_orchestration_files[file_path] = current_mtime

        # Rebuild Router once after all new files are processed
        if new_orchestrations_count > 0:
            self._router = self._router_factory(self._orchestrations)

        if new_orchestrations_count > 0 or modified_orchestrations_count > 0:
            logger.info(
                f"Hot-reload complete: {new_orchestrations_count} new, "
                f"{modified_orchestrations_count} reloaded, "
                f"total active: {len(self._orchestrations)}"
            )

        return new_orchestrations_count, modified_orchestrations_count

    def detect_and_unload_removed_files(self) -> int:
        """Detect and unload orchestrations from removed files.

        Returns:
            Number of orchestrations unloaded.
        """
        if not self._orchestrations_dir.exists() or not self._orchestrations_dir.is_dir():
            return 0

        unloaded_count = 0
        removed_files: list[Path] = []

        # Check which known files no longer exist
        for file_path in list(self._known_orchestration_files.keys()):
            if not file_path.exists():
                logger.info(f"Detected removed orchestration file: {file_path}")
                removed_files.append(file_path)

        # Process each removed file
        for file_path in removed_files:
            unloaded = self._unload_orchestrations_from_file(file_path)
            unloaded_count += unloaded
            del self._known_orchestration_files[file_path]

        if unloaded_count > 0:
            logger.info(
                f"Unloaded {unloaded_count} orchestration(s) from {len(removed_files)} "
                f"removed file(s), total active: {len(self._orchestrations)}"
            )

        return unloaded_count

    def cleanup_pending_removal_versions(self) -> int:
        """Clean up old orchestration versions that no longer have active executions.

        Returns:
            Number of versions cleaned up.
        """
        cleaned_count = 0
        with self._versions_lock:
            still_pending = []
            for version in self._pending_removal_versions:
                if version.has_active_executions:
                    still_pending.append(version)
                else:
                    logger.info(
                        f"Cleaned up old orchestration version '{version.name}' "
                        f"(version {version.version_id[:8]})"
                    )
                    cleaned_count += 1
            self._pending_removal_versions = still_pending

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old orchestration version(s)")

        return cleaned_count

    def _load_orchestrations_from_file(
        self, file_path: Path, mtime: float, rebuild_router: bool = True
    ) -> int:
        """Load orchestrations from a new file.

        Args:
            file_path: Path to the orchestration file.
            mtime: Modification time of the file.
            rebuild_router: Whether to rebuild the router after loading.

        Returns:
            Number of orchestrations loaded.
        """
        try:
            new_orchestrations = load_orchestration_file(file_path)
            if new_orchestrations:
                self._orchestrations.extend(new_orchestrations)
                # Update the router with the new orchestrations
                if rebuild_router:
                    self._router = self._router_factory(self._orchestrations)

                # Create versioned entries for tracking
                with self._versions_lock:
                    for orch in new_orchestrations:
                        version = OrchestrationVersion.create(orch, file_path, mtime)
                        self._active_versions.append(version)

                # Update observability counter
                self._orchestrations_loaded_total += len(new_orchestrations)

                logger.info(
                    f"Loaded {len(new_orchestrations)} orchestration(s) from {file_path.name}"
                )
                return len(new_orchestrations)
            else:
                logger.debug(f"No enabled orchestrations in {file_path.name}")
                return 0
        except OSError as e:
            logger.error(
                f"Failed to load orchestration file {file_path} due to I/O error: {e}",
                extra={"file_path": str(file_path)},
            )
            return 0
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Failed to load orchestration file {file_path} due to data error: {e}",
                extra={"file_path": str(file_path)},
            )
            return 0
        except OrchestrationError as e:
            logger.error(
                f"Failed to load orchestration file {file_path}: {e}",
                extra={"file_path": str(file_path)},
            )
            return 0

    def _reload_modified_file(self, file_path: Path, new_mtime: float) -> int:
        """Reload orchestrations from a modified file.

        Args:
            file_path: Path to the modified orchestration file.
            new_mtime: New modification time of the file.

        Returns:
            Number of orchestrations reloaded.
        """
        try:
            new_orchestrations = load_orchestration_file(file_path)
        except OSError as e:
            logger.error(
                f"Failed to reload modified orchestration file {file_path} due to I/O error: {e}",
                extra={"file_path": str(file_path)},
            )
            return 0
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Failed to reload modified orchestration file {file_path} due to data error: {e}",
                extra={"file_path": str(file_path)},
            )
            return 0
        except OrchestrationError as e:
            logger.error(
                f"Failed to reload modified orchestration file {file_path}: {e}",
                extra={"file_path": str(file_path)},
            )
            return 0

        with self._versions_lock:
            # Move old versions from this file to pending_removal
            old_versions = [v for v in self._active_versions if v.source_file == file_path]
            for old_version in old_versions:
                self._active_versions.remove(old_version)
                if old_version.has_active_executions:
                    self._pending_removal_versions.append(old_version)
                    logger.info(
                        f"Orchestration '{old_version.name}' "
                        f"(version {old_version.version_id[:8]}) moved to pending "
                        f"removal with {old_version.active_executions} active execution(s)"
                    )
                else:
                    logger.debug(
                        f"Orchestration '{old_version.name}' "
                        f"(version {old_version.version_id[:8]}) removed immediately"
                    )

            # Remove old orchestrations from the main list
            old_orch_names = {v.name for v in old_versions}
            self._orchestrations = [
                o
                for o in self._orchestrations
                if o.name not in old_orch_names
                or not any(v.orchestration is o for v in old_versions)
            ]

        # Add new orchestrations
        if new_orchestrations:
            self._orchestrations.extend(new_orchestrations)

            with self._versions_lock:
                for orch in new_orchestrations:
                    version = OrchestrationVersion.create(orch, file_path, new_mtime)
                    self._active_versions.append(version)
                    logger.info(f"Created new version {version.version_id[:8]} for '{orch.name}'")

        # Update the router with the updated orchestrations
        self._router = self._router_factory(self._orchestrations)

        # Update observability counter
        self._orchestrations_reloaded_total += len(new_orchestrations)

        logger.info(f"Reloaded {len(new_orchestrations)} orchestration(s) from {file_path.name}")
        return len(new_orchestrations)

    def _unload_orchestrations_from_file(self, file_path: Path) -> int:
        """Unload orchestrations from a removed file.

        Args:
            file_path: Path to the removed orchestration file.

        Returns:
            Number of orchestrations unloaded.
        """
        unloaded_count = 0

        with self._versions_lock:
            # Find versions from this file
            versions_to_remove = [v for v in self._active_versions if v.source_file == file_path]

            for version in versions_to_remove:
                self._active_versions.remove(version)

                if version.has_active_executions:
                    # Keep alive until executions complete
                    self._pending_removal_versions.append(version)
                    logger.info(
                        f"Orchestration '{version.name}' "
                        f"(version {version.version_id[:8]}) moved to pending "
                        f"removal with {version.active_executions} active execution(s)"
                    )
                else:
                    logger.debug(
                        f"Orchestration '{version.name}' "
                        f"(version {version.version_id[:8]}) removed immediately"
                    )

                unloaded_count += 1

            # Remove orchestrations from the main list using identity comparison
            orchestrations_to_remove = [v.orchestration for v in versions_to_remove]
            self._orchestrations = [
                o
                for o in self._orchestrations
                if not any(o is orch for orch in orchestrations_to_remove)
            ]

        # Update the router with the remaining orchestrations
        if unloaded_count > 0:
            self._router = self._router_factory(self._orchestrations)

            # Update observability counter
            self._orchestrations_unloaded_total += unloaded_count

        return unloaded_count
