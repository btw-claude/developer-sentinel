"""Tests for OrchestrationRegistry.

This module tests the OrchestrationRegistry class in isolation, focusing on:
- Initialization from directory
- File addition detection
- File removal detection
- Version tracking
- Hot-reload mechanics
- Pending removal cleanup
- Hot-reload metrics
- Router rebuilding
"""

import tempfile
from pathlib import Path

from sentinel.orchestration import Orchestration, OrchestrationVersion
from sentinel.orchestration_registry import OrchestrationRegistry
from sentinel.router import Router
from tests.helpers import set_mtime_in_future


def make_router_factory():
    """Create a router factory function for testing.

    Returns:
        A callable that takes a list of orchestrations and returns a Router.
    """
    def router_factory(orchestrations: list[Orchestration]) -> Router:
        return Router(orchestrations)
    return router_factory


def write_orchestration_yaml(file_path: Path, name: str = "test-orch") -> None:
    """Write a minimal valid orchestration YAML file.

    Args:
        file_path: Path to write the YAML file.
        name: Name for the orchestration step (default: "test-orch").
    """
    yaml_content = f"""steps:
  - name: {name}
    trigger:
      project: TEST
      tags:
        - review
    agent:
      prompt: "Test prompt"
"""
    file_path.write_text(yaml_content)


class TestOrchestrationRegistryInit:
    """Tests for OrchestrationRegistry initialization."""

    def test_init_from_directory_with_existing_files(self) -> None:
        """init_from_directory should record existing .yaml/.yml files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)

            # Create some orchestration files
            file1 = orchestrations_dir / "orch1.yaml"
            file2 = orchestrations_dir / "orch2.yml"
            file3 = orchestrations_dir / "not_yaml.txt"

            write_orchestration_yaml(file1, "orch1")
            write_orchestration_yaml(file2, "orch2")
            file3.write_text("not a yaml file")

            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            registry.init_from_directory()

            # Only .yaml and .yml files should be tracked
            known_files = registry._known_orchestration_files
            assert len(known_files) == 2
            assert file1 in known_files
            assert file2 in known_files
            assert file3 not in known_files

            # Mtimes should be recorded
            assert known_files[file1] > 0
            assert known_files[file2] > 0

    def test_init_from_directory_with_empty_dir(self) -> None:
        """init_from_directory should handle an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            registry.init_from_directory()

            known_files = registry._known_orchestration_files
            assert len(known_files) == 0

    def test_init_from_directory_with_non_existent_dir(self) -> None:
        """init_from_directory should handle a non-existent directory."""
        orchestrations_dir = Path("/nonexistent/path")
        router_factory = make_router_factory()
        registry = OrchestrationRegistry(orchestrations_dir, router_factory)

        registry.init_from_directory()

        known_files = registry._known_orchestration_files
        assert len(known_files) == 0


class TestOrchestrationRegistryFileAddition:
    """Tests for detecting and loading new orchestration files."""

    def test_detect_and_load_new_file(self) -> None:
        """detect_and_load_orchestration_changes should load new files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty directory
            registry.init_from_directory()
            assert len(registry.orchestrations) == 0

            # Add a new file
            new_file = orchestrations_dir / "new.yaml"
            write_orchestration_yaml(new_file, "new-orch")

            new_count, modified_count = registry.detect_and_load_orchestration_changes()

            # Should detect 1 new orchestration
            assert new_count == 1
            assert modified_count == 0
            assert len(registry.orchestrations) == 1
            assert registry.orchestrations[0].name == "new-orch"

            # File should be tracked now
            assert new_file in registry._known_orchestration_files

    def test_detect_and_load_multiple_new_files(self) -> None:
        """detect_and_load_orchestration_changes should load multiple new files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            registry.init_from_directory()

            # Add multiple new files
            file1 = orchestrations_dir / "orch1.yaml"
            file2 = orchestrations_dir / "orch2.yaml"
            write_orchestration_yaml(file1, "orch1")
            write_orchestration_yaml(file2, "orch2")

            new_count, modified_count = registry.detect_and_load_orchestration_changes()

            assert new_count == 2
            assert modified_count == 0
            assert len(registry.orchestrations) == 2

            orch_names = {o.name for o in registry.orchestrations}
            assert orch_names == {"orch1", "orch2"}

    def test_router_rebuilt_after_loading_new_files(self) -> None:
        """Router should be rebuilt after loading new files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            registry.init_from_directory()
            assert registry.router is None

            # Add a new file
            new_file = orchestrations_dir / "new.yaml"
            write_orchestration_yaml(new_file, "new-orch")

            registry.detect_and_load_orchestration_changes()

            # Router should be created
            assert registry.router is not None
            assert isinstance(registry.router, Router)

    def test_hot_reload_metrics_incremented_on_load(self) -> None:
        """Hot-reload metrics should increment when loading new files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            registry.init_from_directory()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_loaded_total"] == 0

            # Add a new file
            new_file = orchestrations_dir / "new.yaml"
            write_orchestration_yaml(new_file, "new-orch")

            registry.detect_and_load_orchestration_changes()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_loaded_total"] == 1
            assert metrics["orchestrations_unloaded_total"] == 0
            assert metrics["orchestrations_reloaded_total"] == 0


class TestOrchestrationRegistryFileRemoval:
    """Tests for detecting and unloading removed orchestration files."""

    def test_detect_and_unload_removed_file(self) -> None:
        """detect_and_unload_removed_files should unload deleted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            assert len(registry.orchestrations) == 1

            # Delete the file
            file_path.unlink()

            unloaded_count = registry.detect_and_unload_removed_files()

            assert unloaded_count == 1
            assert len(registry.orchestrations) == 0
            assert file_path not in registry._known_orchestration_files

    def test_detect_and_unload_multiple_removed_files(self) -> None:
        """detect_and_unload_removed_files should unload multiple deleted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load multiple files
            registry.init_from_directory()

            file1 = orchestrations_dir / "orch1.yaml"
            file2 = orchestrations_dir / "orch2.yaml"
            write_orchestration_yaml(file1, "orch1")
            write_orchestration_yaml(file2, "orch2")

            registry.detect_and_load_orchestration_changes()

            assert len(registry.orchestrations) == 2

            # Delete both files
            file1.unlink()
            file2.unlink()

            unloaded_count = registry.detect_and_unload_removed_files()

            assert unloaded_count == 2
            assert len(registry.orchestrations) == 0

    def test_router_rebuilt_after_unloading_files(self) -> None:
        """Router should be rebuilt after unloading files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            old_router = registry.router

            # Delete the file
            file_path.unlink()
            registry.detect_and_unload_removed_files()

            # Router should be rebuilt
            assert registry.router is not None
            assert registry.router is not old_router

    def test_hot_reload_metrics_incremented_on_unload(self) -> None:
        """Hot-reload metrics should increment when unloading files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_unloaded_total"] == 0

            # Delete the file
            file_path.unlink()
            registry.detect_and_unload_removed_files()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_unloaded_total"] == 1


class TestOrchestrationRegistryVersionTracking:
    """Tests for version tracking of orchestrations."""

    def test_get_active_versions_after_loading(self) -> None:
        """get_active_versions should return active versions after loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            active_versions = registry.get_active_versions()

            assert len(active_versions) == 1
            assert isinstance(active_versions[0], OrchestrationVersion)
            assert active_versions[0].name == "test-orch"
            assert active_versions[0].source_file == file_path

    def test_version_id_uniqueness(self) -> None:
        """Each version should have a unique version_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load multiple files
            registry.init_from_directory()

            file1 = orchestrations_dir / "orch1.yaml"
            file2 = orchestrations_dir / "orch2.yaml"
            write_orchestration_yaml(file1, "orch1")
            write_orchestration_yaml(file2, "orch2")

            registry.detect_and_load_orchestration_changes()

            active_versions = registry.get_active_versions()

            assert len(active_versions) == 2
            version_ids = [v.version_id for v in active_versions]
            assert len(set(version_ids)) == 2

    def test_get_version_for_orchestration(self) -> None:
        """get_version_for_orchestration should return the correct version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            orchestration = registry.orchestrations[0]
            version = registry.get_version_for_orchestration(orchestration)

            assert version is not None
            assert isinstance(version, OrchestrationVersion)
            assert version.orchestration is orchestration
            assert version.name == "test-orch"

    def test_get_version_for_orchestration_not_found(self) -> None:
        """get_version_for_orchestration should return None for unknown orch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            from tests.helpers import make_orchestration

            registry.init_from_directory()

            # Create an orchestration not in the registry
            unknown_orch = make_orchestration(name="unknown")

            version = registry.get_version_for_orchestration(unknown_orch)
            assert version is None


class TestOrchestrationRegistryReload:
    """Tests for hot-reload mechanics when files are modified."""

    def test_detect_and_reload_modified_file(self) -> None:
        """detect_and_load_orchestration_changes should reload modified files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "original-name")

            registry.detect_and_load_orchestration_changes()

            assert len(registry.orchestrations) == 1
            assert registry.orchestrations[0].name == "original-name"

            # Modify the file (update mtime and content)
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "modified-name")

            new_count, modified_count = registry.detect_and_load_orchestration_changes()

            assert new_count == 0
            assert modified_count == 1
            assert len(registry.orchestrations) == 1
            assert registry.orchestrations[0].name == "modified-name"

    def test_old_version_moved_to_pending_removal_on_reload(self) -> None:
        """Old version should move to pending_removal when file is reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            old_orchestration = registry.orchestrations[0]
            old_version = registry.get_version_for_orchestration(old_orchestration)

            # Simulate active execution
            old_version.increment_executions()

            # Modify the file
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "test-orch-v2")

            registry.detect_and_load_orchestration_changes()

            # Old version should be in pending_removal
            pending_versions = registry.get_pending_removal_versions()
            assert len(pending_versions) == 1
            assert pending_versions[0] is old_version
            assert pending_versions[0].has_active_executions

            # New version should be active
            active_versions = registry.get_active_versions()
            assert len(active_versions) == 1
            assert active_versions[0] is not old_version

    def test_old_version_without_executions_not_in_pending_removal(self) -> None:
        """Old version without active executions should not be in pending_removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Create and load a file
            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.init_from_directory()
            registry.detect_and_load_orchestration_changes()

            # Don't increment executions

            # Modify the file
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "test-orch-v2")

            registry.detect_and_load_orchestration_changes()

            # Old version should NOT be in pending_removal
            pending_versions = registry.get_pending_removal_versions()
            assert len(pending_versions) == 0

    def test_router_rebuilt_after_reload(self) -> None:
        """Router should be rebuilt after reloading files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Create and load a file
            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.init_from_directory()
            registry.detect_and_load_orchestration_changes()

            old_router = registry.router

            # Modify the file
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "test-orch-v2")

            registry.detect_and_load_orchestration_changes()

            # Router should be rebuilt
            assert registry.router is not None
            assert registry.router is not old_router

    def test_hot_reload_metrics_incremented_on_reload(self) -> None:
        """Hot-reload metrics should increment when reloading files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Create and load a file
            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.init_from_directory()
            registry.detect_and_load_orchestration_changes()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_reloaded_total"] == 0

            # Modify the file
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "test-orch-v2")

            registry.detect_and_load_orchestration_changes()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_reloaded_total"] == 1


class TestOrchestrationRegistryPendingRemovalCleanup:
    """Tests for cleanup of pending removal versions."""

    def test_cleanup_pending_removal_versions_without_active_executions(self) -> None:
        """cleanup_pending_removal_versions should remove versions with no executions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            old_orchestration = registry.orchestrations[0]
            old_version = registry.get_version_for_orchestration(old_orchestration)

            # Simulate active execution
            old_version.increment_executions()

            # Modify the file to trigger reload
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "test-orch-v2")
            registry.detect_and_load_orchestration_changes()

            # Old version should be in pending_removal
            pending_versions = registry.get_pending_removal_versions()
            assert len(pending_versions) == 1

            # Complete the execution
            old_version.decrement_executions()

            # Cleanup should remove it
            cleaned_count = registry.cleanup_pending_removal_versions()

            assert cleaned_count == 1
            pending_versions = registry.get_pending_removal_versions()
            assert len(pending_versions) == 0

    def test_cleanup_pending_removal_versions_with_active_executions(self) -> None:
        """cleanup_pending_removal_versions should keep versions with executions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            old_orchestration = registry.orchestrations[0]
            old_version = registry.get_version_for_orchestration(old_orchestration)

            # Simulate active execution
            old_version.increment_executions()

            # Modify the file to trigger reload
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "test-orch-v2")
            registry.detect_and_load_orchestration_changes()

            # Old version should be in pending_removal
            pending_versions = registry.get_pending_removal_versions()
            assert len(pending_versions) == 1

            # Cleanup should NOT remove it (still has active execution)
            cleaned_count = registry.cleanup_pending_removal_versions()

            assert cleaned_count == 0
            pending_versions = registry.get_pending_removal_versions()
            assert len(pending_versions) == 1

    def test_cleanup_pending_removal_versions_multiple(self) -> None:
        """cleanup_pending_removal_versions should handle multiple versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load two files
            registry.init_from_directory()

            file1 = orchestrations_dir / "orch1.yaml"
            file2 = orchestrations_dir / "orch2.yaml"
            write_orchestration_yaml(file1, "orch1")
            write_orchestration_yaml(file2, "orch2")

            registry.detect_and_load_orchestration_changes()

            old_orch1 = registry.orchestrations[0]
            old_orch2 = registry.orchestrations[1]
            old_version1 = registry.get_version_for_orchestration(old_orch1)
            old_version2 = registry.get_version_for_orchestration(old_orch2)

            # Simulate executions on both
            old_version1.increment_executions()
            old_version2.increment_executions()

            # Modify both files
            set_mtime_in_future(file1)
            set_mtime_in_future(file2)
            write_orchestration_yaml(file1, "orch1-v2")
            write_orchestration_yaml(file2, "orch2-v2")
            registry.detect_and_load_orchestration_changes()

            # Both old versions should be in pending_removal
            pending_versions = registry.get_pending_removal_versions()
            assert len(pending_versions) == 2

            # Complete execution on one
            old_version1.decrement_executions()

            # Cleanup should remove only the one without executions
            cleaned_count = registry.cleanup_pending_removal_versions()

            assert cleaned_count == 1
            pending_versions = registry.get_pending_removal_versions()
            assert len(pending_versions) == 1
            assert pending_versions[0] is old_version2


class TestOrchestrationRegistryProperties:
    """Tests for OrchestrationRegistry properties."""

    def test_orchestrations_property(self) -> None:
        """orchestrations property should return the list of active orchestrations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load files
            registry.init_from_directory()

            file1 = orchestrations_dir / "orch1.yaml"
            file2 = orchestrations_dir / "orch2.yaml"
            write_orchestration_yaml(file1, "orch1")
            write_orchestration_yaml(file2, "orch2")

            registry.detect_and_load_orchestration_changes()

            orchestrations = registry.orchestrations

            assert len(orchestrations) == 2
            assert all(isinstance(o, Orchestration) for o in orchestrations)
            orch_names = {o.name for o in orchestrations}
            assert orch_names == {"orch1", "orch2"}

    def test_router_property(self) -> None:
        """router property should return the current router."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initially None
            assert registry.router is None

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            # Should return a Router
            assert registry.router is not None
            assert isinstance(registry.router, Router)


class TestOrchestrationRegistryHotReloadMetrics:
    """Tests for hot-reload metrics."""

    def test_get_hot_reload_metrics_initial_state(self) -> None:
        """get_hot_reload_metrics should return zero counters initially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            metrics = registry.get_hot_reload_metrics()

            assert metrics["orchestrations_loaded_total"] == 0
            assert metrics["orchestrations_unloaded_total"] == 0
            assert metrics["orchestrations_reloaded_total"] == 0

    def test_get_hot_reload_metrics_after_operations(self) -> None:
        """get_hot_reload_metrics should track all hot-reload operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            registry.init_from_directory()

            # Load a file
            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")
            registry.detect_and_load_orchestration_changes()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_loaded_total"] == 1
            assert metrics["orchestrations_unloaded_total"] == 0
            assert metrics["orchestrations_reloaded_total"] == 0

            # Reload the file
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "test-orch-v2")
            registry.detect_and_load_orchestration_changes()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_loaded_total"] == 1
            assert metrics["orchestrations_unloaded_total"] == 0
            assert metrics["orchestrations_reloaded_total"] == 1

            # Unload the file
            file_path.unlink()
            registry.detect_and_unload_removed_files()

            metrics = registry.get_hot_reload_metrics()
            assert metrics["orchestrations_loaded_total"] == 1
            assert metrics["orchestrations_unloaded_total"] == 1
            assert metrics["orchestrations_reloaded_total"] == 1


class TestOrchestrationRegistryPendingRemovalVersionLookup:
    """Tests for looking up versions in pending_removal."""

    def test_get_version_for_orchestration_in_pending_removal(self) -> None:
        """get_version_for_orchestration should find versions in pending_removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrations_dir = Path(tmpdir)
            router_factory = make_router_factory()
            registry = OrchestrationRegistry(orchestrations_dir, router_factory)

            # Initialize with empty dir, then create and load a file
            registry.init_from_directory()

            file_path = orchestrations_dir / "orch.yaml"
            write_orchestration_yaml(file_path, "test-orch")

            registry.detect_and_load_orchestration_changes()

            old_orchestration = registry.orchestrations[0]
            old_version = registry.get_version_for_orchestration(old_orchestration)

            # Simulate active execution
            old_version.increment_executions()

            # Modify the file to trigger reload
            set_mtime_in_future(file_path)
            write_orchestration_yaml(file_path, "test-orch-v2")
            registry.detect_and_load_orchestration_changes()

            # Old orchestration should still be findable in pending_removal
            found_version = registry.get_version_for_orchestration(old_orchestration)

            assert found_version is old_version
            assert found_version.has_active_executions
