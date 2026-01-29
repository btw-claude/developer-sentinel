"""Tests for Sentinel orchestration hot-reload functionality."""

import tempfile
from pathlib import Path

from sentinel.main import Sentinel
from sentinel.orchestration import Orchestration

# Import shared fixtures and helpers from conftest.py
from tests.conftest import (
    MockAgentClient,
    MockJiraClient,
    MockTagClient,
    make_config,
    make_orchestration,
    set_mtime_in_future,
)


class TestSentinelOrchestrationHotReload:
    """Tests for Sentinel orchestration file hot-reload functionality."""

    def test_init_tracks_existing_orchestration_files(self) -> None:
        """Test that __init__ initializes the set of known orchestration files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            (orch_dir / "existing1.yaml").write_text(
                "orchestrations:\n  - name: existing1\n    trigger: {project: TEST}\n    agent: {prompt: test}"
            )
            (orch_dir / "existing2.yml").write_text(
                "orchestrations:\n  - name: existing2\n    trigger: {project: TEST}\n    agent: {prompt: test}"
            )
            (orch_dir / "readme.txt").write_text("This is not an orchestration file")

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations = [make_orchestration()]

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            assert len(sentinel._known_orchestration_files) == 2
            assert orch_dir / "existing1.yaml" in sentinel._known_orchestration_files
            assert orch_dir / "existing2.yml" in sentinel._known_orchestration_files

    def test_detects_new_orchestration_files(self) -> None:
        """Test that new orchestration files are detected during poll cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            assert len(sentinel._known_orchestration_files) == 0
            assert len(sentinel.orchestrations) == 0

            (orch_dir / "new_orch.yaml").write_text(
                """orchestrations:
  - name: new-orchestration
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Test prompt
      tools: [jira]
"""
            )

            sentinel.run_once()

            assert len(sentinel._known_orchestration_files) == 1
            assert orch_dir / "new_orch.yaml" in sentinel._known_orchestration_files
            assert len(sentinel.orchestrations) == 1
            assert sentinel.orchestrations[0].name == "new-orchestration"

    def test_loads_multiple_orchestrations_from_new_file(self) -> None:
        """Test that multiple orchestrations from a single new file are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            (orch_dir / "multi.yaml").write_text(
                """orchestrations:
  - name: orch-1
    trigger:
      project: TEST
    agent:
      prompt: Test 1
  - name: orch-2
    trigger:
      project: PROJ
    agent:
      prompt: Test 2
"""
            )

            sentinel.run_once()

            assert len(sentinel.orchestrations) == 2
            assert sentinel.orchestrations[0].name == "orch-1"
            assert sentinel.orchestrations[1].name == "orch-2"

    def test_updates_router_with_new_orchestrations(self) -> None:
        """Test that the router is updated when new orchestrations are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(
                issues=[
                    {"key": "TEST-1", "fields": {"summary": "Test", "labels": ["new-tag"]}},
                ]
            )
            agent_client = MockAgentClient(responses=["SUCCESS"])
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            results, submitted_count = sentinel.run_once()
            assert len(results) == 0

            (orch_dir / "matching.yaml").write_text(
                """orchestrations:
  - name: matching-orch
    trigger:
      project: TEST
      tags: [new-tag]
    agent:
      prompt: Process this
      tools: [jira]
"""
            )

            results, submitted_count = sentinel.run_once()
            assert len(results) == 1
            assert results[0].succeeded is True

    def test_handles_invalid_orchestration_file_gracefully(self) -> None:
        """Test that invalid orchestration files don't crash the system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            (orch_dir / "invalid.yaml").write_text("this is not valid: yaml: [[[")

            sentinel.run_once()

            assert orch_dir / "invalid.yaml" in sentinel._known_orchestration_files
            assert len(sentinel.orchestrations) == 0

    def test_no_router_rebuild_for_empty_or_invalid_files(self) -> None:
        """Test that Router is not rebuilt when files contain no valid orchestrations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            original_router = sentinel.router

            (orch_dir / "invalid.yaml").write_text("this is not valid: yaml: [[[")
            (orch_dir / "empty.yaml").write_text("orchestrations: []")
            (orch_dir / "disabled.yaml").write_text(
                """orchestrations:
  - name: disabled-orch
    enabled: false
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            sentinel.run_once()

            assert orch_dir / "invalid.yaml" in sentinel._known_orchestration_files
            assert orch_dir / "empty.yaml" in sentinel._known_orchestration_files
            assert orch_dir / "disabled.yaml" in sentinel._known_orchestration_files
            assert len(sentinel.orchestrations) == 0
            assert sentinel.router is original_router

    def test_does_not_reload_known_files(self) -> None:
        """Test that known files are not reloaded on subsequent poll cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            (orch_dir / "initial.yaml").write_text(
                """orchestrations:
  - name: initial
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            assert len(sentinel._known_orchestration_files) == 1
            sentinel.run_once()
            assert len(sentinel.orchestrations) == 0

            initial_count = len(sentinel.orchestrations)
            sentinel.run_once()
            assert len(sentinel.orchestrations) == initial_count

    def test_handles_nonexistent_orchestrations_directory(self) -> None:
        """Test handling when orchestrations directory doesn't exist."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(orchestrations_dir=Path("/nonexistent/path"))
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        assert len(sentinel._known_orchestration_files) == 0
        sentinel.run_once()
        assert len(sentinel.orchestrations) == 0

    def test_detects_modified_orchestration_files(self) -> None:
        """Test that modified orchestration files are detected and reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            orch_file = orch_dir / "modifiable.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: original-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Original prompt
      tools: [jira]
"""
            )

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            assert len(sentinel._known_orchestration_files) == 1
            assert orch_file in sentinel._known_orchestration_files
            original_mtime = sentinel._known_orchestration_files[orch_file]

            sentinel.run_once()
            assert len(sentinel.orchestrations) == 0

            orch_file.write_text(
                """orchestrations:
  - name: modified-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Modified prompt
      tools: [jira]
"""
            )
            set_mtime_in_future(orch_file)

            sentinel.run_once()

            assert len(sentinel.orchestrations) == 1
            assert sentinel.orchestrations[0].name == "modified-orch"
            assert sentinel.orchestrations[0].agent.prompt == "Modified prompt"
            assert sentinel._known_orchestration_files[orch_file] > original_mtime

    def test_modified_file_moves_old_version_to_pending_removal(self) -> None:
        """Test that modifying a file moves old versions to pending removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            orch_file = orch_dir / "versioned.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: versioned-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Version 1
"""
            )

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            assert len(sentinel._active_versions) == 0
            assert len(sentinel._pending_removal_versions) == 0

            new_file = orch_dir / "new_versioned.yaml"
            new_file.write_text(
                """orchestrations:
  - name: new-versioned-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: New Version 1
"""
            )

            sentinel.run_once()

            assert len(sentinel._active_versions) == 1
            assert sentinel._active_versions[0].name == "new-versioned-orch"

            sentinel._active_versions[0].increment_executions()

            new_file.write_text(
                """orchestrations:
  - name: new-versioned-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: New Version 2 - Modified
"""
            )
            set_mtime_in_future(new_file)

            sentinel.run_once()

            assert len(sentinel._pending_removal_versions) == 1
            assert (
                sentinel._pending_removal_versions[0].orchestration.agent.prompt == "New Version 1"
            )
            assert len(sentinel._active_versions) == 1
            assert (
                sentinel._active_versions[0].orchestration.agent.prompt
                == "New Version 2 - Modified"
            )

    def test_pending_removal_version_cleaned_up_after_execution_completes(self) -> None:
        """Test that pending removal versions are cleaned up after executions complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            orch_file = orch_dir / "cleanup_test.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: cleanup-orch
    trigger:
      project: TEST
    agent:
      prompt: Version 1
"""
            )

            sentinel.run_once()
            sentinel._active_versions[0].increment_executions()

            orch_file.write_text(
                """orchestrations:
  - name: cleanup-orch
    trigger:
      project: TEST
    agent:
      prompt: Version 2
"""
            )
            set_mtime_in_future(orch_file)

            sentinel.run_once()

            assert len(sentinel._pending_removal_versions) == 1
            assert sentinel._pending_removal_versions[0].active_executions == 1

            sentinel._pending_removal_versions[0].decrement_executions()
            sentinel.run_once()

            assert len(sentinel._pending_removal_versions) == 0

    def test_version_without_active_executions_removed_immediately(self) -> None:
        """Test that old versions without active executions are removed immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            orch_file = orch_dir / "no_exec.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: no-exec-orch
    trigger:
      project: TEST
    agent:
      prompt: Version 1
"""
            )

            sentinel.run_once()
            assert sentinel._active_versions[0].active_executions == 0

            orch_file.write_text(
                """orchestrations:
  - name: no-exec-orch
    trigger:
      project: TEST
    agent:
      prompt: Version 2
"""
            )
            set_mtime_in_future(orch_file)

            sentinel.run_once()

            assert len(sentinel._pending_removal_versions) == 0
            assert len(sentinel._active_versions) == 1
            assert sentinel._active_versions[0].orchestration.agent.prompt == "Version 2"

    def test_known_files_stores_mtime(self) -> None:
        """Test that _known_orchestration_files stores mtimes, not just presence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            orch_file = orch_dir / "mtime_test.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: test
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            assert isinstance(sentinel._known_orchestration_files[orch_file], float)
            assert sentinel._known_orchestration_files[orch_file] > 0

    def test_detects_removed_orchestration_files(self) -> None:
        """Test that removed orchestration files are detected and their orchestrations unloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            orch_file = orch_dir / "removable.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: removable-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Test prompt
"""
            )

            sentinel.run_once()

            assert len(sentinel.orchestrations) == 1
            assert sentinel.orchestrations[0].name == "removable-orch"
            assert orch_file in sentinel._known_orchestration_files

            orch_file.unlink()
            sentinel.run_once()

            assert len(sentinel.orchestrations) == 0
            assert orch_file not in sentinel._known_orchestration_files

    def test_removed_file_with_active_execution_moves_to_pending_removal(self) -> None:
        """Test that orchestrations from removed files with active executions go to pending removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            orch_file = orch_dir / "active_removal.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: active-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            sentinel.run_once()
            assert len(sentinel._active_versions) == 1

            sentinel._active_versions[0].increment_executions()
            assert sentinel._active_versions[0].active_executions == 1

            orch_file.unlink()
            sentinel.run_once()

            assert len(sentinel.orchestrations) == 0
            assert len(sentinel._pending_removal_versions) == 1
            assert sentinel._pending_removal_versions[0].name == "active-orch"
            assert sentinel._pending_removal_versions[0].active_executions == 1
            assert len(sentinel._active_versions) == 0

    def test_removed_file_without_active_execution_removed_immediately(self) -> None:
        """Test that orchestrations from removed files without active executions are removed immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            orch_file = orch_dir / "no_active.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: no-active-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            sentinel.run_once()
            assert len(sentinel._active_versions) == 1
            assert sentinel._active_versions[0].active_executions == 0

            orch_file.unlink()
            sentinel.run_once()

            assert len(sentinel.orchestrations) == 0
            assert len(sentinel._pending_removal_versions) == 0
            assert len(sentinel._active_versions) == 0

    def test_pending_removal_from_file_deletion_cleaned_up_after_execution(self) -> None:
        """Test that pending removal versions from deleted files are cleaned up after execution completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            orch_file = orch_dir / "cleanup_after_delete.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: cleanup-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            sentinel.run_once()
            sentinel._active_versions[0].increment_executions()

            orch_file.unlink()
            sentinel.run_once()
            assert len(sentinel._pending_removal_versions) == 1

            sentinel._pending_removal_versions[0].decrement_executions()
            sentinel.run_once()
            assert len(sentinel._pending_removal_versions) == 0

    def test_router_updated_after_file_removal(self) -> None:
        """Test that the router is updated when orchestrations are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(
                issues=[
                    {"key": "TEST-1", "fields": {"summary": "Test", "labels": ["review"]}},
                ]
            )
            agent_client = MockAgentClient(responses=["SUCCESS"])
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            orch_file = orch_dir / "matching.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: matching-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Process this
"""
            )

            results, submitted_count = sentinel.run_once()
            assert len(results) == 1

            jira_client.search_calls.clear()
            orch_file.unlink()
            results, submitted_count = sentinel.run_once()

            assert len(sentinel.orchestrations) == 0

    def test_multiple_orchestrations_from_same_removed_file(self) -> None:
        """Test that multiple orchestrations from a single removed file are all unloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            orch_file = orch_dir / "multi.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: orch-1
    trigger:
      project: TEST
    agent:
      prompt: Test 1
  - name: orch-2
    trigger:
      project: PROJ
    agent:
      prompt: Test 2
  - name: orch-3
    trigger:
      project: OTHER
    agent:
      prompt: Test 3
"""
            )

            sentinel.run_once()
            assert len(sentinel.orchestrations) == 3
            assert len(sentinel._active_versions) == 3

            orch_file.unlink()
            sentinel.run_once()

            assert len(sentinel.orchestrations) == 0
            assert len(sentinel._active_versions) == 0
            assert orch_file not in sentinel._known_orchestration_files

    def test_removal_of_one_file_does_not_affect_others(self) -> None:
        """Test that removing one file doesn't affect orchestrations from other files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            file1 = orch_dir / "keep.yaml"
            file1.write_text(
                """orchestrations:
  - name: keep-orch
    trigger:
      project: TEST
    agent:
      prompt: Keep this
"""
            )

            file2 = orch_dir / "remove.yaml"
            file2.write_text(
                """orchestrations:
  - name: remove-orch
    trigger:
      project: PROJ
    agent:
      prompt: Remove this
"""
            )

            sentinel.run_once()
            assert len(sentinel.orchestrations) == 2
            assert len(sentinel._known_orchestration_files) == 2

            file2.unlink()
            sentinel.run_once()

            assert len(sentinel.orchestrations) == 1
            assert sentinel.orchestrations[0].name == "keep-orch"
            assert file1 in sentinel._known_orchestration_files
            assert file2 not in sentinel._known_orchestration_files
