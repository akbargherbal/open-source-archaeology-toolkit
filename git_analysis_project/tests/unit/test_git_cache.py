# tests/unit/test_git_cache.py

"""
Unit tests for the CommitDataCache class.

These tests validate that the cache correctly processes various Git repository
scenarios into accurate pandas DataFrames.
"""

import pytest
from git_cache import CommitDataCache


class TestCommitDataCache:
    """Tests the core logic of the CommitDataCache."""

    def test_cache_on_empty_repo(self, empty_repo_cache):
        """Ensure the cache handles an empty repository without errors."""
        assert empty_repo_cache.commits.empty
        assert empty_repo_cache.file_changes.empty
        assert empty_repo_cache.file_presence.empty

    def test_cache_properties_on_simple_repo(self, simple_repo_cache):
        """Verify the cache's DataFrames are built correctly for a simple repo."""
        # 1. Test Commits DataFrame
        commits_df = simple_repo_cache.commits
        assert len(commits_df) == 11
        assert list(commits_df.columns) == [
            "hash",
            "author_name",
            "author_email",
            "date",
            "message",
        ]
        assert commits_df["author_name"].unique().tolist() == ["Alice"]
        assert "Initial commit" in commits_df["message"].iloc[-1]

        # 2. Test File Changes DataFrame
        changes_df = simple_repo_cache.file_changes
        assert not changes_df.empty
        # The initial commit should have 3 files added ('A')
        initial_commit_hash = commits_df[commits_df["message"] == "Initial commit"][
            "hash"
        ].iloc[0]
        initial_changes = changes_df[changes_df["commit_hash"] == initial_commit_hash]
        assert len(initial_changes) == 3
        assert initial_changes["change_type"].unique().tolist() == ["A"]
        assert set(initial_changes["filepath"]) == {"README.md", "main.py", "utils.py"}

        # 'main.py' should have been modified ('M') 9 times
        main_py_changes = changes_df[changes_df["filepath"] == "main.py"]
        assert len(main_py_changes[main_py_changes["change_type"] == "M"]) == 9

        # 3. Test File Presence DataFrame
        presence_df = simple_repo_cache.file_presence
        latest_commit_hash = commits_df.iloc[0]["hash"]
        latest_snapshot = presence_df[presence_df["commit_hash"] == latest_commit_hash]
        assert len(latest_snapshot) == 4
        assert set(latest_snapshot["filepath"]) == {
            "README.md",
            "main.py",
            "utils.py",
            "docs/guide.md",
        }

    def test_cache_on_complex_repo(self, complex_repo_cache):
        """Verify the cache handles branches, merges, and multiple authors."""
        commits_df = complex_repo_cache.commits
        authors = commits_df["author_name"].unique()
        assert "Alice" in authors
        assert "Bob" in authors
        assert "Charlie" in authors

        changes_df = complex_repo_cache.file_changes
        # Check for files created in different branches
        assert "feature_x.py" in changes_df["filepath"].values
        assert "core.py" in changes_df["filepath"].values

    def test_cache_on_refactor_repo(self, refactor_repo_cache):
        """
        Verify the cache correctly handles file renames.

        The new `git log --numstat` approach does not provide a distinct 'R'
        (rename) status. Instead, it tracks the file to its new location. This
        test is updated to verify that the *new* file path is correctly
        recorded in the file_changes DataFrame for the refactoring commit.
        """
        commits_df = refactor_repo_cache.commits
        changes_df = refactor_repo_cache.file_changes

        # Find the refactoring commit
        refactor_commit_hash = commits_df[
            commits_df["message"].str.contains("Refactor: Move files")
        ]["hash"].iloc[0]

        # Get the file changes associated with that specific commit
        refactor_changes = changes_df[changes_df["commit_hash"] == refactor_commit_hash]

        # The new implementation correctly follows renames but doesn't label
        # them as 'R'. We verify the new paths are present.
        assert not refactor_changes.empty
        assert len(refactor_changes) == 2

        # The filepath should be the *new* path after the rename
        final_filepaths = refactor_changes["filepath"].tolist()
        assert "new_core/file1.py" in final_filepaths
        assert "new_core/file2.py" in final_filepaths

        # The old path should not be present in the final file path list
        assert "old_module/file1.py" not in final_filepaths
        assert "old_module/file2.py" not in final_filepaths
