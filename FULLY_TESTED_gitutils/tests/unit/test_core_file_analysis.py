"""Unit tests for core file analysis functions."""
import pytest
from unittest.mock import Mock, PropertyMock
from gitutils import find_core_files, find_volatile_files

class TestCoreFileAnalysis:
    """Tests for identifying core, stable, and volatile files."""

    def test_find_core_files_simple_repo(self, simple_repo):
        """Verify core file detection on a real repository fixture."""
        # In simple_repo, main.py and utils.py are present from the start
        core_files = find_core_files(simple_repo, threshold=0.9)
        filenames = [f[0] for f in core_files]
        assert "main.py" in filenames
        assert "utils.py" in filenames
        assert "README.md" in filenames

    def test_find_core_files_empty_repo(self, empty_repo):
        """Should return an empty list for a repo with no commits."""
        assert find_core_files(empty_repo) == []

    def test_find_volatile_files(self, simple_repo):
        """Verify that the most frequently changed file is identified."""
        # main.py is changed in 9 commits
        volatile = find_volatile_files(simple_repo, min_commits=5)
        assert len(volatile) > 0
        assert volatile[0][0] == "main.py"
        assert volatile[0][1] == 10 # Touched in 10 commits (add + 9 mods)