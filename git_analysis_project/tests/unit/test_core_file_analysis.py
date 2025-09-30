# tests/unit/test_core_file_analysis.py

"""
Unit tests for core file analysis functions in gitutils.py.

These tests use mocked CommitDataCache objects to validate the analysis logic
in isolation, without relying on actual Git operations.
"""

import pytest
from unittest.mock import Mock, PropertyMock
import pandas as pd
from gitutils import (
    find_core_files_cached,
    find_volatile_files_cached,
    calculate_file_churn_cached,
)


@pytest.fixture
def mock_cache():
    """Provides a mock CommitDataCache object for unit testing."""
    return Mock()


class TestCoreFileAnalysisCached:
    """Tests for identifying core, volatile, and churned files from a cache."""

    def test_find_core_files_cached(self, mock_cache):
        """Unit test core file logic using a mocked cache."""
        mock_commits = pd.DataFrame({"hash": [f"h{i}" for i in range(10)]})
        mock_presence = pd.DataFrame(
            [
                {"commit_hash": "h0", "filepath": "core.py"},
                {"commit_hash": "h1", "filepath": "core.py"},
                {"commit_hash": "h2", "filepath": "core.py"},
                {"commit_hash": "h3", "filepath": "core.py"},
                {"commit_hash": "h4", "filepath": "core.py"},
                {"commit_hash": "h5", "filepath": "core.py"},
                {"commit_hash": "h6", "filepath": "core.py"},
                {"commit_hash": "h7", "filepath": "core.py"},
                {"commit_hash": "h8", "filepath": "core.py"},
                {"commit_hash": "h9", "filepath": "transient.py"},  # Appears once
            ]
        )
        type(mock_cache).commits = PropertyMock(return_value=mock_commits)
        type(mock_cache).file_presence = PropertyMock(return_value=mock_presence)

        # Test with a threshold that should only return 'core.py' (90%)
        result_df = find_core_files_cached(mock_cache, threshold=0.9)
        assert len(result_df) == 1
        assert result_df.iloc[0]["filepath"] == "core.py"
        assert result_df.iloc[0]["presence_percentage"] == 0.9

        # Test with a lower threshold that returns both
        result_df_both = find_core_files_cached(mock_cache, threshold=0.1)
        assert len(result_df_both) == 2

    def test_find_volatile_files_cached(self, mock_cache):
        """Unit test volatile file logic using a mocked cache."""
        mock_changes = pd.DataFrame(
            [
                {"commit_hash": "h1", "filepath": "a.py"},
                {"commit_hash": "h2", "filepath": "a.py"},
                {"commit_hash": "h3", "filepath": "a.py"},  # a.py: 3 commits
                {"commit_hash": "h4", "filepath": "b.py"},  # b.py: 1 commit
            ]
        )
        type(mock_cache).file_changes = PropertyMock(return_value=mock_changes)

        result_df = find_volatile_files_cached(mock_cache, min_commits=3)
        assert len(result_df) == 1
        assert result_df.iloc[0]["filepath"] == "a.py"
        assert result_df.iloc[0]["commit_count"] == 3

        result_df_empty = find_volatile_files_cached(mock_cache, min_commits=5)
        assert result_df_empty.empty

    def test_calculate_file_churn_cached(self, mock_cache):
        """Unit test file churn logic using a mocked cache."""
        mock_changes = pd.DataFrame(
            [
                {
                    "commit_hash": "h1",
                    "filepath": "a.py",
                    "insertions": 10,
                    "deletions": 2,
                    "total_churn": 12,
                },
                {
                    "commit_hash": "h2",
                    "filepath": "b.py",
                    "insertions": 100,
                    "deletions": 50,
                    "total_churn": 150,
                },
                {
                    "commit_hash": "h3",
                    "filepath": "a.py",
                    "insertions": 5,
                    "deletions": 1,
                    "total_churn": 6,
                },
            ]
        )
        type(mock_cache).file_changes = PropertyMock(return_value=mock_changes)

        result_df = calculate_file_churn_cached(mock_cache)
        assert len(result_df) == 2
        # b.py should be first due to higher churn
        assert result_df.iloc[0]["filepath"] == "b.py"
        assert result_df.iloc[0]["total_churn"] == 150
        assert result_df.iloc[0]["commit_count"] == 1

        # Check aggregated values for a.py
        a_py_row = result_df[result_df["filepath"] == "a.py"].iloc[0]
        assert a_py_row["total_insertions"] == 15
        assert a_py_row["total_deletions"] == 3
        assert a_py_row["total_churn"] == 18
        assert a_py_row["commit_count"] == 2
