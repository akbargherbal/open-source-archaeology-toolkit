"""Unit tests for data extraction functions, using mocks."""

import pytest
from unittest.mock import Mock, PropertyMock
from datetime import datetime, timezone
import pandas as pd
from gitutils import (
    extract_commit_metadata,
    extract_commit_stats,
    build_commit_dataframe,
)


class TestDataExtraction:
    """Tests data extraction from mocked Git objects."""

    @pytest.fixture
    def mock_commit(self) -> Mock:
        """Creates a reusable mock of a git.Commit object."""
        commit = Mock()
        commit.hexsha = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        commit.author.name = "Test Author"
        commit.author.email = "test@example.com"
        commit.committed_datetime = datetime(2023, 1, 1, tzinfo=timezone.utc)
        commit.message = "feat: Implement new feature\n\nDetailed description."
        type(commit).parents = PropertyMock(return_value=[Mock()])  # Has one parent
        return commit

    def test_extract_commit_metadata(self, mock_commit):
        """Verify extraction of basic commit metadata."""
        data = extract_commit_metadata(mock_commit)
        assert data["hash"] == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        assert data["short_hash"] == "a1b2c3d"
        assert data["author_name"] == "Test Author"
        assert data["date"].year == 2023
        assert data["message_first_line"] == "feat: Implement new feature"
        assert data["parent_count"] == 1

    def test_extract_commit_stats(self, mock_commit):
        """Verify extraction of commit change statistics."""
        # Configure stats and diff mocks
        mock_commit.stats.total = {"insertions": 10, "deletions": 5, "lines": 15}
        mock_commit.stats.files = {"a.py": {}, "b.py": {}}

        diff_mock = Mock()
        type(diff_mock).new_file = PropertyMock(return_value=True)
        type(diff_mock).deleted_file = PropertyMock(return_value=False)
        type(diff_mock).renamed = PropertyMock(return_value=False)
        diff_mock.b_path = "a.py"
        mock_commit.parents[0].diff.return_value = [diff_mock]

        data = extract_commit_stats(mock_commit)
        assert data["insertions"] == 10
        assert data["deletions"] == 5
        assert data["total_changes"] == 15
        assert "a.py" in data["files_added"]

    def test_build_commit_dataframe_empty_repo(self, empty_repo):
        """Ensure building a dataframe from an empty repo returns an empty dataframe."""
        df = build_commit_dataframe(empty_repo)
        assert isinstance(df, pd.DataFrame)
        assert df.empty
