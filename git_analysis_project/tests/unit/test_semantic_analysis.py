# tests/unit/test_semantic_analysis.py

"""
Unit tests for semantic analysis functions in gitutils.py.
These tests validate commit message analysis in isolation.
"""

import pytest
from unittest.mock import Mock, PropertyMock
import pandas as pd
from gitutils import find_bug_fix_commits_cached, find_feature_commits_cached


@pytest.fixture
def mock_cache():
    """Provides a mock CommitDataCache object for unit testing."""
    return Mock()


class TestSemanticCommitAnalysisCached:
    """Tests for identifying commits based on their message content."""

    @pytest.fixture
    def sample_commits_df(self):
        """A reusable DataFrame of sample commits for semantic analysis."""
        data = [
            {"hash": "h1", "message": "fix: Correct a major bug in the login flow"},
            {"hash": "h2", "message": "feat: Add new user profile page"},
            {"hash": "h3", "message": "refactor: Clean up the codebase"},
            {"hash": "h4", "message": "patch: Resolve issue with API endpoint"},
            {"hash": "h5", "message": "introduce a new feature for reporting"},
            {"hash": "h6", "message": "This commit fixes a typo"},
            {"hash": "h7", "message": "A regular commit with no keywords"},
        ]
        return pd.DataFrame(data)

    def test_find_bug_fix_commits_cached(self, mock_cache, sample_commits_df):
        """Verify that commits with bug-related keywords are found."""
        type(mock_cache).commits = PropertyMock(return_value=sample_commits_df)

        result_df = find_bug_fix_commits_cached(mock_cache)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        expected_hashes = {"h1", "h4", "h6"}
        assert set(result_df["hash"]) == expected_hashes
        assert "keywords_matched" in result_df.columns
        
        # Check that the correct keywords were identified
        fix_row = result_df[result_df['hash'] == 'h1']
        assert 'fix' in fix_row.iloc[0]['keywords_matched']
        
        patch_row = result_df[result_df['hash'] == 'h4']
        assert 'patch' in patch_row.iloc[0]['keywords_matched']

    def test_find_bug_fix_commits_cached_no_matches(self, mock_cache):
        """Ensure an empty DataFrame is returned when no commits match."""
        no_match_df = pd.DataFrame([{"message": "Just some regular work"}])
        type(mock_cache).commits = PropertyMock(return_value=no_match_df)

        result_df = find_bug_fix_commits_cached(mock_cache)
        assert result_df.empty

    def test_find_feature_commits_cached(self, mock_cache, sample_commits_df):
        """Verify that commits with feature-related keywords are found."""
        type(mock_cache).commits = PropertyMock(return_value=sample_commits_df)

        result_df = find_feature_commits_cached(mock_cache)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        expected_hashes = {"h2", "h5"}
        assert set(result_df["hash"]) == expected_hashes

        # Check that the correct keywords were identified
        feat_row = result_df[result_df['hash'] == 'h2']
        assert 'feat' in feat_row.iloc[0]['keywords_matched']

        introduce_row = result_df[result_df['hash'] == 'h5']
        assert 'introduce' in introduce_row.iloc[0]['keywords_matched']
        assert 'feature' in introduce_row.iloc[0]['keywords_matched']