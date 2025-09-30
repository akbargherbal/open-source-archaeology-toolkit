# tests/unit/test_data_extraction.py

"""
Unit tests for data extraction, focusing on the CommitDataCache accessors.
"""

import pytest
import pandas as pd
from gitutils import build_commit_dataframe


class TestDataExtractionFromCache:
    """Tests data extraction from the CommitDataCache."""

    def test_build_commit_dataframe_on_cache(self, simple_repo_cache):
        """The accessor function should return the commits DataFrame."""
        df = build_commit_dataframe(simple_repo_cache)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert df.equals(simple_repo_cache.commits)

    def test_build_commit_dataframe_empty_repo(self, empty_repo_cache):
        """Building a dataframe from an empty repo returns an empty dataframe."""
        df = build_commit_dataframe(empty_repo_cache)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_cache_properties_on_simple_repo(self, simple_repo_cache):
        """Verify the cache properties on a known repository."""
        commits_df = simple_repo_cache.commits
        files_df = simple_repo_cache.file_changes
        presence_df = simple_repo_cache.file_presence

        # Check commits df
        assert isinstance(commits_df, pd.DataFrame)
        assert len(commits_df) == 11
        assert "author_name" in commits_df.columns
        assert commits_df["author_name"].iloc[0] == "Alice"

        # Check file changes df
        assert isinstance(files_df, pd.DataFrame)
        assert not files_df.empty
        assert "filepath" in files_df.columns
        assert "total_churn" in files_df.columns

        # Verify a known change: the commit that added 'docs/guide.md'
        add_docs_commit_hash = commits_df[commits_df["message"] == "Add docs"][
            "hash"
        ].iloc[0]
        doc_change = files_df[files_df["commit_hash"] == add_docs_commit_hash]
        assert len(doc_change) == 1
        assert doc_change.iloc[0]["filepath"] == "docs/guide.md"
        assert doc_change.iloc[0]["change_type"] == "A"

        # Check file presence df
        assert isinstance(presence_df, pd.DataFrame)
        assert not presence_df.empty
