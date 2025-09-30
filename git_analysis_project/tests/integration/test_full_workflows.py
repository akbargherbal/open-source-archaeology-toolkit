# tests/integration/test_full_workflows.py

"""
Integration tests for gitutils, testing end-to-end workflows
on realistic, programmatically generated Git repositories.
"""
import pytest
import pandas as pd
from gitutils import (
    get_repository_stats,
    build_commit_dataframe,
    calculate_file_churn_cached,
    get_contributor_stats_cached,
    find_reorganization_commits,
    find_abandoned_files,
    plot_complexity_timeline,
)


class TestCachedWorkflow:
    """Test a complete analysis workflow using the CommitDataCache."""

    def test_basic_workflow_on_simple_repo(self, simple_repo_cache, simple_repo):
        """
        End-to-end test: Load -> Get Stats -> Build DataFrame -> Analyze.
        """
        # 1. Get basic stats (still uses repo object for current state)
        stats = get_repository_stats(simple_repo)
        assert stats["total_commits"] == 11
        assert stats["unique_authors"] == 1

        # 2. Build DataFrame from cache
        df = build_commit_dataframe(simple_repo_cache)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert "hash" in df.columns

        # 3. Analyze churn from cache
        churn_df = calculate_file_churn_cached(simple_repo_cache)
        assert not churn_df.empty
        # 'main.py' is the most changed file
        assert churn_df.iloc[0]["filepath"] == "main.py"
        # It was part of the initial commit + 9 updates
        assert churn_df.iloc[0]["commit_count"] == 10


class TestAdvancedScenarios:
    """Test specific analysis functions on tailored repository fixtures."""

    def test_reorganization_detection(self, refactor_repo):
        """Verify detection of a major refactoring commit (Repo-based)."""
        reorgs = find_reorganization_commits(refactor_repo, threshold=0.5)
        assert len(reorgs) == 1
        reorg_commit = reorgs[0]
        assert "Refactor: Move files" in reorg_commit["message"]
        assert reorg_commit["files_moved"] == 2

    def test_contributor_analysis_cached(self, complex_repo_cache):
        """Verify multi-author analysis on the complex_repo using the cache."""
        stats_df = get_contributor_stats_cached(complex_repo_cache)
        assert len(stats_df) == 3  # Alice, Bob, Charlie

        # Alice authored/committed the merge, so she should have the most commits
        assert "Alice" in stats_df.iloc[0]["author_name"]
        assert (
            stats_df["commit_count"].sum() >= 5
        )  # Check for a reasonable number of commits

    def test_abandoned_file_detection(self, abandoned_repo):
        """Verify detection of old files that have not been changed (Repo-based)."""
        abandoned = find_abandoned_files(abandoned_repo, days_threshold=730)
        assert len(abandoned) == 1
        assert abandoned[0]["filepath"] == "main.py"
        assert abandoned[0]["days_since_change"] > 365 * 4  # Committed in early 2020


class TestVisualization:
    """Smoke tests for plotting functions to ensure they don't crash."""

    def test_plot_complexity_timeline_runs_cached(self, simple_repo_cache, mock_plt):
        """Ensure the complexity timeline plot executes without error using cache."""
        try:
            plot_complexity_timeline(simple_repo_cache)
        except Exception as e:
            pytest.fail(f"plot_complexity_timeline raised an exception: {e}")
