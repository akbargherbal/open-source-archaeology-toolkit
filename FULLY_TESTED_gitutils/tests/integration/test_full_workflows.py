"""
Integration tests for gitutils.py, testing end-to-end workflows
on realistic, programmatically generated Git repositories.
"""
import pytest
import pandas as pd
from gitutils import (
    get_repository_stats,
    build_commit_dataframe,
    find_core_files,
    get_complexity_timeline,
    find_reorganization_commits,
    get_contributor_stats,
    find_abandoned_files,
    plot_complexity_timeline,
)

class TestArchaeologyWorkflow:
    """Test a complete analysis workflow on the simple_repo fixture."""

    def test_basic_workflow_on_simple_repo(self, simple_repo):
        """
        End-to-end test: Load -> Get Stats -> Build DataFrame -> Analyze.
        """
        # 1. Get basic stats
        stats = get_repository_stats(simple_repo)
        assert stats['total_commits'] == 11
        assert stats['unique_authors'] == 1

        # 2. Build DataFrame
        df = build_commit_dataframe(simple_repo)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert 'hash' in df.columns

        # 3. Analyze core files
        core = find_core_files(simple_repo, threshold=0.9)
        assert len(core) >= 3  # README.md, main.py, utils.py

        # 4. Get complexity timeline
        timeline = get_complexity_timeline(simple_repo)
        assert len(timeline) == 11
        assert timeline['total_files'].iloc[-1] == 4 # Final file count

class TestAdvancedScenarios:
    """Test specific analysis functions on tailored repository fixtures."""

    def test_reorganization_detection(self, refactor_repo):
        """Verify detection of a major refactoring commit."""
        reorgs = find_reorganization_commits(refactor_repo, threshold=0.5)
        assert len(reorgs) == 1
        reorg_commit = reorgs[0]
        assert reorg_commit['files_moved'] == 2
        assert "Refactor: Move files" in reorg_commit['message']

    def test_contributor_analysis(self, complex_repo):
        """Verify multi-author analysis on the complex_repo."""
        stats_df = get_contributor_stats(complex_repo)
        assert len(stats_df) == 3  # Alice, Bob, Charlie
        # FIX: Cast the string output of the git command to an int for comparison.
        assert stats_df['commit_count'].sum() == int(complex_repo.git.rev_list('--count', 'HEAD'))
        # Alice should have the most commits
        assert 'Alice' in stats_df.iloc[0]['author_name']

    def test_abandoned_file_detection(self, abandoned_repo):
        """Verify detection of files that have not been changed recently."""
        # Threshold of 2 years (730 days)
        abandoned = find_abandoned_files(abandoned_repo, days_threshold=730)
        assert len(abandoned) > 0
        assert abandoned[0]['filepath'] == 'main.py'
        assert abandoned[0]['days_since_change'] > 365 * 2

class TestVisualization:
    """Smoke tests for plotting functions to ensure they don't crash."""

    def test_plot_complexity_timeline_runs(self, simple_repo, mock_plt):
        """Ensure the complexity timeline plot function executes without error."""
        try:
            plot_complexity_timeline(simple_repo)
        except Exception as e:
            pytest.fail(f"plot_complexity_timeline raised an exception: {e}")