"""Unit tests for repository loading and basic stats functions."""
import pytest
from git import InvalidGitRepositoryError, NoSuchPathError
from gitutils import load_repository, get_repository_stats

class TestRepositoryLoading:
    """Tests for loading repositories and getting initial stats."""

    def test_load_repository_valid_path(self, simple_repo):
        """Should load a valid repository without errors."""
        repo = load_repository(simple_repo.working_dir)
        assert repo.working_dir == simple_repo.working_dir

    def test_load_repository_invalid_path(self):
        """Should raise NoSuchPathError for a non-existent path."""
        with pytest.raises(NoSuchPathError):
            load_repository("/non/existent/path")

    def test_load_repository_not_a_repo(self, tmp_path):
        """Should raise InvalidGitRepositoryError for a directory that is not a repo."""
        (tmp_path / "some_file.txt").touch()
        with pytest.raises(InvalidGitRepositoryError):
            load_repository(tmp_path)

    def test_get_repository_stats_simple_repo(self, simple_repo):
        """Verify stats from a simple, known repository."""
        stats = get_repository_stats(simple_repo)
        assert stats["total_commits"] == 11
        assert stats["unique_authors"] == 1
        assert stats["total_files_current"] == 4

    def test_get_repository_stats_empty_repo(self, empty_repo):
        """Verify stats from an empty repository are all zero."""
        stats = get_repository_stats(empty_repo)
        assert stats["total_commits"] == 0
        assert stats["unique_authors"] == 0
        assert stats["total_files_current"] == 0