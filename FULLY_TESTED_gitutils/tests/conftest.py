"""
Pytest fixtures for the entire test suite.

This file defines:
1. Session-scoped fixtures to generate test repositories once.
2. Function-scoped fixtures to provide Repo objects to tests.
3. Mocking fixtures for external dependencies like matplotlib.
"""
import pytest
from pathlib import Path
from git import Repo
from tests.fixtures.create_test_repos import (
    create_simple_repo,
    create_complex_repo,
    create_refactor_repo,
    create_abandoned_repo,
)

@pytest.fixture(scope="session")
def test_repos_dir(tmp_path_factory):
    """
    Creates all test repositories once per test session in a temporary directory.
    This is a performance optimization.
    """
    repos_dir = tmp_path_factory.mktemp("git_repos")
    
    repo_paths = {
        "simple": repos_dir / "simple",
        "complex": repos_dir / "complex",
        "refactor": repos_dir / "refactor",
        "abandoned": repos_dir / "abandoned",
    }
    
    create_simple_repo(repo_paths["simple"])
    create_complex_repo(repo_paths["complex"])
    create_refactor_repo(repo_paths["refactor"])
    create_abandoned_repo(repo_paths["abandoned"])
    
    return repo_paths

# Fixtures to provide specific repository objects to tests
@pytest.fixture
def simple_repo(test_repos_dir) -> Repo:
    """Provides a Repo object for the simple, linear-history repository."""
    return Repo(test_repos_dir["simple"])

@pytest.fixture
def complex_repo(test_repos_dir) -> Repo:
    """Provides a Repo object for the complex, multi-author, branched repository."""
    return Repo(test_repos_dir["complex"])

@pytest.fixture
def refactor_repo(test_repos_dir) -> Repo:
    """Provides a Repo object for the repository with a major refactoring event."""
    return Repo(test_repos_dir["refactor"])

@pytest.fixture
def abandoned_repo(test_repos_dir) -> Repo:
    """Provides a Repo object for the repository with no recent commits."""
    return Repo(test_repos_dir["abandoned"])

@pytest.fixture
def empty_repo(tmp_path) -> Repo:
    """Provides an empty, newly initialized repository."""
    return Repo.init(tmp_path)

@pytest.fixture
def mock_plt(monkeypatch):
    """
    Mocks matplotlib.pyplot to prevent plots from being displayed during tests.
    This is crucial for running tests in a non-interactive CI environment.
    """
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)