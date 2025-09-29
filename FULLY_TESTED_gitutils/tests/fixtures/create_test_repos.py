"""
Programmatic generation of Git repositories for testing.
This script creates various repository scenarios (simple, complex, etc.)
to be used as fixtures in integration tests.
"""

import shutil
from pathlib import Path
from datetime import datetime, timezone
from git import Repo, Actor


# Helper functions for repository creation
def create_file(path: Path, content: str):
    """Creates a file with the given content, ensuring parent dirs exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def modify_file(path: Path, new_content: str):
    """Appends new content to an existing file."""
    with path.open("a") as f:
        f.write(f"\n{new_content}")


def commit(repo: Repo, message: str, author: Actor, commit_date: datetime):
    """Creates a commit with a specific message, author, and date."""
    repo.index.commit(
        message,
        author=author,
        committer=author,
        commit_date=commit_date,
        author_date=commit_date,
    )


# Core repository generation functions
def create_simple_repo(path: Path):
    """
    Creates a repository with a linear history.
    - 11 commits, 4 files, 1 author
    """
    if path.exists():
        shutil.rmtree(path)
    repo = Repo.init(path)
    alice = Actor("Alice", "alice@example.com")

    # Commit 1
    create_file(path / "README.md", "# Simple Repo")
    create_file(path / "main.py", "print('hello')")
    create_file(path / "utils.py", "# Utilities")
    repo.index.add(["README.md", "main.py", "utils.py"])
    commit(
        repo,
        "Initial commit",
        alice,
        datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    # Commits 2-10
    for i in range(2, 11):
        modify_file(path / "main.py", f"print('Update {i}')")
        repo.index.add(["main.py"])
        commit(
            repo,
            f"Update {i}",
            alice,
            datetime(2020, 1, i, 12, 0, 0, tzinfo=timezone.utc),
        )

    # Commit 11
    create_file(path / "docs/guide.md", "Documentation")
    repo.index.add(["docs/guide.md"])
    commit(
        repo, "Add docs", alice, datetime(2020, 1, 11, 12, 0, 0, tzinfo=timezone.utc)
    )


def create_complex_repo(path: Path):
    """
    Creates a repository with branches, merges, and multiple authors.
    - ~20 commits, ~5 files, 3 authors
    """
    if path.exists():
        shutil.rmtree(path)
    repo = Repo.init(path)

    # FIX: Set a local config for the test repo to ensure merge commits
    # are authored by a known user, making tests deterministic.
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Alice")
        cw.set_value("user", "email", "alice@example.com")

    authors = {
        "alice": Actor("Alice", "alice@example.com"),
        "bob": Actor("Bob", "bob@example.com"),
        "charlie": Actor("Charlie", "charlie@example.com"),
    }

    # Initial commit on main (will default to 'master' or configured name)
    create_file(path / "main.py", "import core")
    repo.index.add(["main.py"])
    commit(
        repo,
        "Initial commit",
        authors["alice"],
        datetime(2021, 1, 1, tzinfo=timezone.utc),
    )

    # Feature branch by Bob
    repo.git.checkout("-b", "feature-x")
    create_file(path / "feature_x.py", "# Feature X")
    repo.index.add(["feature_x.py"])
    commit(
        repo, "Add Feature X", authors["bob"], datetime(2021, 1, 5, tzinfo=timezone.utc)
    )

    # Work on main by Alice and Charlie
    # FIX: Use 'master' as it is a more common default than 'main'.
    repo.git.checkout("master")
    create_file(path / "core.py", "# Core logic")
    repo.index.add(["core.py"])
    commit(
        repo,
        "Add core module",
        authors["alice"],
        datetime(2021, 1, 8, tzinfo=timezone.utc),
    )

    # Merge feature branch
    repo.git.merge("feature-x", "-m", "Merge feature-x")

    # More commits
    for i in range(5):
        author = authors[list(authors.keys())[i % 3]]
        modify_file(path / "core.py", f"\n# Change {i}")
        repo.index.add(["core.py"])
        commit(
            repo,
            f"Core update {i}",
            author,
            datetime(2021, 2, 1 + i, tzinfo=timezone.utc),
        )


def create_refactor_repo(path: Path):
    """Creates a repository with a major file reorganization commit."""
    if path.exists():
        shutil.rmtree(path)
    repo = Repo.init(path)
    author = Actor("Refactorer", "refactor@example.com")

    # Create initial files
    (path / "old_module").mkdir()
    create_file(path / "old_module" / "file1.py", "pass")
    create_file(path / "old_module" / "file2.py", "pass")
    repo.index.add(["old_module/file1.py", "old_module/file2.py"])
    commit(repo, "Add old module", author, datetime(2022, 1, 1, tzinfo=timezone.utc))

    # The refactor commit
    (path / "new_core").mkdir()
    repo.git.mv("old_module/file1.py", "new_core/file1.py")
    repo.git.mv("old_module/file2.py", "new_core/file2.py")
    commit(
        repo,
        "Refactor: Move files to new_core",
        author,
        datetime(2022, 1, 21, tzinfo=timezone.utc),
    )


def create_abandoned_repo(path: Path):
    """Creates a repository with no recent activity."""
    if path.exists():
        shutil.rmtree(path)
    repo = Repo.init(path)
    author = Actor("Dev", "dev@example.com")
    commit(repo, "Initial commit", author, datetime(2018, 1, 1, tzinfo=timezone.utc))
    create_file(path / "main.py", "# Active work")
    repo.index.add(["main.py"])
    commit(repo, "Final commit", author, datetime(2020, 2, 1, tzinfo=timezone.utc))
