# gitutils.py

"""
gitutils.py - Helper functions for software archaeology investigations.

A collection of reusable utilities for analyzing Git repository history.
This module contains two types of functions:
1. Cache-Based: High-performance functions that operate on a pre-built
   CommitDataCache object for rapid, full-history analysis.
2. Repo-Based: Functions that perform targeted, live Git operations on a
   Repo object, suitable for specific queries that don't require a full cache.
"""

import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import pandas as pd
from git import Repo, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# The new caching class is a primary dependency
from git_cache import CommitDataCache


# ============================================================================
# REPOSITORY LOADING & CACHE CREATION
# ============================================================================


def load_repository(repo_path: str) -> Repo:
    """Loads a Git repository from a given path."""
    try:
        return Repo(repo_path, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as e:
        print(
            f"Error: Could not load repository at {repo_path}. Is it a valid git repo?"
        )
        raise e


def build_commit_dataframe(cache: CommitDataCache) -> pd.DataFrame:
    """
    (Accessor) Returns the main commits DataFrame from the cache.
    """
    return cache.commits


# ============================================================================
# CACHE-BASED ANALYSIS (High Performance)
# ============================================================================


def find_core_files_cached(
    cache: CommitDataCache, threshold: float = 0.9
) -> pd.DataFrame:
    """
    Identifies files present in X% of commits using the pre-computed cache.
    """
    if cache.file_presence.empty:
        return pd.DataFrame(columns=["filepath", "presence_percentage"])

    file_counts = cache.file_presence["filepath"].value_counts()
    total_commits = cache.commits["hash"].nunique()

    if total_commits == 0:
        return pd.DataFrame(columns=["filepath", "presence_percentage"])

    core_files_series = file_counts[file_counts / total_commits >= threshold]
    core_files_df = (core_files_series / total_commits).reset_index()
    core_files_df.columns = ["filepath", "presence_percentage"]

    return core_files_df.sort_values(
        "presence_percentage", ascending=False
    ).reset_index(drop=True)


def find_volatile_files_cached(
    cache: CommitDataCache, min_commits: int = 50
) -> pd.DataFrame:
    """Finds files with high change frequency using the pre-computed cache."""
    if cache.file_changes.empty:
        return pd.DataFrame(columns=["filepath", "commit_count"])

    change_counts = cache.file_changes["filepath"].value_counts()
    volatile = change_counts[change_counts >= min_commits]

    df = volatile.reset_index()
    df.columns = ["filepath", "commit_count"]
    return df


def calculate_file_churn_cached(cache: CommitDataCache) -> pd.DataFrame:
    """Calculates churn for each file across history using the cache."""
    if cache.file_changes.empty:
        return pd.DataFrame()

    churn_df = (
        cache.file_changes.groupby("filepath")
        .agg(
            total_insertions=("insertions", "sum"),
            total_deletions=("deletions", "sum"),
            total_churn=("total_churn", "sum"),
            commit_count=("commit_hash", "nunique"),
        )
        .reset_index()
    )
    churn_df["avg_churn_per_commit"] = (
        churn_df["total_churn"] / churn_df["commit_count"]
    )
    return churn_df.sort_values("total_churn", ascending=False).reset_index(drop=True)


def get_complexity_timeline_cached(cache: CommitDataCache) -> pd.DataFrame:
    """Tracks file count over time using the file presence cache."""
    if cache.file_presence.empty or cache.commits.empty:
        return pd.DataFrame()

    file_counts_per_commit = (
        cache.file_presence.groupby("commit_hash").size().rename("total_files")
    )

    timeline_df = pd.merge(
        cache.commits,
        file_counts_per_commit,
        left_on="hash",
        right_index=True,
        how="left",
    ).fillna(0)

    timeline_df["total_files"] = timeline_df["total_files"].astype(int)
    return (
        timeline_df[["date", "total_files", "hash"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def get_commit_frequency_timeline_cached(
    cache: CommitDataCache, bin_by: str = "month"
) -> pd.DataFrame:
    """Calculates commit frequency over time from the cache."""
    if cache.commits.empty:
        return pd.DataFrame(columns=["period", "commit_count"])

    df = cache.commits.copy()
    period_map = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
    period_code = period_map.get(bin_by.lower(), "M")

    freq_df = (
        df.set_index("date")
        .resample(period_code)
        .size()
        .reset_index(name="commit_count")
    )
    freq_df.rename(columns={"date": "period"}, inplace=True)
    return freq_df


def get_contributor_stats_cached(cache: CommitDataCache) -> pd.DataFrame:
    """Analyzes contribution patterns for all authors using the cache."""
    if cache.commits.empty or cache.file_changes.empty:
        return pd.DataFrame()

    merged_df = pd.merge(
        cache.commits, cache.file_changes, left_on="hash", right_on="commit_hash"
    )

    author_stats = (
        merged_df.groupby(["author_name", "author_email"])
        .agg(
            commit_count=("hash", "nunique"),
            first_commit_date=("date", "min"),
            last_commit_date=("date", "max"),
            total_insertions=("insertions", "sum"),
            total_deletions=("deletions", "sum"),
        )
        .reset_index()
    )
    author_stats["active_days"] = (
        author_stats["last_commit_date"] - author_stats["first_commit_date"]
    ).dt.days
    return author_stats.sort_values("commit_count", ascending=False).reset_index(
        drop=True
    )


def find_bug_fix_commits_cached(cache: CommitDataCache) -> pd.DataFrame:
    """Identifies commits likely to be bug fixes from the cache."""
    if cache.commits.empty:
        return pd.DataFrame()

    # FIX: Expanded keywords to include common variations (e.g., fixes, fixed).
    keywords = ["fix", "fixes", "fixed", "bug", "repair", "correct", "patch", "resolve"]
    # FIX: Use a non-capturing group (?:...) to silence the pandas UserWarning.
    pattern = r"\b(?:" + "|".join(keywords) + r")\b"

    df = cache.commits[
        cache.commits["message"].str.contains(pattern, case=False, regex=True)
    ].copy()
    df["keywords_matched"] = (
        df["message"]
        .str.findall(pattern, flags=re.IGNORECASE)
        .apply(lambda x: sorted(list(set(i.lower() for i in x))))
    )
    return df


def find_feature_commits_cached(cache: CommitDataCache) -> pd.DataFrame:
    """Identifies commits likely to be new features from the cache."""
    if cache.commits.empty:
        return pd.DataFrame()

    # FIX: Expanded keywords to include common variations.
    keywords = [
        "feature",
        "feat",
        "add",
        "adds",
        "added",
        "implement",
        "implements",
        "implemented",
        "create",
        "creates",
        "created",
        "introduce",
        "introduces",
        "introduced",
    ]
    # FIX: Use a non-capturing group (?:...) to silence the pandas UserWarning.
    pattern = r"\b(?:" + "|".join(keywords) + r")\b"

    df = cache.commits[
        cache.commits["message"].str.contains(pattern, case=False, regex=True)
    ].copy()
    df["keywords_matched"] = (
        df["message"]
        .str.findall(pattern, flags=re.IGNORECASE)
        .apply(lambda x: sorted(list(set(i.lower() for i in x))))
    )
    return df


def find_hotspot_files_cached(
    cache: CommitDataCache, recency_days: int = 180
) -> pd.DataFrame:
    """Finds files with high complexity and recent change frequency using the cache."""
    if cache.commits.empty or cache.file_changes.empty:
        return pd.DataFrame()

    threshold_date = datetime.now(timezone.utc) - timedelta(days=recency_days)
    recent_commits = cache.commits[cache.commits["date"] >= threshold_date]
    if recent_commits.empty:
        return pd.DataFrame()

    recent_changes = cache.file_changes[
        cache.file_changes["commit_hash"].isin(recent_commits["hash"])
    ]
    if recent_changes.empty:
        return pd.DataFrame()

    merged_df = pd.merge(
        recent_changes, recent_commits, left_on="commit_hash", right_on="hash"
    )

    hotspot_data = (
        merged_df.groupby("filepath")
        .agg(
            recent_commits=("commit_hash", "nunique"),
            total_churn=("total_churn", "sum"),
            unique_authors=("author_email", "nunique"),
        )
        .reset_index()
    )

    # Normalize to create a risk score
    hotspot_data["risk_score"] = (
        hotspot_data["recent_commits"] / hotspot_data["recent_commits"].max()
        + hotspot_data["total_churn"] / hotspot_data["total_churn"].max()
        + hotspot_data["unique_authors"] / hotspot_data["unique_authors"].max()
    )
    return hotspot_data.sort_values("risk_score", ascending=False).reset_index(
        drop=True
    )


def calculate_file_coupling_cached(
    cache: CommitDataCache, min_occurrences: int = 5
) -> pd.DataFrame:
    """Finds files that are frequently changed together, using the cache."""
    if cache.file_changes.empty:
        return pd.DataFrame()

    # Get file changes per commit, excluding merge commits (often noisy)
    commits_with_changes = cache.file_changes.groupby("commit_hash")["filepath"].apply(
        list
    )

    coupling = Counter()
    for files in commits_with_changes:
        if len(files) > 1:
            for pair in combinations(sorted(files), 2):
                coupling[pair] += 1

    records = [
        {"file1": pair[0], "file2": pair[1], "times_changed_together": count}
        for pair, count in coupling.items()
        if count >= min_occurrences
    ]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df.sort_values("times_changed_together", ascending=False).reset_index(
        drop=True
    )


# ============================================================================
# REPO-BASED ANALYSIS (Targeted Live Git Operations)
# ============================================================================


def get_repository_stats(repo: Repo) -> Dict:
    """Gets basic statistics about a repository's current state."""
    try:
        if not repo.head.is_valid():
            return defaultdict(int)
        all_commits = list(repo.iter_commits("HEAD"))
    except ValueError:
        return defaultdict(int)

    if not all_commits:
        return defaultdict(int)

    first_commit = all_commits[-1]
    authors = {c.author.email for c in all_commits}
    try:
        total_files = len(
            [blob for blob in repo.head.commit.tree.traverse() if blob.type == "blob"]
        )
    except Exception:
        total_files = 0

    age_in_days = (datetime.now(timezone.utc) - first_commit.committed_datetime).days

    return {
        "total_commits": len(all_commits),
        "total_branches": len(repo.branches),
        "total_tags": len(repo.tags),
        "first_commit_date": first_commit.committed_datetime,
        "last_commit_date": all_commits[0].committed_datetime,
        "age_in_days": age_in_days,
        "unique_authors": len(authors),
        "total_files_current": total_files,
    }


def find_stable_files(
    repo: Repo, days_unchanged: int = 365
) -> List[Tuple[str, datetime]]:
    """Finds files that haven't been modified in X days."""
    stable_files = []
    threshold_date = datetime.now(timezone.utc) - timedelta(days=days_unchanged)
    try:
        head_files = [
            blob.path
            for blob in repo.head.commit.tree.traverse()
            if blob.type == "blob"
        ]
    except ValueError:
        return []

    for filepath in head_files:
        try:
            last_commit = next(repo.iter_commits("HEAD", paths=filepath, max_count=1))
            if last_commit.committed_datetime < threshold_date:
                stable_files.append((filepath, last_commit.committed_datetime))
        except (StopIteration, GitCommandError):
            continue
    return sorted(stable_files, key=lambda item: item[1])


def find_reorganization_commits(repo: Repo, threshold: float = 0.2) -> List[Dict]:
    """Detects commits where many files were moved/renamed."""
    reorg_commits = []
    try:
        for commit in repo.iter_commits("HEAD"):
            if not commit.parents:
                continue
            diff = commit.parents[0].diff(commit)
            renamed_files = [d for d in diff if d.renamed_file]
            if not renamed_files:
                continue

            total_files_in_commit = len(
                [b for b in commit.tree.traverse() if b.type == "blob"]
            )
            if total_files_in_commit == 0:
                continue

            move_ratio = len(renamed_files) / total_files_in_commit
            if move_ratio >= threshold:
                reorg_commits.append(
                    {
                        "commit_hash": commit.hexsha,
                        "date": commit.committed_datetime,
                        "message": commit.message.split("\n")[0],
                        "files_moved": len(renamed_files),
                        "move_ratio": move_ratio,
                    }
                )
    except ValueError:
        pass
    return reorg_commits


def find_abandoned_files(repo: Repo, days_threshold: int = 730) -> List[Dict]:
    """Finds files that exist but haven't been touched in X days."""
    stable = find_stable_files(repo, days_unchanged=days_threshold)
    abandoned = []
    for file, last_mod in stable:
        try:
            last_commit = next(repo.iter_commits("HEAD", paths=file, max_count=1))
            abandoned.append(
                {
                    "filepath": file,
                    "last_modified": last_mod,
                    "days_since_change": (datetime.now(timezone.utc) - last_mod).days,
                    "last_author": last_commit.author.name,
                }
            )
        except (StopIteration, GitCommandError):
            continue
    return abandoned


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================


def plot_complexity_timeline(cache: CommitDataCache, save_path: str = None):
    """Creates and displays file count over time chart from cache."""
    df = get_complexity_timeline_cached(cache)
    if df.empty:
        print("No complexity data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["total_files"])
    ax.set_title("Repository Complexity Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Number of Files")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.autofmt_xdate()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_commit_frequency(
    cache: CommitDataCache, bin_by: str = "month", save_path: str = None
):
    """Creates commit frequency visualization from cache."""
    df = get_commit_frequency_timeline_cached(cache, bin_by=bin_by)
    if df.empty:
        print("No commit data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["period"], df["commit_count"], width=20)
    ax.set_title(f"Commit Frequency per {bin_by.capitalize()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Commits")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    fig.autofmt_xdate()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_churn_analysis(cache: CommitDataCache, top_n: int = 20, save_path: str = None):
    """Visualizes files with the highest churn from cache."""
    df = calculate_file_churn_cached(cache).head(top_n)
    if df.empty:
        print("No churn data to plot.")
        return

    df = df.sort_values("total_churn", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(df["filepath"], df["total_churn"], color="skyblue")
    ax.set_title(f"Top {top_n} Files by Churn")
    ax.set_xlabel("Total Churn (Insertions + Deletions)")
    ax.set_ylabel("File Path")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def calculate_age_in_days(start_date, end_date=None) -> int:
    """
    Calculate days between two dates (end defaults to now).
    """
    if end_date is None:
        # Ensure timezone awareness matches
        if start_date.tzinfo:
            end_date = datetime.now(timezone.utc)
        else:
            end_date = datetime.now()
    return (end_date - start_date).days


def format_date(date_obj, format_str: str = "%Y-%m-%d") -> str:
    """Formats datetime objects consistently."""
    return date_obj.strftime(format_str)


def filter_by_file_extension(files: List[str], extensions: List[str]) -> List[str]:
    """Filters a list of file paths by their extensions."""
    return [f for f in files if Path(f).suffix in extensions]


# gitutils.py

# ... (all other code remains the same) ...


def group_by_directory(files: List[str], depth: int = 1) -> Dict[str, List[str]]:
    """Groups a list of file paths by their parent directory at a specified depth."""
    groups = defaultdict(list)
    for file in files:
        path = Path(file)
        if len(path.parts) <= depth:
            groups["."].append(file)
        else:
            # FIX: Use as_posix() to ensure consistent forward-slash separators
            # on all operating systems, making tests platform-agnostic.
            key = Path(*path.parts[:depth]).as_posix()
            groups[key].append(file)
    return dict(groups)


def is_likely_test_file(filepath: str) -> bool:
    """Heuristic to determine if a file is a test file."""
    path = Path(filepath.lower())
    if len(path.parts) == 1:
        return path.name.startswith("test_") or path.name.endswith("_test.py")
    return (
        any("test" in part for part in path.parts)
        or path.name.startswith("test_")
        or path.name.endswith("_test.py")
    )


def is_likely_config_file(filepath: str) -> bool:
    """Heuristic to determine if a file is configuration."""
    config_extensions = [".json", ".yaml", ".yml", ".xml", ".ini", ".toml", ".cfg"]
    config_filenames = ["config", "configuration", "settings", "dockerfile", "makefile"]
    path = Path(filepath.lower())
    return (
        path.suffix in config_extensions
        or any(name in path.stem for name in config_filenames)
        or path.name in config_filenames
    )


def is_likely_documentation(filepath: str) -> bool:
    """Heuristic to determine if a file is documentation."""
    doc_extensions = [".md", ".rst", ".txt", ".docx"]
    doc_dirs = ["doc", "docs", "documentation"]
    path = Path(filepath.lower())
    return path.suffix in doc_extensions or any(part in doc_dirs for part in path.parts)
