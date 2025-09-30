# git_cache.py

import pandas as pd
from git import Repo, GitCommandError
from typing import List, Dict
from collections import defaultdict
import re
import sys


class CommitDataCache:
    """
    Processes a Git repository in a single pass to create an efficient,
    in-memory cache of commit, file change, and file presence data.

    This avoids expensive, repeated Git operations for different analyses by
    pre-calculating all necessary data and storing it in pandas DataFrames.
    """

    def __init__(
        self, repo: Repo, max_commits: int = None, progress_interval: int = 1000
    ):
        """
        Initializes and processes the repository.

        Args:
            repo: The git.Repo object to analyze.
            max_commits: The maximum number of commits to process.
            progress_interval: Print progress every N commits (default: 1000)
        """
        self.repo = repo
        self.progress_interval = progress_interval
        self._commits_df = pd.DataFrame()
        self._file_changes_df = pd.DataFrame()
        self._file_presence_df = pd.DataFrame()  # MODIFIED: Added for snapshots
        self._process_commits(max_commits)

    def _process_commits(self, max_commits: int = None):
        """
        Iterates through commit history once to extract all necessary data.
        This is the core of the single-pass optimization. It uses direct
        git commands for diffing and tree listing to maximize performance.
        """
        commits_data = []
        file_changes_data = []
        file_presence_data = []  # MODIFIED: Added list for snapshot data

        try:
            commits = list(self.repo.iter_commits("HEAD", max_count=max_commits))
        except (ValueError, GitCommandError):
            return

        total_commits = len(commits)
        if total_commits == 0:
            return

        print(f"      Total commits to process: {total_commits:,}")
        print(f"      Progress updates every {self.progress_interval:,} commits")
        print()

        for idx, commit in enumerate(commits, start=1):
            # --- Metadata Extraction ---
            commits_data.append(
                {
                    "hash": commit.hexsha,
                    "author_name": commit.author.name,
                    "author_email": commit.author.email,
                    "date": commit.committed_datetime,
                    "message": commit.message,
                }
            )

            # --- Optimized Stats & File Change Extraction ---
            if not commit.parents:
                # Initial commit logic... (no changes here)
                try:
                    tree_output = self.repo.git.execute(
                        ["git", "ls-tree", "-r", commit.hexsha]
                    )
                    for line in tree_output.splitlines():
                        parts = line.split("\t")
                        if len(parts) == 2:
                            file_changes_data.append(
                                {
                                    "commit_hash": commit.hexsha,
                                    "filepath": parts[1],
                                    "change_type": "A",
                                    "insertions": 0,
                                    "deletions": 0,
                                }
                            )
                except GitCommandError:
                    for blob in commit.tree.traverse():
                        if blob.type == "blob":
                            file_changes_data.append(
                                {
                                    "commit_hash": commit.hexsha,
                                    "filepath": blob.path,
                                    "change_type": "A",
                                    "insertions": 0,
                                    "deletions": 0,
                                }
                            )
            else:
                # Diff processing logic... (no changes here)
                parent = commit.parents[0]
                try:
                    status_output = self.repo.git.diff(
                        parent.hexsha, commit.hexsha, "--name-status", "-M", "-C"
                    )
                    numstat_output = self.repo.git.diff(
                        parent.hexsha, commit.hexsha, "--numstat", "-M", "-C"
                    )
                    change_types = {}
                    for line in status_output.splitlines():
                        if not line.strip():
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            change_types[parts[-1]] = parts[0][0]
                    for line in numstat_output.splitlines():
                        parts = line.split("\t")
                        if len(parts) != 3:
                            continue
                        ins, dels, path = parts
                        if "=>" in path:
                            match = re.search(r"\{(.*?)\s=>\s(.*?)\}", path)
                            if match:
                                filepath = path.replace(match.group(0), match.group(2))
                            else:
                                filepath = path.split("=>")[-1].strip()
                        else:
                            filepath = path
                        file_changes_data.append(
                            {
                                "commit_hash": commit.hexsha,
                                "filepath": filepath,
                                "change_type": change_types.get(filepath, "M"),
                                "insertions": int(ins) if ins != "-" else 0,
                                "deletions": int(dels) if dels != "-" else 0,
                            }
                        )
                except GitCommandError as e:
                    print(
                        f"      Warning: Could not process diff for commit {commit.hexsha[:7]}: {e}",
                        file=sys.stderr,
                    )
                    continue

            # --- NEW: Snapshot Extraction for Core File Analysis ---
            try:
                tree_output = self.repo.git.ls_tree("-r", "--name-only", commit.hexsha)
                for filepath in tree_output.splitlines():
                    file_presence_data.append(
                        {"commit_hash": commit.hexsha, "filepath": filepath}
                    )
            except GitCommandError:
                print(
                    f"      Warning: Could not list files for commit {commit.hexsha[:7]}",
                    file=sys.stderr,
                )
                continue

            # --- Progress logging ---
            if idx % self.progress_interval == 0:
                percentage = (idx / total_commits) * 100
                print(
                    f"      Progress: {idx:,} / {total_commits:,} commits ({percentage:.1f}%)",
                    flush=True,
                )

        if total_commits % self.progress_interval != 0:
            print(
                f"      Progress: {total_commits:,} / {total_commits:,} commits (100.0%)",
                flush=True,
            )

        # --- Create DataFrames ---
        if commits_data:
            self._commits_df = pd.DataFrame(commits_data)
            self._commits_df["date"] = pd.to_datetime(
                self._commits_df["date"], utc=True
            )

        if file_changes_data:
            self._file_changes_df = pd.DataFrame(file_changes_data)
            self._file_changes_df["total_churn"] = (
                self._file_changes_df["insertions"] + self._file_changes_df["deletions"]
            )

        # MODIFIED: Create the new file_presence DataFrame
        if file_presence_data:
            self._file_presence_df = pd.DataFrame(file_presence_data)

    @property
    def commits(self) -> pd.DataFrame:
        return self._commits_df

    @property
    def file_changes(self) -> pd.DataFrame:
        return self._file_changes_df

    # MODIFIED: Added property for the new DataFrame
    @property
    def file_presence(self) -> pd.DataFrame:
        """
        DataFrame of all files present in each commit (snapshots).
        Columns: commit_hash, filepath
        """
        return self._file_presence_df
