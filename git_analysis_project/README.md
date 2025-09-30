# Git Analysis Toolkit: A Software Archaeology Framework

This project provides a high-performance Python toolkit for conducting "software archaeology"—the systematic analysis of a Git repository's history to understand its structure, evolution, and social dynamics.

It is designed to overcome the performance limitations of traditional Git analysis by performing a single, efficient pass over the repository's history and loading it into memory for rapid, interactive querying.

## The Problem

Analyzing the history of a large code repository often involves running numerous slow `git` commands repeatedly. Answering questions like "Which files change the most?" or "Who were the most active contributors in the project's second year?" can be painfully slow and discourages deep, exploratory analysis.

## The Solution: Single-Pass Caching

This toolkit solves the performance problem with its central component: the `CommitDataCache`. When initialized, it processes the entire Git history once, extracting all relevant data into a set of optimized, in-memory pandas DataFrames.

All subsequent analyses are performed on this fast, in-memory cache, enabling you to query years of project history in milliseconds.

## Features

- **High-Performance Caching:** Processes repositories with thousands of commits in a single pass to build an in-memory analysis cache.
- **Core File Analysis:** Identify the "load-bearing walls" of your codebase—files that have existed for most of the project's life.
- **Code Churn & Volatility:** Pinpoint hotspot files that are rewritten frequently or have high change rates.
- **Contributor Analysis:** Get detailed statistics on author contributions, including commit counts, active periods, and total lines changed.
- **Semantic Commit Analysis:** Find all commits related to bug fixes, feature implementations, or other keywords.
- **Complexity Timelines:** Visualize how the number of files in the repository has grown over time.
- **And more:** The toolkit is built to be easily extensible with new queries and visualizations.

## Getting Started

### Prerequisites

- Python 3.8+
- Git installed on your system

### Installation

1.  Clone this repository:

    ```bash
    git clone https://github.com/your-username/git-analysis-project.git
    cd git-analysis-project
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Example Usage

The following example demonstrates a complete workflow: loading a repository, building the cache, and running several analyses.

```python
from git import Repo
from git_cache import CommitDataCache
import gitutils

# --- 1. Load the Repository and Build the Cache ---
# This is the only slow step. It processes the entire Git history.
print("Loading repository and building cache...")
repo = gitutils.load_repository('/path/to/your/git/repo')
cache = CommitDataCache(repo)
print("Cache built successfully!")


# --- 2. Run Fast, In-Memory Analyses ---
# All subsequent calls use the cache and are extremely fast.

# Find the 10 files that have existed for the longest time
print("\n--- Core Architectural Files (present in >90% of commits) ---")
core_files = gitutils.find_core_files_cached(cache, threshold=0.9)
print(core_files.head(10))

# Find the 10 files with the highest total churn (insertions + deletions)
print("\n--- Top 10 Files by Code Churn ---")
churn_df = gitutils.calculate_file_churn_cached(cache)
print(churn_df.head(10))

# Get statistics for the top 5 contributors
print("\n--- Top 5 Contributors by Commit Count ---")
contributors_df = gitutils.get_contributor_stats_cached(cache)
print(contributors_df.head(5))

# Find all commits that contain the word "fix" or "bug"
print("\n--- Recent Bug Fix Commits ---")
bug_fixes = gitutils.find_bug_fix_commits_cached(cache)
print(bug_fixes[['date', 'author_name', 'message']].head(5))

# Plot the growth of the repository over time
print("\nGenerating repository complexity plot...")
gitutils.plot_complexity_timeline(cache, save_path="complexity_timeline.png")
print("Plot saved to complexity_timeline.png")

```

## Project Structure

- `git_cache.py`: Contains the `CommitDataCache` class responsible for the one-time data extraction from the Git repository.
- `gitutils.py`: A library of analysis functions that operate on the `CommitDataCache` object. This is where you will find functions for churn, contributor stats, etc.
- `tests/`: A comprehensive test suite with unit, integration, and performance tests to ensure reliability and correctness.

## Running Tests

The project maintains a high standard of code quality through a robust test suite. To run the tests, use `pytest`:

```bash
pytest -v
```

## Future Work

- **Enhanced Scalability:** For extremely large repositories (e.g., the Linux kernel), the in-memory approach may be insufficient. Future work will explore disk-backed caching strategies (e.g., using Parquet files) to support larger-than-memory datasets.
- **Expanded Test Coverage:** Increase test coverage for the repo-based utility functions in `gitutils.py`.
- **More Analyses:** Add new analysis modules for detecting code coupling, architectural layers, and dependency evolution.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
