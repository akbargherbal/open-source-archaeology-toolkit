import pandas as pd
import os
import sys
import time
from datetime import datetime
from gitutils import (
    load_repository,
    find_core_files_cached,  # MODIFIED: Import the new cached function
    calculate_file_churn,
    get_contributor_stats,
    find_volatile_files,
)
from git_cache import CommitDataCache

# --- Configuration ---

REPO_PATH = "../REPOS/django-cms"
repo_name = os.path.basename(REPO_PATH.rstrip("/"))
CACHE_FILE = f"{repo_name}-cache.pkl"
LOG_FILE = f"{repo_name}-analysis.log"

print(CACHE_FILE)
print(LOG_FILE)


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_print(message, logger=None, include_timestamp=False):
    if include_timestamp:
        timestamp = get_timestamp()
        message = f"[{timestamp}] {message}"
    print(message, flush=True)


def run_analysis():
    logger = Logger(LOG_FILE)
    sys.stdout = logger

    try:
        log_print("=" * 70)
        log_print("Git Repository Analysis - Started", include_timestamp=True)
        log_print("=" * 70)

        # --- Step 1: Load the Repository ---
        log_print(
            f"\n[1/4] Loading repository from: {REPO_PATH}", include_timestamp=True
        )
        try:
            repo = load_repository(REPO_PATH)
            log_print("✓ Repository loaded successfully.", include_timestamp=True)
        except Exception as e:
            log_print(f"✗ Error loading repository: {e}", include_timestamp=True)
            return

        # --- Step 2: Create or Load the CommitDataCache ---
        cache = None
        if os.path.exists(CACHE_FILE):
            log_print(
                f"\n[2/4] Found existing cache file: '{CACHE_FILE}'",
                include_timestamp=True,
            )
            log_print("      Loading cache from disk...")
            start_time = time.time()
            try:
                cache_data = pd.read_pickle(CACHE_FILE)
                cache = CommitDataCache.__new__(CommitDataCache)
                cache._commits_df = cache_data["commits"]
                cache._file_changes_df = cache_data["file_changes"]
                # MODIFIED: Safely load new data for backward compatibility
                cache._file_presence_df = cache_data.get(
                    "file_presence", pd.DataFrame()
                )
                cache.repo = repo
                elapsed = time.time() - start_time
                log_print(
                    f"✓ Cache loaded in {elapsed:.2f} seconds.", include_timestamp=True
                )
            except Exception as e:
                log_print(f"✗ Could not load cache: {e}", include_timestamp=True)
                log_print("      Will re-process repository.")
                cache = None

        if cache is None:
            log_print(
                f"\n[2/4] No valid cache found. Processing repository commits...",
                include_timestamp=True,
            )
            log_print("      This may take several minutes for large repositories...")
            log_print(f"      Progress updates will appear below and in '{LOG_FILE}'")
            log_print("")

            start_time = time.time()
            cache = CommitDataCache(repo)
            elapsed = time.time() - start_time

            log_print(
                f"\n✓ Repository processing complete in {elapsed:.2f} seconds.",
                include_timestamp=True,
            )
            log_print(f"      Saving cache to '{CACHE_FILE}'...")

            try:
                # MODIFIED: Save all DataFrames, including the new one
                cache_data = {
                    "commits": cache.commits,
                    "file_changes": cache.file_changes,
                    "file_presence": cache.file_presence,
                }
                pd.to_pickle(cache_data, CACHE_FILE)
                log_print("✓ Cache saved successfully.", include_timestamp=True)
            except Exception as e:
                log_print(f"✗ Error saving cache: {e}", include_timestamp=True)

        log_print("\n" + "-" * 70)
        log_print("Cache Summary:")
        log_print(f"  • Total commits processed: {len(cache.commits):,}")
        log_print(f"  • Total file changes found: {len(cache.file_changes):,}")
        # MODIFIED: Added summary for new data
        log_print(f"  • Total file presence records: {len(cache.file_presence):,}")
        log_print("-" * 70)

        # --- Step 3: Run Fast Analyses Using the Cache ---
        log_print(
            "\n[3/4] Running Performance-Optimized Analyses", include_timestamp=True
        )
        log_print("=" * 70)

        # Analysis 1: File Churn (No changes needed)
        log_print(
            "\n▸ Analysis 1: File Churn (Insertions + Deletions)",
            include_timestamp=True,
        )
        start_time = time.time()
        churn_df = calculate_file_churn(cache)
        elapsed = time.time() - start_time
        log_print(f"  Completed in {elapsed:.4f} seconds")
        log_print("\n  Top 5 files by total churn:")
        log_print(churn_df.head().to_string(index=False))

        # Analysis 2: Volatile Files (No changes needed)
        volatile_threshold = 200
        log_print(
            f"\n▸ Analysis 2: Volatile Files (changed >= {volatile_threshold} times)",
            include_timestamp=True,
        )
        start_time = time.time()
        volatile_df = find_volatile_files(cache, min_commits=volatile_threshold)
        elapsed = time.time() - start_time
        log_print(f"  Completed in {elapsed:.4f} seconds")
        log_print("\n  Top 5 most frequently modified files:")
        log_print(volatile_df.head().to_string(index=False))

        # Analysis 3: Contributor Statistics (No changes needed)
        log_print("\n▸ Analysis 3: Contributor Statistics", include_timestamp=True)
        start_time = time.time()
        contributors_df = get_contributor_stats(cache)
        elapsed = time.time() - start_time
        log_print(f"  Completed in {elapsed:.4f} seconds")
        log_print("\n  Top 5 contributors by commit count:")
        columns_to_show = [
            "author_name",
            "commit_count",
            "total_insertions",
            "total_deletions",
            "first_commit_date",
            "last_commit_date",
        ]
        log_print(contributors_df.head()[columns_to_show].to_string(index=False))

        # --- Step 4: Run FULLY CACHED Core File Analysis ---
        log_print(
            "\n[4/4] Running Fully-Cached Core File Analysis", include_timestamp=True
        )
        log_print("=" * 70)

        try:
            log_print(
                "\n▸ Analysis 4: Core Files (present in >98% of commits)",
                include_timestamp=True,
            )
            start_time = time.time()
            # MODIFIED: Call the new, fast, cached function
            core_files_df = find_core_files_cached(cache, threshold=0.98)
            elapsed = time.time() - start_time
            log_print(
                f"  Completed in {elapsed:.4f} seconds"
            )  # MODIFIED: Use .4f for fast functions
            log_print("\n  Top 5 core files:")
            # MODIFIED: Print directly from the resulting DataFrame
            log_print(core_files_df.head().to_string(index=False))
        except Exception as e:
            log_print(
                f"✗ Could not run core file analysis: {e}", include_timestamp=True
            )

        log_print("\n" + "=" * 70)
        log_print("Analysis Complete!", include_timestamp=True)
        log_print(f"Full log saved to: {LOG_FILE}")
        log_print("=" * 70)

    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n✓ Complete log written to: {LOG_FILE}")


if __name__ == "__main__":
    run_analysis()
