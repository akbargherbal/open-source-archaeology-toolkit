"""
Enhanced Git Repository Metadata Extractor

This script extracts comprehensive metadata from a git repository and saves it
in JSON format for data analysis. Enhanced version includes additional
metrics for code quality, language analysis, advanced Git features, and
architectural persistence metrics. Useful to study Software/Repo Evolution.
"""

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import mimetypes

try:
    import pandas as pd
    import numpy as np

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    # This warning is now conditional based on the --architectural-metrics flag in main()


def run_git_command(repo_path: str, command: List[str]) -> str:
    """
    Execute a git command in the specified repository and return the output.
    """
    try:
        result = subprocess.run(
            ["git"] + command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        print("Error: 'git' command not found. Is Git installed and in your PATH?")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {' '.join(['git'] + command)}")
        print(f"Error: {e.stderr.strip()}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a valid filename.
    """
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def detect_language_from_extension(filepath: str) -> str:
    """
    Detect programming language from file extension.
    """
    ext_to_lang = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".h": "C/C++",
        ".hpp": "C++",
        ".cs": "C#",
        ".php": "PHP",
        ".rb": "Ruby",
        ".go": "Go",
        ".rs": "Rust",
        ".kt": "Kotlin",
        ".swift": "Swift",
        ".scala": "Scala",
        ".r": "R",
        ".m": "Objective-C",
        ".sh": "Shell",
        ".bash": "Bash",
        ".ps1": "PowerShell",
        ".bat": "Batch",
        ".html": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
        ".sass": "Sass",
        ".xml": "XML",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".sql": "SQL",
        ".dockerfile": "Dockerfile",
        ".makefile": "Makefile",
        ".md": "Markdown",
        ".tex": "LaTeX",
        ".vim": "Vim",
        ".lua": "Lua",
        ".pl": "Perl",
        ".dart": "Dart",
        ".elm": "Elm",
        ".ex": "Elixir",
        ".clj": "Clojure",
        ".hs": "Haskell",
        ".ml": "OCaml",
        ".f90": "Fortran",
    }

    # Handle special filenames
    if "dockerfile" in Path(filepath).name.lower():
        return "Dockerfile"
    if "makefile" in Path(filepath).name.lower():
        return "Makefile"

    ext = Path(filepath).suffix.lower()
    return ext_to_lang.get(ext, "Other")


def is_binary_file(filepath: str) -> bool:
    """
    Check if a file is likely binary based on MIME type and extension.
    """
    binary_extensions = {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".a",
        ".o",
        ".obj",
        ".lib",
        ".jar",
        ".war",
        ".ear",
        ".class",
        ".pyc",
        ".pyo",
        ".pyd",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".wav",
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".mkv",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".rar",
        ".7z",
        ".deb",
        ".rpm",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
    }

    ext = Path(filepath).suffix.lower()
    if ext in binary_extensions:
        return True

    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type and not mime_type.startswith("text/"):
        return True

    return False


def get_basic_repo_info(repo_path: str) -> Dict[str, Any]:
    """
    Extract basic repository information with enhancements.
    """
    repo_name = os.path.basename(os.path.abspath(repo_path))
    git_dir = Path(repo_path) / ".git"

    info = {
        "repo_name": repo_name,
        "repo_path": os.path.abspath(repo_path),
        "remote_url": run_git_command(
            repo_path, ["config", "--get", "remote.origin.url"]
        ),
        "current_branch": run_git_command(
            repo_path, ["rev-parse", "--abbrev-ref", "HEAD"]
        ),
        "total_commits": 0,
        "git_size_kb": 0,
        "working_tree_clean": False,
        "has_submodules": False,
        "default_branch": "",
        "repo_age_days": 0,
    }

    # Total commits
    total_commits_str = run_git_command(repo_path, ["rev-list", "--count", "HEAD"])
    if total_commits_str.isdigit():
        info["total_commits"] = int(total_commits_str)

    # Git directory size
    if git_dir.exists():
        try:
            git_dir_size = sum(
                f.stat().st_size for f in git_dir.rglob("*") if f.is_file()
            )
            info["git_size_kb"] = round(git_dir_size / 1024, 2)
        except OSError as e:
            print(f"Warning: Could not calculate repository size. Error: {e}")

    # Working tree status
    status_output = run_git_command(repo_path, ["status", "--porcelain"])
    info["working_tree_clean"] = len(status_output.strip()) == 0

    # Check for submodules
    submodules_file = Path(repo_path) / ".gitmodules"
    info["has_submodules"] = submodules_file.exists()

    # Default branch (try to get from remote)
    default_branch = run_git_command(
        repo_path, ["symbolic-ref", "refs/remotes/origin/HEAD"]
    )
    if default_branch:
        info["default_branch"] = default_branch.split("/")[-1]
    else:
        info["default_branch"] = info["current_branch"]

    # Repository age
    first_commit_date_str = run_git_command(
        repo_path, ["log", "--reverse", "--format=%cI", "-1"]
    )
    if first_commit_date_str:
        try:
            first_date = datetime.datetime.fromisoformat(first_commit_date_str)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            repo_age = now_utc - first_date
            info["repo_age_days"] = repo_age.days
        except (ValueError, IndexError) as e:
            print(
                f"Warning: Could not parse first commit date '{first_commit_date_str}'. Error: {e}"
            )
            pass

    return info


def get_commit_history(
    repo_path: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract commit history with enhanced information including diff stats.
    OPTIMIZED VERSION - reduces git command calls significantly.
    """
    # Use pipe delimiter instead of null bytes for easier parsing
    # Include parent hashes (%P) in the main format to avoid separate git calls
    format_string = "--pretty=format:%H|%an|%ae|%aI|%cn|%ce|%cI|%s|%P|%B"
    cmd = ["log", format_string, "--numstat"]  # Removed --all for performance
    if limit:
        cmd.extend(["-n", str(limit)])

    print(f"Running git command: git {' '.join(cmd)}")
    log_output = run_git_command(repo_path, cmd)
    if not log_output:
        return []

    commits = []
    lines = log_output.strip().split("\n")
    i = 0
    total_lines = len(lines)

    print(f"Processing {total_lines} lines of git output...")

    while i < total_lines:
        line = lines[i]

        # Check if this line starts with a 40-character hash (commit line)
        if len(line) >= 40 and re.match(r"^[0-9a-f]{40}\|", line):
            parts = line.split("|", 8)  # Split only first 8 pipes
            if len(parts) >= 9:
                commit = {
                    "hash": parts[0],
                    "author_name": parts[1],
                    "author_email": parts[2],
                    "author_date": parts[3],
                    "committer_name": parts[4],
                    "committer_email": parts[5],
                    "commit_date": parts[6],
                    "subject": parts[7],
                    "parent_hashes": parts[8],
                    "full_message": parts[9] if len(parts) > 9 else parts[7],
                    "is_merge": len(parts[8].split()) > 1 if parts[8] else False,
                    "message_length": len(parts[7]),
                    "files_changed": 0,
                    "insertions": 0,
                    "deletions": 0,
                    "lines_changed": 0,
                }

                # Parse numstat data that follows
                i += 1
                while i < total_lines:
                    if not lines[i].strip():  # Empty line
                        i += 1
                        break
                    if re.match(r"^[0-9a-f]{40}\|", lines[i]):  # Next commit
                        break

                    stat_line = lines[i]
                    if "\t" in stat_line:
                        stat_parts = stat_line.split("\t")
                        if len(stat_parts) >= 3:
                            insertions, deletions = stat_parts[0], stat_parts[1]
                            # Handle binary files where insertions/deletions are '-'
                            if insertions.isdigit():
                                commit["insertions"] += int(insertions)
                            if deletions.isdigit():
                                commit["deletions"] += int(deletions)
                            commit["files_changed"] += 1
                    i += 1

                commit["lines_changed"] = commit["insertions"] + commit["deletions"]
                commits.append(commit)

                # Progress indicator
                if len(commits) % 100 == 0:
                    print(f"Processed {len(commits)} commits...")

                continue
        i += 1

    print(f"Extracted {len(commits)} commits total")
    return commits


def calculate_architectural_metrics(
    commits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Calculate architectural persistence metrics similar to miner.py using pandas/numpy.
    This adds churn_rate, survival_rate, and asi to each commit.
    """
    if not HAS_PANDAS or not commits:
        return commits

    try:
        commits_df = pd.DataFrame(commits)

        commits_df["commit_datetime"] = pd.to_datetime(
            commits_df["commit_date"], errors="coerce"
        )

        # Drop commits where date parsing failed
        commits_df.dropna(subset=["commit_datetime"], inplace=True)
        if commits_df.empty:
            return commits

        repo_start = commits_df["commit_datetime"].min()
        repo_end = commits_df["commit_datetime"].max()
        total_repo_days = (repo_end - repo_start).days

        max_lines_changed = commits_df["lines_changed"].max()

        enhanced_commits = []
        for _, commit in commits_df.iterrows():
            commit_dict = commit.to_dict()

            age_days = (commit["commit_datetime"] - repo_start).days
            age_months = max(1, age_days / 30.44)

            churn_rate = commit["lines_changed"] / age_months
            survival_rate = age_days / total_repo_days if total_repo_days > 0 else 0

            normalized_churn = (
                commit["lines_changed"] / max_lines_changed
                if max_lines_changed > 0
                else 0
            )
            asi = survival_rate / (1 + normalized_churn)

            commit_dict["architectural_metrics"] = {
                "age_days": age_days,
                "age_months": round(age_months, 2),
                "churn_rate": round(churn_rate, 2),
                "survival_rate": round(survival_rate, 4),
                "asi": round(asi, 4),
            }

            # Remove pandas-specific objects for JSON serialization
            del commit_dict["commit_datetime"]
            enhanced_commits.append(commit_dict)

        return enhanced_commits

    except Exception as e:
        print(f"Warning: Could not calculate architectural metrics: {e}")
        # Return original commits on failure, ensuring no pandas objects remain
        original_commits = [c for c in commits if "commit_datetime" not in c]
        return original_commits


def get_file_changes(
    repo_path: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Extract file change statistics per commit with enhanced information.
    """
    cmd = ["log", "--name-status", "--pretty=format:COMMIT:%H%x00%cI", "--all"]
    if limit:
        cmd.extend(["-n", str(limit)])

    log_output = run_git_command(repo_path, cmd)
    if not log_output:
        return []

    changes = []
    current_commit_hash, current_date = None, None

    for line in log_output.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("COMMIT:"):
            _, commit_info = line.split(":", 1)
            parts = commit_info.split("\x00")
            if len(parts) == 2:
                current_commit_hash, current_date = parts
        elif current_commit_hash and "\t" in line:
            status, *paths = line.split("\t")
            file_path = paths[0]

            change = {
                "commit_hash": current_commit_hash,
                "commit_date": current_date,
                "change_type": status.strip(),
                "file_path": file_path,
                "old_file_path": paths[1] if len(paths) > 1 else "",
                "file_extension": Path(file_path).suffix.lower(),
                "language": detect_language_from_extension(file_path),
                "is_binary": is_binary_file(file_path),
                "directory_depth": len(Path(file_path).parents) - 1,
            }
            changes.append(change)

    return changes


def get_branch_info(repo_path: str) -> List[Dict[str, Any]]:
    """
    Extract enhanced information about all branches.
    """
    format_string = "%(refname:short)%x00%(objectname)%x00%(committerdate:iso)%x00%(authorname)%x00%(upstream:short)"
    # Query both local heads and remote heads
    cmd = ["for-each-ref", f"--format={format_string}", "refs/heads", "refs/remotes"]

    branch_output = run_git_command(repo_path, cmd)
    if not branch_output:
        return []

    branches = []
    current_branch = run_git_command(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    default_branch_ref = run_git_command(
        repo_path, ["symbolic-ref", "refs/remotes/origin/HEAD"]
    )
    default_branch = default_branch_ref.split("/")[-1] if default_branch_ref else "main"

    for line in branch_output.split("\n"):
        parts = line.strip().split("\x00")
        if len(parts) >= 4:
            full_branch_name = parts[0]

            is_remote = full_branch_name.startswith("origin/")
            branch_name = (
                full_branch_name.replace("origin/", "")
                if is_remote
                else full_branch_name
            )

            # Get commit count for this branch
            commit_count_str = run_git_command(
                repo_path, ["rev-list", "--count", full_branch_name]
            )
            commit_count = int(commit_count_str) if commit_count_str.isdigit() else 0

            branch_info = {
                "branch_name": branch_name,
                "full_name": full_branch_name,
                "is_current": branch_name == current_branch and not is_remote,
                "is_remote": is_remote,
                "last_commit_hash": parts[1],
                "last_commit_date": parts[2],
                "last_commit_author": parts[3],
                "upstream": parts[4] if len(parts) > 4 else "",
                "commit_count": commit_count,
                "is_merged": False,
            }

            if not is_remote and branch_name != default_branch:
                # Check if branch is merged into the default branch
                merged_check = run_git_command(
                    repo_path, ["branch", "--merged", default_branch]
                )
                branch_info["is_merged"] = any(
                    f.strip() == branch_name
                    for f in merged_check.replace("*", "").split("\n")
                )

            branches.append(branch_info)

    # Deduplicate branches, preferring local over remote if names conflict
    unique_branches_map = {}
    for b in sorted(branches, key=lambda x: x["is_remote"]):
        unique_branches_map[b["branch_name"]] = b

    return list(unique_branches_map.values())


def get_contributor_stats(repo_path: str) -> List[Dict[str, Any]]:
    """
    Extract enhanced contributor statistics.
    """
    shortlog_output = run_git_command(repo_path, ["shortlog", "-sne", "--all"])
    if not shortlog_output:
        return []

    contributors = []
    for line in shortlog_output.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.match(r"(\d+)\s+(.+?)\s+<(.+?)>", line)
        if match:
            commit_count, author_name, author_email = match.groups()

            cmd = ["log", f"--author={author_email}", "--pretty=format:%cI", "--all"]
            dates_output = run_git_command(repo_path, cmd)
            dates = dates_output.split("\n") if dates_output else []

            stats_cmd = [
                "log",
                f"--author={author_email}",
                "--numstat",
                "--pretty=format:",
                "--all",
            ]
            stats_output = run_git_command(repo_path, stats_cmd)

            total_insertions, total_deletions = 0, 0
            files_touched = set()

            for stat_line in stats_output.split("\n"):
                if stat_line.strip() and "\t" in stat_line:
                    parts = stat_line.split("\t")
                    if len(parts) >= 3:
                        insertions, deletions, filepath = parts[0], parts[1], parts[2]
                        if insertions.isdigit():
                            total_insertions += int(insertions)
                        if deletions.isdigit():
                            total_deletions += int(deletions)
                        files_touched.add(filepath)

            contributor = {
                "author_name": author_name,
                "author_email": author_email,
                "commit_count": int(commit_count),
                "first_commit_date": dates[-1] if dates else "",
                "last_commit_date": dates[0] if dates else "",
                "total_insertions": total_insertions,
                "total_deletions": total_deletions,
                "total_lines_changed": total_insertions + total_deletions,
                "files_touched": len(files_touched),
                "avg_lines_per_commit": (
                    round((total_insertions + total_deletions) / int(commit_count), 2)
                    if int(commit_count) > 0
                    else 0
                ),
            }
            contributors.append(contributor)

    return sorted(contributors, key=lambda x: x["commit_count"], reverse=True)


def get_tag_info(repo_path: str) -> List[Dict[str, Any]]:
    """
    Extract information about Git tags.
    """
    tag_output = run_git_command(
        repo_path,
        [
            "tag",
            "-l",
            "--format=%(refname:short)%x00%(objectname)%x00%(taggerdate:iso)%x00%(taggername)%x00%(subject)",
        ],
    )
    if not tag_output:
        return []

    tags = []
    for line in tag_output.split("\n"):
        if line.strip():
            parts = line.split("\x00")
            if len(parts) >= 2:
                tag_info = {
                    "tag_name": parts[0],
                    "commit_hash": parts[1],
                    "tagger_date": parts[2] if len(parts) > 2 else "",
                    "tagger_name": parts[3] if len(parts) > 3 else "",
                    "subject": parts[4] if len(parts) > 4 else "",
                    "is_annotated": len(parts) > 2 and bool(parts[2]),
                }
                tags.append(tag_info)

    return tags


def get_language_stats(repo_path: str) -> List[Dict[str, Any]]:
    """
    Analyze programming languages used in the repository based on tracked files.
    """
    files_output = run_git_command(repo_path, ["ls-files"])
    if not files_output:
        return []

    language_counts = Counter()
    total_files = 0

    for filepath in files_output.split("\n"):
        if filepath.strip():
            # Check if the file exists and is not binary before counting
            full_path = os.path.join(repo_path, filepath)
            if os.path.exists(full_path) and not is_binary_file(full_path):
                language = detect_language_from_extension(filepath)
                language_counts[language] += 1
                total_files += 1

    stats = []
    for language, count in language_counts.most_common():
        percentage = round((count / total_files) * 100, 2) if total_files > 0 else 0
        stats.append(
            {"language": language, "file_count": count, "percentage": percentage}
        )

    return stats


def get_commit_patterns(commits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze commit patterns and trends.
    """
    if not commits:
        return {}

    day_counts = Counter()
    hour_counts = Counter()
    monthly_counts = defaultdict(int)

    merge_commits = 0
    total_message_length = 0

    for commit in commits:
        try:
            commit_date = datetime.datetime.fromisoformat(commit["commit_date"])
            day_counts[commit_date.strftime("%A")] += 1
            hour_counts[commit_date.hour] += 1
            monthly_counts[commit_date.strftime("%Y-%m")] += 1

            if commit.get("is_merge", False):
                merge_commits += 1

            total_message_length += commit.get("message_length", 0)
        except (ValueError, KeyError, IndexError):
            continue

    total_commits = len(commits)
    return {
        "most_active_day": day_counts.most_common(1)[0] if day_counts else ("", 0),
        "most_active_hour": hour_counts.most_common(1)[0] if hour_counts else (0, 0),
        "merge_commit_percentage": (
            round((merge_commits / total_commits) * 100, 2) if total_commits > 0 else 0
        ),
        "avg_message_length": (
            round(total_message_length / total_commits, 2) if total_commits > 0 else 0
        ),
        "monthly_commit_trend": dict(sorted(monthly_counts.items())),
    }


def save_to_csv(data: List[Dict], filename: str, output_dir: Path):
    """
    Save data to a CSV file, flattening nested dictionaries.
    """
    if not data:
        print(f"No data to save for {filename}.csv")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.csv"

    try:
        flat_data = []
        for item in data:
            row = {}
            for key, value in item.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            flat_data.append(row)

        # Collect all possible headers from the flattened data
        fieldnames = sorted(list(set(key for row in flat_data for key in row.keys())))

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(flat_data)
        print(f"Saved {len(data)} records to {filepath}")
    except IOError as e:
        print(f"Error saving file {filepath}: {e}")


def save_to_json(data: Any, filename: str, output_dir: Path):
    """
    Save data to a JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.json"

    try:
        with open(filepath, "w", encoding="utf-8") as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False, default=str)
        print(f"Saved data to {filepath}")
    except IOError as e:
        print(f"Error saving file {filepath}: {e}")


def extract_repo_metadata(
    repo_path: str,
    output_format: str,
    commit_limit: Optional[int],
    output_dir: str,
    include_patterns: bool,
    include_contributors: bool,
    include_architectural_metrics: bool,
):
    """
    Main function to extract all repository metadata with enhancements.
    """
    git_dir = Path(repo_path) / ".git"
    if not git_dir.is_dir():
        print(f"Error: {repo_path} is not a valid git repository.")
        return

    print(f"Extracting metadata from repository: {repo_path}")
    output_path = Path(output_dir)

    print("Extracting basic repository info...")
    basic_info = get_basic_repo_info(repo_path)
    repo_name = sanitize_filename(basic_info["repo_name"])

    print("Extracting commit history...")
    commits = get_commit_history(repo_path, commit_limit)

    print("Extracting file changes...")
    file_changes = get_file_changes(repo_path, commit_limit)

    print("Extracting branch information...")
    branches = get_branch_info(repo_path)

    contributors = []
    if include_contributors:
        print("Extracting contributor statistics...")
        contributors = get_contributor_stats(repo_path)

    print("Extracting tag information...")
    tags = get_tag_info(repo_path)

    print("Analyzing programming languages...")
    language_stats = get_language_stats(repo_path)

    commit_patterns = {}
    if include_patterns:
        print("Analyzing commit patterns...")
        commit_patterns = get_commit_patterns(commits)

    if include_architectural_metrics:
        print("Calculating architectural persistence metrics...")
        commits = calculate_architectural_metrics(commits)

    complete_data = {
        "repository_info": basic_info,
        "commits": commits,
        "file_changes": file_changes,
        "branches": branches,
        "contributors": contributors,
        "tags": tags,
        "language_statistics": language_stats,
        "commit_patterns": commit_patterns,
        "extraction_date": datetime.datetime.now().isoformat(),
        "extraction_summary": {
            "total_commits_processed": len(commits),
            "total_file_changes_logged": len(file_changes),
            "total_branches_found": len(branches),
            "total_contributors_found": len(contributors),
            "total_tags_found": len(tags),
            "languages_detected": len(language_stats),
            "architectural_metrics_calculated": include_architectural_metrics,
        },
    }

    if output_format in ["csv", "both"]:
        save_to_csv([basic_info], f"{repo_name}_basic_info", output_path)
        save_to_csv(commits, f"{repo_name}_commits", output_path)
        save_to_csv(file_changes, f"{repo_name}_file_changes", output_path)
        save_to_csv(branches, f"{repo_name}_branches", output_path)
        if contributors:
            save_to_csv(contributors, f"{repo_name}_contributors", output_path)
        save_to_csv(tags, f"{repo_name}_tags", output_path)
        save_to_csv(language_stats, f"{repo_name}_languages", output_path)

        if commit_patterns:
            patterns_list = [
                {"metric": key, "value": str(value)}
                for key, value in commit_patterns.items()
            ]
            save_to_csv(patterns_list, f"{repo_name}_commit_patterns", output_path)

    if output_format in ["json", "both"]:
        save_to_json(complete_data, f"{repo_name}_complete_metadata", output_path)

    print("\nExtraction complete!")
    print(f"  - Total commits processed: {len(commits)}")
    print(f"  - Total file changes logged: {len(file_changes)}")
    print(f"  - Total branches found: {len(branches)}")
    if include_contributors:
        print(f"  - Total contributors found: {len(contributors)}")
    if include_architectural_metrics:
        print(f"  - Architectural metrics calculated: Yes")
    print(f"  - Output files saved in: {output_path.resolve()}")


def main():
    """
    Command-line interface for the Git repository metadata extractor.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Git Repository Metadata Extractor. Extracts comprehensive metadata from a git repository, including optional architectural metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("repo_path", help="Path to the local Git repository.")
    parser.add_argument(
        "-f",
        "--output-format",
        default="json",
        choices=["json", "csv", "both"],
        help="Output format for the extracted data.",
    )
    parser.add_argument(
        "-l",
        "--commit-limit",
        type=int,
        default=None,
        help="Limit the number of commits to process. Processes all commits by default.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./reop_metadata",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--include-patterns",
        action="store_true",
        help="Include analysis of commit patterns (e.g., activity by day/hour).",
    )
    parser.add_argument(
        "--include-contributors",
        action="store_true",
        help="Include detailed contributor statistics. This can be slow for large repositories.",
    )
    parser.add_argument(
        "--architectural-metrics",
        action="store_true",
        help="Include architectural persistence metrics (churn, survival, ASI) for each commit. Requires pandas and numpy.",
    )

    args = parser.parse_args()

    if args.architectural_metrics and not HAS_PANDAS:
        print(
            "Error: --architectural-metrics flag requires pandas and numpy to be installed."
        )
        print("Please install them with: pip install pandas numpy")
        return

    extract_repo_metadata(
        repo_path=args.repo_path,
        output_format=args.output_format,
        commit_limit=args.commit_limit,
        output_dir=args.output_dir,
        include_patterns=args.include_patterns,
        include_contributors=args.include_contributors,
        include_architectural_metrics=args.architectural_metrics,
    )


if __name__ == "__main__":
    main()
