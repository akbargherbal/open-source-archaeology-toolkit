"""
gitutils.py - Helper functions for software archaeology investigations

A collection of reusable utilities for analyzing Git repository history.
Import these functions into Jupyter notebooks for interactive exploration.

Usage:
    import gitutils
    repo = gitutils.load_repository('../target-repo')
    core_files = gitutils.find_core_files(repo)
"""

import re
from git import Repo, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import combinations
import os
import inspect

# ============================================================================
# REPOSITORY LOADING & BASIC INFO
# ============================================================================

def load_repository(repo_path: str) -> Repo:
    """
    Load a Git repository from path.
    """
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        return repo
    except (InvalidGitRepositoryError, NoSuchPathError) as e:
        print(f"Error: Could not load repository at {repo_path}. Is it a valid git repo?")
        raise e


def get_repository_stats(repo: Repo) -> Dict:
    """
    Get basic statistics about a repository.
    """
    try:
        if not repo.head.is_valid():
            return defaultdict(int)
        all_commits = list(repo.iter_commits('HEAD'))
    except ValueError: # Happens in empty repos
        return defaultdict(int)

    if not all_commits:
        return defaultdict(int)

    first_commit = all_commits[-1]
    last_commit = all_commits[0]
    
    authors = {c.author.email for c in all_commits}
    
    try:
        total_files = len([blob for blob in repo.head.commit.tree.traverse() if blob.type == 'blob'])
    except Exception:
        total_files = 0

    now_aware = datetime.now(timezone.utc)
    age_in_days = (now_aware - first_commit.committed_datetime).days

    return {
        "total_commits": len(all_commits),
        "total_branches": len(repo.branches),
        "total_tags": len(repo.tags),
        "first_commit_date": first_commit.committed_datetime,
        "last_commit_date": last_commit.committed_datetime,
        "age_in_days": age_in_days,
        "unique_authors": len(authors),
        "total_files_current": total_files
    }


def get_commit_range(repo: Repo, start_date: str = None, end_date: str = None) -> List:
    """
    Get commits within a specific date range.
    """
    kwargs = {}
    if start_date:
        kwargs['after'] = start_date
    if end_date:
        kwargs['before'] = end_date
        
    return list(repo.iter_commits('HEAD', **kwargs))


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_commit_metadata(commit) -> Dict:
    """
    Extract metadata from a single commit.
    """
    return {
        "hash": commit.hexsha,
        "short_hash": commit.hexsha[:7],
        "author_name": commit.author.name,
        "author_email": commit.author.email,
        "date": commit.committed_datetime,
        "message": commit.message,
        "message_first_line": commit.message.split('\n')[0].strip(),
        "parent_count": len(commit.parents),
    }


def extract_commit_stats(commit) -> Dict:
    """
    Extract file change statistics from a commit.
    """
    stats = commit.stats.total
    files_changed = list(commit.stats.files.keys())
    
    files_added = []
    files_deleted = []
    files_renamed = {}

    if commit.parents:
        diff = commit.parents[0].diff(commit, create_patch=True)
        for d in diff:
            if d.new_file:
                files_added.append(d.b_path)
            elif d.deleted_file:
                files_deleted.append(d.a_path)
            elif d.renamed_file:
                files_renamed[d.a_path] = d.b_path
    else: # Initial commit
        files_added = files_changed

    return {
        "files_changed": files_changed,
        "insertions": stats['insertions'],
        "deletions": stats['deletions'],
        "total_changes": stats['lines'],
        "files_added": files_added,
        "files_deleted": files_deleted,
        "files_renamed": files_renamed,
    }


def extract_file_tree(commit) -> List[str]:
    """
    Get complete list of files in the repository at a specific commit.
    """
    return [blob.path for blob in commit.tree.traverse() if blob.type == 'blob']


def extract_directory_tree(commit) -> List[str]:
    """
    Get complete list of directories at a specific commit.
    """
    return [tree.path for tree in commit.tree.traverse() if tree.type == 'tree']


def build_commit_dataframe(repo: Repo, max_commits: int = None) -> pd.DataFrame:
    """
    Build a comprehensive DataFrame of all commits.
    """
    # FIX: Handle empty repositories gracefully.
    if not repo.head.is_valid():
        return pd.DataFrame()
    try:
        commits = list(repo.iter_commits('HEAD', max_count=max_commits))
    except ValueError:
        return pd.DataFrame()

    commit_data = []
    for commit in commits:
        try:
            metadata = extract_commit_metadata(commit)
            stats = extract_commit_stats(commit)
            combined = {**metadata, **stats}
            commit_data.append(combined)
        except Exception:
            continue
    
    df = pd.DataFrame(commit_data)
    if 'date' in df.columns and not df.empty:
        df['date'] = pd.to_datetime(df['date'], utc=True)
    return df


# ============================================================================
# CORE FILE ANALYSIS
# ============================================================================

def find_core_files(repo: Repo, threshold: float = 0.9) -> List[Tuple[str, float]]:
    """
    Identify files present in X% of commits (the unchanging core).
    """
    file_presence = Counter()
    # FIX: Handle empty repositories gracefully.
    if not repo.head.is_valid():
        return []
    try:
        commits = list(repo.iter_commits('HEAD'))
    except ValueError:
        return []
    total_commits = len(commits)
    
    if total_commits == 0:
        return []

    for commit in commits:
        try:
            files = extract_file_tree(commit)
            file_presence.update(files)
        except Exception:
            continue
            
    core_files = []
    for file, count in file_presence.items():
        percentage = count / total_commits
        if percentage >= threshold:
            core_files.append((file, percentage))
            
    return sorted(core_files, key=lambda item: item[1], reverse=True)


def find_stable_files(repo: Repo, days_unchanged: int = 365) -> List[Tuple[str, datetime]]:
    """
    Find files that haven't been modified in X days.
    """
    stable_files = []
    threshold_date = datetime.now(timezone.utc) - timedelta(days=days_unchanged)
    
    try:
        head_files = extract_file_tree(repo.head.commit)
    except ValueError:
        return []

    for filepath in head_files:
        try:
            last_commit = next(repo.iter_commits('HEAD', paths=filepath, max_count=1))
            if last_commit.committed_datetime < threshold_date:
                stable_files.append((filepath, last_commit.committed_datetime))
        except (StopIteration, GitCommandError):
            continue
            
    return sorted(stable_files, key=lambda item: item[1])


def find_volatile_files(repo: Repo, min_commits: int = 50) -> List[Tuple[str, int]]:
    """
    Find files with high change frequency.
    """
    file_changes = Counter()
    try:
        for commit in repo.iter_commits('HEAD'):
            try:
                file_changes.update(commit.stats.files.keys())
            except Exception:
                continue
    except ValueError:
        return []

    volatile = [item for item in file_changes.items() if item[1] >= min_commits]
    return sorted(volatile, key=lambda item: item[1], reverse=True)


def calculate_file_churn(repo: Repo) -> pd.DataFrame:
    """
    Calculate churn (insertions + deletions) for each file across history.
    """
    churn_data = defaultdict(lambda: {'total_insertions': 0, 'total_deletions': 0, 'commit_count': 0})
    
    try:
        for commit in repo.iter_commits('HEAD'):
            try:
                for filepath, stats in commit.stats.files.items():
                    churn_data[filepath]['total_insertions'] += stats['insertions']
                    churn_data[filepath]['total_deletions'] += stats['deletions']
                    churn_data[filepath]['commit_count'] += 1
            except Exception:
                continue
    except ValueError:
        return pd.DataFrame()

    records = []
    for filepath, data in churn_data.items():
        total_churn = data['total_insertions'] + data['total_deletions']
        records.append({
            'filepath': filepath,
            'total_insertions': data['total_insertions'],
            'total_deletions': data['total_deletions'],
            'total_churn': total_churn,
            'commit_count': data['commit_count'],
            'avg_churn_per_commit': total_churn / data['commit_count'] if data['commit_count'] > 0 else 0
        })
        
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values('total_churn', ascending=False).reset_index(drop=True)


# ============================================================================
# TIMELINE & EVOLUTION ANALYSIS
# ============================================================================

def get_complexity_timeline(repo: Repo) -> pd.DataFrame:
    """
    Track file count over time.
    """
    timeline_data = []
    try:
        commits = repo.iter_commits('HEAD', reverse=True)
    except ValueError:
        return pd.DataFrame()

    for commit in commits:
        try:
            file_count = len(extract_file_tree(commit))
            timeline_data.append({
                'date': commit.committed_datetime,
                'total_files': file_count,
                'commit_hash': commit.hexsha
            })
        except Exception:
            continue
            
    return pd.DataFrame(timeline_data)


def find_complexity_milestones(repo: Repo, thresholds: List[int] = None) -> Dict:
    """
    Find when file count crossed specific thresholds.
    """
    if thresholds is None:
        thresholds = [50, 100, 200, 500, 1000]
    
    thresholds = sorted(list(set(thresholds)))
    milestones = {}
    
    timeline_df = get_complexity_timeline(repo)
    if timeline_df.empty:
        return {}

    for _, row in timeline_df.iterrows():
        if not thresholds:
            break
        
        if row['total_files'] >= thresholds[0]:
            threshold_val = thresholds.pop(0)
            milestones[threshold_val] = (row['date'], row['commit_hash'], row['total_files'])
            
    return milestones


def get_commit_frequency_timeline(repo: Repo, bin_by: str = 'month') -> pd.DataFrame:
    """
    Calculate commit frequency over time.
    """
    try:
        commit_dates = [c.committed_datetime for c in repo.iter_commits('HEAD')]
    except ValueError:
        return pd.DataFrame(columns=['period', 'commit_count'])

    if not commit_dates:
        return pd.DataFrame(columns=['period', 'commit_count'])

    df = pd.DataFrame(commit_dates, columns=['date'])
    
    period_map = {'day': 'D', 'week': 'W', 'month': 'M', 'quarter': 'Q', 'year': 'Y'}
    period_code = period_map.get(bin_by.lower(), 'M')
    
    freq_df = df.set_index('date').resample(period_code).size().reset_index(name='commit_count')
    freq_df.rename(columns={'date': 'period'}, inplace=True)
    
    return freq_df

def identify_development_phases(repo: Repo, method: str = 'kmeans') -> pd.DataFrame:
    """
    (Placeholder) Automatically identify distinct development phases based on activity.
    """
    print("Note: identify_development_phases is a complex function and is not fully implemented in this version.")
    return pd.DataFrame()


def get_historical_snapshot(repo: Repo, target_date: str) -> Dict:
    """
    Get repository state at a specific date.
    """
    try:
        commit = next(repo.iter_commits('HEAD', until=target_date, max_count=1))
        file_count = len(extract_file_tree(commit))
        dir_count = len(extract_directory_tree(commit))
        return {
            "commit_hash": commit.hexsha,
            "date": commit.committed_datetime,
            "file_count": file_count,
            "directory_count": dir_count,
        }
    except (StopIteration, ValueError):
        return {}


def compare_snapshots(repo: Repo, date1: str, date2: str) -> Dict:
    """
    Compare repository state between two dates.
    """
    try:
        commit1 = next(repo.iter_commits('HEAD', until=date1, max_count=1))
        commit2 = next(repo.iter_commits('HEAD', until=date2, max_count=1))
    except (StopIteration, ValueError):
        return {}

    diff = commit1.diff(commit2)
    
    return {
        "files_added": [d.b_path for d in diff.iter_change_type('A')],
        "files_removed": [d.a_path for d in diff.iter_change_type('D')],
        "files_renamed": [(d.a_path, d.b_path) for d in diff.iter_change_type('R')],
        "files_modified": [d.b_path for d in diff.iter_change_type('M')],
    }


# ============================================================================
# REORGANIZATION DETECTION
# ============================================================================

def find_reorganization_commits(repo: Repo, threshold: float = 0.2) -> List[Dict]:
    """
    Detect commits where many files were moved/renamed.
    """
    reorg_commits = []
    try:
        for commit in repo.iter_commits('HEAD'):
            if not commit.parents:
                continue
            
            diff = commit.parents[0].diff(commit)
            renamed_files = [d for d in diff if d.renamed_file]
            
            if not renamed_files:
                continue

            total_files_in_commit = len(extract_file_tree(commit))
            if total_files_in_commit == 0: continue

            move_ratio = len(renamed_files) / total_files_in_commit
            if move_ratio >= threshold:
                reorg_commits.append({
                    "commit_hash": commit.hexsha,
                    "date": commit.committed_datetime,
                    "message": commit.message.split('\n')[0],
                    "files_moved": len(renamed_files),
                    "move_ratio": move_ratio,
                })
    except ValueError:
        pass
    return reorg_commits

def detect_directory_splits(repo: Repo) -> List[Dict]:
    print("Note: detect_directory_splits is not implemented.")
    return []

def detect_directory_merges(repo: Repo) -> List[Dict]:
    print("Note: detect_directory_merges is not implemented.")
    return []

def track_file_renames(repo: Repo, filepath: str) -> List[Tuple[str, datetime]]:
    """
    Track all historical names of a specific file.
    """
    try:
        log_output = repo.git.log('--follow', '--name-status', '--format=%H %ct', '--', filepath)
        # Complex parsing logic omitted for brevity
        print("Note: track_file_renames is a complex function with a simplified placeholder.")
        return [(filepath, datetime.now(timezone.utc))]
    except GitCommandError:
        return []


# ============================================================================
# CONTRIBUTOR ANALYSIS
# ============================================================================

def get_contributor_stats(repo: Repo) -> pd.DataFrame:
    """
    Analyze contribution patterns for all authors.
    """
    author_stats = defaultdict(lambda: {
        'name': set(), 'email': '', 'commit_count': 0, 'dates': [],
        'insertions': 0, 'deletions': 0
    })

    try:
        for commit in repo.iter_commits('HEAD'):
            author_email = commit.author.email
            stats = author_stats[author_email]
            
            stats['name'].add(commit.author.name)
            stats['email'] = author_email
            stats['commit_count'] += 1
            stats['dates'].append(commit.committed_datetime)
            
            try:
                commit_stats = commit.stats.total
                stats['insertions'] += commit_stats['insertions']
                stats['deletions'] += commit_stats['deletions']
            except Exception:
                continue
    except ValueError:
        return pd.DataFrame()

    records = []
    for email, stats in author_stats.items():
        if not stats['dates']: continue
        first_commit = min(stats['dates'])
        last_commit = max(stats['dates'])
        records.append({
            'author_name': ' / '.join(sorted(list(stats['name']))),
            'author_email': email,
            'commit_count': stats['commit_count'],
            'first_commit_date': first_commit,
            'last_commit_date': last_commit,
            'active_days': (last_commit - first_commit).days,
            'total_insertions': stats['insertions'],
            'total_deletions': stats['deletions']
        })
        
    df = pd.DataFrame(records)
    if df.empty: return df
    return df.sort_values('commit_count', ascending=False).reset_index(drop=True)


def find_top_contributors(repo: Repo, n: int = 10, metric: str = 'commits') -> List[Tuple]:
    """
    Get top N contributors by various metrics.
    """
    stats_df = get_contributor_stats(repo)
    if stats_df.empty:
        return []

    metric_map = {
        'commits': 'commit_count', 'insertions': 'total_insertions',
        'deletions': 'total_deletions',
    }
    
    if metric == 'churn':
        stats_df['churn'] = stats_df['total_insertions'] + stats_df['total_deletions']
        sort_by = 'churn'
    elif metric in metric_map:
        sort_by = metric_map[metric]
    else:
        valid_metrics = list(metric_map.keys()) + ['churn']
        raise ValueError(f"Invalid metric: {metric}. Use one of {valid_metrics}")

    top_df = stats_df.sort_values(sort_by, ascending=False).head(n)
    return list(zip(top_df['author_name'], top_df[sort_by]))


def analyze_contributor_tenure(repo: Repo) -> pd.DataFrame:
    """
    Analyze when contributors joined and left the project.
    """
    stats_df = get_contributor_stats(repo)
    if 'last_commit_date' not in stats_df.columns:
        return pd.DataFrame()
    
    stats_df['tenure_days'] = (stats_df['last_commit_date'] - stats_df['first_commit_date']).dt.days
    ninety_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
    stats_df['is_active'] = stats_df['last_commit_date'] > ninety_days_ago
    
    return stats_df[['author_name', 'first_commit_date', 'last_commit_date', 'tenure_days', 'is_active']]


def find_file_owners(repo: Repo, min_percentage: float = 0.3) -> Dict[str, str]:
    """
    Identify primary author for each file.
    """
    file_authors = defaultdict(Counter)
    try:
        for commit in repo.iter_commits('HEAD'):
            author = commit.author.name
            for file in commit.stats.files:
                file_authors[file][author] += 1
    except ValueError:
        return {}
        
    owners = {}
    for file, authors in file_authors.items():
        if not authors: continue
        total_commits = sum(authors.values())
        primary_author, primary_count = authors.most_common(1)[0]
        
        if (primary_count / total_commits) >= min_percentage:
            owners[file] = primary_author
            
    return owners


def analyze_collaboration_patterns(repo: Repo) -> pd.DataFrame:
    """
    Find which authors frequently modify the same files.
    """
    file_authors = defaultdict(set)
    try:
        for commit in repo.iter_commits('HEAD'):
            author = commit.author.name
            for file in commit.stats.files:
                file_authors[file].add(author)
    except ValueError:
        return pd.DataFrame()

    collaboration = Counter()
    for file, authors in file_authors.items():
        if len(authors) > 1:
            for pair in combinations(sorted(list(authors)), 2):
                collaboration[pair] += 1
    
    records = [{'author1': pair[0], 'author2': pair[1], 'shared_files': count} for pair, count in collaboration.items()]

    df = pd.DataFrame(records)
    if df.empty: return df
    return df.sort_values('shared_files', ascending=False).reset_index(drop=True)


# ============================================================================
# DIRECTORY & MODULE ANALYSIS
# ============================================================================

def get_directory_stats(repo: Repo) -> pd.DataFrame:
    """
    Analyze change frequency and size for each directory.
    """
    dir_stats = defaultdict(lambda: {'file_count': 0, 'total_commits': 0, 'last_modified': None})
    
    try:
        for file in extract_file_tree(repo.head.commit):
            parent_dir = str(Path(file).parent)
            dir_stats[parent_dir]['file_count'] += 1

        for commit in repo.iter_commits('HEAD'):
            for file in commit.stats.files:
                parent_dir = str(Path(file).parent)
                dir_stats[parent_dir]['total_commits'] += 1
                if dir_stats[parent_dir]['last_modified'] is None or commit.committed_datetime > dir_stats[parent_dir]['last_modified']:
                    dir_stats[parent_dir]['last_modified'] = commit.committed_datetime
    except ValueError:
        return pd.DataFrame()
        
    records = [{'directory_path': d, **s} for d, s in dir_stats.items()]
    df = pd.DataFrame(records)
    if df.empty: return df
    return df.sort_values('total_commits', ascending=False).reset_index(drop=True)


def find_stable_directories(repo: Repo, days_unchanged: int = 180) -> List[str]:
    """
    Find directories with no changes in X days.
    """
    dir_df = get_directory_stats(repo)
    if dir_df.empty: return []
    
    threshold = datetime.now(timezone.utc) - timedelta(days=days_unchanged)
    stable_dirs = dir_df[dir_df['last_modified'] < threshold]
    return stable_dirs['directory_path'].tolist()


def find_volatile_directories(repo: Repo, min_commits: int = 100) -> List[Tuple[str, int]]:
    """
    Find directories with high change frequency.
    """
    dir_df = get_directory_stats(repo)
    if dir_df.empty: return []
    
    volatile = dir_df[dir_df['total_commits'] >= min_commits]
    return list(zip(volatile['directory_path'], volatile['total_commits']))


def track_directory_growth(repo: Repo, directory: str) -> pd.DataFrame:
    """
    Track file count in a specific directory over time.
    """
    growth_data = []
    try:
        for commit in repo.iter_commits('HEAD', reverse=True):
            files_in_dir = [f for f in extract_file_tree(commit) if f.startswith(directory)]
            growth_data.append({
                'date': commit.committed_datetime,
                'file_count': len(files_in_dir),
                'commit_hash': commit.hexsha
            })
    except ValueError:
        return pd.DataFrame()
    return pd.DataFrame(growth_data)


# ============================================================================
# FILE LIFETIME ANALYSIS
# ============================================================================

def analyze_file_lifecycle(repo: Repo, filepath: str) -> Dict:
    """
    Comprehensive analysis of a single file's history.
    """
    try:
        commits = list(repo.iter_commits('HEAD', paths=filepath, reverse=True))
        if not commits:
            return {}
        
        created_commit = commits[0]
        authors = {c.author.name for c in commits}
        churn = sum(c.stats.files.get(filepath, {}).get('lines', 0) for c in commits)
        
        return {
            "created_date": created_commit.committed_datetime,
            "created_by": created_commit.author.name,
            "total_commits": len(commits),
            "unique_authors": len(authors),
            "total_churn": churn,
        }
    except GitCommandError:
        return {}


def find_deleted_files(repo: Repo) -> List[Dict]:
    """
    Find files that were deleted and when.
    """
    deleted_files = []
    try:
        log = repo.git.log('--diff-filter=D', '--summary', '--format=%H %ct %an')
        commit_hash = None
        for line in log.splitlines():
            if ' ' in line and len(line.split()) > 2:
                parts = line.split(' ', 2)
                commit_hash, commit_date, author = parts
            elif 'delete mode' in line:
                filepath = line.split(' ')[-1]
                deleted_files.append({
                    'filepath': filepath,
                    'deleted_date': datetime.fromtimestamp(int(commit_date), tz=timezone.utc),
                    'deleted_by': author,
                    'final_commit_hash': commit_hash
                })
    except GitCommandError:
        pass
    return deleted_files


def find_abandoned_files(repo: Repo, days_threshold: int = 730) -> List[Dict]:
    """
    Find files that exist but haven't been touched in X days.
    """
    stable = find_stable_files(repo, days_unchanged=days_threshold)
    abandoned = []
    for file, last_mod in stable:
        try:
            last_commit = next(repo.iter_commits('HEAD', paths=file, max_count=1))
            abandoned.append({
                'filepath': file,
                'last_modified': last_mod,
                'days_since_change': (datetime.now(timezone.utc) - last_mod).days,
                'last_author': last_commit.author.name
            })
        except (StopIteration, GitCommandError):
            continue
    return abandoned


# ============================================================================
# PATTERN DETECTION
# ============================================================================

def detect_major_refactorings(repo: Repo) -> List[Dict]:
    """
    Detect commits with unusually high churn (potential refactorings).
    """
    df = build_commit_dataframe(repo)
    if df.empty or 'total_changes' not in df.columns: return []
    
    churn_threshold = df['total_changes'].quantile(0.95)
    refactors = df[(df['total_changes'] >= churn_threshold) & (df['files_changed'].apply(len) > 5)]
    
    return refactors.to_dict('records')


def find_bug_fix_commits(repo: Repo) -> List[Dict]:
    """
    Identify commits likely to be bug fixes (based on message keywords).
    """
    keywords = ['fix', 'bug', 'repair', 'correct', 'patch', 'resolve']
    pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    
    fix_commits = []
    try:
        for commit in repo.iter_commits('HEAD'):
            matches = pattern.findall(commit.message)
            if matches:
                fix_commits.append({
                    'commit_hash': commit.hexsha, 'date': commit.committed_datetime,
                    'author': commit.author.name, 'message': commit.message.split('\n')[0],
                    'keywords_matched': list(set(m.lower() for m in matches))
                })
    except ValueError:
        pass
    return fix_commits


def find_feature_commits(repo: Repo) -> List[Dict]:
    """
    Identify commits likely to be new features.
    """
    keywords = ['feature', 'add', 'implement', 'create', 'introduce', 'feat']
    pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    
    feature_commits = []
    try:
        for commit in repo.iter_commits('HEAD'):
            matches = pattern.findall(commit.message)
            if matches:
                feature_commits.append({
                    'commit_hash': commit.hexsha, 'date': commit.committed_datetime,
                    'author': commit.author.name, 'message': commit.message.split('\n')[0],
                    'keywords_matched': list(set(m.lower() for m in matches))
                })
    except ValueError:
        pass
    return feature_commits

def detect_copy_paste_events(repo: Repo) -> List[Dict]:
    print("Note: detect_copy_paste_events requires advanced diffing and is not implemented.")
    return []


# ============================================================================
# HOTSPOT ANALYSIS
# ============================================================================

def find_hotspot_files(repo: Repo, recency_days: int = 180) -> pd.DataFrame:
    """
    Find files with both high complexity and high change frequency.
    """
    threshold_date = datetime.now(timezone.utc) - timedelta(days=recency_days)
    
    hotspot_data = defaultdict(lambda: {'recent_commits': 0, 'total_churn': 0, 'authors': set()})

    try:
        recent_commits = repo.iter_commits('HEAD', after=threshold_date.isoformat())
    except ValueError:
        return pd.DataFrame()

    for commit in recent_commits:
        try:
            for filepath, stats in commit.stats.files.items():
                hotspot_data[filepath]['recent_commits'] += 1
                hotspot_data[filepath]['total_churn'] += stats['insertions'] + stats['deletions']
                hotspot_data[filepath]['authors'].add(commit.author.email)
        except Exception:
            continue
            
    records = [{'filepath': f, 'unique_authors': len(d['authors']), **d} for f, d in hotspot_data.items()]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['risk_score'] = (
        df['recent_commits'] / df['recent_commits'].max() +
        df['total_churn'] / df['total_churn'].max() +
        df['unique_authors'] / df['unique_authors'].max()
    )
    
    return df.sort_values('risk_score', ascending=False).reset_index(drop=True)


def find_hotspot_directories(repo: Repo, recency_days: int = 180) -> pd.DataFrame:
    """
    Directory-level hotspot analysis.
    """
    hotspot_df = find_hotspot_files(repo, recency_days=recency_days)
    if hotspot_df.empty: return pd.DataFrame()

    hotspot_df['directory'] = hotspot_df['filepath'].apply(lambda p: str(Path(p).parent))
    
    dir_hotspots = hotspot_df.groupby('directory').agg(
        recent_commits=('recent_commits', 'sum'),
        total_churn=('total_churn', 'sum'),
        file_count=('filepath', 'count')
    ).reset_index()

    dir_hotspots['risk_score'] = (
        dir_hotspots['recent_commits'] / dir_hotspots['recent_commits'].max() +
        dir_hotspots['total_churn'] / dir_hotspots['total_churn'].max()
    )
    return dir_hotspots.sort_values('risk_score', ascending=False).reset_index(drop=True)


# ============================================================================
# COMPLEXITY METRICS
# ============================================================================

def calculate_file_coupling(repo: Repo, min_occurrences: int = 5) -> pd.DataFrame:
    """
    Find files that are frequently changed together (temporal coupling).
    """
    coupling = Counter()
    commit_file_counts = Counter()

    try:
        for commit in repo.iter_commits('HEAD'):
            if len(commit.parents) > 1: continue
            try:
                changed_files = sorted(list(commit.stats.files.keys()))
                for file in changed_files:
                    commit_file_counts[file] += 1
                for pair in combinations(changed_files, 2):
                    coupling[pair] += 1
            except Exception:
                continue
    except ValueError:
        return pd.DataFrame()

    records = []
    for (file1, file2), count in coupling.items():
        if count >= min_occurrences:
            total_commits = commit_file_counts[file1] + commit_file_counts[file2] - count
            strength = count / total_commits if total_commits > 0 else 0
            records.append({
                'file1': file1, 'file2': file2, 'times_changed_together': count,
                'coupling_strength': strength
            })
            
    df = pd.DataFrame(records)
    if df.empty: return df
    return df.sort_values('coupling_strength', ascending=False).reset_index(drop=True)


def calculate_ownership_fragmentation(repo: Repo) -> pd.DataFrame:
    """
    Measure how many authors contribute to each file.
    """
    file_authors = defaultdict(Counter)
    try:
        for commit in repo.iter_commits('HEAD'):
            author = commit.author.name
            for file in commit.stats.files:
                file_authors[file][author] += 1
    except ValueError:
        return pd.DataFrame()
        
    records = []
    for file, authors in file_authors.items():
        if not authors: continue
        total_commits = sum(authors.values())
        primary_author, primary_count = authors.most_common(1)[0]
        records.append({
            'filepath': file,
            'unique_authors': len(authors),
            'primary_author_percentage': primary_count / total_commits,
            'fragmentation_score': 1 - (primary_count / total_commits)
        })

    df = pd.DataFrame(records)
    if df.empty: return df
    return df.sort_values('fragmentation_score', ascending=False).reset_index(drop=True)


# ============================================================================
# LEARNING PATH HELPERS
# ============================================================================

def suggest_learning_commit(repo: Repo, target_complexity: str = 'medium') -> Dict:
    """
    Suggest a historical commit that's good for learning.
    """
    df = build_commit_dataframe(repo)
    if df.empty: return {}

    df['files_changed_count'] = df['files_changed'].apply(len)
    df = df[df['parent_count'] == 1]
    df = df[df['total_changes'] > 0]

    if target_complexity == 'simple':
        candidates = df[(df['files_changed_count'] == 1) & (df['total_changes'] < 50)]
    elif target_complexity == 'medium':
        candidates = df[(df['files_changed_count'] > 1) & (df['files_changed_count'] < 5) & (df['total_changes'] < 200)]
    else:
        candidates = df[(df['files_changed_count'] > 5) & (df['total_changes'] > 200)]

    if candidates.empty: return {"reason": "No suitable commits found."}
    
    commit = candidates.sample(1).iloc[0]
    return {
        "commit_hash": commit['hash'], "date": commit['date'],
        "complexity_level": target_complexity,
        "reason": f"Changed {commit['files_changed_count']} files with {commit['total_changes']} lines of churn."
    }


def suggest_contribution_areas(repo: Repo) -> List[Dict]:
    """
    Suggest directories or files good for first contributions.
    """
    hotspots = find_hotspot_files(repo, recency_days=90)
    if hotspots.empty: return []

    candidates = hotspots[(hotspots['unique_authors'] > 1) & (hotspots['risk_score'] < hotspots['risk_score'].quantile(0.8))]
    
    return candidates.head(10).to_dict('records')


def find_good_entry_points(repo: Repo) -> List[str]:
    """
    Suggest files to read first (core, stable, well-maintained).
    """
    core_files = find_core_files(repo, threshold=0.95)
    if not core_files: return []
    return [f[0] for f in core_files[:10]]


# ============================================================================
# EXPORT & CACHING
# ============================================================================

def cache_repository_data(repo: Repo, output_path: str = 'repo_cache.parquet'):
    """
    Extract and cache all repository data for faster subsequent analysis.
    """
    df = build_commit_dataframe(repo)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Repository data cached to {path}")


def load_cached_data(cache_path: str = 'repo_cache.parquet') -> pd.DataFrame:
    """
    Load previously cached repository data.
    """
    return pd.read_parquet(cache_path)


def export_findings_to_markdown(findings: Dict, output_path: str = 'findings.md'):
    """
    Export analysis results to a formatted markdown document.
    """
    with open(output_path, 'w') as f:
        for title, content in findings.items():
            f.write(f"# {title.replace('_', ' ').title()}\n\n")
            if isinstance(content, pd.DataFrame):
                f.write(content.to_markdown(index=False))
            elif isinstance(content, list):
                for item in content:
                    f.write(f"- {item}\n")
            elif isinstance(content, dict):
                for k, v in content.items():
                    f.write(f"- **{k}**: {v}\n")
            else:
                f.write(str(content))
            f.write("\n\n")
    print(f"Findings exported to {output_path}")


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def plot_complexity_timeline(repo: Repo, save_path: str = None):
    """
    Create and display file count over time chart.
    """
    df = get_complexity_timeline(repo)
    if df.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['date'], df['total_files'])
    
    ax.set_title(f'Repository Complexity Over Time ({Path(repo.working_dir).name})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Number of Files')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.autofmt_xdate()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_commit_frequency(repo: Repo, bin_by: str = 'month', save_path: str = None):
    """
    Create commit frequency visualization.
    """
    df = get_commit_frequency_timeline(repo, bin_by=bin_by)
    if df.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['period'], df['commit_count'], width=20)
    
    ax.set_title(f'Commit Frequency per {bin_by.capitalize()} ({Path(repo.working_dir).name})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Commits')
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    fig.autofmt_xdate()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_contributor_timeline(repo: Repo, top_n: int = 10, save_path: str = None):
    """
    Create timeline showing when top contributors were active.
    """
    stats_df = get_contributor_stats(repo)
    if stats_df.empty:
        print("No data to plot.")
        return
        
    top_authors = stats_df.head(top_n)['author_name'].tolist()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, author in enumerate(top_authors):
        author_df = stats_df[stats_df['author_name'] == author]
        start = author_df['first_commit_date'].iloc[0]
        end = author_df['last_commit_date'].iloc[0]
        ax.plot([start, end], [i, i], lw=10, solid_capstyle='round')
        
    ax.set_yticks(range(len(top_authors)))
    ax.set_yticklabels(top_authors)
    ax.set_title(f'Top {top_n} Contributor Activity Timeline')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_directory_heatmap(repo: Repo, save_path: str = None):
    """
    Create heatmap of directory activity.
    """
    print("Note: plot_directory_heatmap is a complex visualization and is not implemented.")


def plot_churn_analysis(repo: Repo, top_n: int = 20, save_path: str = None):
    """
    Visualize files with highest churn.
    """
    df = calculate_file_churn(repo).head(top_n)
    if df.empty:
        print("No data to plot.")
        return
        
    df = df.sort_values('total_churn', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(df['filepath'], df['total_churn'], color='skyblue')
    
    ax.set_title(f'Top {top_n} Files by Churn ({Path(repo.working_dir).name})')
    ax.set_xlabel('Total Churn (Insertions + Deletions)')
    ax.set_ylabel('File Path')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_commit_access(commit, attribute: str, default=None):
    """
    Safely access commit attributes that might not exist.
    """
    return getattr(commit, attribute, default)


def format_date(date_obj, format_str: str = '%Y-%m-%d') -> str:
    """
    Format datetime objects consistently.
    """
    return date_obj.strftime(format_str)


def calculate_age_in_days(start_date, end_date=None) -> int:
    """
    Calculate days between two dates (end defaults to now).
    """
    if end_date is None:
        end_date = datetime.now(start_date.tzinfo)
    return (end_date - start_date).days


def filter_by_file_extension(files: List[str], extensions: List[str]) -> List[str]:
    """
    Filter file list by extensions.
    """
    return [f for f in files if Path(f).suffix in extensions]


def group_by_directory(files: List[str], depth: int = 1) -> Dict[str, List[str]]:
    """
    Group files by directory at specified depth.
    """
    groups = defaultdict(list)
    for file in files:
        path = Path(file)
        # FIX: Correctly handle root-level files and directory depth.
        if len(path.parts) < depth + 1:
            groups['.'].append(file)
        else:
            key = str(Path(*path.parts[:depth]))
            groups[key].append(file)
    return dict(groups)


def is_likely_test_file(filepath: str) -> bool:
    """
    Heuristic to determine if a file is a test file.
    """
    path = Path(filepath.lower())
    # FIX: Apply stricter rules for root-level files to avoid false positives like 'test.py'.
    if len(path.parts) == 1:  # It's a root file
        return path.name.startswith('test_') or path.name.endswith('_test.py')
    
    # For files in subdirectories, the broader check is appropriate.
    return any('test' in part for part in path.parts) or path.name.startswith('test_') or path.name.endswith('_test.py')


def is_likely_config_file(filepath: str) -> bool:
    """
    Heuristic to determine if a file is configuration.
    """
    config_extensions = ['.json', '.yaml', '.yml', '.xml', '.ini', '.toml', '.cfg']
    config_filenames = ['config', 'configuration', 'settings', 'dockerfile', 'makefile']
    path = Path(filepath.lower())
    return path.suffix in config_extensions or any(name in path.stem for name in config_filenames) or path.name in config_filenames


def is_likely_documentation(filepath: str) -> bool:
    """
    Heuristic to determine if a file is documentation.
    """
    doc_extensions = ['.md', '.rst', '.txt', '.docx']
    doc_dirs = ['doc', 'docs', 'documentation']
    path = Path(filepath.lower())
    return path.suffix in doc_extensions or any(part in doc_dirs for part in path.parts)