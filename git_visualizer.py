#!/usr/bin/env python3
"""
Unified Git Repository Analyzer

A comprehensive, modular analysis tool that consolidates the functionality
of git_viz_generator.py and enhanced_repo_analyzer.py into a single,
well-structured system with pluggable analysis modules.

Author: Unified by Claude
"""

import argparse
import json
import logging
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import importlib.util

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm

# Optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Enhanced styling configuration
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white"
})

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    # Module selection
    modules: List[str] = None
    
    # Data filtering
    date_range: Optional[str] = None
    exclude_paths: Optional[str] = None
    min_file_changes: int = 2
    contributors_min_commits: int = 5
    
    # Analysis parameters
    top_n: int = 20
    core_file_threshold: float = 0.80
    active_file_threshold: float = 0.50
    
    # Output settings
    output_formats: List[str] = None
    interactive_plots: bool = False
    generate_html_report: bool = True
    theme: str = "modern"
    
    # Performance settings
    max_memory_mb: int = 1000
    sample_large_datasets: bool = True
    large_dataset_threshold: int = 50000
    
    def __post_init__(self):
        if self.modules is None:
            self.modules = ["commits", "contributors", "files", "hotspots"]
        if self.output_formats is None:
            self.output_formats = ["png"]

class DataProcessor:
    """Handles data loading, validation, and preprocessing."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_and_validate(self, json_file: Path) -> Dict[str, pd.DataFrame]:
        """Load and validate repository data from JSON."""
        self.logger.info(f"Loading data from {json_file}")
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")
        
        # Store raw data for metadata access
        self.raw_data = raw_data
        
        # Convert to DataFrames and validate
        dataframes = {}
        
        # Process commits data
        if "commits" in raw_data and raw_data["commits"]:
            commits_df = pd.DataFrame(raw_data["commits"])
            commits_df = self._validate_commits_data(commits_df)
            dataframes["commits"] = commits_df
        
        # Process file changes data
        if "file_changes" in raw_data and raw_data["file_changes"]:
            file_changes_df = pd.DataFrame(raw_data["file_changes"])
            file_changes_df = self._validate_file_changes_data(file_changes_df)
            if not file_changes_df.empty:
                file_changes_df = self._enhance_file_changes_data(file_changes_df)
            dataframes["file_changes"] = file_changes_df
        
        # Process contributors data
        if "contributors" in raw_data and raw_data["contributors"]:
            contributors_df = pd.DataFrame(raw_data["contributors"])
            dataframes["contributors"] = contributors_df
        
        # Process branches data
        if "branches" in raw_data and raw_data["branches"]:
            branches_df = pd.DataFrame(raw_data["branches"])
            dataframes["branches"] = branches_df
            
        # Process language statistics
        if "language_statistics" in raw_data and raw_data["language_statistics"]:
            languages_df = pd.DataFrame(raw_data["language_statistics"])
            dataframes["languages"] = languages_df
        
        # Apply filters
        dataframes = self._apply_filters(dataframes)
        
        self.logger.info(f"Loaded datasets: {list(dataframes.keys())}")
        return dataframes
    
    def _validate_commits_data(self, commits_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean commits data."""
        original_len = len(commits_df)
        
        # Handle column naming variations
        if 'hash' in commits_df.columns and 'commit_hash' not in commits_df.columns:
            commits_df = commits_df.rename(columns={'hash': 'commit_hash'})
        
        # Validate dates
        commits_df['commit_date'] = pd.to_datetime(commits_df['commit_date'], errors='coerce', utc=True)
        commits_df['author_date'] = pd.to_datetime(commits_df['author_date'], errors='coerce', utc=True)
        
        # Remove invalid dates
        commits_df = commits_df.dropna(subset=['commit_date'])
        commits_df = commits_df.sort_values('commit_date')
        
        # Add derived columns
        commits_df['year'] = commits_df['commit_date'].dt.year
        commits_df['month'] = commits_df['commit_date'].dt.to_period('M')
        commits_df['day_of_week'] = commits_df['commit_date'].dt.day_name()
        commits_df['hour'] = commits_df['commit_date'].dt.hour
        
        # Handle architectural metrics if present
        if any('architectural_metrics' in str(commit) for commit in self.raw_data.get('commits', [])):
            arch_data = []
            for commit in self.raw_data['commits']:
                if isinstance(commit, dict) and 'architectural_metrics' in commit:
                    arch_data.append(commit['architectural_metrics'])
                else:
                    arch_data.append({})
            
            arch_df = pd.json_normalize(arch_data)
            commits_df = pd.concat([commits_df.reset_index(drop=True), arch_df.reset_index(drop=True)], axis=1)
        
        self.logger.info(f"Commits: {original_len} → {len(commits_df)} after validation")
        return commits_df
    
    def _validate_file_changes_data(self, file_changes_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean file changes data."""
        original_len = len(file_changes_df)
        
        # Convert dates
        file_changes_df['commit_date'] = pd.to_datetime(file_changes_df['commit_date'], errors='coerce', utc=True)
        
        # Remove invalid data
        invalid_mask = (file_changes_df['commit_date'].isna() | 
                       file_changes_df['file_path'].isna() |
                       (file_changes_df['file_path'] == ''))
        
        if invalid_mask.sum() > 0:
            self.logger.warning(f"Removing {invalid_mask.sum()} invalid file change records")
            file_changes_df = file_changes_df[~invalid_mask]
        
        self.logger.info(f"File changes: {original_len} → {len(file_changes_df)} after validation")
        return file_changes_df
    
    def _enhance_file_changes_data(self, file_changes_df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced columns for file analysis."""
        self.logger.info("Enhancing file changes data")
        
        # File path analysis
        file_changes_df['path_obj'] = file_changes_df['file_path'].apply(Path)
        file_changes_df['file_extension'] = file_changes_df['path_obj'].apply(
            lambda x: ''.join(x.suffixes).lower() if x.suffixes else 'no_ext'
        )
        file_changes_df['filename'] = file_changes_df['path_obj'].apply(lambda x: x.name)
        file_changes_df['directory'] = file_changes_df['path_obj'].apply(
            lambda x: str(x.parent) if x.parent != Path('.') else 'root'
        )
        file_changes_df['path_depth'] = file_changes_df['path_obj'].apply(lambda x: len(x.parts))
        file_changes_df['top_level_dir'] = file_changes_df['path_obj'].apply(
            lambda x: x.parts[0] if x.parts else 'root'
        )
        
        # Time-based features
        file_changes_df['year'] = file_changes_df['commit_date'].dt.year
        file_changes_df['month'] = file_changes_df['commit_date'].dt.month
        file_changes_df['day_of_week'] = file_changes_df['commit_date'].dt.day_name()
        file_changes_df['hour'] = file_changes_df['commit_date'].dt.hour
        
        return file_changes_df
    
    def _apply_filters(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply configured filters to the datasets."""
        # Date range filter
        if self.config.date_range:
            try:
                start_str, end_str = self.config.date_range.split(",")
                start_date = pd.to_datetime(start_str, utc=True)
                end_date = pd.to_datetime(end_str, utc=True)
                
                self.logger.info(f"Applying date filter: {start_date} to {end_date}")
                
                for key, df in dataframes.items():
                    if 'commit_date' in df.columns:
                        mask = ((df['commit_date'] >= start_date) & (df['commit_date'] <= end_date))
                        dataframes[key] = df[mask]
                        
            except ValueError as e:
                self.logger.warning(f"Invalid date range: {e}")
        
        # Path exclusion filter
        if self.config.exclude_paths and 'file_changes' in dataframes:
            patterns = [p.strip() for p in self.config.exclude_paths.split(",")]
            self.logger.info(f"Excluding paths: {patterns}")
            
            df = dataframes['file_changes']
            for pattern in patterns:
                mask = ~df['file_path'].str.contains(pattern, regex=True, na=False)
                df = df[mask]
            dataframes['file_changes'] = df
        
        # Minimum file changes filter
        if self.config.min_file_changes > 1 and 'file_changes' in dataframes:
            df = dataframes['file_changes']
            file_counts = df['file_path'].value_counts()
            valid_files = file_counts[file_counts >= self.config.min_file_changes].index
            dataframes['file_changes'] = df[df['file_path'].isin(valid_files)]
        
        return dataframes

class AnalysisModule(ABC):
    """Abstract base class for analysis modules."""
    
    def __init__(self, config: AnalysisConfig, dataframes: Dict[str, pd.DataFrame], 
                 output_dir: Path, raw_data: Dict[str, Any]):
        self.config = config
        self.dataframes = dataframes
        self.output_dir = output_dir
        self.raw_data = raw_data
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create module-specific directories
        self.plots_dir = output_dir / "plots"
        self.data_dir = output_dir / "data"
        self.interactive_dir = output_dir / "interactive"
        
        for directory in [self.plots_dir, self.data_dir, self.interactive_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """Run the analysis and return results."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name for identification."""
        pass
    
    @property
    @abstractmethod
    def required_data(self) -> Set[str]:
        """Required dataframe keys for this module."""
        pass
    
    def can_run(self) -> bool:
        """Check if the module can run with available data."""
        return self.required_data.issubset(set(self.dataframes.keys()))
    
    def _save_plot(self, filename: str, dpi: int = 300):
        """Save current matplotlib plot."""
        for fmt in self.config.output_formats:
            filepath = self.plots_dir / f"{filename}.{fmt}"
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
            self.logger.info(f"Saved plot: {filepath}")
    
    def _save_data(self, data: pd.DataFrame, filename: str):
        """Save dataframe to CSV."""
        filepath = self.data_dir / f"{filename}.csv"
        data.to_csv(filepath, index=False)
        self.logger.info(f"Saved data: {filepath}")

class CommitAnalysisModule(AnalysisModule):
    """Analyzes commit patterns and trends."""
    
    @property
    def name(self) -> str:
        return "commits"
    
    @property
    def required_data(self) -> Set[str]:
        return {"commits"}
    
    def analyze(self) -> Dict[str, Any]:
        commits_df = self.dataframes["commits"]
        
        if commits_df.empty:
            self.logger.warning("No commit data available")
            return {}
        
        self.logger.info("Analyzing commit patterns...")
        
        # Monthly commit trends
        monthly_commits = commits_df.set_index('commit_date').resample('M').size()
        
        # Create comprehensive commit analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Repository Commit Analysis', fontsize=20, fontweight='bold')
        
        # 1. Monthly trend with rolling average
        monthly_commits.plot(ax=ax1, color='#2E8B57', linewidth=2, alpha=0.7)
        if len(monthly_commits) > 3:
            rolling_mean = monthly_commits.rolling(window=3, center=True).mean()
            rolling_mean.plot(ax=ax1, color='red', linestyle='--', 
                            linewidth=2, label='3-month average')
        ax1.set_title('Commit Activity Over Time', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Commits per Month')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Day of week pattern
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_commits = commits_df['day_of_week'].value_counts().reindex(dow_order).fillna(0)
        dow_commits.plot(kind='bar', ax=ax2, color='skyblue', alpha=0.8)
        ax2.set_title('Commits by Day of Week', fontweight='bold')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Number of Commits')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Hourly pattern
        hourly_commits = commits_df['hour'].value_counts().sort_index()
        hourly_commits.plot(kind='line', ax=ax3, marker='o', color='orange', linewidth=2)
        ax3.set_title('Commits by Hour of Day', fontweight='bold')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Number of Commits')
        ax3.grid(True, alpha=0.3)
        
        # 4. Yearly overview
        yearly_commits = commits_df['year'].value_counts().sort_index()
        yearly_commits.plot(kind='bar', ax=ax4, color='purple', alpha=0.8)
        ax4.set_title('Commits by Year', fontweight='bold')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Commits')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._save_plot("01_commit_analysis")
        plt.close()
        
        # Save data
        self._save_data(monthly_commits.reset_index(), "monthly_commits")
        
        # Generate interactive plot if requested
        if self.config.interactive_plots and HAS_PLOTLY:
            self._create_interactive_timeline(monthly_commits)
        
        return {
            "monthly_commits": monthly_commits.to_dict(),
            "total_commits": len(commits_df),
            "date_range": [commits_df['commit_date'].min().isoformat(), 
                          commits_df['commit_date'].max().isoformat()],
            "most_active_day": dow_commits.idxmax(),
            "most_active_hour": hourly_commits.idxmax()
        }
    
    def _create_interactive_timeline(self, monthly_commits: pd.Series):
        """Create interactive Plotly timeline."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_commits.index,
            y=monthly_commits.values,
            mode='lines+markers',
            name='Monthly Commits',
            line=dict(color='#2E8B57', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Interactive Commit Timeline',
            xaxis_title='Date',
            yaxis_title='Commits per Month',
            hovermode='x',
            template='plotly_white'
        )
        
        filepath = self.interactive_dir / "commit_timeline.html"
        fig.write_html(filepath)
        self.logger.info(f"Saved interactive plot: {filepath}")

class ContributorAnalysisModule(AnalysisModule):
    """Analyzes contributor patterns and activity."""
    
    @property
    def name(self) -> str:
        return "contributors"
    
    @property
    def required_data(self) -> Set[str]:
        return {"commits"}  # Can work with commits data or dedicated contributors data
    
    def analyze(self) -> Dict[str, Any]:
        # Use dedicated contributors data if available, otherwise derive from commits
        if "contributors" in self.dataframes and not self.dataframes["contributors"].empty:
            contributors_df = self.dataframes["contributors"]
            return self._analyze_contributors_data(contributors_df)
        elif "commits" in self.dataframes and not self.dataframes["commits"].empty:
            return self._analyze_from_commits(self.dataframes["commits"])
        else:
            self.logger.warning("No contributor data available")
            return {}
    
    def _analyze_contributors_data(self, contributors_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze using dedicated contributors dataset."""
        self.logger.info("Analyzing contributors from dedicated dataset...")
        
        # Find the appropriate columns
        commit_col = self._find_column(contributors_df, ['commit_count', 'commits', 'total_commits'])
        name_col = self._find_column(contributors_df, ['author_name', 'name', 'contributor'])
        
        if not commit_col or not name_col:
            self.logger.warning("Could not find required columns in contributors data")
            return {}
        
        # Filter active contributors
        active_contributors = contributors_df[
            contributors_df[commit_col] >= self.config.contributors_min_commits
        ].sort_values(commit_col, ascending=False).head(self.config.top_n)
        
        self._create_contributor_plots(active_contributors, contributors_df, commit_col, name_col)
        
        return {
            "total_contributors": len(contributors_df),
            "active_contributors": len(active_contributors),
            "top_contributor": active_contributors.iloc[0][name_col] if not active_contributors.empty else None
        }
    
    def _analyze_from_commits(self, commits_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze contributors derived from commit data."""
        self.logger.info("Analyzing contributors from commit data...")
        
        # Create contributor statistics from commits
        contributor_stats = commits_df.groupby('author_email').agg({
            'commit_hash': 'count',
            'author_name': 'first',
            'commit_date': ['min', 'max']
        }).round(2)
        
        contributor_stats.columns = ['commits', 'name', 'first_commit', 'last_commit']
        contributor_stats = contributor_stats.sort_values('commits', ascending=False)
        
        # Filter active contributors
        active_contributors = contributor_stats[
            contributor_stats['commits'] >= self.config.contributors_min_commits
        ].head(self.config.top_n)
        
        self._create_contributor_plots_from_commits(active_contributors, contributor_stats, commits_df)
        
        return {
            "total_contributors": len(contributor_stats),
            "active_contributors": len(active_contributors),
            "top_contributor": active_contributors.iloc[0]['name'] if not active_contributors.empty else None
        }
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _create_contributor_plots(self, active_contributors: pd.DataFrame, 
                                all_contributors: pd.DataFrame, commit_col: str, name_col: str):
        """Create contributor analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Contributor Analysis', fontsize=18, fontweight='bold')
        
        # 1. Top contributors
        y_pos = np.arange(len(active_contributors))
        colors = plt.cm.plasma(np.linspace(0, 1, len(active_contributors)))
        
        ax1.barh(y_pos, active_contributors[commit_col], color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(active_contributors[name_col])
        ax1.invert_yaxis()
        ax1.set_xlabel('Number of Commits')
        ax1.set_title(f'Top {len(active_contributors)} Contributors', fontweight='bold')
        
        # 2. Contribution distribution
        commit_counts = all_contributors[commit_col]
        ax2.hist(commit_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(commit_counts.median(), color='red', linestyle='--', 
                   label=f'Median: {commit_counts.median():.1f}')
        ax2.set_xlabel('Commits per Contributor')
        ax2.set_ylabel('Number of Contributors')
        ax2.set_title('Contribution Distribution', fontweight='bold')
        ax2.legend()
        ax2.set_yscale('log')
        
        # 3. Contributor categories
        self._plot_contributor_categories(ax3, commit_counts)
        
        # 4. Pareto analysis
        self._plot_pareto_analysis(ax4, commit_counts)
        
        plt.tight_layout()
        self._save_plot("02_contributor_analysis")
        plt.close()
        
        # Save data
        self._save_data(active_contributors, "active_contributors")
    
    def _create_contributor_plots_from_commits(self, active_contributors: pd.DataFrame,
                                             all_contributors: pd.DataFrame, commits_df: pd.DataFrame):
        """Create contributor plots from derived commit data."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Contributor Analysis', fontsize=18, fontweight='bold')
        
        # 1. Top contributors
        y_pos = np.arange(len(active_contributors))
        colors = plt.cm.plasma(np.linspace(0, 1, len(active_contributors)))
        
        ax1.barh(y_pos, active_contributors['commits'], color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(active_contributors['name'])
        ax1.invert_yaxis()
        ax1.set_xlabel('Number of Commits')
        ax1.set_title(f'Top {len(active_contributors)} Contributors', fontweight='bold')
        
        # 2. Contribution distribution
        commit_counts = all_contributors['commits']
        ax2.hist(commit_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(commit_counts.median(), color='red', linestyle='--', 
                   label=f'Median: {commit_counts.median():.1f}')
        ax2.set_xlabel('Commits per Contributor')
        ax2.set_ylabel('Number of Contributors')
        ax2.set_title('Contribution Distribution', fontweight='bold')
        ax2.legend()
        ax2.set_yscale('log')
        
        # 3. Contributor categories
        self._plot_contributor_categories(ax3, commit_counts)
        
        # 4. Monthly active contributors
        monthly_contributors = commits_df.set_index('commit_date').resample('M')['author_email'].nunique()
        monthly_contributors.plot(kind='line', ax=ax4, linewidth=2, marker='o')
        ax4.set_title('Active Contributors per Month', fontweight='bold')
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Unique Active Contributors')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("02_contributor_analysis")
        plt.close()
        
        # Save data
        self._save_data(active_contributors, "active_contributors")
    
    def _plot_contributor_categories(self, ax, commit_counts):
        """Plot contributor categories pie chart."""
        categories = ['Casual (1-5)', 'Regular (6-20)', 'Active (21-100)', 'Core (100+)']
        thresholds = [1, 6, 21, 100]
        counts = []
        
        for i in range(len(thresholds)):
            if i == 0:
                count = len(commit_counts[commit_counts < thresholds[i+1]]) if i+1 < len(thresholds) else 0
            elif i == len(thresholds) - 1:
                count = len(commit_counts[commit_counts >= thresholds[i]])
            else:
                count = len(commit_counts[(commit_counts >= thresholds[i]) & 
                                        (commit_counts < thresholds[i+1])])
            counts.append(count)
        
        if sum(counts) > 0:
            ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            ax.set_title('Contributor Categories', fontweight='bold')
    
    def _plot_pareto_analysis(self, ax, commit_counts):
        """Plot Pareto analysis of contributions."""
        sorted_commits = commit_counts.sort_values(ascending=False)
        cumulative_pct = (sorted_commits.cumsum() / sorted_commits.sum() * 100)
        contributor_pct = np.arange(1, len(sorted_commits) + 1) / len(sorted_commits) * 100
        
        ax.plot(contributor_pct, cumulative_pct, linewidth=2, color='green')
        ax.axhline(80, color='red', linestyle='--', alpha=0.7, label='80% of commits')
        ax.set_xlabel('Contributors (%)')
        ax.set_ylabel('Cumulative Commits (%)')
        ax.set_title('Pareto Analysis (80/20 Rule)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

class FileEvolutionModule(AnalysisModule):
    """Analyzes file evolution patterns and language trends."""
    
    @property
    def name(self) -> str:
        return "files"
    
    @property
    def required_data(self) -> Set[str]:
        return {"file_changes"}
    
    def analyze(self) -> Dict[str, Any]:
        file_changes_df = self.dataframes["file_changes"]
        
        if file_changes_df.empty:
            self.logger.warning("No file changes data available")
            return {}
        
        self.logger.info("Analyzing file evolution patterns...")
        
        # File type evolution over time
        monthly_extensions = (
            file_changes_df.groupby([
                file_changes_df['commit_date'].dt.to_period('M'), 
                'file_extension'
            ]).size().unstack(fill_value=0)
        )
        
        # Focus on top extensions
        top_extensions = file_changes_df['file_extension'].value_counts().head(8).index
        monthly_extensions_top = monthly_extensions.reindex(columns=top_extensions).fillna(0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle('File Evolution Analysis', fontsize=18, fontweight='bold')
        
        # 1. Stacked area chart of file types over time
        if not monthly_extensions_top.empty:
            monthly_extensions_top.plot(kind='area', stacked=True, ax=ax1, 
                                      alpha=0.8, colormap='tab10')
            ax1.set_title('File Type Changes Over Time', fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Changes')
            ax1.legend(title='File Extensions', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. File type distribution
        ext_counts = file_changes_df['file_extension'].value_counts().head(10)
        ext_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
        ax2.set_title('File Type Distribution', fontweight='bold')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        self._save_plot("03_file_evolution")
        plt.close()
        
        # Analyze language statistics if available
        if "languages" in self.dataframes and not self.dataframes["languages"].empty:
            self._analyze_languages()
        
        # Save data
        self._save_data(monthly_extensions.reset_index(), "monthly_file_extensions")
        self._save_data(ext_counts.reset_index(), "file_extension_counts")
        
        return {
            "total_file_changes": len(file_changes_df),
            "unique_files": file_changes_df['file_path'].nunique(),
            "top_file_types": ext_counts.head(5).to_dict(),
            "most_active_extension": ext_counts.index[0] if not ext_counts.empty else None
        }
    
    def _analyze_languages(self):
        """Analyze programming language distribution."""
        languages_df = self.dataframes["languages"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle('Programming Language Analysis', fontsize=18, fontweight='bold')
        
        # 1. Language pie chart
        top_langs = languages_df.nlargest(8, 'file_count')
        others_count = languages_df.loc[~languages_df.index.isin(top_langs.index), 'file_count'].sum()
        
        plot_data = top_langs.set_index('language')['file_count'].to_dict()
        if others_count > 0:
            plot_data['Others'] = others_count
            
        ax1.pie(plot_data.values(), labels=plot_data.keys(), autopct='%1.1f%%', startangle=90)
        ax1.set_title('Language Distribution by File Count', fontweight='bold')
        
        # 2. Language bar chart
        top_langs.plot(x='language', y='file_count', kind='bar', ax=ax2, legend=False, color='lightcoral')
        ax2.set_title('File Count by Language (Top 8)', fontweight='bold')
        ax2.set_xlabel('Programming Language')
        ax2.set_ylabel('Number of Files')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._save_plot("04_language_analysis")
        plt.close()

class HotspotsModule(AnalysisModule):
    """Identifies code hotspots and analyzes repository complexity."""
    
    @property
    def name(self) -> str:
        return "hotspots"
    
    @property
    def required_data(self) -> Set[str]:
        return {"file_changes"}
    
    def analyze(self) -> Dict[str, Any]:
        file_changes_df = self.dataframes["file_changes"]
        
        if file_changes_df.empty:
            self.logger.warning("No file changes data for hotspot analysis")
            return {}
        
        self.logger.info("Analyzing code hotspots and complexity...")
        
        # File change frequency analysis
        file_changes = file_changes_df['file_path'].value_counts()
        directory_activity = file_changes_df['directory'].value_counts().head(15)
        depth_analysis = file_changes_df['path_depth'].value_counts().sort_index()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Repository Hotspots and Complexity Analysis', fontsize=18, fontweight='bold')
        
        # 1. Top changed files
        top_files = file_changes.head(self.config.top_n)
        y_pos = np.arange(len(top_files))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_files)))
        
        ax1.barh(y_pos, top_files.values, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([Path(f).name for f in top_files.index])
        ax1.invert_yaxis()
        ax1.set_xlabel('Number of Changes')
        ax1.set_title(f'Top {self.config.top_n} Most Changed Files', fontweight='bold')
        
        # 2. Directory activity
        directory_activity.plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title('Most Active Directories', fontweight='bold')
        ax2.set_xlabel('Number of Changes')
        
        # 3. Path depth distribution
        depth_analysis.plot(kind='bar', ax=ax3, color='lightblue', alpha=0.8)
        ax3.set_title('File Path Depth Distribution', fontweight='bold')
        ax3.set_xlabel('Path Depth (directories)')
        ax3.set_ylabel('Number of Changes')
        ax3.tick_params(axis='x', rotation=0)
        
        # 4. Hotspot categorization
        self._plot_hotspot_categories(ax4, file_changes)
        
        plt.tight_layout()
        self._save_plot("05_hotspots_complexity")
        plt.close()
        
        # Create detailed hotspot analysis
        hotspot_analysis = self._create_hotspot_analysis(file_changes)
        self._save_data(hotspot_analysis, "hotspot_analysis")
        
        return {
            "total_hotspots": len(file_changes),
            "critical_hotspots": len(file_changes[file_changes >= file_changes.quantile(0.95)]),
            "most_changed_file": file_changes.index[0] if not file_changes.empty else None,
            "max_changes": file_changes.iloc[0] if not file_changes.empty else 0
        }
    
    def _plot_hotspot_categories(self, ax, file_changes):
        """Plot hotspot categorization pie chart."""
        percentiles = [0.5, 0.8, 0.95]
        thresholds = [file_changes.quantile(p) for p in percentiles]
        categories = ['Stable', 'Active', 'Hot', 'Critical']
        
        counts = []
        for i, thresh in enumerate([0] + thresholds):
            if i < len(thresholds):
                upper = thresholds[i]
                count = len(file_changes[(file_changes > thresh) & (file_changes <= upper)])
            else:
                count = len(file_changes[file_changes > thresh])
            counts.append(count)
        
        if sum(counts) > 0:
            ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90,
                   colors=['lightgreen', 'yellow', 'orange', 'red'])
            ax.set_title('File Change Hotspot Categories', fontweight='bold')
    
    def _create_hotspot_analysis(self, file_changes):
        """Create detailed hotspot analysis dataframe."""
        percentiles = [0.5, 0.8, 0.95]
        thresholds = [file_changes.quantile(p) for p in percentiles]
        categories = ['Stable', 'Active', 'Hot', 'Critical']
        
        hotspot_analysis = pd.DataFrame({
            'file_path': file_changes.index,
            'change_count': file_changes.values,
        })
        
        hotspot_analysis['category'] = pd.cut(
            file_changes.values, 
            bins=[-np.inf] + thresholds + [np.inf],
            labels=categories
        )
        
        return hotspot_analysis

class ArchitecturalMetricsModule(AnalysisModule):
    """Analyzes architectural stability metrics if available."""
    
    @property
    def name(self) -> str:
        return "architectural"
    
    @property
    def required_data(self) -> Set[str]:
        return {"commits"}
    
    def analyze(self) -> Dict[str, Any]:
        commits_df = self.dataframes["commits"]
        
        # Check if architectural metrics are available
        if 'asi' not in commits_df.columns or commits_df['asi'].isnull().all():
            self.logger.warning("Architectural metrics not available")
            return {"message": "Architectural metrics not found in dataset"}
        
        self.logger.info("Analyzing architectural stability metrics...")
        
        # Create comprehensive architectural metrics dashboard
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Architectural Stability Metrics Dashboard', fontsize=18, fontweight='bold')
        
        # 1. ASI Distribution
        commits_df['asi'].dropna().hist(bins=30, ax=ax1, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Architectural Stability Index Distribution', fontweight='bold')
        ax1.set_xlabel('ASI Score')
        ax1.set_ylabel('Frequency')
        
        # 2. Survival vs Churn scatter
        if 'survival_rate' in commits_df.columns and 'churn_rate' in commits_df.columns:
            scatter = ax2.scatter(commits_df['survival_rate'], commits_df['churn_rate'], 
                                c=commits_df['asi'], cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax2, label='ASI Score')
            ax2.set_xlabel('Survival Rate')
            ax2.set_ylabel('Churn Rate')
            ax2.set_title('Survival vs Churn Rate', fontweight='bold')
        
        # 3. ASI trend over time
        asi_trend = commits_df.set_index('commit_date')['asi'].rolling('30D', min_periods=1).mean()
        asi_trend.plot(ax=ax3, linewidth=2, color='green')
        ax3.set_title('ASI Trend (30-day average)', fontweight='bold')
        ax3.set_ylabel('Average ASI')
        
        # 4. Age distribution
        if 'age_days' in commits_df.columns:
            commits_df['age_days'].dropna().hist(bins=30, ax=ax4, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_title('Commit Age Distribution', fontweight='bold')
            ax4.set_xlabel('Age (Days)')
        
        # 5. High vs Low ASI comparison
        high_asi = commits_df.query('asi > 0.8')['lines_changed'].dropna() if 'lines_changed' in commits_df.columns else pd.Series()
        low_asi = commits_df.query('asi < 0.3')['lines_changed'].dropna() if 'lines_changed' in commits_df.columns else pd.Series()
        
        if not high_asi.empty and not low_asi.empty:
            ax5.boxplot([high_asi, low_asi], labels=['High ASI (>0.8)', 'Low ASI (<0.3)'])
            ax5.set_title('Lines Changed: High vs Low ASI', fontweight='bold')
            ax5.set_ylabel('Lines Changed')
            ax5.set_yscale('log')
        
        # 6. Metrics correlation heatmap
        arch_columns = ['asi', 'survival_rate', 'churn_rate', 'age_days', 'lines_changed', 'insertions', 'deletions']
        available_cols = [col for col in arch_columns if col in commits_df.columns]
        
        if len(available_cols) > 1:
            corr_data = commits_df[available_cols].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', ax=ax6)
            ax6.set_title('Architectural Metrics Correlation', fontweight='bold')
        
        plt.tight_layout()
        self._save_plot("06_architectural_metrics")
        plt.close()
        
        # Save summary statistics
        arch_summary = commits_df[available_cols].describe() if available_cols else pd.DataFrame()
        if not arch_summary.empty:
            self._save_data(arch_summary.reset_index(), "architectural_metrics_summary")
        
        return {
            "asi_available": True,
            "avg_asi": commits_df['asi'].mean(),
            "median_asi": commits_df['asi'].median(),
            "high_stability_commits": len(commits_df[commits_df['asi'] > 0.8]),
            "low_stability_commits": len(commits_df[commits_df['asi'] < 0.3])
        }

class ActivityHeatmapModule(AnalysisModule):
    """Creates developer activity heatmaps."""
    
    @property
    def name(self) -> str:
        return "activity"
    
    @property
    def required_data(self) -> Set[str]:
        return {"commits"}
    
    def analyze(self) -> Dict[str, Any]:
        commits_df = self.dataframes["commits"]
        
        if commits_df.empty:
            self.logger.warning("No commit data for activity heatmap")
            return {}
        
        self.logger.info("Creating developer activity heatmap...")
        
        # Create activity heatmap
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = commits_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.reindex(day_order).fillna(0)
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='d', 
                   cbar_kws={'label': 'Number of Commits'})
        plt.title('Developer Activity Patterns\n(Commits by Day of Week and Hour)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Day of Week', fontsize=12)
        
        self._save_plot("07_activity_heatmap")
        plt.close()
        
        # Create interactive heatmap if requested
        if self.config.interactive_plots and HAS_PLOTLY:
            self._create_interactive_heatmap(heatmap_data)
        
        return {
            "peak_activity_day": heatmap_data.sum(axis=1).idxmax(),
            "peak_activity_hour": heatmap_data.sum(axis=0).idxmax(),
            "total_activity_points": heatmap_data.sum().sum()
        }
    
    def _create_interactive_heatmap(self, heatmap_data):
        """Create interactive Plotly heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=list(range(24)),  # Hours
            y=heatmap_data.index,  # Days
            colorscale='YlOrRd',
            colorbar=dict(title="Commits")
        ))
        
        fig.update_layout(
            title='Interactive Development Activity Heatmap',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            template='plotly_white'
        )
        
        filepath = self.interactive_dir / "activity_heatmap.html"
        fig.write_html(filepath)
        self.logger.info(f"Saved interactive heatmap: {filepath}")

class UnifiedGitAnalyzer:
    """Main analyzer class that coordinates all analysis modules."""
    
    def __init__(self, json_file: Path, output_dir: Path, config: AnalysisConfig):
        self.json_file = json_file
        self.output_dir = output_dir
        self.config = config
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data processor
        self.data_processor = DataProcessor(config)
        self.dataframes = {}
        self.raw_data = {}
        
        # Available analysis modules
        self.available_modules = {
            "commits": CommitAnalysisModule,
            "contributors": ContributorAnalysisModule,
            "files": FileEvolutionModule,
            "hotspots": HotspotsModule,
            "architectural": ArchitecturalMetricsModule,
            "activity": ActivityHeatmapModule,
        }
        
        self.results = {}
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """Load and validate data from JSON file."""
        self.dataframes = self.data_processor.load_and_validate(self.json_file)
        self.raw_data = self.data_processor.raw_data
        
    def run_analysis(self):
        """Run selected analysis modules."""
        self.logger.info(f"Running analysis with modules: {self.config.modules}")
        
        successful_modules = 0
        total_modules = len(self.config.modules)
        
        for module_name in tqdm(self.config.modules, desc="Running analysis modules"):
            if module_name not in self.available_modules:
                self.logger.warning(f"Unknown module: {module_name}")
                continue
            
            try:
                self.logger.info(f"Running {module_name} module...")
                module_class = self.available_modules[module_name]
                module = module_class(self.config, self.dataframes, self.output_dir, self.raw_data)
                
                if not module.can_run():
                    self.logger.warning(f"Module {module_name} cannot run - missing required data: {module.required_data}")
                    continue
                
                result = module.analyze()
                self.results[module_name] = result
                successful_modules += 1
                
            except Exception as e:
                self.logger.error(f"Error in {module_name} module: {e}", exc_info=True)
        
        self.logger.info(f"Completed {successful_modules}/{total_modules} modules successfully")
        
    def generate_comprehensive_report(self):
        """Generate comprehensive HTML report."""
        if not self.config.generate_html_report:
            return
        
        self.logger.info("Generating comprehensive HTML report...")
        
        repo_info = self.raw_data.get('repository_info', {})
        repo_name = repo_info.get('repo_name', 'Unknown Repository')
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats()
        
        html_content = self._generate_html_report(repo_name, summary_stats)
        
        with open(self.output_dir / "comprehensive_analysis_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        self.logger.info("HTML report saved as comprehensive_analysis_report.html")
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics across all modules."""
        stats = {}
        
        if "commits" in self.dataframes:
            commits_df = self.dataframes["commits"]
            stats.update({
                "total_commits": len(commits_df),
                "date_range": [commits_df['commit_date'].min().isoformat(), 
                             commits_df['commit_date'].max().isoformat()],
                "unique_authors": commits_df['author_email'].nunique() if 'author_email' in commits_df.columns else 0
            })
        
        if "file_changes" in self.dataframes:
            file_changes_df = self.dataframes["file_changes"]
            stats.update({
                "total_file_changes": len(file_changes_df),
                "unique_files": file_changes_df['file_path'].nunique(),
                "most_active_extension": file_changes_df['file_extension'].value_counts().index[0] if not file_changes_df.empty else None
            })
        
        return stats
    
    #jjj 
    def _generate_html_report(self, repo_name: str, summary_stats: Dict[str, Any]) -> str:
        """Generate the HTML report content."""
        # Get plot files
        plot_files = []
        for plot_path in sorted(self.output_dir.glob("plots/*.png")):
            plot_files.append((plot_path.stem, plot_path.name))
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Repository Analysis - {repo_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
        }}
        .metric-number {{
            font-size: 2.8em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #666;
            font-size: 1.1em;
            font-weight: 500;
        }}
        .section {{
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 25px;
            font-size: 1.8em;
        }}
        .plot-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 25px;
        }}
        .plot-container {{
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        .plot-container h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: #555;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Comprehensive Repository Analysis</h1>
        <p style="font-size: 1.4em; opacity: 0.9;">{repo_name}</p>
    </div>

    <div class="section">
        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-number">{summary_stats.get('total_commits', 'N/A')}</div>
                <div class="metric-label">Total Commits</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">{summary_stats.get('unique_authors', 'N/A')}</div>
                <div class="metric-label">Unique Contributors</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">{summary_stats.get('unique_files', 'N/A')}</div>
                <div class="metric-label">Unique Files Changed</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">{summary_stats.get('total_file_changes', 'N/A')}</div>
                <div class="metric-label">Total File Changes</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Visual Analysis</h2>
        <div class="plot-gallery">
"""
        for title, filename in plot_files:
            clean_title = ' '.join(word.capitalize() for word in title.replace('_', ' ').split())
            html_content += f"""
            <div class="plot-container">
                <h3>{clean_title}</h3>
                <img src="plots/{filename}" alt="{clean_title}">
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
        # Add links to interactive plots if they exist
        interactive_files = sorted(self.output_dir.glob("interactive/*.html"))
        if interactive_files:
            html_content += """
    <div class="section">
        <h2>Interactive Reports</h2>
        <ul>
"""
            for f in interactive_files:
                clean_name = ' '.join(word.capitalize() for word in f.stem.replace('_', ' ').split())
                html_content += f'            <li><a href="interactive/{f.name}" target="_blank">{clean_name}</a></li>\\n'
            html_content += """
        </ul>
    </div>
"""

        html_content += f"""
    <div class="footer">
        <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        return html_content


def main():
    """Main function to run the analyzer from the command line."""
    parser = argparse.ArgumentParser(description="Unified Git Repository Analyzer")
    
    # Required arguments
    parser.add_argument("json_file", type=Path, help="Path to the input JSON data file.")
    parser.add_argument("output_dir", type=Path, help="Directory to save analysis results.")
    
    # Module selection
    parser.add_argument("-m", "--modules", nargs='+', 
                        default=["commits", "contributors", "files", "hotspots", "activity", "architectural"],
                        help="List of analysis modules to run.")
    
    # Data filtering
    parser.add_argument("--date-range", type=str, help="Date range to filter commits (e.g., 'YYYY-MM-DD,YYYY-MM-DD').")
    parser.add_argument("--exclude-paths", type=str, help="Comma-separated regex patterns to exclude file paths.")
    parser.add_argument("--min-file-changes", type=int, default=2, help="Minimum changes for a file to be included.")
    parser.add_argument("--contributors-min-commits", type=int, default=5, help="Minimum commits for a contributor to be considered active.")
    
    # Analysis parameters
    parser.add_argument("--top-n", type=int, default=20, help="Number of top items to display in charts.")
    
    # Output settings
    parser.add_argument("--output-formats", nargs='+', default=["png"], help="Output formats for plots (e.g., png, svg, pdf).")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive plots (requires Plotly).")
    parser.add_argument("--no-html-report", action="store_false", dest="generate_html_report",
                        help="Disable generation of the comprehensive HTML report.")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config object
    config = AnalysisConfig(
        modules=args.modules,
        date_range=args.date_range,
        exclude_paths=args.exclude_paths,
        min_file_changes=args.min_file_changes,
        contributors_min_commits=args.contributors_min_commits,
        top_n=args.top_n,
        output_formats=args.output_formats,
        interactive_plots=args.interactive,
        generate_html_report=args.generate_html_report
    )
    
    # Check for optional dependencies if interactive plots are requested
    if config.interactive_plots and not HAS_PLOTLY:
        print("Warning: Interactive plots requested, but Plotly is not installed. Skipping.", file=sys.stderr)
        config.interactive_plots = False
        
    try:
        analyzer = UnifiedGitAnalyzer(
            json_file=args.json_file,
            output_dir=args.output_dir,
            config=config
        )
        
        analyzer.load_data()
        analyzer.run_analysis()
        analyzer.generate_comprehensive_report()
        
        print(f"\nAnalysis complete. Results are saved in: {args.output_dir.resolve()}")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()