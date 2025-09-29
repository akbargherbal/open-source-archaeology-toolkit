import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import warnings

# --- 1. Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
warnings.filterwarnings('ignore')

# --- 2. Default Configuration ---
class Config:
    """
    Default configuration settings.
    These will be overridden by dynamic thresholds where applicable.
    """
    TOP_CONTRIBUTORS_COUNT = 10
    HOTSPOT_FILE_COUNT = 15
    DEFAULT_HIGH_CHURN_COMMIT_COUNT = 50
    DEFAULT_STABLE_CODE_DAYS = 180

# --- 3. JSON Processing Module ---
class JSONProcessor:
    """
    Loads, validates, and preprocesses the input JSON data,
    flattening nested structures for easier analysis.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found at '{self.file_path}'")
        self.data = {}

    def load_and_validate(self):
        logging.info(f"Loading data from '{self.file_path}'...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        required_keys = ['repository_info', 'commits', 'file_changes', 'contributors']
        for key in required_keys:
            if key not in raw_data or not raw_data[key]:
                logging.warning(f"Data for key '{key}' is missing or empty. Some analyses may be skipped.")
                raw_data[key] = [] # Ensure key exists to prevent crashes

        self.data['repository_info'] = raw_data.get('repository_info', {})
        
        # Flatten nested architectural metrics from commits
        commits_df = pd.json_normalize(raw_data.get('commits', []), sep='_')
        if not commits_df.empty:
            commits_df['author_date'] = pd.to_datetime(commits_df['author_date'], utc=True)
        self.data['commits_df'] = commits_df

        changes_df = pd.DataFrame(raw_data.get('file_changes', []))
        if not changes_df.empty:
            changes_df['commit_date'] = pd.to_datetime(changes_df['commit_date'], utc=True)
        self.data['changes_df'] = changes_df
        
        self.data['contributors_df'] = pd.DataFrame(raw_data.get('contributors', []))
        
        logging.info("Data loaded and pre-processed successfully.")
        return self.data

# --- 4. Analysis Engine Module ---
class AnalysisEngine:
    """
    Performs dynamic, adaptive analysis on the repository data.
    """
    def __init__(self, data: dict):
        self.data = data
        self.results = {}
        self.thresholds = {}
        self._validate_data_completeness()
        self._calculate_adaptive_thresholds()

    def _validate_data_completeness(self):
        logging.info("Validating data completeness...")
        if self.data["commits_df"].empty:
            logging.warning("No commit data found. Historical and architectural analysis will be skipped.")
        if self.data["changes_df"].empty:
            logging.warning("No file change data found. Development pattern analysis will be skipped.")
        if "architectural_metrics_asi" not in self.data["commits_df"].columns:
            logging.info("Architectural Stability Index (ASI) not found. Hotspot analysis will be less accurate.")

    def _calculate_adaptive_thresholds(self):
        logging.info("Calculating adaptive thresholds...")
        commits_df = self.data["commits_df"]
        if commits_df.empty:
            self.thresholds['high_churn'] = Config.DEFAULT_HIGH_CHURN_COMMIT_COUNT
            self.thresholds['stable_days'] = Config.DEFAULT_STABLE_CODE_DAYS
            logging.warning("Cannot calculate adaptive thresholds due to missing commit data. Using defaults.")
            return

        repo_age_days = (commits_df["author_date"].max() - commits_df["author_date"].min()).days
        
        # High churn is the 90th percentile of changes per file, with a minimum floor.
        if not self.data['changes_df'].empty:
            churn_distribution = self.data['changes_df']['file_path'].value_counts()
            high_churn_quantile = int(churn_distribution.quantile(0.90))
            self.thresholds['high_churn'] = max(10, high_churn_quantile)
        else:
            self.thresholds['high_churn'] = Config.DEFAULT_HIGH_CHURN_COMMIT_COUNT

        # Stable code is defined as a fraction of the repo's age, with a cap.
        self.thresholds['stable_days'] = min(180, int(repo_age_days * 0.25))

        logging.info(f"Adaptive thresholds set: High Churn > {self.thresholds['high_churn']} commits, Stable Code > {self.thresholds['stable_days']} days.")

    def run_all_analyses(self):
        logging.info("Starting repository analysis...")
        self.analyze_repository_health()
        self.analyze_development_patterns()
        self.analyze_historical_insights()
        self.analyze_team_dynamics()
        logging.info("Analysis complete.")
        return self.results

    def analyze_repository_health(self):
        logging.info("Analyzing repository health...")
        repo_info = self.data['repository_info']
        commits_df = self.data['commits_df']
        
        repo_age_days = 0
        if not commits_df.empty:
            repo_age_days = (commits_df['author_date'].max() - commits_df['author_date'].min()).days

        language_stats = self.data['changes_df']['language'].value_counts(normalize=True) * 100
        
        self.results['repository_health'] = {
            'name': repo_info.get('repo_name', 'N/A'),
            'total_commits': repo_info.get('total_commits', len(commits_df)),
            'total_contributors': len(self.data['contributors_df']),
            'repo_age_days': repo_age_days,
            'language_distribution': language_stats.to_dict()
        }

    def analyze_development_patterns(self):
        logging.info("Analyzing development patterns...")
        changes_df = self.data['changes_df']
        if changes_df.empty:
            logging.warning("Skipping development pattern analysis due to empty file change data.")
            self.results['development_patterns'] = {}
            return

        # Performance Enhancement: Use groupby().agg() for single-pass analysis
        file_metrics = changes_df.groupby('file_path').agg(
            commit_count=('commit_hash', 'nunique'),
            last_modified_date=('commit_date', 'max')
        ).reset_index()

        # Integrate Architectural Metrics (ASI) if available
        if 'architectural_metrics_asi' in self.data['commits_df'].columns:
            commit_asi = self.data['commits_df'][['hash', 'architectural_metrics_asi']].rename(columns={'hash': 'commit_hash'})
            changes_with_asi = pd.merge(changes_df, commit_asi, on='commit_hash', how='left')
            avg_asi = changes_with_asi.groupby('file_path')['architectural_metrics_asi'].mean().reset_index()
            file_metrics = pd.merge(file_metrics, avg_asi, on='file_path', how='left')
            file_metrics['architectural_metrics_asi'].fillna(0, inplace=True)
            # Enhanced Impact Score
            file_metrics['impact_score'] = file_metrics['commit_count'] * (1 + file_metrics['architectural_metrics_asi'])
        else:
            file_metrics['impact_score'] = file_metrics['commit_count']

        now_utc = pd.Timestamp.now(tz='UTC')
        file_metrics['days_since_last_change'] = (now_utc - file_metrics['last_modified_date']).dt.days
        
        stable_files = file_metrics[file_metrics['days_since_last_change'] > self.thresholds['stable_days']]
        volatile_files = file_metrics[file_metrics['commit_count'] > self.thresholds['high_churn']]
        hotspots = file_metrics.sort_values(by='impact_score', ascending=False).head(Config.HOTSPOT_FILE_COUNT)

        self.results['development_patterns'] = {
            'stable_files': stable_files.sort_values(by='days_since_last_change', ascending=False),
            'volatile_files': volatile_files.sort_values(by='commit_count', ascending=False),
            'architectural_hotspots': hotspots
        }

    def analyze_historical_insights(self):
        logging.info("Analyzing historical insights...")
        commits_df = self.data['commits_df'].copy()
        if commits_df.empty: return

        commits_df.set_index('author_date', inplace=True)
        commit_velocity = commits_df.resample('M').size()
        commit_velocity.index = commit_velocity.index.strftime('%Y-%m')
        self.results['historical_insights'] = {'commit_velocity': commit_velocity.to_dict()}

    def analyze_team_dynamics(self):
        logging.info("Analyzing team dynamics...")
        contributors_df = self.data['contributors_df']
        if contributors_df.empty: return
        
        # Adapt to schema: find the commit count column regardless of name
        count_col = next((col for col in contributors_df.columns if 'commit' in col.lower() and 'count' in col.lower()), None)
        
        if count_col:
            top_contributors = contributors_df.sort_values(by=count_col, ascending=False).head(Config.TOP_CONTRIBUTORS_COUNT)
            self.results['team_dynamics'] = {'top_contributors': top_contributors, 'count_col': count_col}
        else:
            logging.warning("Could not find a 'commit_count' column in contributors data. Skipping team dynamics.")
            self.results['team_dynamics'] = {'top_contributors': pd.DataFrame()}

# --- 5. Enhanced Markdown Generation Module ---
class MarkdownGenerator:
    """
    Generates a rich, actionable markdown report from the analysis results.
    Enhanced version addresses the gaps identified in the review.
    """
    def __init__(self, results: dict, thresholds: dict):
        self.results = results
        self.thresholds = thresholds
        self.report_lines = []

    def _add_title(self, title): self.report_lines.append(f"# {title}\n")
    def _add_heading(self, heading, level=2): self.report_lines.append(f"{'#' * level} {heading}\n")
    def _add_text(self, text): self.report_lines.append(f"{text}\n")
    def _add_table(self, df: pd.DataFrame):
        if df.empty: self.report_lines.append("> No data available for this section.\n")
        else: self.report_lines.append(df.to_markdown(index=False) + "\n")
    def _add_list(self, items: dict):
        for key, value in items.items(): self.report_lines.append(f"- **{key}**: {value}")
        self.report_lines.append("\n")
    def _add_collapsible_section(self, summary: str, content: str):
        self.report_lines.append("<details>")
        self.report_lines.append(f"<summary>{summary}</summary>\n")
        self.report_lines.append(content)
        self.report_lines.append("\n</details>\n")

    def _generate_glossary_section(self):
        self._add_heading("📚 Metrics Glossary", level=2)
        
        glossary_items = {
            "**Impact Score**": "Calculated as `commit_count × (1 + ASI)`. Higher scores indicate files that are both frequently changed and architecturally significant.",
            "**ASI (Architectural Stability Index)**": "Range 0.0-1.0. Measures `survival_rate / (1 + normalized_churn)`. Higher values suggest code that has proven stable over time relative to its change frequency.",
            "**Architectural Hotspots**": "Files with high impact scores, indicating they are central to the system architecture and frequently modified. Understanding these is crucial for comprehending the codebase.",
            "**Volatile Files**": f"Files changed more than {self.thresholds['high_churn']} times. These areas see frequent development activity.",
            "**Stable Files**": f"Files unchanged for more than {self.thresholds['stable_days']} days. These may represent mature, reliable code or potentially neglected areas.",
            "**Survival Rate**": "Proportion of repository lifetime that has passed since a commit was made. Higher values indicate older, more persistent code.",
            "**Churn Rate**": "Lines changed per month since the commit. Higher values suggest areas under active modification."
        }
        
        for term, definition in glossary_items.items():
            self.report_lines.append(f"- {term}: {definition}")
        self.report_lines.append("\n")

    def _categorize_hotspots(self, hotspots_df):
        """Categorize files by type for better explanation."""
        categories = {
            "Package Management": [],
            "Core Logic": [],
            "Configuration": [],
            "UI/Frontend": [],
            "Documentation": [],
            "Other": []
        }
        
        for _, row in hotspots_df.head(15).iterrows():
            file_path = row['file_path'].lower()
            
            if any(x in file_path for x in ['package.json', 'package-lock.json', 'pnpm-lock.yaml', 'yarn.lock']):
                categories["Package Management"].append(row)
            elif any(x in file_path for x in ['.js', '.ts', '.py']) and 'test' not in file_path:
                categories["Core Logic"].append(row)
            elif any(x in file_path for x in ['.yml', '.yaml', '.json', 'dockerfile', 'makefile']):
                categories["Configuration"].append(row)
            elif any(x in file_path for x in ['.css', '.html', '.vue', '.jsx', '.tsx']):
                categories["UI/Frontend"].append(row)
            elif any(x in file_path for x in ['readme', 'changelog', '.md']):
                categories["Documentation"].append(row)
            else:
                categories["Other"].append(row)
        
        return categories

    def _generate_onboarding_recommendations(self):
        self._add_heading("🎯 Actionable Recommendations for New Developers")
        patterns = self.results.get('development_patterns', {})
        if not patterns or patterns['architectural_hotspots'].empty:
            self._add_text("> Analysis data insufficient to generate recommendations.")
            return

        hotspots = patterns['architectural_hotspots']
        stable_files = patterns['stable_files']
        
        self._add_text("To get up to speed quickly, focus on the following areas:")
        
        # Enhanced hotspot explanation with categorization
        self._add_heading("1. 🏗️ Study These Core Files (Architectural Hotspots)", level=3)
        self._add_text("These files have high impact scores, meaning they're both frequently changed and architecturally significant:")
        
        hotspot_categories = self._categorize_hotspots(hotspots)
        
        for category, files in hotspot_categories.items():
            if files:
                self.report_lines.append(f"**{category}:**")
                for file_info in files[:3]:  # Top 3 per category
                    file_path = file_info['file_path']
                    impact = file_info['impact_score']
                    commits = file_info['commit_count']
                    self.report_lines.append(f"- `{file_path}` (Impact: {impact:.1f}, {commits} commits)")
                self.report_lines.append("")

        self._add_heading("2. 💡 Good First Issues Might Be Here", level=3)
        self._add_text("These files are modified often but have lower architectural impact, making them safer to change:")
        volatile_low_impact = patterns['volatile_files'][~patterns['volatile_files']['file_path'].isin(hotspots['file_path'])]['file_path'].tolist()
        if volatile_low_impact:
            self.report_lines.append("- `"+"`\n- `".join(volatile_low_impact[:5])+"`")
        else:
            self.report_lines.append("- (No specific files identified, look at general volatile files)")

        self._add_heading("3. ⚠️ Avoid Modifying These (Initially)", level=3)
        self._add_text(f"These files are highly stable (unchanged for >{self.thresholds['stable_days']} days) and likely form the repository's foundation:")
        if not stable_files.empty:
             self.report_lines.append("- `"+"`\n- `".join(stable_files['file_path'].tolist()[:5])+"`")
        self.report_lines.append("\n")

    def _explain_language_role(self, language):
        """Provide context for what each language typically represents."""
        roles = {
            'JavaScript': 'Core application logic',
            'TypeScript': 'Type-safe application code',
            'JSON': 'Configuration and data',
            'YAML': 'Configuration files',
            'CSS': 'User interface styling',
            'HTML': 'User interface structure',
            'Python': 'Scripts and utilities',
            'Shell': 'Build and deployment',
            'Dockerfile': 'Containerization',
            'Markdown': 'Documentation'
        }
        return roles.get(language, 'Supporting files')

    def _generate_language_analysis(self):
        health = self.results.get('repository_health', {})
        lang_dist = health.get('language_distribution', {})
        
        self._add_heading("Programming Language Breakdown", level=3)
        
        if not lang_dist:
            self._add_text("No language data available.")
            return
        
        # Categorize languages
        major_langs = {k: v for k, v in lang_dist.items() if v >= 5.0}
        minor_langs = {k: v for k, v in lang_dist.items() if v < 5.0}
        
        # Major languages table
        if major_langs:
            major_df = pd.DataFrame(list(major_langs.items()), columns=['Language', 'Percentage'])
            major_df['Percentage'] = major_df['Percentage'].map('{:.2f}%'.format)
            major_df['Role'] = major_df['Language'].map(self._explain_language_role)
            self._add_table(major_df)
        
        # Minor languages explanation
        if minor_langs:
            self._add_heading("Minor Languages (< 5%)", level=4)
            minor_explanation = "These languages likely represent: "
            categories = []
            
            if any(lang in minor_langs for lang in ['Python', 'Shell', 'Batch']):
                categories.append("build/deployment scripts")
            if any(lang in minor_langs for lang in ['Java', 'C++', 'C']):
                categories.append("legacy components or dependencies")
            if any(lang in minor_langs for lang in ['LaTeX', 'Markdown']):
                categories.append("documentation")
            
            if categories:
                minor_explanation += ", ".join(categories) + "."
            else:
                minor_explanation += "specialized tools, legacy code, or project infrastructure."
            
            self._add_text(minor_explanation)

    def _generate_repo_health_section(self):
        self._add_heading("📊 Repository Health Dashboard")
        health = self.results.get('repository_health', {})
        if not health: return
        
        age_years = health['repo_age_days'] / 365.25
        commits_per_year = health.get('total_commits', 0) / max(age_years, 1)
        
        health_metrics = {
            "Total Commits": f"{health.get('total_commits', 0):,}",
            "Total Contributors": f"{health.get('total_contributors', 0):,}",
            "Repository Age": f"{health.get('repo_age_days', 0):,} days (~{age_years:.1f} years)",
            "Average Commits/Year": f"{commits_per_year:.0f}"
        }
        self._add_list(health_metrics)
        
        # Add health interpretation
        self._add_heading("Health Assessment", level=3)
        health_insights = []
        
        if commits_per_year > 500:
            health_insights.append("✅ **High Activity**: This repository shows consistent, active development.")
        elif commits_per_year > 100:
            health_insights.append("✅ **Moderate Activity**: Steady development pace typical of mature projects.")
        else:
            health_insights.append("⚠️ **Low Activity**: May indicate maintenance mode or need for more contributors.")
        
        if health.get('total_contributors', 0) > 100:
            health_insights.append("✅ **Strong Community**: Large contributor base suggests healthy project ecosystem.")
        elif health.get('total_contributors', 0) > 20:
            health_insights.append("✅ **Active Community**: Good contributor diversity.")
        else:
            health_insights.append("⚠️ **Small Team**: Limited contributor base may indicate dependency risk.")
        
        for insight in health_insights:
            self.report_lines.append(insight)
        self.report_lines.append("\n")
        
        # Enhanced language analysis
        self._generate_language_analysis()

    def _generate_dev_patterns_section(self):
        self._add_heading("🛠️ Development Pattern Analysis")
        patterns = self.results.get('development_patterns', {})
        if not patterns: return

        self._add_heading("Architectural Hotspots", level=3)
        self._add_text(f"Files with the highest impact score (commits × ASI). These are critical, complex, and central to the architecture.")
        hotspots_df = patterns.get('architectural_hotspots', pd.DataFrame())
        if not hotspots_df.empty:
            self._add_table(hotspots_df.rename(columns={'commit_count': 'Commits', 'days_since_last_change': 'Days Since Change', 'impact_score': 'Impact Score', 'file_path': 'File', 'architectural_metrics_asi': 'Avg ASI'}).round(2))

        self._add_heading(f"Active Development Zones (>{self.thresholds['high_churn']} commits)", level=3)
        volatile_df = patterns.get('volatile_files', pd.DataFrame())
        self._add_collapsible_section(f"Click to see {len(volatile_df)} volatile files", volatile_df[['file_path', 'commit_count']].to_markdown(index=False))

        self._add_heading(f"Stability Indicators (unchanged >{self.thresholds['stable_days']} days)", level=3)
        stable_df = patterns.get('stable_files', pd.DataFrame())
        self._add_collapsible_section(f"Click to see {len(stable_df)} stable files", stable_df[['file_path', 'days_since_last_change']].to_markdown(index=False))

    def _generate_historical_section(self):
        self._add_heading("📈 Historical Insights")
        history = self.results.get('historical_insights', {})
        if not history: return
        
        self._add_heading("Commit Velocity (Commits per Month)", level=3)
        velocity_df = pd.DataFrame(list(history.get('commit_velocity', {}).items()), columns=['Month', 'Commit Count'])
        if not velocity_df.empty:
            max_commits = velocity_df['Commit Count'].max()
            if max_commits > 0:
                velocity_df['Trend'] = velocity_df['Commit Count'].apply(lambda x: '█' * int(x / max_commits * 50))
            self._add_table(velocity_df)

    def _generate_team_dynamics_section(self):
        self._add_heading("👥 Team Dynamics")
        team = self.results.get('team_dynamics', {})
        if not team or team['top_contributors'].empty: return
        
        self._add_heading(f"Top {Config.TOP_CONTRIBUTORS_COUNT} Contributors", level=3)
        contributors_df = team['top_contributors']
        # Select columns dynamically based on what's available
        display_cols = [col for col in ['author_name', team.get('count_col'), 'first_commit_date', 'last_commit_date'] if col in contributors_df.columns]
        self._add_table(contributors_df[display_cols])

    def _generate_developer_guidance_section(self):
        self._add_heading("🧭 Developer Learning Path", level=2)
        
        self._add_text("""
### For New Team Members:

**Week 1-2: Foundation**
- Read documentation files (README, CONTRIBUTING)
- Study architectural hotspots to understand core system design
- Set up development environment using configuration files

**Week 3-4: Hands-on Learning**
- Start with "good first issues" in volatile but low-impact files
- Review recent commit history in hotspot files to understand changes
- Avoid modifying stable files until you understand their role

**Month 2+: Advanced Contributions**
- Begin contributing to architectural hotspots with team review
- Help maintain stable files when updates are needed
- Mentor other new contributors using this same progression

### Hotspot vs. Good First Issues Balance:
- **Hotspots (70% study time)**: Understand the 'why' behind the architecture
- **Good First Issues (30% contribution time)**: Learn the 'how' of making changes safely
        """)

    def generate_report(self, output_path: Path):
        logging.info(f"Generating enhanced markdown report at '{output_path}'...")
        repo_name = self.results.get('repository_health', {}).get('name', 'Repository')
        self._add_title(f"🚀 Git Repository Analysis: `{repo_name}`")
        self._add_text(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        self._generate_glossary_section()
        self._generate_onboarding_recommendations()
        self._generate_repo_health_section()
        self._generate_dev_patterns_section()
        self._generate_historical_section()
        self._generate_team_dynamics_section()
        self._generate_developer_guidance_section()

        with open(output_path, 'w', encoding='utf-8') as f: f.write("\n".join(self.report_lines))
        logging.info("Enhanced report generated successfully.")

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate an enhanced markdown analysis report from a Git metadata JSON file.")
    parser.add_argument("input_file", type=Path, help="Path to the input JSON file.")
    parser.add_argument("--output", "-o", type=Path, default="enhanced_repository_report.md", help="Path to save the output markdown report.")
    args = parser.parse_args()

    try:
        processor = JSONProcessor(args.input_file)
        repository_data = processor.load_and_validate()

        engine = AnalysisEngine(repository_data)
        analysis_results = engine.run_all_analyses()

        generator = MarkdownGenerator(analysis_results, engine.thresholds)
        generator.generate_report(args.output)

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"A configuration or data error occurred: {e}")
    except Exception:
        logging.error("An unexpected error occurred. See traceback below:", exc_info=True)

if __name__ == "__main__":
    main()