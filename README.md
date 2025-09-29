# Open-Source Archaeology Toolkit

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A comprehensive suite of Python tools for the systematic analysis of Git repository history. This toolkit helps developers, architects, and team leads understand how complex codebases evolve over time.

## The Problem: Bridging the Gap

Tutorials teach us how to build applications from scratch. Production codebases, however, are rarely so simple. When approaching a mature open-source project, developers are often faced with thousands of files, years of commits, and architectural decisions that lack obvious context.

The gap between "greenfield" projects and living, evolving systems can be immense. This toolkit aims to bridge that gap using the principles of **Software Archaeology**: studying the history of a codebase to understand its present structure and predict its future trajectory.

## 🚀 Core Philosophy

Instead of just looking at the final code, we analyze its journey. By treating a Git repository like an archaeological site, we can uncover foundational concepts, identify major architectural shifts, and learn from the "ghosts" of refactoring past. This approach transforms a complex codebase from an intimidating monolith into a series of understandable, incremental changes.

## 🛠️ The Toolkit Components

This repository contains several powerful, standalone scripts, each designed for a specific analysis task. They work together to provide a complete workflow, from broad data extraction to fine-grained exploration.

### 1. `git_metadata_extractor.py` - The Foundation

This is the starting point for any analysis. It's a powerful, command-line script that walks an entire Git history and extracts a rich dataset into a single, structured JSON file.

- **Purpose**: To create a comprehensive, queryable dataset of a repository's history.
- **Key Features**:
  - Extracts commit metadata (hash, author, date, message).
  - Logs detailed file changes for every commit.
  - Calculates architectural persistence metrics (ASI, churn, survival rate).
  - Gathers contributor, branch, and tag information.
  - Analyzes language distribution across the repository.
- **Use Case**: Run this once per repository to generate the foundational data for all other tools.

### 2. `git_visualizer.py` - The Automated Dashboard

This script consumes the JSON output from the metadata extractor and generates a beautiful, insightful HTML dashboard with a suite of visualizations.

- **Purpose**: To provide a high-level, visual overview of the repository's evolution.
- **Key Features**:
  - Generates charts for commit activity, contributor patterns, and file evolution.
  - Identifies and visualizes code "hotspots" (frequently changed files).
  - Creates developer activity heatmaps (commits by day/hour).
  - Produces a self-contained HTML report for easy sharing.
- **Use Case**: Get a quick, shareable summary of a project's health and history without writing any code.

### 3. `enhanced_generate_report.py` - The Actionable Onboarding Guide

This script focuses on interpreting the analysis to provide actionable guidance, particularly for new developers joining a project.

- **Purpose**: To create a practical, human-readable Markdown report that serves as a learning path.
- **Key Features**:
  - Identifies and categorizes architectural hotspots.
  - Recommends core files to study first.
  - Suggests "good first issue" areas (volatile but low-impact files).
  - Provides a "Developer Learning Path" with a week-by-week plan.
- **Use Case**: Automatically generate onboarding documentation for a team or for personal study.

### 4. `repo_time_travel.py` - The Time Machine

A specialized utility to reconstruct the exact state of a repository at any given point in time.

- **Purpose**: To inspect the repository's file tree, content, and metadata as it existed on a specific date.
- **Key Features**:
  - Finds the commit closest to a target date.
  - Reconstructs and displays the full file tree.
  - Provides content previews for files as they existed in the past.
  - Outputs the snapshot to a detailed JSON file or a human-readable tree.
- **Use Case**: Investigate the architecture right before a major refactoring or understand what the "v1.0" codebase looked like.

### 5. `gitutils.py` - The Interactive Workbench (Toolkit Centerpiece)

This module is the culmination of the project's goals: a highly portable, fully tested, and well-documented Python library for deep, interactive analysis. Unlike the other scripts, it is designed to be imported into a Jupyter notebook or another Python environment.

- **Purpose**: To empower developers to ask custom, nuanced questions about repository history.
- **Key Features**:
  - **Pure Python & Portable**: Uses the `GitPython` library, making it easy to run anywhere.
  - **Fully Tested**: A comprehensive test suite (`FULLY_TESTED_gitutils/`) ensures reliability.
  - **Rich Helper Functions**: Provides dozens of functions to find core files, analyze contributor tenure, calculate file churn, detect refactorings, and much more.
  - **Pandas Integration**: Outputs data directly into pandas DataFrames for powerful, flexible analysis.
- **Use Case**: For deep-dive, exploratory analysis in a Jupyter notebook. When the automated reports raise questions, `gitutils.py` is the tool you use to find the answers.

## workflow

This toolkit is designed to support a flexible workflow, moving from a high-level overview to specific, deep investigations.

1.  **Extract Data (Once)**
    Use `git_metadata_extractor.py` to process your target repository. This is the most time-intensive step, but you only need to do it once.

    ```bash
    python git_metadata_extractor.py /path/to/your/repo --architectural-metrics --include-contributors -o .
    ```

2.  **Get a Quick Overview**
    Run `git_visualizer.py` on the generated JSON file to get an instant HTML dashboard.

    ```bash
    python git_visualizer.py your_repo_complete_metadata.json analysis_output/
    ```

3.  **Generate an Onboarding Guide**
    Use `enhanced_generate_report.py` to create a learning plan.

    ```bash
    python enhanced_generate_report.py your_repo_complete_metadata.json -o onboarding_guide.md
    ```

4.  **Ask Deeper Questions**
    When you have specific questions, fire up a Jupyter notebook and use `gitutils.py` for a detailed, interactive analysis.

    ```python
    # In your notebook
    import gitutils

    repo = gitutils.load_repository('/path/to/your/repo')
    core_files = gitutils.find_core_files(repo, threshold=0.95)
    churn_df = gitutils.calculate_file_churn(repo)

    print("Core Architectural Files:")
    print(core_files)

    print("\nTop 5 Files by Churn:")
    print(churn_df.head(5))
    ```

5.  **Travel Through Time**
    To inspect the code at a key historical moment (e.g., a major refactoring date you discovered), use `repo_time_travel.py`.
    ```bash
    python repo_time_travel.py /path/to/your/repo "2020-01-01" --max-depth 3
    ```

## ⚙️ Getting Started

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/open-source-archaeology-toolkit.git
    cd open-source-archaeology-toolkit
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    _(Note: You may need to create a `requirements.txt` file)_

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run an analysis:**
    Follow the workflow described above to start exploring a repository.

## 🤝 Contributing

Contributions are welcome! Whether it's improving the analysis algorithms, adding new visualizations, or enhancing the documentation, your help is appreciated. Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
