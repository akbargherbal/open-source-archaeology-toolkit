#!/usr/bin/env python3
"""
Repository Time Travel Script

Reconstructs the repository structure and state at a specific point in time.
Shows the file tree, content, and metadata as it existed on a given date.

Author: Repository Time Traveler
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

@dataclass
class FileNode:
    """Represents a file or directory in the repository tree."""
    name: str
    path: str
    type: str  # 'file', 'directory', 'symlink'
    size: Optional[int] = None
    mode: Optional[str] = None
    last_modified: Optional[str] = None
    content_preview: Optional[str] = None
    children: Optional[Dict[str, 'FileNode']] = None
    
    def __post_init__(self):
        if self.children is None and self.type == 'directory':
            self.children = {}

@dataclass
class RepoSnapshot:
    """Complete repository snapshot at a specific point in time."""
    timestamp: str
    commit_hash: str
    commit_message: str
    author: str
    branch: Optional[str]
    total_files: int
    total_size: int
    file_types: Dict[str, int]
    root: FileNode
    extraction_date: str

def run_git_command(repo_path: str, command: List[str], allow_failure: bool = False) -> str:
    """Execute a git command and return output."""
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
    except subprocess.CalledProcessError as e:
        if allow_failure:
            return ""
        logging.error(f"Git command failed: {' '.join(command)}")
        logging.error(f"Error: {e.stderr.strip()}")
        return ""
    except Exception as e:
        if allow_failure:
            return ""
        logging.error(f"Unexpected error: {e}")
        return ""

class RepositoryTimeTraveler:
    """Main class for reconstructing repository state at specific timestamps."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.git_dir = self.repo_path / ".git"
        
        if not self.git_dir.exists():
            raise ValueError(f"Not a git repository: {repo_path}")
        
        self.logger = logging.getLogger(__name__)
    
    def find_commit_at_timestamp(self, target_date: str) -> Tuple[str, str, str, str]:
        """Find the closest commit to the target timestamp."""
        self.logger.info(f"Finding commit closest to {target_date}")
        
        # Parse target date
        try:
            target_dt = datetime.fromisoformat(target_date.replace('Z', '+00:00'))
        except ValueError:
            try:
                target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid date format: {target_date}. Use YYYY-MM-DD or ISO format")
        
        # Get commits before the target date
        iso_date = target_dt.strftime('%Y-%m-%dT%H:%M:%S')
        cmd = [
            "log",
            f"--until={iso_date}",
            "--pretty=format:%H%x00%ai%x00%s%x00%an",
            "--all",
            "-1"  # Get the most recent commit before the date
        ]
        
        output = run_git_command(str(self.repo_path), cmd)
        if not output:
            # If no commits before date, get the first commit ever
            cmd = [
                "log",
                "--pretty=format:%H%x00%ai%x00%s%x00%an",
                "--reverse",
                "-1"
            ]
            output = run_git_command(str(self.repo_path), cmd)
            
        if not output:
            raise ValueError("No commits found in repository")
        
        parts = output.split('\x00')
        if len(parts) != 4:
            raise ValueError("Unexpected git log output format")
        
        commit_hash, commit_date, commit_message, author = parts
        self.logger.info(f"Found commit {commit_hash[:8]} from {commit_date}")
        
        return commit_hash, commit_date, commit_message, author
    
    def get_file_tree_at_commit(self, commit_hash: str) -> FileNode:
        """Build complete file tree at the specified commit."""
        self.logger.info(f"Building file tree for commit {commit_hash[:8]}")
        
        # Get the tree listing
        cmd = ["ls-tree", "-r", "--long", commit_hash]
        output = run_git_command(str(self.repo_path), cmd)
        
        if not output:
            self.logger.warning("No files found in commit")
            return FileNode(name="root", path="", type="directory")
        
        root = FileNode(name="root", path="", type="directory")
        file_count = 0
        total_size = 0
        file_types = {}
        
        for line in output.split('\n'):
            if not line.strip():
                continue
                
            # Parse git ls-tree output: mode type hash size path
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            meta_parts = parts[0].split()
            if len(meta_parts) < 4:
                continue
                
            mode, obj_type, obj_hash = meta_parts[0], meta_parts[1], meta_parts[2]
            size = int(meta_parts[3]) if meta_parts[3] != '-' else 0
            file_path = parts[1]
            
            if obj_type != 'blob':  # Only process files for now
                continue
            
            file_count += 1
            total_size += size
            
            # Track file extensions
            path_obj = Path(file_path)
            ext = ''.join(path_obj.suffixes).lower() or 'no_ext'
            file_types[ext] = file_types.get(ext, 0) + 1
            
            # Build tree structure
            self._add_file_to_tree(root, file_path, size, mode, commit_hash, obj_hash)
        
        self.logger.info(f"Processed {file_count} files, total size: {total_size:,} bytes")
        return root
    
    def _add_file_to_tree(self, root: FileNode, file_path: str, size: int, 
                         mode: str, commit_hash: str, obj_hash: str):
        """Add a file to the tree structure, creating directories as needed."""
        parts = Path(file_path).parts
        current = root
        
        # Create directory structure
        for part in parts[:-1]:
            if part not in current.children:
                current.children[part] = FileNode(
                    name=part,
                    path=str(Path(current.path) / part) if current.path else part,
                    type="directory"
                )
            current = current.children[part]
        
        # Add the file
        filename = parts[-1]
        file_node = FileNode(
            name=filename,
            path=file_path,
            type="file",
            size=size,
            mode=mode
        )
        
        # Get file content preview (first few lines for text files)
        content_preview = self._get_file_content_preview(commit_hash, file_path)
        if content_preview:
            file_node.content_preview = content_preview
        
        current.children[filename] = file_node
    
    def _get_file_content_preview(self, commit_hash: str, file_path: str, 
                                max_lines: int = 5, max_chars: int = 500) -> Optional[str]:
        """Get a preview of file content at the specified commit."""
        try:
            cmd = ["show", f"{commit_hash}:{file_path}"]
            content = run_git_command(str(self.repo_path), cmd, allow_failure=True)
            
            if not content:
                return None
            
            # Check if it's likely a binary file
            if '\x00' in content:
                return f"<binary file, {len(content)} bytes>"
            
            lines = content.split('\n')
            preview_lines = lines[:max_lines]
            preview = '\n'.join(preview_lines)
            
            if len(preview) > max_chars:
                preview = preview[:max_chars] + "..."
            
            if len(lines) > max_lines:
                preview += f"\n... ({len(lines) - max_lines} more lines)"
            
            return preview
        except Exception:
            return None
    
    def get_repository_snapshot(self, target_date: str, include_content: bool = True) -> RepoSnapshot:
        """Generate complete repository snapshot at target date."""
        self.logger.info(f"Creating repository snapshot for {target_date}")
        
        commit_hash, commit_date, commit_message, author = self.find_commit_at_timestamp(target_date)
        
        # Build file tree
        root = self.get_file_tree_at_commit(commit_hash)
        
        # Calculate statistics
        file_types = {}
        total_files = 0
        total_size = 0
        
        def count_files(node: FileNode):
            nonlocal total_files, total_size
            if node.type == "file":
                total_files += 1
                if node.size:
                    total_size += node.size
                
                # Count file types
                path_obj = Path(node.path)
                ext = ''.join(path_obj.suffixes).lower() or 'no_ext'
                file_types[ext] = file_types.get(ext, 0) + 1
            
            if node.children:
                for child in node.children.values():
                    count_files(child)
        
        count_files(root)
        
        # Try to determine branch
        branch = self._get_branch_for_commit(commit_hash)
        
        snapshot = RepoSnapshot(
            timestamp=commit_date,
            commit_hash=commit_hash,
            commit_message=commit_message,
            author=author,
            branch=branch,
            total_files=total_files,
            total_size=total_size,
            file_types=file_types,
            root=root,
            extraction_date=datetime.now().isoformat()
        )
        
        return snapshot
    
    def _get_branch_for_commit(self, commit_hash: str) -> Optional[str]:
        """Try to determine which branch contains this commit."""
        cmd = ["branch", "--contains", commit_hash]
        output = run_git_command(str(self.repo_path), cmd, allow_failure=True)
        
        if output:
            # Return first branch (remove * if current)
            first_branch = output.split('\n')[0].strip()
            return first_branch.lstrip('* ')
        
        return None
    
    def print_tree_structure(self, node: FileNode, prefix: str = "", is_last: bool = True, max_depth: int = None, current_depth: int = 0):
        """Print tree structure in a nice ASCII format."""
        if max_depth is not None and current_depth > max_depth:
            return
        
        # Choose the appropriate tree symbols
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{node.name}", end="")
        
        # Add file info
        if node.type == "file" and node.size is not None:
            print(f" ({self._format_size(node.size)})", end="")
        elif node.type == "directory" and node.children:
            print(f" ({len(node.children)} items)", end="")
        
        print()  # New line
        
        # Print children
        if node.children and (max_depth is None or current_depth < max_depth):
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = sorted(node.children.values(), key=lambda x: (x.type != "directory", x.name.lower()))
            
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                self.print_tree_structure(child, child_prefix, is_last_child, max_depth, current_depth + 1)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

def save_snapshot_to_json(snapshot: RepoSnapshot, output_file: str):
    """Save snapshot to JSON file."""
    # Convert FileNode objects to dictionaries for JSON serialization
    def node_to_dict(node: FileNode) -> Dict[str, Any]:
        result = {
            "name": node.name,
            "path": node.path,
            "type": node.type,
            "size": node.size,
            "mode": node.mode,
            "content_preview": node.content_preview
        }
        
        if node.children:
            result["children"] = {name: node_to_dict(child) for name, child in node.children.items()}
        
        return result
    
    snapshot_dict = asdict(snapshot)
    snapshot_dict["root"] = node_to_dict(snapshot.root)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(snapshot_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Snapshot saved to: {output_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reconstruct repository state at a specific point in time",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("target_date", help="Target date (YYYY-MM-DD or ISO format)")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument("--max-depth", type=int, help="Maximum depth for tree display")
    parser.add_argument("--no-content", action="store_true", help="Skip content preview generation")
    parser.add_argument("--tree-only", action="store_true", help="Only show tree structure, don't save JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        traveler = RepositoryTimeTraveler(args.repo_path)
        snapshot = traveler.get_repository_snapshot(
            args.target_date, 
            include_content=not args.no_content
        )
        
        # Print summary
        print(f"\n📅 Repository Time Travel Results")
        print(f"{'='*50}")
        print(f"Target Date: {args.target_date}")
        print(f"Closest Commit: {snapshot.commit_hash[:8]} ({snapshot.timestamp})")
        print(f"Author: {snapshot.author}")
        print(f"Branch: {snapshot.branch or 'Unknown'}")
        print(f"Message: {snapshot.commit_message}")
        print(f"Total Files: {snapshot.total_files:,}")
        print(f"Total Size: {traveler._format_size(snapshot.total_size)}")
        
        print(f"\n📁 File Types:")
        sorted_types = sorted(snapshot.file_types.items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_types[:10]:  # Top 10
            print(f"  {ext}: {count} files")
        
        print(f"\n🌳 Repository Structure:")
        print(f"{'='*50}")
        traveler.print_tree_structure(snapshot.root, max_depth=args.max_depth)
        
        # Save to JSON if requested
        if not args.tree_only:
            output_file = args.output or f"repo_snapshot_{args.target_date.replace('-', '')}.json"
            save_snapshot_to_json(snapshot, output_file)
        
        return 0
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())