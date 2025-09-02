#!/usr/bin/env python3
"""
BensBot Codebase Cleanup Tool

This script identifies and selectively removes orphaned, backup, and redundant
files from the BensBot codebase as part of the productionization process.
"""

import os
import sys
import re
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cleanup")

# Root directory
ROOT_DIR = Path(__file__).parent.parent

# Directory for archiving instead of deleting
ARCHIVE_DIR = ROOT_DIR / "archive" / f"legacy_cleanup_{datetime.now().strftime('%Y%m%d')}"

# Files and patterns to identify for cleanup
BACKUP_PATTERNS = [
    r".*\.bak$",
    r".*\.backup$",
    r".*\.old$",
    r".*\.newer[0-9]*$",
    r".*\.save$",
    r".*_old\.[a-zA-Z0-9]+$",
    r".*_backup\.[a-zA-Z0-9]+$",
]

# Standalone app files that are now redundant
REDUNDANT_APPS = [
    "app.py",
    "app_fixed.py",
    "app_modular.py",
    "app_new.py",
]

# Directories that should be consolidated or cleaned
DIRS_TO_CLEAN = [
    "archive/duplicate_scripts",
    "archive/old_app_versions",
]

# Files that will be preserved and NOT cleaned up (important keep list)
PRESERVE_FILES = [
    "trading_bot/dashboard/app.py",  # Dashboard app
    "trading_bot/api/app.py",        # API service
    "trading_bot/main.py",           # Main entry point
    "trading_bot/run_bot.py",        # New runner
    "trading_bot/core/main_orchestrator.py"  # Core orchestrator
]

# Configuration files to consolidate
LEGACY_CONFIG_FILES = [
    "config/broker_config.json",
    "config/risk_management.json",
    "config/persistence_config.json",
    "config/market_regime_config.json",
    "config/market_data_config.json"
]


def is_backup_file(file_path: Path) -> bool:
    """Check if a file matches any backup pattern."""
    for pattern in BACKUP_PATTERNS:
        if re.match(pattern, str(file_path)):
            return True
    return False


def find_duplicate_scripts(directory: Path = ROOT_DIR) -> List[Path]:
    """Find duplicate script files in the codebase."""
    duplicates = []
    
    # Check for the redundant standalone app files
    for app_file in REDUNDANT_APPS:
        app_path = directory / app_file
        if app_path.exists():
            duplicates.append(app_path)
    
    # Check special cleanup directories
    for clean_dir in DIRS_TO_CLEAN:
        dir_path = directory / clean_dir
        if dir_path.exists():
            for file_path in dir_path.glob("**/*"):
                if file_path.is_file():
                    duplicates.append(file_path)
    
    return duplicates


def find_backup_files(directory: Path = ROOT_DIR) -> List[Path]:
    """Find all backup files in the codebase."""
    backup_files = []
    
    for file_path in directory.glob("**/*"):
        if file_path.is_file() and is_backup_file(file_path):
            backup_files.append(file_path)
    
    return backup_files


def is_preserved_file(file_path: Path) -> bool:
    """Check if a file should be preserved."""
    for preserve in PRESERVE_FILES:
        if str(file_path).endswith(preserve):
            return True
    return False


def archive_file(file_path: Path) -> bool:
    """Archive a file instead of deleting it."""
    try:
        # Create relative path structure in archive
        rel_path = file_path.relative_to(ROOT_DIR)
        archive_path = ARCHIVE_DIR / rel_path
        
        # Create parent directories
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file to archive
        shutil.copy2(file_path, archive_path)
        return True
    except Exception as e:
        logger.error(f"Failed to archive {file_path}: {e}")
        return False


def remove_file(file_path: Path, archive_first: bool = True) -> bool:
    """Remove a file, optionally archiving it first."""
    try:
        if is_preserved_file(file_path):
            logger.info(f"Skipping preserved file: {file_path}")
            return False
            
        if archive_first:
            archived = archive_file(file_path)
            if not archived:
                logger.warning(f"Archive failed, will not remove: {file_path}")
                return False
        
        # Remove the file
        os.remove(file_path)
        logger.info(f"Removed file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove {file_path}: {e}")
        return False


def analyze_config_files() -> Dict[str, List[str]]:
    """Analyze configuration files for migration to unified config."""
    config_analysis = {
        "to_migrate": [],
        "already_migrated": [],
        "unknown_format": []
    }
    
    # Check if the unified config exists
    unified_yaml = ROOT_DIR / "config" / "config.yaml"
    unified_json = ROOT_DIR / "config" / "config.json"
    
    has_unified = unified_yaml.exists() or unified_json.exists()
    
    # Check legacy configs
    for config_file in LEGACY_CONFIG_FILES:
        config_path = ROOT_DIR / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    json.load(f)  # Test if valid JSON
                config_analysis["to_migrate"].append(str(config_path))
            except json.JSONDecodeError:
                config_analysis["unknown_format"].append(str(config_path))
    
    return config_analysis


def generate_migration_report() -> str:
    """Generate a report of files to be migrated or cleaned up."""
    backup_files = find_backup_files()
    duplicate_scripts = find_duplicate_scripts()
    config_analysis = analyze_config_files()
    
    report = [
        "# BensBot Codebase Cleanup and Migration Report",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Backup Files to Remove",
        ""
    ]
    
    for file in backup_files:
        report.append(f"- {file}")
    
    report.extend([
        "",
        "## 2. Duplicate Scripts to Consolidate",
        ""
    ])
    
    for file in duplicate_scripts:
        report.append(f"- {file}")
    
    report.extend([
        "",
        "## 3. Configuration Files to Migrate",
        ""
    ])
    
    for file in config_analysis["to_migrate"]:
        report.append(f"- {file}")
    
    report.extend([
        "",
        "## 4. Files with Unknown Format (Manual Review)",
        ""
    ])
    
    for file in config_analysis["unknown_format"]:
        report.append(f"- {file}")
    
    report.extend([
        "",
        "## 5. Multiple Main Entry Points (Consolidate)",
        ""
    ])
    
    # Find redundant main scripts
    main_files = [p for p in Path(ROOT_DIR).glob("**/main*.py") 
                 if not is_preserved_file(p)]
    
    for file in main_files:
        report.append(f"- {file}")
    
    # Instructions
    report.extend([
        "",
        "## Migration Instructions",
        "",
        "1. **Backup files**: These should be removed after archiving.",
        "2. **Duplicate scripts**: Consolidate functionality into the new package structure.",
        "3. **Configuration**: Use the migration utility to convert to the unified format:",
        "   ```bash",
        "   python -m trading_bot.config.migrate_configs --base-dir ./config --output ./config/config.yaml",
        "   ```",
        "4. **Multiple main entry points**: Standardize on `trading_bot.run_bot` as the primary entry point.",
        "",
        "## Automatic Cleanup",
        "",
        "Run this script with `--execute` to perform the cleanup:",
        "```bash",
        "python tools/cleanup_orphaned_files.py --execute",
        "```",
        "",
        "Files will be archived to: " + str(ARCHIVE_DIR)
    ])
    
    return "\n".join(report)


def main():
    """Main entry point for the cleanup script."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="BensBot Codebase Cleanup Tool")
    parser.add_argument("--execute", action="store_true", help="Actually perform the cleanup (default: dry run)")
    parser.add_argument("--report", action="store_true", help="Generate a cleanup and migration report")
    parser.add_argument("--backup-only", action="store_true", help="Only clean backup files, not duplicates")
    args = parser.parse_args()
    
    # Generate report if requested
    if args.report:
        report = generate_migration_report()
        report_path = ROOT_DIR / "cleanup_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report generated: {report_path}")
        return 0
    
    # Dry run by default
    if not args.execute:
        print("DRY RUN - no files will be modified")
        print("Run with --execute to perform actual cleanup")
        print("Run with --report to generate a detailed report")
        
    # Create archive directory if executing
    if args.execute:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created archive directory: {ARCHIVE_DIR}")
    
    # Find files to clean up
    backup_files = find_backup_files()
    print(f"Found {len(backup_files)} backup files to clean up")
    
    duplicate_scripts = []
    if not args.backup_only:
        duplicate_scripts = find_duplicate_scripts()
        print(f"Found {len(duplicate_scripts)} duplicate scripts to consolidate")
    
    # Process backup files
    for file_path in backup_files:
        if args.execute:
            removed = remove_file(file_path)
            if removed:
                print(f"Removed backup file: {file_path}")
        else:
            print(f"Would remove backup file: {file_path}")
    
    # Process duplicate scripts
    if not args.backup_only:
        for file_path in duplicate_scripts:
            if args.execute:
                removed = remove_file(file_path)
                if removed:
                    print(f"Removed duplicate script: {file_path}")
            else:
                print(f"Would remove duplicate script: {file_path}")
    
    # Final summary
    total_files = len(backup_files) + len(duplicate_scripts)
    if args.execute:
        print(f"Cleanup complete. Processed {total_files} files.")
        print(f"Files were archived to: {ARCHIVE_DIR}")
    else:
        print(f"Dry run complete. Would process {total_files} files.")
        print("Run with --execute to perform the actual cleanup.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
