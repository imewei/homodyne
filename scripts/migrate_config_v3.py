#!/usr/bin/env python3
"""Migrate homodyne configuration files from v2.x to v3.0.

This script automatically removes deprecated configuration sections that are
no longer used in homodyne v3.0:
- performance.subsampling (removed in v3.0)
- optimization_performance.time_subsampling (removed in v2.1)

NLSQ now handles large datasets natively with automatic strategy selection.

Usage:
    # Migrate single config file
    python migrate_config_v3.py path/to/config.yaml

    # Migrate all YAML files in a directory
    python migrate_config_v3.py path/to/configs/

    # Dry run (show what would be changed without modifying files)
    python migrate_config_v3.py --dry-run path/to/config.yaml

    # Skip backup creation
    python migrate_config_v3.py --no-backup path/to/config.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import shutil

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


def find_yaml_files(path: Path) -> list[Path]:
    """Find all YAML files in path (file or directory)."""
    if path.is_file():
        if path.suffix in ['.yaml', '.yml']:
            return [path]
        else:
            print(f"Warning: {path} is not a YAML file")
            return []
    elif path.is_dir():
        yaml_files = list(path.glob('**/*.yaml')) + list(path.glob('**/*.yml'))
        return yaml_files
    else:
        print(f"Error: {path} does not exist")
        return []


def check_deprecated_sections(config: dict) -> dict[str, bool]:
    """Check which deprecated sections are present in config.

    Returns:
        dict with keys: 'performance_subsampling', 'optimization_performance_time_subsampling'
        Values are True if section exists, False otherwise.
    """
    found = {
        'performance_subsampling': False,
        'optimization_performance_time_subsampling': False,
    }

    # Check performance.subsampling
    if 'performance' in config:
        if 'subsampling' in config['performance']:
            found['performance_subsampling'] = True

    # Check optimization_performance.time_subsampling
    if 'optimization_performance' in config:
        if 'time_subsampling' in config['optimization_performance']:
            found['optimization_performance_time_subsampling'] = True

    return found


def remove_deprecated_sections(config: dict) -> tuple[dict, list[str]]:
    """Remove deprecated sections from config.

    Returns:
        Tuple of (modified_config, list of removed sections)
    """
    removed = []

    # Remove performance.subsampling
    if 'performance' in config:
        if 'subsampling' in config['performance']:
            del config['performance']['subsampling']
            removed.append('performance.subsampling')
            # If performance section is now empty, consider removing it
            # (but keep it if other settings exist)

    # Remove optimization_performance.time_subsampling
    if 'optimization_performance' in config:
        if 'time_subsampling' in config['optimization_performance']:
            del config['optimization_performance']['time_subsampling']
            removed.append('optimization_performance.time_subsampling')
            # If optimization_performance is now empty, remove it
            if not config['optimization_performance']:
                del config['optimization_performance']
                removed.append('optimization_performance (empty section)')

    return config, removed


def backup_file(file_path: Path) -> Path:
    """Create timestamped backup of config file.

    Returns:
        Path to backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f'.{timestamp}.backup{file_path.suffix}')
    shutil.copy2(file_path, backup_path)
    return backup_path


def migrate_config_file(
    file_path: Path,
    dry_run: bool = False,
    create_backup: bool = True,
) -> tuple[bool, list[str]]:
    """Migrate a single config file.

    Returns:
        Tuple of (was_modified, list of removed sections)
    """
    try:
        # Load config
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            print(f"  ⚠️  {file_path}: Empty file, skipping")
            return False, []

        # Check for deprecated sections
        deprecated = check_deprecated_sections(config)

        if not any(deprecated.values()):
            print(f"  ✅ {file_path}: Already up to date (no deprecated sections)")
            return False, []

        # Report what was found
        found_sections = [k for k, v in deprecated.items() if v]
        print(f"  🔍 {file_path}: Found deprecated sections:")
        for section in found_sections:
            section_name = section.replace('_', '.')
            print(f"     - {section_name}")

        if dry_run:
            print(f"  🔍 {file_path}: [DRY RUN] Would remove deprecated sections")
            return True, found_sections

        # Create backup
        if create_backup:
            backup_path = backup_file(file_path)
            print(f"  💾 {file_path}: Backup created at {backup_path.name}")

        # Remove deprecated sections
        modified_config, removed = remove_deprecated_sections(config)

        # Save modified config
        with open(file_path, 'w') as f:
            yaml.dump(modified_config, f, default_flow_style=False, sort_keys=False)

        print(f"  ✅ {file_path}: Migrated successfully")
        print(f"     Removed: {', '.join(removed)}")

        return True, removed

    except Exception as e:
        print(f"  ❌ {file_path}: Error during migration: {e}")
        return False, []


def main():
    parser = argparse.ArgumentParser(
        description="Migrate homodyne config files from v2.x to v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.yaml                    # Migrate single file
  %(prog)s configs/                       # Migrate all YAML files in directory
  %(prog)s --dry-run config.yaml          # Show changes without modifying
  %(prog)s --no-backup config.yaml        # Skip backup creation
        """
    )

    parser.add_argument(
        'path',
        type=Path,
        help='Path to config file or directory containing config files'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )

    args = parser.parse_args()

    # Banner
    print("=" * 70)
    print("Homodyne v3.0 Configuration Migration")
    print("=" * 70)
    print()

    if args.dry_run:
        print("🔍 DRY RUN MODE: No files will be modified")
        print()

    # Find YAML files
    yaml_files = find_yaml_files(args.path)

    if not yaml_files:
        print("No YAML files found to migrate.")
        return 0

    print(f"Found {len(yaml_files)} YAML file(s) to process:")
    print()

    # Migrate each file
    total_migrated = 0
    for file_path in yaml_files:
        was_modified, removed = migrate_config_file(
            file_path,
            dry_run=args.dry_run,
            create_backup=not args.no_backup,
        )
        if was_modified:
            total_migrated += 1
        print()

    # Summary
    print("=" * 70)
    print("Migration Summary")
    print("=" * 70)
    print(f"Total files processed: {len(yaml_files)}")
    print(f"Files migrated: {total_migrated}")
    print(f"Files already up to date: {len(yaml_files) - total_migrated}")
    print()

    if args.dry_run:
        print("🔍 This was a dry run. Run without --dry-run to apply changes.")
    elif total_migrated > 0:
        print("✅ Migration complete!")
        print()
        print("Next steps:")
        print("  1. Review your config files to ensure they look correct")
        print("  2. Run your analysis with homodyne v3.0")
        print("  3. NLSQ will automatically handle large datasets with optimal strategy")
        print()
        print("For more information, see:")
        print("  https://nlsq.readthedocs.io/en/latest/guides/large_datasets.html")
    else:
        print("✅ All config files are already up to date!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
