# Homodyne Utility Scripts

This directory contains utility scripts for homodyne development and maintenance.

## Configuration Migration

### migrate_config_v3.py

Migrate homodyne configuration files from v2.x to v3.0.

**What it does:**
- Removes deprecated `performance.subsampling` section (removed in v3.0)
- Removes deprecated `optimization_performance.time_subsampling` section (removed in v2.1)
- Creates timestamped backups of original files
- Provides clear reporting of changes made

**Why migrate:**
Homodyne v3.0 uses NLSQ's native large dataset handling with automatic strategy selection. The custom subsampling system has been removed in favor of NLSQ's robust, tested memory management.

**Usage:**

```bash
# Migrate single config file
python scripts/migrate_config_v3.py config.yaml

# Migrate all YAML files in a directory
python scripts/migrate_config_v3.py configs/

# Preview changes without modifying files (dry run)
python scripts/migrate_config_v3.py --dry-run config.yaml

# Skip backup creation
python scripts/migrate_config_v3.py --no-backup config.yaml
```

**Example output:**

```
======================================================================
Homodyne v3.0 Configuration Migration
======================================================================

Found 2 YAML file(s) to process:

  🔍 config.yaml: Found deprecated sections:
     - performance.subsampling
  💾 config.yaml: Backup created at config.20251019_235900.backup.yaml
  ✅ config.yaml: Migrated successfully
     Removed: performance.subsampling

  ✅ config2.yaml: Already up to date (no deprecated sections)

======================================================================
Migration Summary
======================================================================
Total files processed: 2
Files migrated: 1
Files already up to date: 1

✅ Migration complete!
```

**Safety:**
- Automatic backups created before modifications
- Dry-run mode available to preview changes
- Only removes deprecated sections, preserves all other config

**Documentation:**
For more information about v3.0 changes, see:
- `CLAUDE.md` - Development guide
- `agent-os/specs/2025-10-19-nlsq-native-large-dataset-handling/` - Implementation details
- https://nlsq.readthedocs.io/en/latest/guides/large_datasets.html - NLSQ documentation

## Future Scripts

Additional utility scripts may be added here for:
- Performance benchmarking
- Data validation
- Result comparison
- Configuration validation
