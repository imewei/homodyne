#!/usr/bin/env python3
"""Regenerate line-count table in homodyne-architecture-overview.md.

Walks the homodyne/ source tree, counts lines per .py file, and updates
the '## 1. Package Structure' section of the architecture overview document
with accurate totals.

Usage:
    python scripts/utils/regenerate_line_counts.py          # dry-run (show diff)
    python scripts/utils/regenerate_line_counts.py --apply  # overwrite doc in-place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project layout
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = REPO_ROOT / "homodyne"
DOC_PATH = REPO_ROOT / "docs" / "architecture" / "homodyne-architecture-overview.md"

# Directories in the order they appear in the overview doc
DIR_ORDER = [
    "core",
    "data",
    "config",
    "optimization",
    "io",
    "viz",
    "cli",
    "device",
    "utils",
]

# Subdirectories that get their own subtotal in the table
OPTIMIZATION_SUBDIRS = ["nlsq", "cmc"]
NLSQ_SUBDIRS = ["strategies", "validation"]
CMC_SUBDIRS = ["backends"]


def count_lines(path: Path) -> int:
    """Count non-empty lines in a Python file (wc -l equivalent)."""
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return 0


def count_dir(dirpath: Path, *, recursive: bool = True) -> int:
    """Sum line counts for all .py files under *dirpath*."""
    pattern = "**/*.py" if recursive else "*.py"
    return sum(count_lines(f) for f in sorted(dirpath.glob(pattern)))


def file_counts(dirpath: Path) -> dict[str, int]:
    """Map filename -> line count for .py files directly in *dirpath*."""
    return {
        f.name: count_lines(f)
        for f in sorted(dirpath.glob("*.py"))
    }


def generate_summary_table() -> str:
    """Generate the Size Distribution markdown table."""
    totals: dict[str, int] = {}
    for d in DIR_ORDER:
        totals[d] = count_dir(PKG_DIR / d)

    # optimization subtotals
    opt_nlsq = count_dir(PKG_DIR / "optimization" / "nlsq")
    opt_cmc = count_dir(PKG_DIR / "optimization" / "cmc")
    opt_shared = totals["optimization"] - opt_nlsq - opt_cmc

    # runtime (special: scattered across runtime/ + post_install.py + uninstall_scripts.py)
    runtime_total = 0
    runtime_dir = PKG_DIR / "runtime"
    if runtime_dir.exists():
        runtime_total += count_dir(runtime_dir)
    for fname in ("post_install.py", "uninstall_scripts.py"):
        f = PKG_DIR / fname
        if f.exists():
            runtime_total += count_lines(f)

    # Root package files
    root_total = sum(
        count_lines(f) for f in PKG_DIR.glob("*.py")
        if f.name not in ("post_install.py", "uninstall_scripts.py")
    )

    grand_total = sum(totals.values()) + runtime_total + root_total

    descriptions = {
        "optimization": f"NLSQ ({opt_nlsq:,}) + CMC ({opt_cmc:,}) + shared ({opt_shared:,})",
        "data": "HDF5 loading, preprocessing, QC, caching",
        "core": "Physics models, JIT kernels, fitting",
        "config": "YAML/JSON config, parameter management",
        "cli": "CLI entry points, pipeline orchestration",
        "viz": "NLSQ/MCMC plots, datashader backend",
        "runtime": "System validator, shell completion, installer",
        "utils": "Structured logging, path security",
        "device": "CPU topology, NUMA, HPC optimization",
        "io": "JSON/NPZ/NetCDF result writers",
        "root": "Package init, version",
    }

    # Build rows sorted by line count descending
    rows: list[tuple[str, int, str]] = []
    for d in DIR_ORDER:
        rows.append((f"{d}/", totals[d], descriptions[d]))
    rows.append(("runtime/", runtime_total, descriptions["runtime"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    rows.append(("root", root_total, descriptions["root"]))

    lines = [
        "| Module | Lines | % | Primary Responsibility |",
        "|--------|------:|--:|------------------------|",
    ]
    for name, count, desc in rows:
        pct = count / grand_total * 100
        lines.append(f"| {name} | {count:,} | {pct:.1f}% | {desc} |")
    lines.append(
        f"| **Total** | **~{round(grand_total, -2):,}** | | |"
    )
    return "\n".join(lines)


def generate_report() -> str:
    """Generate a full plain-text report of line counts per module."""
    parts: list[str] = ["Homodyne Line-Count Report", "=" * 40, ""]

    for d in DIR_ORDER:
        dirpath = PKG_DIR / d
        total = count_dir(dirpath)
        parts.append(f"{d}/  ({total:,} lines)")

        # Top-level files
        for fname, lc in sorted(file_counts(dirpath).items(), key=lambda x: -x[1]):
            parts.append(f"  {fname:<40s} {lc:>6,}")

        # Known subdirectories
        subdirs: list[str] = []
        if d == "optimization":
            subdirs = OPTIMIZATION_SUBDIRS
        for sd in subdirs:
            sdpath = dirpath / sd
            if not sdpath.exists():
                continue
            sd_total = count_dir(sdpath)
            parts.append(f"  {sd}/  ({sd_total:,} lines)")
            for fname, lc in sorted(file_counts(sdpath).items(), key=lambda x: -x[1]):
                parts.append(f"    {fname:<38s} {lc:>6,}")

            # Sub-subdirectories (strategies/, backends/, validation/)
            inner_subdirs = NLSQ_SUBDIRS if sd == "nlsq" else CMC_SUBDIRS if sd == "cmc" else []
            for isd in inner_subdirs:
                isdpath = sdpath / isd
                if not isdpath.exists():
                    continue
                isd_total = count_dir(isdpath)
                parts.append(f"    {isd}/  ({isd_total:,} lines)")
                for fname, lc in sorted(file_counts(isdpath).items(), key=lambda x: -x[1]):
                    parts.append(f"      {fname:<36s} {lc:>6,}")

        parts.append("")

    # Runtime
    runtime_total = 0
    runtime_dir = PKG_DIR / "runtime"
    parts.append("runtime/")
    if runtime_dir.exists():
        rt = count_dir(runtime_dir)
        runtime_total += rt
        parts.append(f"  runtime/ subtree  ({rt:,} lines)")
    for fname in ("post_install.py", "uninstall_scripts.py"):
        f = PKG_DIR / fname
        if f.exists():
            lc = count_lines(f)
            runtime_total += lc
            parts.append(f"  {fname:<40s} {lc:>6,}")
    parts.append(f"  TOTAL: {runtime_total:,} lines")
    parts.append("")

    # Root
    root_files = file_counts(PKG_DIR)
    # Exclude runtime-associated files
    for fname in ("post_install.py", "uninstall_scripts.py"):
        root_files.pop(fname, None)
    root_total = sum(root_files.values())
    parts.append(f"root/  ({root_total:,} lines)")
    for fname, lc in sorted(root_files.items(), key=lambda x: -x[1]):
        parts.append(f"  {fname:<40s} {lc:>6,}")
    parts.append("")

    parts.append("=" * 40)
    parts.append(generate_summary_table())

    return "\n".join(parts)


def update_doc(*, dry_run: bool = True) -> bool:
    """Update the Size Distribution table in the overview doc.

    Returns True if changes were made (or would be made in dry-run mode).
    """
    if not DOC_PATH.exists():
        print(f"ERROR: {DOC_PATH} not found", file=sys.stderr)
        return False

    content = DOC_PATH.read_text(encoding="utf-8")

    # Find the Size Distribution table
    marker_start = "| Module | Lines | % | Primary Responsibility |"
    marker_end_pattern = "| **Total**"

    start_idx = content.find(marker_start)
    if start_idx == -1:
        print("ERROR: Could not find Size Distribution table in doc", file=sys.stderr)
        return False

    # Find end of table (line after **Total** row)
    total_idx = content.find(marker_end_pattern, start_idx)
    if total_idx == -1:
        print("ERROR: Could not find **Total** row in table", file=sys.stderr)
        return False

    # End is after the Total row's newline
    end_idx = content.index("\n", total_idx) + 1

    old_table = content[start_idx:end_idx]
    new_table = generate_summary_table() + "\n"

    if old_table == new_table:
        print("Size Distribution table is already up to date.")
        return False

    if dry_run:
        print("--- CURRENT ---")
        print(old_table)
        print("--- PROPOSED ---")
        print(new_table)
        print("\nRun with --apply to update the document.")
        return True

    updated = content[:start_idx] + new_table + content[end_idx:]
    DOC_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {DOC_PATH.relative_to(REPO_ROOT)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate line-count data for architecture overview."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update the overview doc in place (default: dry-run).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed per-file line counts.",
    )
    args = parser.parse_args()

    if args.report:
        print(generate_report())
        return

    update_doc(dry_run=not args.apply)


if __name__ == "__main__":
    main()
