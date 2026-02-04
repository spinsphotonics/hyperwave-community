#!/usr/bin/env python3
"""
Sync modules from hyperwave (private) to hyperwave-community (public).

Run this script to update hyperwave-community with the latest code from hyperwave,
excluding proprietary modules (solve.py, FDTD internals, etc.).

Usage:
    python sync_modules.py           # Dry run (show what would be copied)
    python sync_modules.py --apply   # Actually copy the files
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Paths (adjust if your repo locations differ)
SCRIPT_DIR = Path(__file__).parent.resolve()
HYPERWAVE_SRC = SCRIPT_DIR.parent / "hyperwave" / "hyperwave"
HYPERWAVE_COMMUNITY_DST = SCRIPT_DIR / "hyperwave_community"

# Modules to sync (exist in both repos, safe to copy)
MODULES_TO_SYNC = [
    "absorption.py",
    "data_io.py",
    "metasurface.py",
    "monitors.py",
    "simulate.py",
    "sources.py",
    "structure.py",
]

# Modules to NEVER sync (proprietary or community-specific)
MODULES_EXCLUDED = [
    "solve.py",              # Core FDTD solver (proprietary)
    "adjoint_modal.py",      # Adjoint optimization
    "early_stopping.py",     # Early stopping logic
    "early_stopping_modal.py",
    "gaussian_source_modal.py",
    "granular_modal.py",
    "grating_coupler.py",
    "recipe_builder_modal.py",
    "simulate_modal.py",
    "simulate_modal_old_backup.py",
    "visualization.py",
    "api_client.py",         # Community-specific (API wrapper)
    "__init__.py",           # Different exports in each repo
]


def sync_modules(dry_run: bool = True) -> dict:
    """
    Sync modules from hyperwave to hyperwave-community.

    Args:
        dry_run: If True, only print what would be done without copying.

    Returns:
        dict with 'synced', 'skipped', 'missing' lists
    """
    results = {
        "synced": [],
        "skipped": [],
        "missing": [],
        "errors": [],
    }

    # Validate paths
    if not HYPERWAVE_SRC.exists():
        print(f"ERROR: Source directory not found: {HYPERWAVE_SRC}")
        print("Make sure hyperwave repo is at the expected location.")
        return results

    if not HYPERWAVE_COMMUNITY_DST.exists():
        print(f"ERROR: Destination directory not found: {HYPERWAVE_COMMUNITY_DST}")
        return results

    print(f"Source:      {HYPERWAVE_SRC}")
    print(f"Destination: {HYPERWAVE_COMMUNITY_DST}")
    print(f"Mode:        {'DRY RUN' if dry_run else 'APPLY'}")
    print("-" * 60)

    for module in MODULES_TO_SYNC:
        src_file = HYPERWAVE_SRC / module
        dst_file = HYPERWAVE_COMMUNITY_DST / module

        if not src_file.exists():
            print(f"  MISSING: {module} (not in hyperwave)")
            results["missing"].append(module)
            continue

        # Check if files differ
        if dst_file.exists():
            src_content = src_file.read_text()
            dst_content = dst_file.read_text()
            if src_content == dst_content:
                print(f"  SAME:    {module} (no changes)")
                results["skipped"].append(module)
                continue

        # Copy the file
        if dry_run:
            print(f"  WOULD COPY: {module}")
        else:
            try:
                shutil.copy2(src_file, dst_file)
                print(f"  COPIED:  {module}")
            except Exception as e:
                print(f"  ERROR:   {module} - {e}")
                results["errors"].append((module, str(e)))
                continue

        results["synced"].append(module)

    print("-" * 60)
    print(f"Summary: {len(results['synced'])} to sync, {len(results['skipped'])} unchanged, {len(results['missing'])} missing")

    if dry_run and results["synced"]:
        print("\nRun with --apply to actually copy the files.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sync modules from hyperwave to hyperwave-community")
    parser.add_argument("--apply", action="store_true", help="Actually copy files (default is dry run)")
    parser.add_argument("--list", action="store_true", help="List modules to sync and exit")
    args = parser.parse_args()

    if args.list:
        print("Modules to sync:")
        for m in MODULES_TO_SYNC:
            print(f"  - {m}")
        print("\nModules excluded:")
        for m in MODULES_EXCLUDED:
            print(f"  - {m}")
        return

    print(f"\n{'='*60}")
    print(f"Hyperwave Module Sync - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    sync_modules(dry_run=not args.apply)


if __name__ == "__main__":
    main()
