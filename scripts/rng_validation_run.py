#!/usr/bin/env python3
"""Run single CREST iteration with portable LCG and full RNG call logging.

This script implements Objective #1: RNG Call Sequence Matching
- Runs Python version with portable LCG (Linear Congruential Generator)
- Logs every single RNG call with location, order, and value
- Output log can be compared with Excel VBA equivalent

Usage:
    python scripts/rng_validation_run.py [dwellings_file] [extra_args...]

Examples:
    # Run with default config (excel/lcg_fixed/Dwellings.csv)
    # If excel/lcg_fixed/ folder is missing, it will be auto-extracted from excel/lcg_fixed.xlsm
    python scripts/rng_validation_run.py

    # Run with custom config
    python scripts/rng_validation_run.py excel/original/Dwellings.csv

    # Run with additional flags
    python scripts/rng_validation_run.py excel/lcg_fixed/Dwellings.csv --day 15
"""

import subprocess
import sys
from pathlib import Path

# Import helper utilities
from utils import create_output_dir, get_project_root, get_python_main

# Default configuration
DEFAULT_EXCEL = 'excel/lcg_fixed.xlsm'
DEFAULT_CONFIG = 'excel/lcg_fixed/Dwellings.csv'


def main():
    """Run single iteration with LCG logging enabled."""
    # Parse arguments
    config_file = DEFAULT_CONFIG
    extra_args = []

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if len(sys.argv) > 2:
        extra_args = sys.argv[2:]

    # Change to project root
    project_root = get_project_root()
    import os
    os.chdir(project_root)

    print("=" * 80)
    print("CREST RNG VALIDATION RUN")
    print("=" * 80)
    print("Objective #1: Verify identical RNG call sequences (Python vs Excel)")
    print()
    print(f"Config file: {config_file}")
    if extra_args:
        print(f"Extra args:  {' '.join(extra_args)}")
    print()

    # Check if config exists
    config_path = Path(config_file)
    if not config_path.exists():
        # If using default config and it's missing, try to export from Excel
        if config_file == DEFAULT_CONFIG:
            excel_file = Path(DEFAULT_EXCEL)
            if excel_file.exists():
                print(f"Config not found: {config_file}")
                print(f"Extracting data from: {excel_file}")
                print()
                export_cmd = [sys.executable, 'scripts/export_excel.py', str(excel_file)]
                export_result = subprocess.run(export_cmd, capture_output=True, text=True)
                if export_result.returncode != 0:
                    print("ERROR: Failed to export Excel file:")
                    print(export_result.stderr)
                    sys.exit(1)
                print(export_result.stdout)
                # Check again
                if not config_path.exists():
                    print(f"ERROR: Config still not found after export: {config_file}")
                    sys.exit(1)
            else:
                print(f"ERROR: Config file not found: {config_file}")
                print(f"       Excel file also not found: {excel_file}")
                sys.exit(1)
        else:
            print(f"ERROR: Config file not found: {config_file}")
            print("\nAvailable configs:")
            for f in sorted(Path("excel").glob("*/Dwellings.csv")):
                print(f"  - {f}")
            sys.exit(1)

    # Create output directory
    output_dir = create_output_dir(
        "rng_validation",
        prefix="python_5houses"
    )

    print(f"Output directory: {output_dir}")
    print()

    # Build command
    rng_log_path = output_dir / 'rng_calls.log'
    cmd = [
        sys.executable,
        str(get_python_main()),
        '--config-file', str(config_file),
        '--save-detailed',  # Save minute-level data
        '--output-dir', str(output_dir),
        '--rng-log-file', str(rng_log_path),  # CRITICAL: Enable portable LCG + log all RNG calls
        '--seed', '42'  # Fixed seed for reproducibility
    ]

    # Add any extra arguments
    cmd.extend(extra_args)

    print("Running simulation with LCG logging...")
    print("(This may take longer due to extensive logging)")

    print('-'*80)
    print(" ".join(cmd))
    print('-'*80)
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)

        print("✓ Simulation completed successfully")
        print()

        # Check if RNG log was created
        if rng_log_path.exists():
            log_size = rng_log_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ RNG log created: {rng_log_path}")
            print(f"  Size: {log_size:.1f} MB")

            # Count number of RNG calls
            with open(rng_log_path, 'r') as f:
                num_calls = sum(1 for line in f)
            print(f"  Total RNG calls logged: {num_calls:,}")
        else:
            print("⚠ WARNING: RNG log not found!")
            print("  Check that --rng-log-file flag is working correctly")

        # List output files
        print()
        print("Output files:")
        for file in sorted(output_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} bytes"
                print(f"  {file.name:40} {size_str:>12}")

    except subprocess.TimeoutExpired:
        print("✗ ERROR: Simulation timed out (>5 minutes)")
        print("  The simulation may be too slow with extensive RNG logging")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print("✗ ERROR: Simulation failed")
        print()
        print("STDOUT:")
        print(e.stdout)
        print()
        print("STDERR:")
        print(e.stderr)
        sys.exit(1)

    print()
    print("=" * 80)
    print("RNG VALIDATION RUN COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Manually run Excel (lcg_fixed.xlsm) with same config and seed")
    print("2. Save Excel output to: output/rng_validation/excel_5houses_YYYYMMDD_NN/")
    print("3. Compare logs:")
    print(f"   python scripts/rng_log_compare.py \\")
    print(f"     {output_dir} \\")
    print(f"     output/rng_validation/excel_5houses_YYYYMMDD_NN")
    print()


if __name__ == '__main__':
    main()
