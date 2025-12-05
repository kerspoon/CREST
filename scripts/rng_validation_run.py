#!/usr/bin/env python3
"""Run single CREST iteration with portable LCG and full RNG call logging.

This script implements Objective #1: RNG Call Sequence Matching
- Exports from Excel file to ensure fresh data (by default)
- Reads simulation settings from exported JSON (lat/long, day, month, etc.)
- Runs Python version with portable LCG (Linear Congruential Generator)
- Logs every single RNG call with location, order, and value
- Output log can be compared with Excel VBA equivalent

Usage:
    python scripts/rng_validation_run.py [--no-export] [extra_args...]

Examples:
    # Run with default (exports from Excel first, uses exported settings)
    python scripts/rng_validation_run.py

    # Skip export (use existing exported data)
    python scripts/rng_validation_run.py --no-export

    # Override seed
    python scripts/rng_validation_run.py --seed 12345
"""

import subprocess
import sys
import json
from pathlib import Path

# Import helper utilities
from utils import create_output_dir, get_project_root, get_python_main

# Default configuration
DEFAULT_EXCEL = 'excel/lcg_fixed.xlsm'
DEFAULT_CONFIG = 'excel/lcg_fixed/Dwellings.csv'
DEFAULT_SETTINGS = 'excel/lcg_fixed/simulation_settings.json'


def main():
    """Run single iteration with LCG logging enabled."""
    # Change to project root first
    project_root = get_project_root()
    import os
    os.chdir(project_root)

    # Parse arguments
    skip_export = '--no-export' in sys.argv
    extra_args = [arg for arg in sys.argv[1:] if arg != '--no-export']

    print("=" * 80)
    print("CREST RNG VALIDATION RUN")
    print("=" * 80)
    print("Objective #1: Verify identical RNG call sequences (Python vs Excel)")
    print()

    # Step 1: Export from Excel (unless --no-export)
    excel_file = Path(DEFAULT_EXCEL)
    if not skip_export:
        if excel_file.exists():
            print(f"Exporting data from: {excel_file}")
            print("-" * 80)
            export_cmd = [sys.executable, 'scripts/export_excel.py', str(excel_file)]
            export_result = subprocess.run(export_cmd, capture_output=True, text=True)
            if export_result.returncode != 0:
                print("ERROR: Failed to export Excel file:")
                print(export_result.stderr)
                sys.exit(1)
            print(export_result.stdout)
            print("-" * 80)
        else:
            print(f"WARNING: Excel file not found: {excel_file}")
            print("         Using existing exported data...")
            print()
    else:
        print("Skipping Excel export (--no-export flag)")
        print()

    # Step 2: Load settings from exported JSON
    settings_path = Path(DEFAULT_SETTINGS)
    config_file = DEFAULT_CONFIG

    if not settings_path.exists():
        print(f"ERROR: Settings file not found: {settings_path}")
        print("       Run without --no-export to export from Excel first.")
        sys.exit(1)

    with open(settings_path, 'r') as f:
        settings = json.load(f)

    print("Loaded settings from Excel:")
    print(f"  Day:       {settings.get('day')}")
    print(f"  Month:     {settings.get('month')}")
    print(f"  Latitude:  {settings.get('latitude')}")
    print(f"  Longitude: {settings.get('longitude')}")
    print(f"  Meridian:  {settings.get('meridian')}")
    print(f"  City:      {settings.get('city')}")
    print(f"  Country:   {settings.get('country')}")
    print(f"  Dwellings: {settings.get('num_dwellings')}")
    print()

    # Check if config exists
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)

    # Create output directory
    num_dwellings = settings.get('num_dwellings', 5)
    output_dir = create_output_dir(
        "rng_validation",
        prefix=f"python_{num_dwellings}houses"
    )

    print(f"Config file: {config_file}")
    print(f"Output directory: {output_dir}")
    if extra_args:
        print(f"Extra args:  {' '.join(extra_args)}")
    print()

    # Build command using settings from exported JSON
    rng_log_path = output_dir / 'rng_calls.log'
    cmd = [
        sys.executable,
        str(get_python_main()),
        '--config-file', str(config_file),
        '--output-dir', str(output_dir),
        '--rng-log-file', str(rng_log_path),  # CRITICAL: Enable portable LCG + log all RNG calls
        '--seed', '42',  # Fixed seed for reproducibility (overriding Excel since it doesn't save seed)
        '--day', str(settings.get('day', 1)),
        '--month', str(settings.get('month', 1)),
        '--latitude', str(settings.get('latitude', 53.4794892)),
        '--longitude', str(settings.get('longitude', -2.2451148)),
        '--meridian', str(settings.get('meridian', 0.0)),
        '--country', str(settings.get('country', 'UK')),
        '--city', str(settings.get('city', 'England')),
    ]

    # Add --save-detailed if enabled in settings
    if settings.get('save_detailed', True):
        cmd.append('--save-detailed')

    # Add any extra arguments (these can override settings above)
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
    print(f"Settings used (from {settings_path}):")
    print(f"  Day={settings.get('day')}, Month={settings.get('month')}")
    print(f"  Lat={settings.get('latitude')}, Lon={settings.get('longitude')}")
    print(f"  Dwellings={num_dwellings}, Seed=42")
    print()
    print("Next steps:")
    print(f"1. Manually run Excel ({DEFAULT_EXCEL}) with same settings and seed=42")
    print(f"2. Save Excel output to: output/rng_validation/excel_{num_dwellings}houses_YYYYMMDD_NN/")
    print("3. Compare logs:")
    print(f"   python scripts/rng_log_compare.py \\")
    print(f"     {output_dir} \\")
    print(f"     output/rng_validation/excel_{num_dwellings}houses_YYYYMMDD_NN")
    print()


if __name__ == '__main__':
    main()
