#!/usr/bin/env python3
"""
Run Python CREST simulation using settings from an Excel file and compare results.

This script:
1. Exports VBA and CSV from Excel file to excel/{basename}/
2. Extracts run settings from Main Sheet
3. Runs Python simulation with those settings
4. Compares Python output with Excel output (if available)
5. Saves everything to output/run_YYYYMMDD_NN/

Usage:
    python scripts/excel_run_and_compare.py excel/original.xlsm
    python scripts/excel_run_and_compare.py excel/original_100houses.xlsm
    python scripts/excel_run_and_compare.py excel/lcg_fixed.xlsm --no-compare
"""

import sys
import subprocess
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime

# Import helper utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils import create_output_dir, get_project_root, get_python_main


def export_excel_file(excel_path: Path, export_dir: Path) -> bool:
    """
    Export VBA and CSV from Excel file.

    Args:
        excel_path: Path to .xlsm file
        export_dir: Directory to export to

    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: Exporting Excel file")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        'scripts/export_excel.py',
        str(excel_path),
        '--output', str(export_dir)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Export failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def extract_run_settings(excel_path: Path) -> dict:
    """
    Extract run settings from Excel Main Sheet.

    Args:
        excel_path: Path to .xlsm file

    Returns:
        Dictionary of settings
    """
    print(f"\n{'='*60}")
    print(f"STEP 2: Extracting run settings")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        'scripts/extract_settings.py',
        str(excel_path),
        '--format', 'json'
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Parse JSON output
        import json
        # Extract JSON from output (after the debug prints)
        lines = result.stdout.strip().split('\n')
        # Find the JSON block (between === markers)
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break

        if json_start is not None:
            json_str = '\n'.join(lines[json_start:])
            # Remove trailing === if present
            json_str = json_str.split('=====')[0].strip()
            settings = json.loads(json_str)
            return settings
        else:
            print("WARNING: Could not parse settings JSON")
            return {}

    except Exception as e:
        print(f"ERROR: Failed to extract settings: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_python_simulation(settings: dict, dwellings_file: Path, output_dir: Path) -> bool:
    """
    Run Python CREST simulation.

    Args:
        settings: Run settings from Excel
        dwellings_file: Path to Dwellings.csv
        output_dir: Output directory

    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: Running Python simulation")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(get_python_main()),
        '--config-file', str(dwellings_file),
        '--output-dir', str(output_dir),
        '--save-detailed'
    ]

    # Add settings as command-line arguments
    if 'day' in settings:
        cmd.extend(['--day', str(settings['day'])])

    if 'month' in settings:
        cmd.extend(['--month', str(settings['month'])])

    if 'year' in settings and settings.get('country') == 'India':
        cmd.extend(['--year', str(settings['year'])])

    if 'country' in settings:
        cmd.extend(['--country', settings['country']])

    if 'city' in settings:
        cmd.extend(['--city', settings['city']])

    if 'urban_rural' in settings:
        cmd.extend(['--urban-rural', settings['urban_rural']])

    if 'seed' in settings and settings['seed'] is not None:
        cmd.extend(['--seed', str(settings['seed'])])

    if settings.get('use_portable_rng', False):
        cmd.append('--portable-rng')

    print(f"\nCommand: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        print("ERROR: Simulation timed out (>10 minutes)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Simulation failed")
        print(e.stderr)
        return False


def compare_results(python_dir: Path, excel_dir: Path, output_dir: Path) -> None:
    """
    Compare Python and Excel results.

    Args:
        python_dir: Directory with Python results
        excel_dir: Directory with Excel results (from CSV export)
        output_dir: Directory to save comparison
    """
    print(f"\n{'='*60}")
    print(f"STEP 4: Comparing results")
    print(f"{'='*60}")

    # Check if Excel results exist (minute-level disaggregated results)
    # Excel exports with spaces in filenames: "Results - disaggregated.csv"
    excel_results_minute = excel_dir / "Results - disaggregated.csv"
    excel_results_daily = excel_dir / "Results - daily totals.csv"

    if not excel_results_minute.exists():
        print(f"\nNo Excel minute-level results found at: {excel_results_minute}")
        print(f"Skipping comparison.")
        print(f"\nTo compare:")
        print(f"1. Run Excel file manually and let it complete simulation")
        print(f"2. Export results using: python scripts/export_excel.py {excel_dir.parent / (excel_dir.name + '.xlsm')}")
        print(f"3. Re-run this script")
        return

    python_results_minute = output_dir / "results_minute_level.csv"
    python_results_daily = output_dir / "results_daily_summary.csv"

    if not python_results_minute.exists():
        print(f"\nPython results not found at: {python_results_minute}")
        print(f"Skipping comparison.")
        return

    # Compare minute-level results
    try:
        import pandas as pd

        print(f"\nLoading minute-level results...")
        print(f"  Excel:  {excel_results_minute}")
        print(f"  Python: {python_results_minute}")

        # Excel file has header rows:
        # Rows 0-2: Description/empty
        # Row 3: Column names
        # Rows 4-5: Variable symbols and units
        # Skip rows 0-2, use row 3 as header, skip rows 4-5
        excel_df = pd.read_csv(excel_results_minute, skiprows=[0, 1, 2, 4, 5])
        python_df = pd.read_csv(python_results_minute)

        print(f"  Excel:  {len(excel_df)} rows, {len(excel_df.columns)} columns")
        print(f"  Python: {len(python_df)} rows, {len(python_df.columns)} columns")

        # Save comparison report
        report_file = output_dir / "comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write("CREST COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            f.write("MINUTE-LEVEL RESULTS\n")
            f.write(f"Excel results:  {excel_results_minute}\n")
            f.write(f"Python results: {python_results_minute}\n\n")
            f.write(f"Excel:  {len(excel_df)} rows, {len(excel_df.columns)} columns\n")
            f.write(f"Python: {len(python_df)} rows, {len(python_df.columns)} columns\n\n")

            if len(excel_df) == len(python_df):
                f.write("✓ Row counts match\n")
            else:
                f.write(f"✗ Row count mismatch: {abs(len(excel_df) - len(python_df))} difference\n")

        # Also compare daily summaries if available
        if excel_results_daily.exists() and python_results_daily.exists():
            print(f"\nLoading daily summary results...")
            # Daily file structure:
            # Row 0: Description
            # Row 1: Column names (use as header)
            # Rows 2-3: Variable symbols and units (skip)
            excel_daily_df = pd.read_csv(excel_results_daily, skiprows=[0, 2, 3])
            python_daily_df = pd.read_csv(python_results_daily)

            print(f"  Excel:  {len(excel_daily_df)} rows, {len(excel_daily_df.columns)} columns")
            print(f"  Python: {len(python_daily_df)} rows, {len(python_daily_df.columns)} columns")

            with open(report_file, 'a') as f:
                f.write("\n\nDAILY SUMMARY RESULTS\n")
                f.write(f"Excel results:  {excel_results_daily}\n")
                f.write(f"Python results: {python_results_daily}\n\n")
                f.write(f"Excel:  {len(excel_daily_df)} rows, {len(excel_daily_df.columns)} columns\n")
                f.write(f"Python: {len(python_daily_df)} rows, {len(python_daily_df.columns)} columns\n\n")

                if len(excel_daily_df) == len(python_daily_df):
                    f.write("✓ Row counts match\n")
                else:
                    f.write(f"✗ Row count mismatch: {abs(len(excel_daily_df) - len(python_daily_df))} difference\n")

        print(f"\nComparison report saved to: {report_file}")

    except Exception as e:
        print(f"\nERROR during comparison: {e}")
        import traceback
        traceback.print_exc()


def create_run_script(settings: dict, dwellings_file: Path, output_dir: Path) -> None:
    """
    Create a shell script to re-run this simulation.

    Args:
        settings: Run settings
        dwellings_file: Path to Dwellings.csv
        output_dir: Output directory to save script
    """
    # Get the Python interpreter path
    python_path = sys.executable

    script_content = [
        "#!/bin/bash",
        "# CREST Simulation Re-run Script",
        f"# Generated: {datetime.now().isoformat()}",
        "",
        "# To re-run this exact simulation:",
        "#   bash rerun_simulation.sh",
        "",
        "DWELLINGS_FILE=\"" + str(dwellings_file.absolute()) + "\"",
        "OUTPUT_DIR=\"" + str(output_dir.absolute()) + "\"",
        "",
        f"{python_path} python/main.py \\",
    ]

    args = []
    args.append(f"  --config-file \"$DWELLINGS_FILE\" \\")

    if 'day' in settings:
        args.append(f"  --day {settings['day']} \\")

    if 'month' in settings:
        args.append(f"  --month {settings['month']} \\")

    if 'country' in settings:
        args.append(f"  --country {settings['country']} \\")

    if 'city' in settings:
        args.append(f"  --city '{settings['city']}' \\")

    if 'urban_rural' in settings:
        args.append(f"  --urban-rural {settings['urban_rural']} \\")

    if 'seed' in settings and settings['seed'] is not None:
        args.append(f"  --seed {settings['seed']} \\")

    if settings.get('use_portable_rng', False):
        args.append(f"  --portable-rng \\")

    args.append(f"  --save-detailed \\")
    args.append(f"  --output-dir \"$OUTPUT_DIR\"")

    script_content.extend(args)

    script_file = output_dir / "rerun_simulation.sh"
    script_file.write_text('\n'.join(script_content))
    script_file.chmod(0o755)  # Make executable

    print(f"\nRe-run script saved to: {script_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Python CREST using Excel settings and compare results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('excel_file', type=Path,
                       help='Path to Excel .xlsm file')
    parser.add_argument('--no-compare', action='store_true',
                       help='Skip comparison with Excel results')
    parser.add_argument('--force-export', action='store_true',
                       help='Force re-export even if directory exists')

    args = parser.parse_args()

    # Change to project root
    project_root = get_project_root()
    import os
    os.chdir(project_root)

    if not args.excel_file.exists():
        print(f"ERROR: Excel file not found: {args.excel_file}")
        sys.exit(1)

    print("="*60)
    print("CREST EXCEL RUN AND COMPARE")
    print("="*60)
    print(f"Excel file: {args.excel_file}")
    print()

    # Determine export directory: excel/{basename}/
    basename = args.excel_file.stem
    export_dir = Path("excel") / basename

    # Step 1: Export Excel file (if needed)
    if not export_dir.exists() or args.force_export:
        if not export_excel_file(args.excel_file, export_dir):
            print("\nERROR: Export failed")
            sys.exit(1)
    else:
        print(f"\nExport directory already exists: {export_dir}")
        print(f"Using existing exports (use --force-export to re-export)")

    # Step 2: Extract settings
    settings = extract_run_settings(args.excel_file)
    if not settings:
        print("\nWARNING: No settings extracted, using defaults")
        settings = {}

    # Find Dwellings.csv
    dwellings_file = export_dir / "Dwellings.csv"
    if not dwellings_file.exists():
        print(f"\nERROR: Dwellings.csv not found at: {dwellings_file}")
        sys.exit(1)

    # Step 3: Create output directory
    output_dir = create_output_dir("", prefix="run")
    print(f"\nOutput directory: {output_dir}")

    # Copy config files to output
    shutil.copy(dwellings_file, output_dir / "dwellings_config.csv")
    print(f"  Copied: dwellings_config.csv")

    # Create re-run script
    create_run_script(settings, dwellings_file, output_dir)

    # Step 4: Run Python simulation
    if not run_python_simulation(settings, dwellings_file, output_dir):
        print("\nERROR: Simulation failed")
        sys.exit(1)

    # Step 5: Compare results (unless --no-compare)
    if not args.no_compare:
        compare_results(output_dir, export_dir, output_dir)

    # Save metadata
    metadata = {
        "excel_file": str(args.excel_file.absolute()),
        "export_dir": str(export_dir.absolute()),
        "settings": settings,
        "run_date": datetime.now().isoformat()
    }
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_file}")

    print()
    print("="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
