#!/usr/bin/env python3
"""
Compare Python and Excel simulation results.

Usage:
    python scripts/compare_results.py <excel_folder> <python_folder>

Example:
    python scripts/compare_results.py excel/lcg_fixed/ output/rng_validation/python_2houses_20251206_01

Output:
    - Creates comparison CSV in output/comparison/comparison_YYYYMMDD_NN/
    - Prints summary statistics to stdout
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add scripts dir to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils import create_output_dir


def load_excel_minute_results(excel_folder: Path) -> pd.DataFrame:
    """Load Excel minute-level results."""
    file_path = excel_folder / "Results - disaggregated.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Excel results not found: {file_path}")

    # Skip rows 0,1,2,4,5 to get data with row 3 as header
    df = pd.read_csv(file_path, skiprows=[0, 1, 2, 4, 5])
    return df


def load_python_minute_results(python_folder: Path) -> pd.DataFrame:
    """Load Python minute-level results."""
    file_path = python_folder / "results_minute_level.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Python results not found: {file_path}")

    # Skip rows 0,2,3 to get data with row 1 as header
    df = pd.read_csv(file_path, skiprows=[0, 2, 3])
    return df


def load_excel_daily_results(excel_folder: Path) -> pd.DataFrame:
    """Load Excel daily summary results."""
    file_path = excel_folder / "Results - daily totals.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Excel daily results not found: {file_path}")

    # Skip rows 0,2,3 to get data with row 1 as header
    df = pd.read_csv(file_path, skiprows=[0, 2, 3])
    return df


def load_python_daily_results(python_folder: Path) -> pd.DataFrame:
    """Load Python daily summary results."""
    file_path = python_folder / "results_daily_summary.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Python daily results not found: {file_path}")

    # Skip rows 0,2,3 to get data with row 1 as header
    df = pd.read_csv(file_path, skiprows=[0, 2, 3])
    return df


def load_header_rows(file_path: Path, num_rows: int = 4) -> list:
    """Load header rows from a file."""
    headers = []
    with open(file_path, 'r') as f:
        for i in range(num_rows):
            headers.append(f.readline().strip())
    return headers


def compare_results(excel_df: pd.DataFrame, python_df: pd.DataFrame) -> tuple:
    """
    Compare Excel and Python results.

    Returns:
        tuple: (abs_diff_df, sum_series, max_series)
    """
    # Ensure same columns
    common_cols = [c for c in excel_df.columns if c in python_df.columns]

    excel_data = excel_df[common_cols].copy()
    python_data = python_df[common_cols].copy()

    # Convert to numeric, coercing errors
    for col in common_cols:
        excel_data[col] = pd.to_numeric(excel_data[col], errors='coerce')
        python_data[col] = pd.to_numeric(python_data[col], errors='coerce')

    # Calculate absolute differences
    abs_diff = (python_data - excel_data).abs()

    # Calculate sum and max for each column
    sum_diff = abs_diff.sum()
    max_diff = abs_diff.max()

    return abs_diff, sum_diff, max_diff


def write_comparison_csv(output_path: Path, headers: list, sum_diff: pd.Series,
                         max_diff: pd.Series, abs_diff: pd.DataFrame):
    """Write comparison results to CSV."""
    with open(output_path, 'w') as f:
        # Write original headers
        for header in headers:
            f.write(header + '\n')

        # Write sum row
        sum_row = ['SUM()'] + [f'{sum_diff.get(col, 0):.4f}' for col in abs_diff.columns]
        f.write(','.join(sum_row) + '\n')

        # Write max row
        max_row = ['MAX()'] + [f'{max_diff.get(col, 0):.4f}' for col in abs_diff.columns]
        f.write(','.join(max_row) + '\n')

        # Write blank separator
        f.write('\n')

        # Write abs diff data
        abs_diff.to_csv(f, index=False, float_format='%.4f')


def print_summary(sum_diff: pd.Series, max_diff: pd.Series):
    """Print summary statistics to stdout."""
    # Count columns with no difference
    no_diff_cols = (max_diff == 0).sum()

    # Count columns with max diff < 0.1
    small_diff_cols = ((max_diff > 0) & (max_diff < 0.1)).sum()

    # Total columns
    total_cols = len(max_diff)

    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nTotal columns: {total_cols}")
    print(f"Columns with NO difference (max=0): {no_diff_cols}")
    print(f"Columns with small difference (0 < max < 0.1): {small_diff_cols}")
    print(f"Columns with significant difference (max >= 0.1): {total_cols - no_diff_cols - small_diff_cols}")

    # Top 5 by sum
    print("\n" + "-" * 70)
    print("TOP 5 COLUMNS BY SUM OF ABSOLUTE DIFFERENCES:")
    print("-" * 70)
    top_sum = sum_diff.nlargest(5)
    for i, (col, val) in enumerate(top_sum.items(), 1):
        print(f"  {i}. {col}: {val:,.4f}")

    # Top 5 by max
    print("\n" + "-" * 70)
    print("TOP 5 COLUMNS BY MAX ABSOLUTE DIFFERENCE:")
    print("-" * 70)
    top_max = max_diff.nlargest(5)
    for i, (col, val) in enumerate(top_max.items(), 1):
        print(f"  {i}. {col}: {val:,.4f}")

    # List columns with no difference
    if no_diff_cols > 0:
        print("\n" + "-" * 70)
        print("COLUMNS WITH PERFECT MATCH (max=0):")
        print("-" * 70)
        perfect_cols = max_diff[max_diff == 0].index.tolist()
        for col in perfect_cols:
            print(f"  - {col}")

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    excel_folder = Path(sys.argv[1])
    python_folder = Path(sys.argv[2])

    # Validate input folders
    if not excel_folder.exists():
        print(f"Error: Excel folder not found: {excel_folder}")
        sys.exit(1)
    if not python_folder.exists():
        print(f"Error: Python folder not found: {python_folder}")
        sys.exit(1)

    # Create dated output folder
    output_folder = create_output_dir("comparison", prefix="comparison")

    print(f"Loading Excel results from: {excel_folder}")
    print(f"Loading Python results from: {python_folder}")
    print(f"Output folder: {output_folder}")
    print()

    # === MINUTE-LEVEL COMPARISON ===
    print("=" * 70)
    print("MINUTE-LEVEL COMPARISON")
    print("=" * 70)

    try:
        excel_minute = load_excel_minute_results(excel_folder)
        python_minute = load_python_minute_results(python_folder)
        minute_headers = load_header_rows(excel_folder / "Results - disaggregated.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Excel rows: {len(excel_minute)}, Python rows: {len(python_minute)}")

    # Check row counts match
    if len(excel_minute) != len(python_minute):
        print(f"Warning: Row count mismatch! Excel={len(excel_minute)}, Python={len(python_minute)}")
        min_rows = min(len(excel_minute), len(python_minute))
        excel_minute = excel_minute.iloc[:min_rows]
        python_minute = python_minute.iloc[:min_rows]
        print(f"Truncating to {min_rows} rows for comparison")

    # Compare minute-level
    abs_diff, sum_diff, max_diff = compare_results(excel_minute, python_minute)

    # Write comparison CSV
    output_file = output_folder / "minute_level_comparison.csv"
    write_comparison_csv(output_file, minute_headers, sum_diff, max_diff, abs_diff)
    print(f"\nComparison written to: {output_file}")

    # Print summary
    print_summary(sum_diff, max_diff)

    # === DAILY SUMMARY COMPARISON ===
    print("\n")
    print("=" * 70)
    print("DAILY SUMMARY COMPARISON")
    print("=" * 70)

    try:
        excel_daily = load_excel_daily_results(excel_folder)
        python_daily = load_python_daily_results(python_folder)
        daily_headers = load_header_rows(excel_folder / "Results - daily totals.csv")
    except FileNotFoundError as e:
        print(f"Warning: Could not load daily results: {e}")
        return

    print(f"Excel rows: {len(excel_daily)}, Python rows: {len(python_daily)}")

    # Check row counts match
    if len(excel_daily) != len(python_daily):
        print(f"Warning: Row count mismatch! Excel={len(excel_daily)}, Python={len(python_daily)}")
        min_rows = min(len(excel_daily), len(python_daily))
        excel_daily = excel_daily.iloc[:min_rows]
        python_daily = python_daily.iloc[:min_rows]
        print(f"Truncating to {min_rows} rows for comparison")

    # Compare daily
    abs_diff_daily, sum_diff_daily, max_diff_daily = compare_results(excel_daily, python_daily)

    # Write comparison CSV
    output_file_daily = output_folder / "daily_summary_comparison.csv"
    write_comparison_csv(output_file_daily, daily_headers, sum_diff_daily, max_diff_daily, abs_diff_daily)
    print(f"\nComparison written to: {output_file_daily}")

    # Print summary
    print_summary(sum_diff_daily, max_diff_daily)


if __name__ == "__main__":
    main()
