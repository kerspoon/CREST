#!/usr/bin/env python3
"""
CREST Model Validation - Results Comparison Script

Compares Python port results against Excel/VBA model results for validation.
Performs statistical tests and generates a validation report.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def load_data(python_dir: Path, excel_dir: Path):
    """Load both Python and Excel datasets."""
    print("Loading datasets...")

    data = {}

    # Load Python data
    try:
        data['python_daily'] = pd.read_csv(python_dir / "results_daily_summary.csv")
        print(f"  Python daily: {len(data['python_daily'])} dwellings")
    except FileNotFoundError:
        print("  Error: Python daily summary not found")
        return None

    # Load Excel data
    try:
        data['excel_daily'] = pd.read_csv(excel_dir / "results_daily_summary.csv")
        print(f"  Excel daily: {len(data['excel_daily'])} dwellings")
    except FileNotFoundError:
        print("  Error: Excel daily summary not found")
        return None

    # Optionally load minute-level data if available (very large)
    python_minute_file = python_dir / "results_minute_level.csv"
    excel_minute_file = excel_dir / "results_minute_level.csv"

    if python_minute_file.exists() and excel_minute_file.exists():
        print("  Loading minute-level data (this may take a while)...")
        data['python_minute'] = pd.read_csv(python_minute_file)
        data['excel_minute'] = pd.read_csv(excel_minute_file)
        print(f"    Python minute-level: {len(data['python_minute'])} rows")
        print(f"    Excel minute-level: {len(data['excel_minute'])} rows")
    else:
        data['python_minute'] = None
        data['excel_minute'] = None
        print("  Minute-level data not available for both datasets")

    return data


def compare_daily_summaries(python_df: pd.DataFrame, excel_df: pd.DataFrame):
    """Compare daily summary statistics."""
    print("\n" + "=" * 80)
    print("DAILY SUMMARY COMPARISON")
    print("=" * 80)

    # Ensure same number of dwellings
    n_dwellings = min(len(python_df), len(excel_df))
    print(f"\nComparing {n_dwellings} dwellings\n")

    # Define metrics to compare
    metrics = [
        ('Total_Electricity_kWh', 'Total Electricity (kWh)', 0.05),  # 5% tolerance
        ('Total_Gas_m3', 'Total Gas (m³)', 0.05),
        ('Total_Hot_Water_L', 'Total Hot Water (L)', 0.10),
        ('Peak_Electricity_W', 'Peak Electricity (W)', 0.10),
        ('Mean_Internal_Temp_C', 'Mean Internal Temp (°C)', 0.02),
    ]

    results = {}

    for col, name, tolerance in metrics:
        if col not in python_df.columns or col not in excel_df.columns:
            print(f"⚠ {name}: Column not found in both datasets")
            continue

        python_vals = python_df[col].values[:n_dwellings]
        excel_vals = excel_df[col].values[:n_dwellings]

        # Calculate statistics
        python_mean = np.mean(python_vals)
        excel_mean = np.mean(excel_vals)
        python_std = np.std(python_vals)
        excel_std = np.std(excel_vals)

        # Calculate differences
        mean_diff_pct = abs(python_mean - excel_mean) / excel_mean * 100 if excel_mean != 0 else 0
        std_diff_pct = abs(python_std - excel_std) / excel_std * 100 if excel_std != 0 else 0

        # Kolmogorov-Smirnov test for distribution similarity
        ks_stat, ks_pvalue = stats.ks_2samp(python_vals, excel_vals)

        # Determine pass/fail
        mean_pass = mean_diff_pct <= (tolerance * 100)
        std_pass = std_diff_pct <= (tolerance * 100 * 2)  # 2x tolerance for std dev
        ks_pass = ks_pvalue > 0.05  # p > 0.05 means distributions are similar

        overall_pass = mean_pass and ks_pass

        # Store results
        results[col] = {
            'name': name,
            'python_mean': python_mean,
            'excel_mean': excel_mean,
            'python_std': python_std,
            'excel_std': excel_std,
            'mean_diff_pct': mean_diff_pct,
            'std_diff_pct': std_diff_pct,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mean_pass': mean_pass,
            'std_pass': std_pass,
            'ks_pass': ks_pass,
            'overall_pass': overall_pass
        }

        # Print results
        status = "✓ PASS" if overall_pass else "✗ FAIL"
        print(f"{status} {name}")
        print(f"  Python:  mean={python_mean:.3f}, std={python_std:.3f}")
        print(f"  Excel:   mean={excel_mean:.3f}, std={excel_std:.3f}")
        print(f"  Difference: mean={mean_diff_pct:.2f}%, std={std_diff_pct:.2f}%")
        print(f"  KS test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
        print()

    return results


def compare_time_series(python_df: pd.DataFrame, excel_df: pd.DataFrame, dwelling_idx: int = 0):
    """Compare time-series data for a single dwelling."""
    if python_df is None or excel_df is None:
        return None

    print("\n" + "=" * 80)
    print(f"TIME-SERIES COMPARISON (Dwelling {dwelling_idx + 1})")
    print("=" * 80)

    # Filter data for specific dwelling
    python_dwelling = python_df[python_df['Dwelling'] == dwelling_idx + 1].sort_values('Minute')
    excel_dwelling = excel_df[excel_df['Dwelling'] == dwelling_idx + 1].sort_values('Minute')

    if len(python_dwelling) == 0 or len(excel_dwelling) == 0:
        print(f"  Warning: No data found for dwelling {dwelling_idx + 1}")
        return None

    # Compare key time-series
    metrics = [
        ('Total_Electricity_W', 'Total Electricity'),
        ('Internal_Temp_C', 'Internal Temperature'),
        ('Hot_Water_Demand_L_per_min', 'Hot Water Demand'),
    ]

    results = {}

    for col, name in metrics:
        if col not in python_dwelling.columns or col not in excel_dwelling.columns:
            print(f"⚠ {name}: Column not found")
            continue

        python_series = python_dwelling[col].values
        excel_series = excel_dwelling[col].values

        min_len = min(len(python_series), len(excel_series))
        python_series = python_series[:min_len]
        excel_series = excel_series[:min_len]

        # Calculate correlation
        correlation = np.corrcoef(python_series, excel_series)[0, 1]

        # Calculate RMSE
        rmse = np.sqrt(np.mean((python_series - excel_series) ** 2))

        # Calculate MAE
        mae = np.mean(np.abs(python_series - excel_series))

        # Determine pass/fail
        corr_pass = correlation > 0.95 if not np.isnan(correlation) else False

        results[col] = {
            'name': name,
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            'pass': corr_pass
        }

        status = "✓ PASS" if corr_pass else "✗ FAIL"
        print(f"{status} {name}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print()

    return results


def generate_summary_report(daily_results: dict, timeseries_results: dict):
    """Generate overall validation summary."""
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # Count passes and fails
    daily_passed = sum(1 for r in daily_results.values() if r.get('overall_pass', False))
    daily_total = len(daily_results)

    print(f"\nDaily Summary Tests: {daily_passed}/{daily_total} passed")

    if timeseries_results:
        ts_passed = sum(1 for r in timeseries_results.values() if r.get('pass', False))
        ts_total = len(timeseries_results)
        print(f"Time-Series Tests: {ts_passed}/{ts_total} passed")

    # Overall assessment
    overall_pass_rate = daily_passed / daily_total if daily_total > 0 else 0

    print("\n" + "-" * 80)
    if overall_pass_rate >= 0.8:
        print("✓ VALIDATION PASSED")
        print(f"  {overall_pass_rate*100:.1f}% of tests passed")
        print("  The Python port shows good statistical agreement with the Excel model.")
    elif overall_pass_rate >= 0.6:
        print("⚠ VALIDATION PARTIAL")
        print(f"  {overall_pass_rate*100:.1f}% of tests passed")
        print("  The Python port shows reasonable agreement but needs review.")
    else:
        print("✗ VALIDATION FAILED")
        print(f"  Only {overall_pass_rate*100:.1f}% of tests passed")
        print("  Significant differences detected - requires investigation.")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python and Excel CREST model results"
    )
    parser.add_argument(
        "python_dir",
        type=Path,
        help="Directory containing Python simulation results"
    )
    parser.add_argument(
        "excel_dir",
        type=Path,
        help="Directory containing extracted Excel results"
    )
    parser.add_argument(
        "--dwelling",
        type=int,
        default=0,
        help="Dwelling index for time-series comparison (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for detailed report (optional)"
    )

    args = parser.parse_args()

    # Check directories exist
    if not args.python_dir.exists():
        print(f"Error: Python results directory not found: {args.python_dir}")
        sys.exit(1)

    if not args.excel_dir.exists():
        print(f"Error: Excel results directory not found: {args.excel_dir}")
        sys.exit(1)

    # Load data
    data = load_data(args.python_dir, args.excel_dir)
    if data is None:
        sys.exit(1)

    # Compare daily summaries
    daily_results = compare_daily_summaries(data['python_daily'], data['excel_daily'])

    # Compare time-series if available
    timeseries_results = None
    if data['python_minute'] is not None and data['excel_minute'] is not None:
        timeseries_results = compare_time_series(
            data['python_minute'],
            data['excel_minute'],
            args.dwelling
        )

    # Generate summary
    generate_summary_report(daily_results, timeseries_results)

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
