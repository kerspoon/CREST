#!/usr/bin/env python3
"""Compare Excel and Python Monte Carlo runs using IQR validation.

This script implements Objective #2: Statistical Distribution Validation
- Compare 20 Excel runs against 1000 Python runs
- For each minute, dwelling, and variable: check if Excel value falls in Python IQR
- Expected: >50% of Excel samples should fall within Python IQR
- Test across 72,000+ data points (5 houses × 20 runs × 5 variables × 1440 minutes)

Usage:
    python scripts/monte_carlo_compare.py \\
        output/monte_carlo/python_1000runs_YYYYMMDD_NN \\
        output/monte_carlo/excel_20runs_YYYYMMDD_NN
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import helper utilities
from utils import create_validation_dir, save_metadata, get_project_root


# Column name mappings: Excel VBA → Python
EXCEL_TO_PYTHON = {
    'Dwelling index': 'dwelling',
    'Lighting demand': 'Lighting_W',
    'Appliance demand': 'Appliances_W',
    'Net dwelling electricity demand': 'Total_Electricity_W',
    'Internal building node temperature': 'Internal_Temp_C',
    'Hot water demand (litres)': 'Hot_Water_Demand_L_per_min',
    'Fuel flow rate (gas)': 'Gas_Consumption_m3_per_min'
}

# Variables to compare (Python column names)
VARIABLES = {
    'electricity': 'Total_Electricity_W',
    'gas': 'Gas_Consumption_m3_per_min',
    'water': 'Hot_Water_Demand_L_per_min',
    'temperature': 'Internal_Temp_C',
    'lighting': 'Lighting_W'
}


def find_column(df: pd.DataFrame, possible_names: List[str]) -> str:
    """Find first matching column name from a list of possibilities."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def load_python_baseline(python_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Python Monte Carlo baseline (minute-level and daily).

    Returns:
        Tuple of (minute_df, daily_df)
    """
    print(f"\nLoading Python baseline from: {python_dir}")

    # Try loading minute-level data (parquet or CSV)
    minute_df = None
    for filename in ['minute_level.parquet', 'monte_carlo_minute.parquet', 'results_minute_level.csv']:
        filepath = python_dir / filename
        if filepath.exists():
            if filepath.suffix == '.parquet':
                minute_df = pd.read_parquet(filepath)
            else:
                minute_df = pd.read_csv(filepath)
            print(f"  Loaded minute data: {filename} ({len(minute_df):,} rows)")
            break

    if minute_df is None:
        print("  ERROR: No minute-level data found!")
        return None, None

    # Load daily summary
    daily_df = None
    for filename in ['daily_summary.csv', 'monte_carlo_daily.csv', 'results_daily_summary.csv']:
        filepath = python_dir / filename
        if filepath.exists():
            daily_df = pd.read_csv(filepath)
            print(f"  Loaded daily data: {filename} ({len(daily_df)} rows)")
            break

    return minute_df, daily_df


def load_excel_runs(excel_dir: Path) -> List[Dict[str, pd.DataFrame]]:
    """
    Load Excel runs (expecting either run_NN/ subdirs or vba_run_N.csv files).

    Returns:
        List of dicts with 'minute' and 'daily' dataframes for each run
    """
    print(f"\nLoading Excel runs from: {excel_dir}")

    runs = []

    # Try subdirectories first (run_01/, run_02/, etc.)
    run_dirs = sorted([d for d in excel_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])

    if run_dirs:
        for run_dir in run_dirs:
            run_data = {}

            # Load minute-level
            minute_file = run_dir / 'results_minute_level.csv'
            if minute_file.exists():
                run_data['minute'] = pd.read_csv(minute_file)
            else:
                print(f"  WARN: No minute data in {run_dir.name}")
                continue

            # Load daily summary
            daily_file = run_dir / 'results_daily_summary.csv'
            if daily_file.exists():
                run_data['daily'] = pd.read_csv(daily_file)

            run_data['run_name'] = run_dir.name
            runs.append(run_data)
    else:
        # Try VBA run files (vba_run_1.csv, vba_run_2.csv, etc.)
        vba_files = sorted(excel_dir.glob('vba_run_*.csv'))

        if not vba_files:
            print("  ERROR: No run_* subdirectories or vba_run_*.csv files found!")
            return []

        for vba_file in vba_files:
            run_data = {}
            # Excel disaggregated results have header rows - skip them
            df = pd.read_csv(vba_file, skiprows=[0, 1, 2, 4, 5])

            # Rename Excel columns to match Python column names
            df = df.rename(columns=EXCEL_TO_PYTHON)

            # Add minute column (row number, 1-indexed)
            if 'Minute' not in df.columns:
                df['Minute'] = range(1, len(df) + 1)

            run_data['minute'] = df
            run_data['run_name'] = vba_file.stem  # e.g., 'vba_run_1'
            runs.append(run_data)

    print(f"  Loaded {len(runs)} Excel runs")
    return runs


def compute_python_iqr(python_minute: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IQR statistics for each (dwelling, minute, variable) combination.

    Returns:
        DataFrame with columns: dwelling, minute, variable, q1, median, q3, iqr
    """
    print("\nComputing Python IQR statistics...")

    # Detect time column
    time_col = find_column(python_minute, ['Minute', 'minute', 'time', 'timestep'])
    if time_col is None:
        print("  ERROR: No time column found!")
        return pd.DataFrame()

    # Detect dwelling column
    dwelling_col = find_column(python_minute, ['dwelling', 'Dwelling', 'dwelling_index'])
    if dwelling_col is None:
        print("  ERROR: No dwelling column found!")
        return pd.DataFrame()

    # Normalize column names
    python_minute = python_minute.rename(columns={time_col: 'minute', dwelling_col: 'dwelling'})

    stats_list = []
    dwellings = sorted(python_minute['dwelling'].unique())
    minutes = range(1, 1441)  # 1440 minutes in a day

    for dwelling in dwellings:
        print(f"  Computing for dwelling {dwelling}...")
        d = python_minute[python_minute['dwelling'] == dwelling]

        for minute in minutes:
            m = d[d['minute'] == minute]

            if len(m) < 10:  # Need enough samples
                continue

            row = {'dwelling': dwelling, 'minute': minute}

            # Compute IQR for each variable
            for var_name, col_name in VARIABLES.items():
                if col_name in m.columns:
                    values = m[col_name].dropna()
                    if len(values) > 0:
                        row[f'{var_name}_q1'] = np.percentile(values, 25)
                        row[f'{var_name}_median'] = np.median(values)
                        row[f'{var_name}_q3'] = np.percentile(values, 75)
                        row[f'{var_name}_iqr'] = row[f'{var_name}_q3'] - row[f'{var_name}_q1']

            stats_list.append(row)

    df_stats = pd.DataFrame(stats_list)
    print(f"  Computed {len(df_stats)} (dwelling, minute) combinations")

    return df_stats


def validate_excel_against_iqr(
    excel_runs: List[Dict[str, pd.DataFrame]],
    python_iqr: pd.DataFrame
) -> pd.DataFrame:
    """
    Check how many Excel values fall within Python IQR.

    Returns:
        DataFrame with validation results
    """
    print("\nValidating Excel runs against Python IQR...")

    results = []

    for run_data in excel_runs:
        run_name = run_data['run_name']
        excel_minute = run_data['minute']

        # Detect and normalize columns
        time_col = find_column(excel_minute, ['Minute', 'minute', 'time'])
        dwelling_col = find_column(excel_minute, ['dwelling', 'Dwelling', 'dwelling_index'])

        if time_col is None or dwelling_col is None:
            print(f"  WARN: Skipping {run_name} - missing columns")
            continue

        excel_minute = excel_minute.rename(columns={time_col: 'minute', dwelling_col: 'dwelling'})

        # Merge with Python IQR
        for dwelling in sorted(python_iqr['dwelling'].unique()):
            python_d = python_iqr[python_iqr['dwelling'] == dwelling]
            excel_d = excel_minute[excel_minute['dwelling'] == dwelling]

            merged = excel_d.merge(python_d, on='minute', how='inner')

            if len(merged) == 0:
                continue

            # Check each variable
            for var_name, col_name in VARIABLES.items():
                if col_name not in merged.columns:
                    continue

                q1_col = f'{var_name}_q1'
                q3_col = f'{var_name}_q3'

                if q1_col not in merged.columns or q3_col not in merged.columns:
                    continue

                # Count how many Excel values fall in Python IQR
                in_iqr = (merged[col_name] >= merged[q1_col]) & (merged[col_name] <= merged[q3_col])
                total = len(merged)
                in_iqr_count = in_iqr.sum()
                in_iqr_pct = 100 * in_iqr_count / total if total > 0 else 0

                results.append({
                    'run': run_name,
                    'dwelling': dwelling,
                    'variable': var_name,
                    'total_minutes': total,
                    'in_iqr_count': in_iqr_count,
                    'in_iqr_pct': in_iqr_pct,
                    'excel_mean': merged[col_name].mean(),
                    'python_median': merged[f'{var_name}_median'].mean() if f'{var_name}_median' in merged.columns else np.nan
                })

    df_results = pd.DataFrame(results)
    print(f"  Validated {len(df_results)} (run, dwelling, variable) combinations")

    return df_results


def generate_summary(validation_results: pd.DataFrame, validation_dir: Path) -> None:
    """Generate summary statistics and save reports."""
    print("\nGenerating summary report...")

    # Overall statistics
    overall = validation_results.groupby('variable').agg({
        'in_iqr_pct': ['mean', 'std', 'min', 'max'],
        'total_minutes': 'sum'
    }).reset_index()

    # Save detailed results
    results_file = validation_dir / 'iqr_analysis.csv'
    validation_results.to_csv(results_file, index=False)
    print(f"  Saved: {results_file}")

    # Save summary
    summary_file = validation_dir / 'summary_statistics.csv'
    overall.to_csv(summary_file, index=False)
    print(f"  Saved: {summary_file}")

    # Generate text report
    report = []
    report.append("=" * 80)
    report.append("MONTE CARLO IQR VALIDATION RESULTS")
    report.append("=" * 80)
    report.append("")
    report.append("Objective: >50% of Excel samples should fall within Python IQR")
    report.append("")

    for var in validation_results['variable'].unique():
        var_data = validation_results[validation_results['variable'] == var]
        mean_pct = var_data['in_iqr_pct'].mean()
        std_pct = var_data['in_iqr_pct'].std()
        total_tests = var_data['total_minutes'].sum()

        status = "✓ PASS" if mean_pct >= 50 else "✗ FAIL"

        report.append(f"{var.upper()}: {mean_pct:.1f}% ± {std_pct:.1f}% in IQR {status}")
        report.append(f"  Total data points tested: {total_tests:,}")
        report.append("")

    total_data_points = validation_results['total_minutes'].sum()
    report.append(f"TOTAL DATA POINTS: {total_data_points:,}")
    report.append("")
    report.append("=" * 80)

    # Print to console
    print("\n" + '\n'.join(report))

    # Save to file
    report_file = validation_dir / 'validation_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    print(f"  Saved: {report_file}")


# Column names for daily totals analysis (15 variables, columns 3-17 from VBA)
# VBA Reference: DailyTotals columns 3-17 (mdlThermalElectricalModel.bas lines 1105-1119)
DAILY_VARIABLES = {
    'Mean_Active_Occupancy': 'Mean active occupancy',
    'Proportion_Day_Actively_Occupied': 'Proportion actively occupied',
    'Lighting_Demand_kWh': 'Lighting demand',
    'Appliance_Demand_kWh': 'Appliance demand',
    'PV_Output_kWh': 'PV output',
    'Total_Electricity_Demand_kWh': 'Total electricity demand',
    'Self_Consumption_kWh': 'Self consumption',
    'Net_Electricity_Demand_kWh': 'Net electricity demand',
    'Hot_Water_Demand_L': 'Hot water demand',
    'Average_Indoor_Temperature_C': 'Average indoor temperature',
    'Thermal_Energy_Space_Heating_kWh': 'Thermal energy space heating',
    'Thermal_Energy_Water_Heating_kWh': 'Thermal energy water heating',
    'Gas_Demand_m3': 'Gas demand',
    'Space_Thermostat_Setpoint_C': 'Space thermostat setpoint',
    'Solar_Thermal_Heat_Gains_kWh': 'Solar thermal heat gains'
}


def compute_daily_totals_iqr(python_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IQR statistics for daily totals (per dwelling, per variable).

    VBA Reference: Compare 15 variables from columns 3-17 of DailyTotals

    Args:
        python_daily: DataFrame with daily summary data from Python runs

    Returns:
        DataFrame with columns: dwelling, variable, q1, median, q3, iqr, count
    """
    print("\nComputing Python daily totals IQR...")

    if python_daily is None or len(python_daily) == 0:
        print("  ERROR: No Python daily data to analyze!")
        return pd.DataFrame()

    # Auto-detect dwelling column
    dwelling_col = find_column(python_daily, ['Dwelling', 'dwelling', 'dwelling_index'])
    if dwelling_col is None:
        print("  ERROR: No dwelling column found in Python daily data!")
        return pd.DataFrame()

    # Normalize column name
    if dwelling_col != 'Dwelling':
        python_daily = python_daily.rename(columns={dwelling_col: 'Dwelling'})

    stats_list = []
    dwellings = sorted(python_daily['Dwelling'].unique())

    for dwelling in dwellings:
        print(f"  Computing for dwelling {dwelling}...")
        d = python_daily[python_daily['Dwelling'] == dwelling]

        for var_name, var_label in DAILY_VARIABLES.items():
            if var_name not in d.columns:
                print(f"    WARN: Column {var_name} not found, skipping")
                continue

            values = d[var_name].dropna()
            if len(values) < 10:  # Need enough samples
                print(f"    WARN: Only {len(values)} samples for {var_name}, skipping")
                continue

            stats_list.append({
                'dwelling': dwelling,
                'variable': var_name,
                'label': var_label,
                'q1': np.percentile(values, 25),
                'median': np.median(values),
                'q3': np.percentile(values, 75),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                'count': len(values)
            })

    df_stats = pd.DataFrame(stats_list)
    print(f"  Computed {len(df_stats)} (dwelling, variable) combinations")

    return df_stats


def validate_excel_daily_against_iqr(
    excel_runs: List[Dict[str, pd.DataFrame]],
    python_iqr: pd.DataFrame
) -> pd.DataFrame:
    """
    Check how many Excel daily total values fall within Python IQR.

    Args:
        excel_runs: List of Excel run data (each with 'daily' DataFrame)
        python_iqr: Python IQR statistics for daily totals

    Returns:
        DataFrame with validation results per (run, dwelling, variable)
    """
    print("\nValidating Excel daily totals against Python IQR...")

    results = []

    for run_data in excel_runs:
        run_name = run_data['run_name']

        if 'daily' not in run_data or run_data['daily'] is None:
            print(f"  WARN: No daily data for {run_name}, skipping")
            continue

        excel_daily = run_data['daily']

        # Normalize dwelling column
        dwelling_col = find_column(excel_daily, ['Dwelling', 'dwelling', 'Dwelling index', 'dwelling_index'])
        if dwelling_col is None:
            print(f"  WARN: No dwelling column in {run_name}, skipping")
            continue

        if dwelling_col != 'Dwelling':
            excel_daily = excel_daily.rename(columns={dwelling_col: 'Dwelling'})

        # Check each variable for each dwelling
        for dwelling in sorted(python_iqr['dwelling'].unique()):
            python_d = python_iqr[python_iqr['dwelling'] == dwelling]
            excel_d = excel_daily[excel_daily['Dwelling'] == dwelling]

            if len(excel_d) == 0:
                continue

            # For daily totals, we expect just 1 row per dwelling
            if len(excel_d) > 1:
                print(f"  WARN: Multiple rows for dwelling {dwelling} in {run_name}, using first")
                excel_d = excel_d.iloc[0:1]

            for _, python_row in python_d.iterrows():
                var_name = python_row['variable']

                if var_name not in excel_d.columns:
                    continue

                excel_value = excel_d[var_name].iloc[0]
                q1 = python_row['q1']
                q3 = python_row['q3']

                # Check if Excel value falls in Python IQR
                in_iqr = (excel_value >= q1) and (excel_value <= q3)

                results.append({
                    'run': run_name,
                    'dwelling': dwelling,
                    'variable': var_name,
                    'label': python_row['label'],
                    'excel_value': excel_value,
                    'python_q1': q1,
                    'python_median': python_row['median'],
                    'python_q3': q3,
                    'in_iqr': 1 if in_iqr else 0
                })

    df_results = pd.DataFrame(results)
    print(f"  Validated {len(df_results)} (run, dwelling, variable) combinations")

    return df_results


def generate_daily_totals_wide_format(
    excel_runs: List[Dict[str, pd.DataFrame]],
    python_daily_iqr: pd.DataFrame,
    validation_dir: Path
) -> None:
    """
    Generate wide-format comparison table for daily totals.

    Format:
    - Columns: name, Dwelling, + 15 variables
    - Rows: excel 1...N, py25%, py50%, py75%, % within IQR (all per dwelling)

    Args:
        excel_runs: List of Excel run data
        python_daily_iqr: Python IQR statistics (dwelling, variable, q1, median, q3)
        validation_dir: Output directory
    """
    print("\nGenerating daily totals wide-format comparison...")

    # Get list of dwellings
    dwellings = sorted(python_daily_iqr['dwelling'].unique())

    # Column headers: use human-readable labels from VBA
    # Map from our column names to VBA-style labels
    column_labels = {
        'Mean_Active_Occupancy': 'Mean active occupancy',
        'Proportion_Day_Actively_Occupied': 'Proportion of day actively occupied',
        'Lighting_Demand_kWh': 'Lighting demand',
        'Appliance_Demand_kWh': 'Appliance demand',
        'PV_Output_kWh': 'PV output',
        'Total_Electricity_Demand_kWh': 'Total dwelling electricity demand',
        'Self_Consumption_kWh': 'Total self-consumption',
        'Net_Electricity_Demand_kWh': 'Net dwelling electricity demand',
        'Hot_Water_Demand_L': 'Hot water demand (litres)',
        'Average_Indoor_Temperature_C': 'Average indoor air temperature',
        'Thermal_Energy_Space_Heating_kWh': 'Thermal energy used for space heating',
        'Thermal_Energy_Water_Heating_kWh': 'Thermal energy used for hot water heating',
        'Gas_Demand_m3': 'Gas demand',
        'Space_Thermostat_Setpoint_C': 'Space thermostat set point',
        'Solar_Thermal_Heat_Gains_kWh': 'Solar thermal collector heat gains'
    }

    # Build rows list
    rows = []

    # 1. Add Excel run rows (excel 1, excel 2, ..., excel N for each dwelling)
    for run_data in excel_runs:
        run_name = run_data['run_name']

        if 'daily' not in run_data or run_data['daily'] is None:
            continue

        excel_daily = run_data['daily']

        # Normalize dwelling column
        dwelling_col = find_column(excel_daily, ['Dwelling', 'dwelling', 'Dwelling index', 'dwelling_index'])
        if dwelling_col and dwelling_col != 'Dwelling':
            excel_daily = excel_daily.rename(columns={dwelling_col: 'Dwelling'})

        # Extract run number from run_name (e.g., "run_01" -> 1, "vba_run_3" -> 3)
        import re
        match = re.search(r'(\d+)', run_name)
        run_num = int(match.group(1)) if match else run_name

        for dwelling in dwellings:
            excel_d = excel_daily[excel_daily['Dwelling'] == dwelling]

            if len(excel_d) == 0:
                continue

            if len(excel_d) > 1:
                excel_d = excel_d.iloc[0:1]

            row = {'name': f'excel {run_num}', 'Dwelling': dwelling}

            # Add values for each variable
            for col_name, label in column_labels.items():
                if col_name in excel_d.columns:
                    row[label] = excel_d[col_name].iloc[0]
                else:
                    row[label] = np.nan

            rows.append(row)

    # 2. Add Python quartile rows (py25%, py50%, py75% for each dwelling)
    for quartile_name, quartile_col in [('py25%', 'q1'), ('py50%', 'median'), ('py75%', 'q3')]:
        for dwelling in dwellings:
            row = {'name': quartile_name, 'Dwelling': dwelling}

            # Get Python IQR data for this dwelling
            python_d = python_daily_iqr[python_daily_iqr['dwelling'] == dwelling]

            for col_name, label in column_labels.items():
                python_var = python_d[python_d['variable'] == col_name]
                if len(python_var) > 0:
                    row[label] = python_var[quartile_col].iloc[0]
                else:
                    row[label] = np.nan

            rows.append(row)

    # 3. Add "% within IQR" rows (for each dwelling)
    for dwelling in dwellings:
        row = {'name': '% within IQR', 'Dwelling': dwelling}

        # For each variable, calculate what % of Excel runs fell within IQR
        python_d = python_daily_iqr[python_daily_iqr['dwelling'] == dwelling]

        for col_name, label in column_labels.items():
            python_var = python_d[python_d['variable'] == col_name]

            if len(python_var) == 0:
                row[label] = np.nan
                continue

            q1 = python_var['q1'].iloc[0]
            q3 = python_var['q3'].iloc[0]

            # Count how many Excel runs fall within [q1, q3] for this dwelling & variable
            in_iqr_count = 0
            total_count = 0

            for run_data in excel_runs:
                if 'daily' not in run_data or run_data['daily'] is None:
                    continue

                excel_daily = run_data['daily']
                dwelling_col = find_column(excel_daily, ['Dwelling', 'dwelling', 'Dwelling index', 'dwelling_index'])
                if dwelling_col and dwelling_col != 'Dwelling':
                    excel_daily = excel_daily.rename(columns={dwelling_col: 'Dwelling'})

                excel_d = excel_daily[excel_daily['Dwelling'] == dwelling]

                if len(excel_d) > 0 and col_name in excel_d.columns:
                    excel_value = excel_d[col_name].iloc[0]
                    total_count += 1
                    if q1 <= excel_value <= q3:
                        in_iqr_count += 1

            # Calculate percentage
            if total_count > 0:
                row[label] = 100.0 * in_iqr_count / total_count
            else:
                row[label] = np.nan

        rows.append(row)

    # Convert to DataFrame
    df_comparison = pd.DataFrame(rows)

    # Ensure columns are in the right order
    column_order = ['name', 'Dwelling'] + [column_labels[col] for col in column_labels.keys()]
    df_comparison = df_comparison[column_order]

    # Save to CSV
    output_file = validation_dir / 'daily_totals_comparison.csv'
    df_comparison.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print(f"  Rows: {len(df_comparison)} ({len(excel_runs)} Excel runs × {len(dwellings)} dwellings + quartiles + summary)")

    return df_comparison


def main():
    """Main validation workflow."""
    # Change to project root for consistent paths
    import os
    project_root = get_project_root()
    os.chdir(project_root)

    if len(sys.argv) < 3:
        print("Usage: python scripts/monte_carlo_compare.py <python_dir> <excel_dir>")
        print("\nExample:")
        print("  python scripts/monte_carlo_compare.py \\")
        print("    output/monte_carlo/python_1000runs_20250113_01 \\")
        print("    output/monte_carlo/excel_20runs_20250113_01")
        sys.exit(1)

    python_dir = Path(sys.argv[1])
    excel_dir = Path(sys.argv[2])

    # Validate directories exist
    if not python_dir.exists():
        print(f"ERROR: Python directory not found: {python_dir}")
        sys.exit(1)

    if not excel_dir.exists():
        print(f"ERROR: Excel directory not found: {excel_dir}")
        sys.exit(1)

    print("=" * 80)
    print("CREST MONTE CARLO IQR VALIDATION")
    print("=" * 80)

    # Load Python baseline
    python_minute, python_daily = load_python_baseline(python_dir)
    if python_minute is None:
        sys.exit(1)

    # Load Excel runs
    excel_runs = load_excel_runs(excel_dir)
    if not excel_runs:
        sys.exit(1)

    # Compute Python IQR
    python_iqr = compute_python_iqr(python_minute)
    if len(python_iqr) == 0:
        print("ERROR: Failed to compute Python IQR")
        sys.exit(1)

    # Validate Excel against Python IQR
    validation_results = validate_excel_against_iqr(excel_runs, python_iqr)
    if len(validation_results) == 0:
        print("ERROR: Validation failed")
        sys.exit(1)

    # Create validation directory
    validation_dir = create_validation_dir(str(python_dir), str(excel_dir), "monte_carlo")
    print(f"\nValidation directory: {validation_dir}")

    # Save metadata
    save_metadata(
        validation_dir,
        str(python_dir),
        str(excel_dir),
        python_runs=int(len(python_minute['seed'].unique())) if 'seed' in python_minute.columns else "unknown",
        excel_runs=int(len(excel_runs)),
        total_data_points=int(validation_results['total_minutes'].sum())
    )

    # Generate summary for minute-level validation
    generate_summary(validation_results, validation_dir)

    # ========================================================================
    # DAILY TOTALS VALIDATION
    # ========================================================================
    if python_daily is not None and len(python_daily) > 0:
        print("\n" + "=" * 80)
        print("DAILY TOTALS VALIDATION")
        print("=" * 80)

        # Compute Python IQR for daily totals
        daily_iqr = compute_daily_totals_iqr(python_daily)
        if len(daily_iqr) > 0:
            # Generate wide-format comparison table
            generate_daily_totals_wide_format(excel_runs, daily_iqr, validation_dir)
        else:
            print("  WARNING: Failed to compute daily totals IQR")
    else:
        print("\n" + "=" * 80)
        print("WARNING: No Python daily data found - skipping daily totals validation")
        print("=" * 80)

    print("\n" + "=" * 80)
    print(f"VALIDATION COMPLETE - Results saved to: {validation_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
