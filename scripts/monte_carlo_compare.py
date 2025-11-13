#!/usr/bin/env python3
"""Comprehensive Monte Carlo IQR validation comparing Excel and Python runs.

This script implements Objective #2a: Statistical Distribution Validation

DAILY TOTALS: 15 columns (C-Q) × 5 houses × 20 Excel runs
DISAGGREGATED: 37 columns (D-AN) × 5 houses × 1440 minutes × 20 Excel runs

Outputs:
- Daily totals summary table with IQR statistics
- Disaggregated matrix: 37 variables (rows) × 5 houses (columns), showing % of
  timestamps (1440 × 20 = 28,800 per house) that fall within Python IQR
- Statistical analysis of expected variance

Usage:
    python scripts/monte_carlo_compare.py \\
        output/monte_carlo/python_1000runs_YYYYMMDD_NN \\
        output/monte_carlo/excel_20runs_YYYYMMDD_NN
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats as scipy_stats

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import helper utilities
from utils import create_validation_dir, save_metadata, get_project_root


# ============================================================================
# COLUMN MAPPINGS: Excel → Python
# ============================================================================

# Daily totals: columns C-Q from "Results - daily totals" sheet
# NOTE: Python output now matches Excel exactly - same column names!
DAILY_COLUMNS = {
    # Excel column name: (Python column name - now identical!, units, description)
    'Mean active occupancy': ('Mean active occupancy', '', 'Mean active occupancy'),
    'Proportion of day actively occupied': ('Proportion of day actively occupied', '', 'Proportion actively occupied'),
    'Lighting demand': ('Lighting demand', 'kWh', 'Lighting demand'),
    'Appliance demand': ('Appliance demand', 'kWh', 'Appliance demand'),
    'PV output': ('PV output', 'kWh', 'PV output'),
    'Total dwelling electricity demand': ('Total dwelling electricity demand', 'kWh', 'Total electricity'),
    'Total self-consumption': ('Total self-consumption', 'kWh', 'Self-consumption'),
    'Net dwelling electricity demand': ('Net dwelling electricity demand', 'kWh', 'Net electricity'),
    'Hot water demand (litres)': ('Hot water demand (litres)', 'L', 'Hot water demand'),
    'Average indoor air temperature': ('Average indoor air temperature', '°C', 'Average indoor temp'),
    'Thermal energy used for space heating': ('Thermal energy used for space heating', 'kWh', 'Space heating energy'),
    'Thermal energy used for hot water heating': ('Thermal energy used for hot water heating', 'kWh', 'Water heating energy'),
    'Gas demand': ('Gas demand', 'm³', 'Gas demand'),
    'Space thermostat set point': ('Space thermostat set point', '°C', 'Thermostat setpoint'),
    'Solar thermal collector heat gains': ('Solar thermal collector heat gains', 'kWh', 'Solar thermal gains'),
}

# Disaggregated: columns D-AN from "Results - disaggregated" sheet
DISAGGREGATED_COLUMNS = {
    # Excel column name: (Python column name, units, description)
    'Occupancy': ('At_Home', '', 'Occupancy state'),
    'Activity': ('Active', '', 'Activity state'),
    'Lighting demand': ('Lighting_W', 'W', 'Lighting power'),
    'Appliance demand': ('Appliances_W', 'W', 'Appliance power'),
    'Casual thermal gains from occupants, lighting and appliances': ('Casual_Gains_W', 'W', 'Casual gains'),
    'Outdoor temperature': ('Outdoor_Temp_C', '°C', 'Outdoor temp'),
    'Outdoor global radiation (horizontal)': ('Irradiance_Wm2', 'W/m²', 'Irradiance'),
    'Passive solar gains': ('Passive_Solar_Gains_W', 'W', 'Passive solar'),
    'Primary heating system thermal output': ('Total_Heat_Output_W', 'W', 'Total heating'),
    'External building node temperature': ('External_Building_Temp_C', '°C', 'External temp'),
    'Internal building node temperature': ('Internal_Temp_C', '°C', 'Internal temp'),
    'Hot water demand (litres)': ('Hot_Water_Demand_L_per_min', 'L/min', 'Hot water'),
    'Hot water temperature in hot water tank': ('Cylinder_Temp_C', '°C', 'Cylinder temp'),
    'Space heating timer settings': (None, '', 'Heating timer'),  # Not in Python output
    'Hot water heating timer settings': (None, '', 'HW timer'),  # Not in Python output
    'Heating system switched on': (None, '', 'Heating on'),  # Not in Python output
    'Hot water heating required': (None, '', 'HW heating req'),  # Not in Python output
    'Emitter temperature': ('Emitter_Temp_C', '°C', 'Emitter temp'),
    'Radiation incident on PV array': (None, 'W/m²', 'PV irradiance'),  # Not in Python output
    'PV output': ('PV_Output_W', 'W', 'PV power'),
    'Net dwelling electricity demand': ('Total_Electricity_W', 'W', 'Net electricity'),
    'Heat output from primary heating system to space': ('Space_Heating_W', 'W', 'Space heating'),
    'Heat output from primary heating system to hot water': ('Water_Heating_W', 'W', 'Water heating'),
    'Fuel flow rate (gas)': ('Gas_Consumption_m3_per_min', 'm³/min', 'Gas flow'),
    'Solar power incident on collector': (None, 'W', 'Solar collector'),  # Not in Python output
    'Solar thermal collector control state': (None, '', 'Collector state'),  # Not in Python output
    'Solar thermal collector temperature': (None, '°C', 'Collector temp'),  # Not in Python output
    'Heat gains to cylinder from solar thermal collector': (None, 'W', 'Solar thermal heat'),  # Not in Python output
    'Dwelling self-consumption': (None, 'kWh', 'Self-consumption'),  # Not in Python output
    'Space cooling timer settings': (None, '', 'Cooling timer'),  # Not in Python output
    'Cooling system switched on': (None, '', 'Cooling on'),  # Not in Python output
    'Cooling output from cooling system to space': (None, 'W', 'Cooling output'),  # Not in Python output
    'Cooler Emitter temperature': ('Cooling_Emitter_Temp_C', '°C', 'Cooling emitter'),
    'Heating Thermostat Set Point': (None, '°C', 'Heating setpoint'),  # Not in Python output
    'Cooling Thermostat Set Point': (None, '°C', 'Cooling setpoint'),  # Not in Python output
    'Electricity used by cooling system': ('Cooling_Electricity_W', 'W', 'Cooling electricity'),
    'Electricity used by heating system': ('Heating_Electricity_W', 'W', 'Heating electricity'),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """Find first matching column name from a list of possibilities."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def load_python_baseline(python_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load Python Monte Carlo baseline (minute-level and daily)."""
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
            print(f"  ✓ Loaded minute data: {filename} ({len(minute_df):,} rows)")
            break

    if minute_df is None:
        print("  ✗ ERROR: No minute-level data found!")
        sys.exit(1)

    # Load daily summary
    daily_df = None
    for filename in ['daily_summary.csv', 'monte_carlo_daily.csv', 'results_daily_summary.csv']:
        filepath = python_dir / filename
        if filepath.exists():
            daily_df = pd.read_csv(filepath)
            print(f"  ✓ Loaded daily data: {filename} ({len(daily_df)} rows)")
            break

    if daily_df is None:
        print("  ⚠ WARNING: No daily data found")

    return minute_df, daily_df


def load_excel_runs(excel_dir: Path) -> List[Dict[str, pd.DataFrame]]:
    """Load Excel runs (expecting run_NN/ subdirectories)."""
    print(f"\nLoading Excel runs from: {excel_dir}")

    runs = []

    # Look for run subdirectories (run_01/, run_02/, etc.)
    run_dirs = sorted([d for d in excel_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])

    if not run_dirs:
        print("  ✗ ERROR: No run_* subdirectories found!")
        print("  Expected format: excel_dir/run_01/, excel_dir/run_02/, etc.")
        sys.exit(1)

    for run_dir in run_dirs:
        run_data = {'run_name': run_dir.name}

        # Load minute-level (disaggregated)
        minute_file = run_dir / 'results_minute_level.csv'
        if minute_file.exists():
            try:
                # Excel exports have:
                # Row 0: Description (with BOM)
                # Row 1: Column names
                # Row 2: Units symbols (Greek letters)
                # Row 3: Units text
                # Row 4+: Data
                df_minute = pd.read_csv(minute_file, skiprows=[0, 2, 3], encoding='utf-8-sig')

                # Verify we have the expected columns
                if 'Dwelling index' in df_minute.columns and 'Time' in df_minute.columns:
                    run_data['minute'] = df_minute
                    print(f"  ✓ {run_dir.name}: minute data ({len(df_minute)} rows)")
                else:
                    print(f"  ⚠ {run_dir.name}: Missing columns in minute data")
                    print(f"    Found: {list(df_minute.columns[:5])}...")
            except Exception as e:
                print(f"  ⚠ {run_dir.name}: Could not load minute data: {e}")

        # Load daily summary
        daily_file = run_dir / 'results_daily_summary.csv'
        if daily_file.exists():
            try:
                # Excel exports have:
                # Row 0: Description
                # Row 1: Column names
                # Row 2: Units symbols
                # Row 3: Units text
                # Row 4+: Data
                df_daily = pd.read_csv(daily_file, skiprows=[0, 2, 3], encoding='utf-8-sig')

                # Verify we have the expected columns
                if 'Dwelling index' in df_daily.columns:
                    run_data['daily'] = df_daily
                    print(f"  ✓ {run_dir.name}: daily data ({len(df_daily)} rows)")
                else:
                    print(f"  ⚠ {run_dir.name}: Missing columns in daily data")
                    print(f"    Found: {list(df_daily.columns[:5])}...")
            except Exception as e:
                print(f"  ⚠ {run_dir.name}: Could not load daily data: {e}")

        if 'minute' in run_data or 'daily' in run_data:
            runs.append(run_data)

    print(f"\n  ✓ Loaded {len(runs)} Excel runs")
    if len(runs) == 0:
        print("  ✗ ERROR: No valid Excel runs found!")
        sys.exit(1)

    return runs


# ============================================================================
# DISAGGREGATED ANALYSIS (37 columns × 5 houses × 1440 minutes × 20 runs)
# ============================================================================

def compute_python_iqr_disaggregated(python_minute: pd.DataFrame) -> pd.DataFrame:
    """Compute IQR statistics for each (dwelling, minute, variable) combination."""
    print("\n" + "=" * 80)
    print("COMPUTING PYTHON IQR - DISAGGREGATED DATA")
    print("=" * 80)

    # Normalize column names
    time_col = find_column(python_minute, ['Minute', 'minute', 'time', 'timestep'])
    dwelling_col = find_column(python_minute, ['dwelling', 'Dwelling', 'Dwelling_index'])

    if not time_col or not dwelling_col:
        print(f"  ✗ ERROR: Missing required columns (time: {time_col}, dwelling: {dwelling_col})")
        sys.exit(1)

    python_minute = python_minute.rename(columns={time_col: 'minute', dwelling_col: 'dwelling'})

    # Get available variables
    available_vars = [(excel_name, py_col, desc)
                      for excel_name, (py_col, units, desc) in DISAGGREGATED_COLUMNS.items()
                      if py_col and py_col in python_minute.columns]

    print(f"  Testing {len(available_vars)} variables (out of {len(DISAGGREGATED_COLUMNS)} total)")

    stats_list = []
    dwellings = sorted(python_minute['dwelling'].unique())
    print(f"  Processing {len(dwellings)} dwellings...")

    for dwelling in dwellings:
        print(f"    Dwelling {dwelling}...", end=" ", flush=True)
        d = python_minute[python_minute['dwelling'] == dwelling]

        for minute in range(1, 1441):
            m = d[d['minute'] == minute]

            if len(m) < 10:  # Need enough samples for IQR
                continue

            row = {'dwelling': int(dwelling), 'minute': int(minute)}

            for excel_name, py_col, desc in available_vars:
                values = m[py_col].dropna()
                if len(values) > 0:
                    row[f'{excel_name}_q1'] = np.percentile(values, 25)
                    row[f'{excel_name}_median'] = np.median(values)
                    row[f'{excel_name}_q3'] = np.percentile(values, 75)

            stats_list.append(row)

        print(f"✓ ({len([r for r in stats_list if r['dwelling'] == dwelling])} minutes)")

    df_stats = pd.DataFrame(stats_list)
    print(f"\n  ✓ Computed IQR for {len(df_stats):,} (dwelling, minute) combinations")

    return df_stats


def validate_excel_disaggregated(
    excel_runs: List[Dict[str, pd.DataFrame]],
    python_iqr: pd.DataFrame
) -> pd.DataFrame:
    """Validate Excel disaggregated data against Python IQR."""
    print("\n" + "=" * 80)
    print("VALIDATING EXCEL DISAGGREGATED DATA")
    print("=" * 80)

    results = []

    for run_data in excel_runs:
        run_name = run_data['run_name']
        if 'minute' not in run_data:
            print(f"  ⚠ Skipping {run_name} - no minute data")
            continue

        excel_minute = run_data['minute']

        # Normalize columns
        time_col = find_column(excel_minute, ['Time', 'Minute', 'minute'])
        dwelling_col = find_column(excel_minute, ['Dwelling index', 'Dwelling', 'dwelling'])

        if not time_col or not dwelling_col:
            print(f"  ⚠ Skipping {run_name} - missing time/dwelling columns")
            continue

        # Parse time column (may be "HH:MM:SS" format)
        excel_minute = excel_minute.copy()
        if excel_minute[time_col].dtype == 'object':
            # Convert "HH:MM:SS" to minute number (1-1440)
            def parse_time(t):
                if pd.isna(t):
                    return None
                if ':' in str(t):
                    parts = str(t).split(':')
                    return int(parts[0]) * 60 + int(parts[1]) + 1
                return int(t)
            excel_minute['minute'] = excel_minute[time_col].apply(parse_time)
        else:
            excel_minute['minute'] = excel_minute[time_col].astype(int)

        excel_minute['dwelling'] = excel_minute[dwelling_col].astype(int)

        print(f"  {run_name}:", end=" ", flush=True)

        # Test each variable for each dwelling
        dwellings = sorted(python_iqr['dwelling'].unique())

        for dwelling in dwellings:
            python_d = python_iqr[python_iqr['dwelling'] == dwelling]
            excel_d = excel_minute[excel_minute['dwelling'] == dwelling]

            merged = excel_d.merge(python_d, on='minute', how='inner', suffixes=('_excel', '_py'))

            if len(merged) == 0:
                continue

            # Check each variable
            for excel_name, (py_col, units, desc) in DISAGGREGATED_COLUMNS.items():
                if not py_col:  # Skip unmapped columns
                    continue

                # Try to find Excel column
                excel_col = find_column(merged, [excel_name, py_col, f'{py_col}_excel'])
                if not excel_col:
                    continue

                q1_col = f'{excel_name}_q1'
                q3_col = f'{excel_name}_q3'

                if q1_col not in merged.columns or q3_col not in merged.columns:
                    continue

                # Count how many values fall in IQR
                values = merged[excel_col].dropna()
                q1 = merged[q1_col].dropna()
                q3 = merged[q3_col].dropna()

                if len(values) == 0 or len(q1) == 0 or len(q3) == 0:
                    continue

                in_iqr = (merged[excel_col] >= merged[q1_col]) & (merged[excel_col] <= merged[q3_col])
                total = len(merged)
                in_iqr_count = in_iqr.sum()
                in_iqr_pct = 100 * in_iqr_count / total if total > 0 else 0

                results.append({
                    'run': run_name,
                    'dwelling': int(dwelling),
                    'variable': excel_name,
                    'python_column': py_col,
                    'units': units,
                    'total_minutes': int(total),
                    'in_iqr_count': int(in_iqr_count),
                    'in_iqr_pct': float(in_iqr_pct),
                })

        print("✓")

    df_results = pd.DataFrame(results)
    print(f"\n  ✓ Validated {len(df_results):,} (run, dwelling, variable) combinations")

    return df_results


# ============================================================================
# DAILY TOTALS ANALYSIS (15 columns × 5 houses × 20 runs)
# ============================================================================

def compute_python_iqr_daily(python_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute IQR statistics for daily totals by dwelling."""
    print("\n" + "=" * 80)
    print("COMPUTING PYTHON IQR - DAILY TOTALS")
    print("=" * 80)

    # Normalize column names
    dwelling_col = find_column(python_daily, ['dwelling', 'Dwelling', 'Dwelling_index'])
    if not dwelling_col:
        print("  ✗ ERROR: No dwelling column found in daily data")
        sys.exit(1)

    python_daily = python_daily.rename(columns={dwelling_col: 'dwelling'})

    # Get available variables
    available_vars = []
    for excel_name, (py_col, units, desc) in DAILY_COLUMNS.items():
        if py_col and py_col in python_daily.columns:
            available_vars.append((excel_name, py_col, desc))

    print(f"  Testing {len(available_vars)} variables (out of {len(DAILY_COLUMNS)} total)")

    stats_list = []
    dwellings = sorted(python_daily['dwelling'].unique())

    for dwelling in dwellings:
        d = python_daily[python_daily['dwelling'] == dwelling]

        row = {'dwelling': int(dwelling)}

        for excel_name, py_col, desc in available_vars:
            values = d[py_col].dropna()
            if len(values) >= 10:  # Need enough samples
                row[f'{excel_name}_q1'] = np.percentile(values, 25)
                row[f'{excel_name}_median'] = np.median(values)
                row[f'{excel_name}_q3'] = np.percentile(values, 75)
                row[f'{excel_name}_mean'] = np.mean(values)
                row[f'{excel_name}_std'] = np.std(values)

        stats_list.append(row)

    df_stats = pd.DataFrame(stats_list)
    print(f"  ✓ Computed IQR for {len(dwellings)} dwellings")

    return df_stats


def validate_excel_daily(
    excel_runs: List[Dict[str, pd.DataFrame]],
    python_iqr: pd.DataFrame
) -> pd.DataFrame:
    """Validate Excel daily totals against Python IQR."""
    print("\n" + "=" * 80)
    print("VALIDATING EXCEL DAILY TOTALS")
    print("=" * 80)

    results = []

    for run_data in excel_runs:
        run_name = run_data['run_name']
        if 'daily' not in run_data:
            print(f"  ⚠ Skipping {run_name} - no daily data")
            continue

        excel_daily = run_data['daily']

        # Normalize dwelling column
        dwelling_col = find_column(excel_daily, ['Dwelling index', 'Dwelling', 'dwelling'])
        if not dwelling_col:
            print(f"  ⚠ Skipping {run_name} - no dwelling column")
            continue

        excel_daily = excel_daily.copy()
        excel_daily['dwelling'] = excel_daily[dwelling_col].astype(int)

        print(f"  {run_name}:", end=" ", flush=True)

        # Merge with Python IQR
        merged = excel_daily.merge(python_iqr, on='dwelling', how='inner', suffixes=('_excel', '_py'))

        if len(merged) == 0:
            print("no matches")
            continue

        # Check each variable for each dwelling
        for dwelling in sorted(merged['dwelling'].unique()):
            d = merged[merged['dwelling'] == dwelling]

            for excel_name, (py_col, units, desc) in DAILY_COLUMNS.items():
                if not py_col:  # Skip unmapped columns
                    continue

                # Try to find Excel column (should match exactly now)
                excel_col = find_column(d, [excel_name, py_col, f'{py_col}_excel'])
                if not excel_col:
                    continue

                q1_col = f'{excel_name}_q1'
                q3_col = f'{excel_name}_q3'

                if q1_col not in d.columns or q3_col not in d.columns:
                    continue

                # Get value and check if in IQR
                value = d[excel_col].iloc[0]
                q1 = d[q1_col].iloc[0]
                q3 = d[q3_col].iloc[0]

                if pd.isna(value) or pd.isna(q1) or pd.isna(q3):
                    continue

                in_iqr = (value >= q1) and (value <= q3)

                results.append({
                    'run': run_name,
                    'dwelling': int(dwelling),
                    'variable': excel_name,
                    'python_column': py_col,
                    'units': units,
                    'excel_value': float(value),
                    'python_q1': float(q1),
                    'python_median': float(d[f'{excel_name}_median'].iloc[0]) if f'{excel_name}_median' in d.columns else np.nan,
                    'python_q3': float(q3),
                    'in_iqr': bool(in_iqr),
                })

        print("✓")

    df_results = pd.DataFrame(results)
    print(f"\n  ✓ Validated {len(df_results)} (run, dwelling, variable) combinations")

    return df_results


# ============================================================================
# STATISTICAL VARIANCE ANALYSIS
# ============================================================================

def compute_expected_iqr_statistics(n_python: int, n_excel: int) -> Dict:
    """Compute expected IQR statistics for given sample sizes.

    By definition, 50% of samples should fall within the IQR. But with finite
    sample sizes, there's natural variance. This computes the expected distribution.
    """
    print("\n" + "=" * 80)
    print("EXPECTED IQR STATISTICS")
    print("=" * 80)
    print(f"  Python samples: {n_python}")
    print(f"  Excel samples: {n_excel}")
    print()

    # For n_excel samples, how many should fall in IQR?
    # This follows a binomial distribution: B(n, p=0.5)
    expected_mean = n_excel * 0.5
    expected_std = np.sqrt(n_excel * 0.5 * 0.5)

    # Confidence intervals
    ci_68 = (expected_mean - expected_std, expected_mean + expected_std)  # ~68% CI
    ci_95 = (expected_mean - 2*expected_std, expected_mean + 2*expected_std)  # ~95% CI
    ci_99 = (expected_mean - 3*expected_std, expected_mean + 3*expected_std)  # ~99.7% CI

    # Convert to percentages
    expected_pct = 100 * expected_mean / n_excel
    ci_68_pct = (100 * ci_68[0] / n_excel, 100 * ci_68[1] / n_excel)
    ci_95_pct = (100 * ci_95[0] / n_excel, 100 * ci_95[1] / n_excel)
    ci_99_pct = (100 * ci_99[0] / n_excel, 100 * ci_99[1] / n_excel)

    print(f"  Expected: {expected_mean:.1f} / {n_excel} = {expected_pct:.1f}%")
    print(f"  68% CI: {ci_68_pct[0]:.1f}% - {ci_68_pct[1]:.1f}%")
    print(f"  95% CI: {ci_95_pct[0]:.1f}% - {ci_95_pct[1]:.1f}%")
    print(f"  99.7% CI: {ci_99_pct[0]:.1f}% - {ci_99_pct[1]:.1f}%")
    print()

    # How unlikely is it to be off by various amounts?
    for delta_pct in [0.1, 1.0, 10.0]:
        delta_count = delta_pct / 100 * n_excel
        z_score = abs(delta_count) / expected_std if expected_std > 0 else 0
        p_value = 2 * (1 - scipy_stats.norm.cdf(z_score))  # Two-tailed
        print(f"  Probability of being off by ±{delta_pct}%: {p_value:.2%}")

    return {
        'n_python': n_python,
        'n_excel': n_excel,
        'expected_mean': expected_mean,
        'expected_std': expected_std,
        'expected_pct': expected_pct,
        'ci_68_pct': ci_68_pct,
        'ci_95_pct': ci_95_pct,
        'ci_99_pct': ci_99_pct,
    }


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def generate_disaggregated_summary_table(validation_results: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table: 37 variables (rows) × 5 houses (columns).

    Each cell shows % of timestamps (1440 × 20 = 28,800) in IQR.
    """
    print("\n" + "=" * 80)
    print("GENERATING DISAGGREGATED SUMMARY TABLE")
    print("=" * 80)

    # For each (dwelling, variable), compute aggregate IQR percentage
    summary = validation_results.groupby(['variable', 'dwelling']).agg({
        'in_iqr_count': 'sum',
        'total_minutes': 'sum',
    }).reset_index()

    summary['in_iqr_pct'] = 100 * summary['in_iqr_count'] / summary['total_minutes']

    # Pivot to create matrix: variables × dwellings
    table = summary.pivot(index='variable', columns='dwelling', values='in_iqr_pct')

    # Add row averages
    table['Mean'] = table.mean(axis=1)

    # Round to 1 decimal
    table = table.round(1)

    # Sort by variable name (to match Excel order)
    variable_order = [name for name in DISAGGREGATED_COLUMNS.keys()
                      if name in table.index]
    table = table.loc[variable_order]

    print(f"  ✓ Created table: {len(table)} variables × {len(table.columns)} columns")

    return table


def generate_daily_summary_table(validation_results: pd.DataFrame) -> pd.DataFrame:
    """Generate daily totals summary table."""
    print("\n" + "=" * 80)
    print("GENERATING DAILY TOTALS SUMMARY TABLE")
    print("=" * 80)

    # For each (dwelling, variable), count how many runs fall in IQR
    summary = validation_results.groupby(['variable', 'dwelling']).agg({
        'in_iqr': 'sum',
        'excel_value': 'count',
    }).reset_index()

    summary.rename(columns={'in_iqr': 'in_iqr_count', 'excel_value': 'total_runs'}, inplace=True)
    summary['in_iqr_pct'] = 100 * summary['in_iqr_count'] / summary['total_runs']

    # Pivot to create matrix: variables × dwellings
    table = summary.pivot(index='variable', columns='dwelling', values='in_iqr_pct')

    # Add row averages
    table['Mean'] = table.mean(axis=1)

    # Round to 1 decimal
    table = table.round(1)

    # Sort by variable order (to match Excel C-Q)
    variable_order = [name for name in DAILY_COLUMNS.keys()
                      if name in table.index]
    table = table.loc[variable_order]

    print(f"  ✓ Created table: {len(table)} variables × {len(table.columns)} columns")

    return table


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_comprehensive_report(
    daily_results: pd.DataFrame,
    disagg_results: pd.DataFrame,
    daily_table: pd.DataFrame,
    disagg_table: pd.DataFrame,
    stats_info: Dict,
    validation_dir: Path
) -> None:
    """Generate comprehensive validation report."""
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 80)

    # Save detailed results
    daily_results.to_csv(validation_dir / 'daily_totals_detailed.csv', index=False)
    disagg_results.to_csv(validation_dir / 'disaggregated_detailed.csv', index=False)

    # Save summary tables
    daily_table.to_csv(validation_dir / 'daily_totals_summary.csv')
    disagg_table.to_csv(validation_dir / 'disaggregated_summary.csv')

    print(f"  ✓ Saved: daily_totals_detailed.csv")
    print(f"  ✓ Saved: disaggregated_detailed.csv")
    print(f"  ✓ Saved: daily_totals_summary.csv")
    print(f"  ✓ Saved: disaggregated_summary.csv")

    # Generate text report
    report = []
    report.append("=" * 80)
    report.append("CREST MONTE CARLO IQR VALIDATION - COMPREHENSIVE REPORT")
    report.append("=" * 80)
    report.append("")

    # Statistical expectations
    report.append("STATISTICAL EXPECTATIONS")
    report.append("-" * 80)
    report.append(f"Python samples: {stats_info['n_python']}")
    report.append(f"Excel samples: {stats_info['n_excel']}")
    report.append(f"Expected IQR percentage: {stats_info['expected_pct']:.1f}%")
    report.append(f"95% confidence interval: {stats_info['ci_95_pct'][0]:.1f}% - {stats_info['ci_95_pct'][1]:.1f}%")
    report.append("")

    # Daily totals summary
    report.append("DAILY TOTALS (15 variables × 5 houses × 20 runs)")
    report.append("-" * 80)
    if len(daily_results) > 0:
        overall_daily = daily_results.groupby('variable')['in_iqr'].agg(['sum', 'count']).reset_index()
        overall_daily['pct'] = 100 * overall_daily['sum'] / overall_daily['count']

        for _, row in overall_daily.iterrows():
            pct = row['pct']
            status = "✓" if pct >= 40 else "✗"  # Allow some variance from 50%
            report.append(f"  {row['variable']:<50} {pct:5.1f}% {status}")

        overall_pct = 100 * overall_daily['sum'].sum() / overall_daily['count'].sum()
        report.append(f"\n  Overall: {overall_pct:.1f}% in IQR")
    else:
        report.append("  No daily totals data available")
    report.append("")

    # Disaggregated summary
    report.append("DISAGGREGATED (37 variables × 5 houses × 1440 minutes × 20 runs)")
    report.append("-" * 80)
    if len(disagg_results) > 0:
        overall_disagg = disagg_results.groupby('variable').agg({
            'in_iqr_count': 'sum',
            'total_minutes': 'sum',
        }).reset_index()
        overall_disagg['pct'] = 100 * overall_disagg['in_iqr_count'] / overall_disagg['total_minutes']

        # Show top 10 best and worst
        overall_disagg_sorted = overall_disagg.sort_values('pct', ascending=False)

        report.append("Top 10 (highest IQR match):")
        for _, row in overall_disagg_sorted.head(10).iterrows():
            pct = row['pct']
            status = "✓" if pct >= 40 else "✗"
            report.append(f"  {row['variable']:<50} {pct:5.1f}% {status}")

        report.append("\nBottom 10 (lowest IQR match):")
        for _, row in overall_disagg_sorted.tail(10).iterrows():
            pct = row['pct']
            status = "✓" if pct >= 40 else "✗"
            report.append(f"  {row['variable']:<50} {pct:5.1f}% {status}")

        overall_pct = 100 * overall_disagg['in_iqr_count'].sum() / overall_disagg['total_minutes'].sum()
        report.append(f"\n  Overall: {overall_pct:.1f}% in IQR")
    else:
        report.append("  No disaggregated data available")
    report.append("")

    # Per-dwelling breakdown
    report.append("PER-DWELLING BREAKDOWN")
    report.append("-" * 80)
    if len(disagg_results) > 0 and 'dwelling' in disagg_results.columns:
        for dwelling in sorted(disagg_results['dwelling'].unique()):
            d = disagg_results[disagg_results['dwelling'] == dwelling]
            pct = 100 * d['in_iqr_count'].sum() / d['total_minutes'].sum()
            status = "✓" if pct >= 40 else "✗"
            report.append(f"  Dwelling {dwelling}: {pct:.1f}% in IQR {status}")
    else:
        report.append("  No disaggregated data available for per-dwelling analysis")
    report.append("")

    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save and print report
    report_file = validation_dir / 'validation_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"  ✓ Saved: validation_report.txt")

    # Print to console
    print("\n" + '\n'.join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main validation workflow."""
    import os
    project_root = get_project_root()
    os.chdir(project_root)

    if len(sys.argv) < 3:
        print("Usage: python scripts/monte_carlo_compare.py <python_dir> <excel_dir>")
        print("\nExample:")
        print("  python scripts/monte_carlo_compare.py \\")
        print("    output/monte_carlo/python_1000runs_20251113_01 \\")
        print("    output/monte_carlo/excel_20runs_20251113_01")
        sys.exit(1)

    python_dir = Path(sys.argv[1])
    excel_dir = Path(sys.argv[2])

    # Validate directories exist
    if not python_dir.exists():
        print(f"✗ ERROR: Python directory not found: {python_dir}")
        sys.exit(1)

    if not excel_dir.exists():
        print(f"✗ ERROR: Excel directory not found: {excel_dir}")
        sys.exit(1)

    print("=" * 80)
    print("CREST MONTE CARLO IQR VALIDATION - COMPREHENSIVE")
    print("=" * 80)

    # Load data
    python_minute, python_daily = load_python_baseline(python_dir)
    excel_runs = load_excel_runs(excel_dir)

    # Detect sample sizes for statistical analysis
    n_python = len(python_minute['seed'].unique()) if 'seed' in python_minute.columns else len(python_minute) // 1440
    n_excel = len(excel_runs)

    # Compute statistical expectations
    stats_info_disagg = compute_expected_iqr_statistics(n_python, n_excel * 1440)  # Each run has 1440 minutes
    stats_info_daily = compute_expected_iqr_statistics(n_python, n_excel)

    # DISAGGREGATED ANALYSIS
    python_iqr_disagg = compute_python_iqr_disaggregated(python_minute)
    disagg_results = validate_excel_disaggregated(excel_runs, python_iqr_disagg)
    disagg_table = generate_disaggregated_summary_table(disagg_results) if len(disagg_results) > 0 else pd.DataFrame()

    # DAILY ANALYSIS
    daily_results = pd.DataFrame()
    daily_table = pd.DataFrame()
    if python_daily is not None:
        python_iqr_daily = compute_python_iqr_daily(python_daily)
        daily_results = validate_excel_daily(excel_runs, python_iqr_daily)
        daily_table = generate_daily_summary_table(daily_results) if len(daily_results) > 0 else pd.DataFrame()

    # Create validation directory
    validation_dir = create_validation_dir(str(python_dir), str(excel_dir), "monte_carlo")
    print(f"\n✓ Validation directory: {validation_dir}")

    # Save metadata
    save_metadata(
        validation_dir,
        str(python_dir),
        str(excel_dir),
        python_runs=n_python,
        excel_runs=n_excel,
        total_daily_comparisons=len(daily_results),
        total_disaggregated_comparisons=len(disagg_results)
    )

    # Generate report
    generate_comprehensive_report(
        daily_results,
        disagg_results,
        daily_table,
        disagg_table,
        stats_info_disagg,
        validation_dir
    )

    print("\n" + "=" * 80)
    print(f"✓ VALIDATION COMPLETE - Results saved to: {validation_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
