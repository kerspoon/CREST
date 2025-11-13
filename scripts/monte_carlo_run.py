#!/usr/bin/env python3
"""Run Monte Carlo simulations for CREST validation.

Usage:
    python scripts/monte_carlo_run.py [iterations] [dwellings_file] [output_flags...]

Examples:
    # Run 1000 iterations with default settings
    python scripts/monte_carlo_run.py

    # Run 10 iterations (faster for testing)
    python scripts/monte_carlo_run.py 10

    # Run 500 iterations with custom dwellings
    python scripts/monte_carlo_run.py 500 excel/monte_carlo_base/Dwellings.csv

    # Run with additional flags
    python scripts/monte_carlo_run.py 10 excel/monte_carlo_base/Dwellings.csv --day 15
"""

import subprocess
import pandas as pd
import sys
from pathlib import Path
import shutil

# Import helper utilities
from utils import create_output_dir, get_project_root, get_python_main

# Default values
DEFAULT_ITERATIONS = 1000
DEFAULT_CONFIG = 'excel/monte_carlo_base/Dwellings.csv'


def run_simulation(seed: int, output_dir: Path, config_file: str, extra_args: list) -> bool:
    """
    Run one simulation with given seed.

    Args:
        seed: Random seed for this iteration
        output_dir: Directory to save results
        config_file: Path to dwellings configuration CSV
        extra_args: Additional command-line arguments to pass to main.py

    Returns:
        True if successful, False otherwise
    """
    seed_dir = output_dir / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,  # Use current Python interpreter
        str(get_python_main()),
        '--config-file', str(config_file),
        '--save-detailed',  # CRITICAL: Save minute-level data
        '--output-dir', str(seed_dir),
        '--seed', str(seed)
    ]

    # Add any extra arguments
    cmd.extend(extra_args)

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Seed {seed} timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Seed {seed} failed: {e.stderr.decode()[:200]}")
        return False
    except Exception as e:
        print(f"  [ERROR] Seed {seed} failed: {e}")
        return False


def extract_daily_totals(seed_dir: Path, seed: int) -> list:
    """
    Extract all 17 daily totals columns for each dwelling from a seed directory.

    VBA Reference: DailyTotals (mdlThermalElectricalModel.bas lines 1057-1121)
    Extracts all 17 columns matching VBA daily totals output.

    Args:
        seed_dir: Path to seed output directory
        seed: Seed number

    Returns:
        List of dictionaries with all 17 daily total values per dwelling
    """
    results = []
    daily_csv = seed_dir / "results_daily_summary.csv"

    if not daily_csv.exists():
        return results

    try:
        df = pd.read_csv(daily_csv)

        for _, row in df.iterrows():
            # Extract all 17 columns matching VBA DailyTotals (lines 1103-1119)
            results.append({
                'seed': seed,
                'dwelling': row['Dwelling'],
                'mean_active_occupancy': row.get('Mean_Active_Occupancy', 0),
                'proportion_day_actively_occupied': row.get('Proportion_Day_Actively_Occupied', 0),
                'lighting_kwh': row.get('Lighting_Demand_kWh', 0),
                'appliance_kwh': row.get('Appliance_Demand_kWh', 0),
                'pv_output_kwh': row.get('PV_Output_kWh', 0),
                'total_electricity_kwh': row.get('Total_Electricity_Demand_kWh', 0),
                'self_consumption_kwh': row.get('Self_Consumption_kWh', 0),
                'net_electricity_kwh': row.get('Net_Electricity_Demand_kWh', 0),
                'hot_water_L': row.get('Hot_Water_Demand_L', 0),
                'avg_indoor_temp_C': row.get('Average_Indoor_Temperature_C', 0),
                'thermal_energy_space_kwh': row.get('Thermal_Energy_Space_Heating_kWh', 0),
                'thermal_energy_water_kwh': row.get('Thermal_Energy_Water_Heating_kWh', 0),
                'gas_m3': row.get('Gas_Demand_m3', 0),
                'thermostat_setpoint_C': row.get('Space_Thermostat_Setpoint_C', 20.0),
                'solar_thermal_kwh': row.get('Solar_Thermal_Heat_Gains_kWh', 0)
            })
    except Exception as e:
        print(f"  [WARN] Failed to extract daily totals for seed {seed}: {e}")

    return results


def extract_minute_data(seed_dir: Path, seed: int) -> pd.DataFrame:
    """
    Extract minute-level time-series data from a seed directory.

    Args:
        seed_dir: Path to seed output directory
        seed: Seed number

    Returns:
        DataFrame with minute-level data, or None if not available
    """
    minute_csv = seed_dir / "results_minute_level.csv"

    if not minute_csv.exists():
        return None

    try:
        df = pd.read_csv(minute_csv)

        # Auto-detect dwelling column
        dwelling_col = None
        for col in ['dwelling_index', 'dwelling', 'Dwelling', 'dwelling_id']:
            if col in df.columns:
                dwelling_col = col
                break

        if dwelling_col is None:
            print(f"  [WARN] No dwelling column found in seed {seed}")
            return None

        # Rename to standard 'dwelling' if needed
        if dwelling_col != 'dwelling':
            df = df.rename(columns={dwelling_col: 'dwelling'})

        # Add seed metadata
        df['seed'] = seed

        return df

    except Exception as e:
        print(f"  [WARN] Failed to extract minute data for seed {seed}: {e}")
        return None


def main():
    """Run Monte Carlo simulations."""
    # Parse command-line arguments
    num_iterations = DEFAULT_ITERATIONS
    config_file = DEFAULT_CONFIG
    extra_args = []

    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print(f"ERROR: First argument must be number of iterations, got: {sys.argv[1]}")
            sys.exit(1)

    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    if len(sys.argv) > 3:
        extra_args = sys.argv[3:]

    # Change to project root
    project_root = get_project_root()
    import os
    os.chdir(project_root)

    print("=" * 60)
    print("CREST Monte Carlo Runner")
    print("=" * 60)
    print(f"Iterations:  {num_iterations}")
    print(f"Config file: {config_file}")
    if extra_args:
        print(f"Extra args:  {' '.join(extra_args)}")
    print()

    # Check if config exists
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_file}")
        print("Available configs:")
        excel_files = Path("excel/excel_files")
        if excel_files.exists():
            for f in excel_files.glob("*.csv"):
                print(f"  - {f}")
        sys.exit(1)

    # Count number of dwellings
    try:
        df_config = pd.read_csv(config_path)
        num_dwellings = len(df_config)
    except Exception as e:
        print(f"ERROR: Failed to read config file: {e}")
        sys.exit(1)

    # Create output directory with auto-incrementing number
    output_dir = create_output_dir(
        "monte_carlo",
        prefix=f"python_{num_iterations}runs"
    )

    print(f"Output directory: {output_dir}")
    print(f"Dwellings: {num_dwellings}")
    print()

    # Run simulations
    all_daily = []
    all_minute = []
    successful_runs = 0
    failed_runs = 0

    print(f"Running {num_iterations} iterations...")
    print("(Saving minute-level data - this will take time)")
    print()

    for seed in range(1, num_iterations + 1):
        if seed % 50 == 0 or seed == 1:
            print(f"  Progress: {seed}/{num_iterations} ({successful_runs} OK, {failed_runs} failed)")

        seed_dir = output_dir / f"seed_{seed:03d}"
        success = run_simulation(seed, output_dir, config_file, extra_args)

        if success:
            successful_runs += 1

            # Extract daily totals
            daily = extract_daily_totals(seed_dir, seed)
            all_daily.extend(daily)

            # Extract minute data
            minute = extract_minute_data(seed_dir, seed)
            if minute is not None:
                all_minute.append(minute)
        else:
            failed_runs += 1

    print()
    print(f"Completed: {successful_runs} successful, {failed_runs} failed")
    print()

    # Save daily results
    if all_daily:
        df_daily = pd.DataFrame(all_daily)
        daily_file = output_dir / "daily_summary.csv"
        df_daily.to_csv(daily_file, index=False)
        print(f"Daily results: {daily_file}")
        print(f"  {len(df_daily)} rows ({num_dwellings} dwellings × {successful_runs} seeds)")
    else:
        print("WARNING: No daily results collected!")

    # Save minute results (compressed parquet for efficiency)
    if all_minute:
        df_minute = pd.concat(all_minute, ignore_index=True)
        minute_file = output_dir / "minute_level.parquet"
        df_minute.to_parquet(minute_file, compression='snappy', index=False)
        print(f"Minute results: {minute_file}")
        print(f"  {len(df_minute):,} rows (compressed)")
    else:
        print("WARNING: No minute-level data collected!")

    # Quick statistics
    if all_daily:
        df_daily = pd.DataFrame(all_daily)
        print()
        print("=" * 60)
        print("DAILY STATISTICS")
        print("=" * 60)

        for dwelling in sorted(df_daily['dwelling'].unique()):
            d = df_daily[df_daily['dwelling'] == dwelling]
            print(f"\nDwelling {dwelling} (n={len(d)}):")
            print(f"  Electricity: {d['total_electricity_kwh'].mean():8.2f} ± {d['total_electricity_kwh'].std():.2f} kWh")
            print(f"  Gas:         {d['gas_m3'].mean():8.2f} ± {d['gas_m3'].std():.2f} m³")
            print(f"  Water:       {d['hot_water_L'].mean():8.2f} ± {d['hot_water_L'].std():.2f} L")
            print(f"  Mean Occupancy: {d['mean_active_occupancy'].mean():6.2f} ± {d['mean_active_occupancy'].std():.2f}")
            print(f"  Avg Temp:    {d['avg_indoor_temp_C'].mean():8.2f} ± {d['avg_indoor_temp_C'].std():.2f} °C")

    print()
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
