#!/usr/bin/env python3
"""Run Monte Carlo simulations for CREST validation.

Usage:
    python scripts/monte_carlo_run.py [iterations] [--excel FILE] [output_flags...]

Examples:
    # Run 1000 iterations with default settings
    python scripts/monte_carlo_run.py

    # Run 10 iterations (faster for testing)
    python scripts/monte_carlo_run.py 10

    # Run 100 iterations using settings from an Excel file
    python scripts/monte_carlo_run.py 100 --excel excel/monte_carlo_base.xlsm

    # Run with additional flags (override settings)
    python scripts/monte_carlo_run.py 10 --excel excel/lcg_fixed.xlsm --day 15
"""

import subprocess
import pandas as pd
import sys
import json
from pathlib import Path
import shutil

# Import helper utilities
from utils import create_output_dir, get_project_root, get_python_main

# Default values
DEFAULT_ITERATIONS = 1000
DEFAULT_CONFIG = 'excel/monte_carlo_base_fixed/Dwellings.csv'
DEFAULT_SETTINGS = 'excel/monte_carlo_base_fixed/simulation_settings.json'
DEFAULT_EXCEL = None  # Optional: Excel file to load settings from


def load_excel_settings(excel_path: Path) -> tuple:
    """
    Export Excel file and load simulation settings.

    Args:
        excel_path: Path to .xlsm file

    Returns:
        Tuple of (settings_dict, dwellings_file_path)
    """
    basename = excel_path.stem
    export_dir = Path("excel") / basename

    # Export Excel file (this also extracts settings)
    print(f"Exporting Excel file: {excel_path}")
    cmd = [sys.executable, 'scripts/export_excel.py', str(excel_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to export Excel file")
        print(result.stderr)
        return {}, None

    # Load settings from exported JSON
    settings_file = export_dir / 'simulation_settings.json'
    if not settings_file.exists():
        print(f"WARNING: Settings file not found: {settings_file}")
        return {}, export_dir / 'Dwellings.csv'

    with open(settings_file, 'r') as f:
        settings = json.load(f)

    print(f"Loaded settings from: {settings_file}")

    return settings, export_dir / 'Dwellings.csv'


def settings_to_args(settings: dict) -> list:
    """
    Convert settings dictionary to command-line arguments.

    Args:
        settings: Settings dictionary from simulation_settings.json

    Returns:
        List of command-line arguments

    Supported checkbox settings:
        - save_detailed (objDynamicOutput): --save-detailed flag
        - weekday: --weekend flag if weekday='we'

    Not yet supported as flags (would need main.py changes):
        - assign_dwelling_params: Determined by --config-file presence
        - save_daily_totals: Not implemented
        - pv_enabled: Determined by dwelling config
        - daylight_saving: Not implemented as flag
    """
    args = []

    # Date/time settings
    if 'day' in settings:
        args.extend(['--day', str(settings['day'])])
    if 'month' in settings:
        args.extend(['--month', str(settings['month'])])

    # Weekend flag based on weekday setting
    if settings.get('weekday', 'wd').lower() == 'we':
        args.append('--weekend')

    # Location settings
    if 'latitude' in settings:
        args.extend(['--latitude', str(settings['latitude'])])
    if 'longitude' in settings:
        args.extend(['--longitude', str(settings['longitude'])])
    if 'meridian' in settings:
        args.extend(['--meridian', str(settings['meridian'])])
    if 'country' in settings:
        args.extend(['--country', str(settings['country'])])
    if 'city' in settings:
        args.extend(['--city', str(settings['city'])])
    if 'urban_rural' in settings:
        args.extend(['--urban-rural', str(settings['urban_rural'])])

    # Checkbox settings (only save_detailed is supported as a flag)
    # Note: --save-detailed is typically always wanted for Monte Carlo,
    # so we add it in run_simulation() rather than here

    return args


def run_simulation(seed: int, output_dir: Path, config_file: str, num_dwellings: int,
                   assign_dwelling_params: bool, extra_args: list) -> bool:
    """
    Run one simulation with given seed.

    Args:
        seed: Random seed for this iteration
        output_dir: Directory to save results
        config_file: Path to dwellings configuration CSV (used if assign_dwelling_params=False)
        num_dwellings: Number of dwellings (used if assign_dwelling_params=True)
        assign_dwelling_params: If True, generate dwellings stochastically; if False, use config file
        extra_args: Additional command-line arguments to pass to main.py

    Returns:
        True if successful, False otherwise
    """
    seed_dir = output_dir / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,  # Use current Python interpreter
        str(get_python_main()),
        '--save-detailed',  # CRITICAL: Save minute-level data
        '--output-dir', str(seed_dir),
        '--seed', str(seed)
    ]

    # Dwelling configuration based on assign_dwelling_params setting
    if assign_dwelling_params:
        cmd.extend(['--num-dwellings', str(num_dwellings)])
    else:
        cmd.extend(['--config-file', str(config_file)])

    # Add any extra arguments
    cmd.extend(extra_args)

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Seed {seed} timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Seed {seed} failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"  [ERROR] Seed {seed} failed: {e}")
        return False


def extract_daily_totals(seed_dir: Path, seed: int) -> list:
    """
    Extract daily totals for each dwelling from a seed directory.

    Extracts ALL 15 data columns (C-Q) from the daily summary to match Excel format:
    - Mean active occupancy
    - Proportion of day actively occupied
    - Lighting demand (kWh)
    - Appliance demand (kWh)
    - PV output (kWh)
    - Total dwelling electricity demand (kWh)
    - Total self-consumption (kWh)
    - Net dwelling electricity demand (kWh)
    - Hot water demand (litres)
    - Average indoor air temperature (°C)
    - Thermal energy used for space heating (kWh)
    - Thermal energy used for hot water heating (kWh)
    - Gas demand (m³/day)
    - Space thermostat set point (°C)
    - Solar thermal collector heat gains (kWh)

    Args:
        seed_dir: Path to seed output directory
        seed: Seed number

    Returns:
        List of dictionaries with daily totals (all 15 columns)
    """
    results = []
    daily_csv = seed_dir / "results_daily_summary.csv"

    if not daily_csv.exists():
        return results

    # All 15 data columns from Excel "Results - daily totals" sheet (columns C-Q)
    DAILY_COLUMNS = [
        'Mean active occupancy',
        'Proportion of day actively occupied',
        'Lighting demand',
        'Appliance demand',
        'PV output',
        'Total dwelling electricity demand',
        'Total self-consumption',
        'Net dwelling electricity demand',
        'Hot water demand (litres)',
        'Average indoor air temperature',
        'Thermal energy used for space heating',
        'Thermal energy used for hot water heating',
        'Gas demand',
        'Space thermostat set point',
        'Solar thermal collector heat gains',
    ]

    try:
        # Header format: Row 1=description, Row 2=column names, Row 3=symbols, Row 4=units, Row 5+=data
        # Skip rows 0 (description), 2 (symbols), 3 (units); keep row 1 (column names) as header
        df = pd.read_csv(daily_csv, skiprows=[0, 2, 3])

        for _, row in df.iterrows():
            record = {
                'seed': seed,
                'dwelling': int(row.get('Dwelling index', row.get('Dwelling', 0))),
            }

            # Extract all 15 data columns
            for col in DAILY_COLUMNS:
                if col in row.index:
                    record[col] = row[col]
                else:
                    record[col] = None  # Mark missing columns explicitly

            results.append(record)

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
        # Header format: Row 1=description, Row 2=column names, Row 3=symbols, Row 4=units, Row 5+=data
        # Skip rows 0 (description), 2 (symbols), 3 (units); keep row 1 (column names) as header
        df = pd.read_csv(minute_csv, skiprows=[0, 2, 3])

        # Auto-detect dwelling column
        dwelling_col = None
        for col in ['Dwelling index', 'dwelling_index', 'dwelling', 'Dwelling', 'dwelling_id']:
            if col in df.columns:
                dwelling_col = col
                break

        if dwelling_col is None:
            print(f"  [WARN] No dwelling column found in seed {seed}")
            return None

        # Rename to standard 'dwelling' if needed
        if dwelling_col != 'dwelling':
            df = df.rename(columns={dwelling_col: 'dwelling'})

        # Add numeric Minute column (1-1440) from Time column if needed
        time_col = None
        for col in ['Time', 'time', 'Minute', 'minute']:
            if col in df.columns:
                time_col = col
                break

        if time_col and df[time_col].dtype == 'object':
            def parse_time_to_minute(t):
                """Convert time string to minute number (1-1440)."""
                if pd.isna(t):
                    return None
                t_str = str(t)
                if ':' in t_str:
                    # Handle "HH:MM:SS" or "HH:MM:SS AM/PM" format
                    parts = t_str.replace(' AM', '').replace(' PM', '').split(':')
                    hour = int(parts[0])
                    mins = int(parts[1])
                    # Handle 12-hour format
                    if 'PM' in t_str and hour != 12:
                        hour += 12
                    elif 'AM' in t_str and hour == 12:
                        hour = 0
                    return hour * 60 + mins + 1  # 1-based minute
                try:
                    return int(float(t))
                except (ValueError, TypeError):
                    return None
            df['Minute'] = df[time_col].apply(parse_time_to_minute)

        # Add seed metadata
        df['seed'] = seed

        return df

    except Exception as e:
        print(f"  [WARN] Failed to extract minute data for seed {seed}: {e}")
        return None


def main():
    """Run Monte Carlo simulations."""
    # Change to project root first
    project_root = get_project_root()
    import os
    os.chdir(project_root)

    # Parse command-line arguments
    num_iterations = DEFAULT_ITERATIONS
    config_file = DEFAULT_CONFIG
    excel_file = DEFAULT_EXCEL
    extra_args = []
    settings = {}

    # Parse args manually to handle --excel flag
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--excel' and i + 1 < len(args):
            excel_file = args[i + 1]
            i += 2
        elif arg.startswith('--'):
            # Pass through other flags as extra_args
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                extra_args.extend([arg, args[i + 1]])
                i += 2
            else:
                extra_args.append(arg)
                i += 1
        elif i == 0:
            # First positional arg is iterations
            try:
                num_iterations = int(arg)
            except ValueError:
                print(f"ERROR: First argument must be number of iterations, got: {arg}")
                sys.exit(1)
            i += 1
        else:
            # Additional positional args go to extra_args
            extra_args.append(arg)
            i += 1

    print("=" * 60)
    print("CREST Monte Carlo Runner")
    print("=" * 60)

    # Load settings from Excel if specified
    if excel_file:
        excel_path = Path(excel_file)
        if not excel_path.exists():
            print(f"ERROR: Excel file not found: {excel_file}")
            sys.exit(1)

        settings, dwellings_path = load_excel_settings(excel_path)
        if dwellings_path and dwellings_path.exists():
            config_file = str(dwellings_path)

        # Convert settings to command-line args (prepend so extra_args can override)
        settings_args = settings_to_args(settings)
        extra_args = settings_args + extra_args

        print(f"Excel file:  {excel_file}")
        print(f"  Day: {settings.get('day')}, Month: {settings.get('month')}")
        print(f"  Lat: {settings.get('latitude')}, Lon: {settings.get('longitude')}")

    else:
        # No Excel file specified - load settings from JSON (like rng_validation_run.py)
        settings_path = Path(DEFAULT_SETTINGS)
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)

            # Convert settings to command-line args (prepend so extra_args can override)
            settings_args = settings_to_args(settings)
            extra_args = settings_args + extra_args

            print(f"Settings:    {settings_path}")
            print(f"  Day: {settings.get('day')}, Month: {settings.get('month')}")
            print(f"  Lat: {settings.get('latitude')}, Lon: {settings.get('longitude')}")
        else:
            print(f"WARNING: Settings file not found: {settings_path}")
            print("         Using Python defaults (day=15, month=6)")
            print("         Run: python scripts/export_excel.py excel/monte_carlo_base.xlsm")
            print()

    print(f"Iterations:  {num_iterations}")
    if extra_args:
        print(f"Extra args:  {' '.join(extra_args)}")
    print()

    # Get assign_dwelling_params from settings (defaults to True = stochastic)
    assign_dwelling_params = settings.get('assign_dwelling_params', True)
    config_path = Path(config_file)

    # Determine num_dwellings based on assign_dwelling_params
    if assign_dwelling_params:
        # Stochastic mode: get num_dwellings from settings
        num_dwellings = settings.get('num_dwellings', 1)
        print(f"Mode: Stochastic dwelling assignment (num_dwellings={num_dwellings})")
    else:
        # Fixed config mode: count dwellings from config file
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_file}")
            print("  assign_dwelling_params=False requires a valid config file")
            print("\nTip: Use --excel to specify an Excel file to export and use:")
            print(f"  python scripts/monte_carlo_run.py {num_iterations} --excel excel/your_file.xlsm")
            sys.exit(1)
        try:
            df_config = pd.read_csv(config_path)
            # Count valid dwelling rows (skip header rows - look for numeric first column)
            num_dwellings = 0
            for _, row in df_config.iterrows():
                if pd.notna(row.iloc[0]) and str(row.iloc[0]).isdigit():
                    num_dwellings += 1
        except Exception as e:
            print(f"ERROR: Failed to read config file: {e}")
            sys.exit(1)
        print(f"Mode: Fixed dwelling config from {config_file} ({num_dwellings} dwellings)")

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
        success = run_simulation(seed, output_dir, config_file, num_dwellings,
                                 assign_dwelling_params, extra_args)

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
            # Use the actual Excel column names
            elec_col = 'Total dwelling electricity demand'
            gas_col = 'Gas demand'
            water_col = 'Hot water demand (litres)'
            if elec_col in d.columns:
                print(f"  Electricity: {d[elec_col].mean():8.2f} ± {d[elec_col].std():.2f} kWh")
            if gas_col in d.columns:
                print(f"  Gas:         {d[gas_col].mean():8.2f} ± {d[gas_col].std():.2f} m³")
            if water_col in d.columns:
                print(f"  Water:       {d[water_col].mean():8.2f} ± {d[water_col].std():.2f} L")

    print()
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
