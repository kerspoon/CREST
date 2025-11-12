#!/usr/bin/env python3
"""Run 1000 Monte Carlo simulations for 5 test dwellings."""

import subprocess
import pandas as pd
import os
from pathlib import Path
import numpy as np
import sys

# Default values
NUM_ITERATIONS = 200 # 1000
CONFIG_FILE = 'test_5_identical_dwellings.csv'
TEMP_DIR = 'output/monte_carlo_temp'
DAILY_RESULTS_FILE = 'monte_carlo_daily.csv'
MINUTE_RESULTS_FILE = 'monte_carlo_minute.parquet'

# Allow override from command line
if len(sys.argv) > 1:
    NUM_ITERATIONS = int(sys.argv[1])
if len(sys.argv) > 2:
    CONFIG_FILE = sys.argv[2]

def run_simulation(seed):
    """Run one simulation with given seed."""
    output_dir = f"{TEMP_DIR}/seed_{seed}"
    
    cmd = [
        'venv/bin/python3', 'crest_simulate.py',
        '--num-dwellings', '5',
        '--day', '1',
        '--month', '1',
        '--country', 'UK',
        '--city', 'England',
        '--urban-rural', 'Urban',
        '--config-file', CONFIG_FILE,
        '--save-detailed',  # CRITICAL: Save minute-level data
        '--output-dir', output_dir,
        '--seed', str(seed)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return output_dir
    except Exception as e:
        print(f"Seed {seed} failed: {e}")
        return None

def extract_daily_totals(output_dir, seed):
    """Extract daily totals for each dwelling."""
    results = []
    
    daily_csv = f"{output_dir}/results_daily_summary.csv"
    if not os.path.exists(daily_csv):
        return results
    
    df = pd.read_csv(daily_csv)
    
    for _, row in df.iterrows():
        results.append({
            'seed': seed,
            'dwelling': row['Dwelling'],
            'electricity_kwh': row['Total_Electricity_kWh'],
            'gas_m3': row.get('Total_Gas_m3', 0),
            'water_L': row.get('Total_Hot_Water_L', 0),
            'temp_C': row.get('Mean_Internal_Temp_C', 0)
        })
    
    return results

def extract_minute_data(output_dir, seed):
    """Extract minute-level time-series data."""
    results = []
    
    # All dwellings are in ONE file: results_minute_level.csv
    minute_csv = f"{output_dir}/results_minute_level.csv"
    if not os.path.exists(minute_csv):
        return results
    
    df = pd.read_csv(minute_csv)
    
    # Debug: print columns on first seed
    if seed == 1:
        print(f"  [Debug] Minute CSV columns: {list(df.columns)[:10]}...")
        if len(df) > 0:
            print(f"  [Debug] Shape: {df.shape}")
    
    # Auto-detect dwelling column
    dwelling_col = None
    for col in ['dwelling_index', 'dwelling', 'Dwelling', 'dwelling_id']:
        if col in df.columns:
            dwelling_col = col
            break
    
    if dwelling_col is None:
        if seed == 1:
            print(f"  [ERROR] No dwelling column in minute data. Available: {list(df.columns)[:10]}")
        return results
    
    # Rename to standard 'dwelling' if needed
    if dwelling_col != 'dwelling':
        df = df.rename(columns={dwelling_col: 'dwelling'})
    
    # Add seed metadata
    df['seed'] = seed
    
    results.append(df)
    
    return results

def main():
    import shutil
    
    print(f"Monte Carlo Runner")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print(f"  Config: {CONFIG_FILE}")
    print(f"Usage: python run_monte_carlo.py [num_iterations] [config_file]")
    print()
    
    # Check if config exists
    if not Path(CONFIG_FILE).exists():
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        print("Run: python scripts/create_test_config.py first")
        sys.exit(1)
    
    # Clean up old results
    print("Cleaning up old results...")
    if Path(TEMP_DIR).exists():
        shutil.rmtree(TEMP_DIR)
        print(f"  Removed {TEMP_DIR}")
    
    if Path(DAILY_RESULTS_FILE).exists():
        Path(DAILY_RESULTS_FILE).unlink()
        print(f"  Removed {DAILY_RESULTS_FILE}")
    
    if Path(MINUTE_RESULTS_FILE).exists():
        Path(MINUTE_RESULTS_FILE).unlink()
        print(f"  Removed {MINUTE_RESULTS_FILE}")
    
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    print()
    
    all_daily = []
    all_minute = []
    
    print(f"Running {NUM_ITERATIONS} iterations...")
    print("(Saving minute-level data - this will take longer)")
    
    for seed in range(1, NUM_ITERATIONS + 1):
        if seed % 50 == 0:
            print(f"  Progress: {seed}/{NUM_ITERATIONS}")
        
        output_dir = run_simulation(seed)
        if output_dir:
            # Extract daily totals
            daily = extract_daily_totals(output_dir, seed)
            all_daily.extend(daily)
            
            # Extract minute data
            minute = extract_minute_data(output_dir, seed)
            all_minute.extend(minute)
    
    # Save daily results
    df_daily = pd.DataFrame(all_daily)
    
    if len(df_daily) == 0:
        print("\n⚠ WARNING: No daily results collected!")
        print("Check that daily_totals.csv files are being created in output directories")
        return
    
    df_daily.to_csv(DAILY_RESULTS_FILE, index=False)
    print(f"\nDaily results: {DAILY_RESULTS_FILE} ({len(df_daily)} rows)")
    
    # Save minute results (compressed parquet)
    if all_minute:
        df_minute = pd.concat(all_minute, ignore_index=True)
        df_minute.to_parquet(MINUTE_RESULTS_FILE, compression='snappy', index=False)
        print(f"Minute results: {MINUTE_RESULTS_FILE} ({len(df_minute):,} rows)")
    else:
        print("\n⚠ WARNING: No minute-level data collected!")
        print("Check that dwelling_*_detailed.csv files are being created")
        return
    
    # Quick stats
    print("\n=== DAILY STATISTICS ===")
    for dwelling in sorted(df_daily['dwelling'].unique()):
        d = df_daily[df_daily['dwelling'] == dwelling]
        print(f"\nDwelling {dwelling} (n={len(d)}):")
        print(f"  Electricity: {d['electricity_kwh'].mean():.2f} ± {d['electricity_kwh'].std():.2f} kWh")

if __name__ == '__main__':
    main()
