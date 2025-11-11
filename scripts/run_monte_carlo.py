#!/usr/bin/env python3
"""Run 1000 Monte Carlo simulations for 5 test dwellings."""

import subprocess
import pandas as pd
import os
from pathlib import Path
import json

NUM_ITERATIONS = 200
CONFIG_FILE = 'test_5_dwellings.csv'
TEMP_DIR = 'output/monte_carlo_temp'
RESULTS_FILE = 'monte_carlo_results.csv'

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
    """Extract electricity, gas, water, temp for each dwelling."""
    results = []
    
    # Read the daily totals CSV
    daily_csv = f"{output_dir}/daily_totals.csv"
    if not os.path.exists(daily_csv):
        return results
    
    df = pd.read_csv(daily_csv)
    
    for _, row in df.iterrows():
        results.append({
            'seed': seed,
            'dwelling': row['dwelling_index'],
            'electricity_kwh': row['total_electricity_kwh'],
            'gas_m3': row.get('gas_m3', 0),
            'water_L': row['hot_water_L'],
            'temp_C': row['mean_indoor_temp_C']
        })
    
    return results

def main():
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    print(f"Running {NUM_ITERATIONS} iterations...")
    
    for seed in range(1, NUM_ITERATIONS + 1):
        if seed % 50 == 0:
            print(f"  Progress: {seed}/{NUM_ITERATIONS}")
        
        output_dir = run_simulation(seed)
        if output_dir:
            results = extract_daily_totals(output_dir, seed)
            all_results.extend(results)
    
    # Save aggregated results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    
    print(f"\nCompleted {len(df)} runs")
    print(f"Results saved to: {RESULTS_FILE}")
    
    # Quick stats per dwelling
    print("\n=== STATISTICS PER DWELLING ===")
    for dwelling in sorted(df['dwelling'].unique()):
        d = df[df['dwelling'] == dwelling]
        print(f"\nDwelling {dwelling} (n={len(d)}):")
        print(f"  Electricity: {d['electricity_kwh'].mean():.2f} ± {d['electricity_kwh'].std():.2f} kWh")
        print(f"  Gas:         {d['gas_m3'].mean():.2f} ± {d['gas_m3'].std():.2f} m³")
        print(f"  Water:       {d['water_L'].mean():.1f} ± {d['water_L'].std():.1f} L")
        print(f"  Temp:        {d['temp_C'].mean():.2f} ± {d['temp_C'].std():.2f} °C")

if __name__ == '__main__':
    main()


