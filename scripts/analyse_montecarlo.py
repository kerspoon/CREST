#!/usr/bin/env python3
"""Analyze Monte Carlo results and compare to VBA."""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def collect_results_from_folders(temp_dir='output/monte_carlo_temp'):
    """Scan temp directory and collect all available results."""
    all_results = []
    temp_path = Path(temp_dir)
    
    if not temp_path.exists():
        print(f"Error: {temp_dir} not found")
        return pd.DataFrame()
    
    # Find all seed_* directories
    seed_dirs = sorted(temp_path.glob('seed_*'))
    
    print(f"Found {len(seed_dirs)} result directories")
    
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split('_')[1])
        daily_csv = seed_dir / 'results_daily_summary.csv'
        
        if daily_csv.exists():
            try:
                df = pd.read_csv(daily_csv)
                
                for _, row in df.iterrows():
                    all_results.append({
                        'seed': seed,
                        'dwelling': row['Dwelling'],
                        'electricity_kwh': row['Total_Electricity_kWh'],
                        'gas_m3': row['Total_Gas_m3'],
                        'water_L': row['Total_Hot_Water_L'],
                        'temp_C': row['Mean_Internal_Temp_C']
                    })
            except Exception as e:
                # Skip corrupted/empty files
                pass
    
    return pd.DataFrame(all_results)

def analyze_results(results_csv=None, vba_csv=None, temp_dir=None):
    """Analyze Monte Carlo distribution and compare to VBA if available."""
    
    # Load data from CSV or scan folders
    if temp_dir:
        df = collect_results_from_folders(temp_dir)
        if df.empty:
            print("No results found")
            return []
    elif results_csv:
        df = pd.read_csv(results_csv)
    else:
        df = collect_results_from_folders()
        if df.empty:
            print("No results found")
            return []
    
    print(f"\n=== PYTHON MONTE CARLO ANALYSIS ===")
    print(f"Total runs: {len(df)}\n")
    
    dwelling_stats = []
    
    for dwelling in sorted(df['dwelling'].unique()):
        d = df[df['dwelling'] == dwelling]
        n = len(d)
        
        elec_mean = d['electricity_kwh'].mean()
        elec_std = d['electricity_kwh'].std()
        elec_ci_lower = elec_mean - 1.96 * elec_std
        elec_ci_upper = elec_mean + 1.96 * elec_std
        
        print(f"Dwelling {dwelling} (original dwelling {[27,8,30,37,7][int(dwelling)-1]}):")
        print(f"  n = {n} runs")
        print(f"  Electricity: {elec_mean:.2f} ± {elec_std:.2f} kWh")
        print(f"  95% CI:      [{elec_ci_lower:.2f}, {elec_ci_upper:.2f}] kWh")
        print(f"  Range:       [{d['electricity_kwh'].min():.2f}, {d['electricity_kwh'].max():.2f}]")
        
        if n < 30:
            print(f"  WARNING: Small sample (n={n}), CI may be unreliable")
        
        print()
        
        dwelling_stats.append({
            'dwelling': dwelling,
            'original_id': [27,8,30,37,7][int(dwelling)-1],
            'n': n,
            'mean': elec_mean,
            'std': elec_std,
            'ci_lower': elec_ci_lower,
            'ci_upper': elec_ci_upper
        })
    
    if vba_csv:
        print("\n=== COMPARISON TO VBA ===\n")
        vba = pd.read_csv(vba_csv)
        
        for stat in dwelling_stats:
            orig_id = stat['original_id']
            if orig_id in vba['dwelling_index'].values:
                vba_elec = vba[vba['dwelling_index'] == orig_id]['total_electricity_kwh'].values[0]
                
                # Check if VBA falls within Python 95% CI
                in_ci = stat['ci_lower'] <= vba_elec <= stat['ci_upper']
                ratio = vba_elec / stat['mean']
                
                print(f"Dwelling {orig_id}:")
                print(f"  n:      {stat['n']} runs")
                print(f"  VBA:    {vba_elec:.2f} kWh")
                print(f"  Python: {stat['mean']:.2f} ± {stat['std']:.2f} kWh")
                print(f"  Ratio:  {ratio:.3f} (VBA/Python)")
                print(f"  In CI:  {'✓ YES' if in_ci else '✗ NO'}")
                print()
    
    return dwelling_stats

if __name__ == '__main__':
    import sys
    
    # Usage:
    # python analyze_monte_carlo.py                                    # Scan output/monte_carlo_temp/
    # python analyze_monte_carlo.py --dir output/mc_temp               # Scan custom dir
    # python analyze_monte_carlo.py monte_carlo_results.csv            # Use CSV
    # python analyze_monte_carlo.py results.csv vba_daily.csv          # CSV + VBA comparison
    
    if len(sys.argv) == 1:
        # No args - scan default temp dir
        analyze_results(temp_dir='output/monte_carlo_temp')
    elif sys.argv[1] == '--dir':
        # Scan specified dir
        temp_dir = sys.argv[2] if len(sys.argv) > 2 else 'output/monte_carlo_temp'
        vba_csv = sys.argv[3] if len(sys.argv) > 3 else None
        analyze_results(temp_dir=temp_dir, vba_csv=vba_csv)
    else:
        # Use CSV files
        results_csv = sys.argv[1]
        vba_csv = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_results(results_csv=results_csv, vba_csv=vba_csv)
