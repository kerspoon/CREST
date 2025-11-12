#!/usr/bin/env python3
"""
Validation Analysis: 20 VBA runs vs 1000 Python Monte Carlo runs.

Tests if algorithms match by checking if VBA samples fall within Python's IQR
at the expected 50% rate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns

# Dwelling mapping
DWELLING_MAP = {1: 27, 2: 8, 3: 30, 4: 37, 5: 7}
DWELLING_DESC = {
    1: "D27: 2res, 14app",
    2: "D8: 1res, 25app",
    3: "D30: 5res, 17app",
    4: "D37: 5res, 22app",
    5: "D7: 2res, 21app"
}

METRICS = [
    'Total_Electricity_W',
    'Lighting_W',
    'Appliances_W',
    'Gas_Consumption_m3_per_min',
    'Internal_Temp_C',
    'Hot_Water_Demand_L_per_min'
]

class ValidationAnalyzer:
    def __init__(self, python_parquet: str, vba_dir: str):
        """
        Load data for validation.
        
        Parameters
        ----------
        python_parquet : str
            Path to Python Monte Carlo results (1000 iterations)
        vba_dir : str
            Directory containing VBA results from 20 runs
            Expected files: vba_run_1.csv, vba_run_2.csv, ..., vba_run_20.csv
            Each file should have columns: Dwelling, Minute, <metrics>
        """
        print("Loading Python Monte Carlo data...")
        self.df_python = pd.read_parquet(python_parquet)
        print(f"  {len(self.df_python):,} records from {self.df_python['seed'].nunique()} iterations")
        
        print(f"\nLoading VBA data from {vba_dir}...")
        self.df_vba = self._load_vba_runs(vba_dir)
        print(f"  {len(self.df_vba):,} records from {self.df_vba['run_id'].nunique()} runs")
        
        self.output_dir = Path('validation_output')
        self.output_dir.mkdir(exist_ok=True)
    
    def _load_vba_runs(self, vba_dir: str) -> pd.DataFrame:
        """Load and combine 20 VBA run files."""
        vba_path = Path(vba_dir)
        all_runs = []
        
        # Try different naming patterns
        for i in range(1, 21):
            for pattern in [f'vba_run_{i}.csv', f'run_{i}.csv', f'results_minute_level_run_{i}.csv']:
                file_path = vba_path / pattern
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df['run_id'] = i
                    
                    # Normalize column names
                    if 'Dwelling' in df.columns:
                        df['dwelling'] = df['Dwelling']
                    if 'Minute' in df.columns:
                        df['minute'] = self._parse_minute_column(df['Minute'])
                    
                    all_runs.append(df)
                    print(f"    Loaded run {i}")
                    break
        
        if len(all_runs) == 0:
            raise FileNotFoundError(f"No VBA run files found in {vba_dir}")
        
        return pd.concat(all_runs, ignore_index=True)
    
    def _parse_minute_column(self, minute_series):
        """Parse minute column (handles time strings like '00:01:00')."""
        if minute_series.dtype == 'object':
            def parse_time(t):
                if pd.isna(t):
                    return None
                parts = str(t).split(':')
                if len(parts) >= 2:
                    return int(parts[0]) * 60 + int(parts[1]) + 1
                return int(t)
            return minute_series.apply(parse_time)
        return minute_series.astype(int)
    
    def compute_iqr_match_rate(self) -> pd.DataFrame:
        """
        For each (dwelling, minute, metric), compute:
        - Python IQR from 1000 samples
        - % of 20 VBA samples that fall in IQR
        """
        print("\nComputing IQR match rates...")
        
        results = []
        
        for dwelling in range(1, 6):
            print(f"  Dwelling {dwelling}...")
            
            py_dwelling = self.df_python[self.df_python['dwelling'] == dwelling]
            vba_dwelling = self.df_vba[self.df_vba['dwelling'] == dwelling]
            
            for metric in METRICS:
                if metric not in py_dwelling.columns or metric not in vba_dwelling.columns:
                    continue
                
                # For each minute, compute Python IQR
                total_vba_points = 0
                points_in_iqr = 0
                
                for minute in range(1, 1441):
                    py_minute = py_dwelling[py_dwelling['Minute'] == minute][metric]
                    vba_minute = vba_dwelling[vba_dwelling['minute'] == minute][metric]
                    
                    if len(py_minute) < 100 or len(vba_minute) == 0:
                        continue
                    
                    # Compute Python IQR
                    q1 = np.percentile(py_minute, 25)
                    q3 = np.percentile(py_minute, 75)
                    
                    # Count VBA points in IQR
                    in_iqr = ((vba_minute >= q1) & (vba_minute <= q3)).sum()
                    
                    total_vba_points += len(vba_minute)
                    points_in_iqr += in_iqr
                
                # Calculate overall percentage
                pct_in_iqr = (points_in_iqr / total_vba_points * 100) if total_vba_points > 0 else 0
                
                # Statistical test: is this significantly different from 50%?
                # Binomial test: H0: p = 0.5
                p_value = scipy_stats.binom_test(points_in_iqr, total_vba_points, 0.5, alternative='two-sided')
                
                # 95% confidence interval for proportion
                ci_low, ci_high = self._proportion_ci(points_in_iqr, total_vba_points)
                
                results.append({
                    'dwelling': dwelling,
                    'dwelling_desc': DWELLING_DESC[dwelling],
                    'metric': metric,
                    'total_vba_points': total_vba_points,
                    'points_in_iqr': points_in_iqr,
                    'pct_in_iqr': pct_in_iqr,
                    'ci_low': ci_low * 100,
                    'ci_high': ci_high * 100,
                    'p_value': p_value,
                    'passes': (ci_low < 0.5 < ci_high)  # Does CI contain 50%?
                })
        
        df_results = pd.DataFrame(results)
        
        # Save results
        results_file = self.output_dir / 'iqr_match_rates.csv'
        df_results.to_csv(results_file, index=False)
        print(f"\n  Saved: {results_file}")
        
        return df_results
    
    def _proportion_ci(self, successes, trials, confidence=0.95):
        """Calculate Wilson score confidence interval for proportion."""
        if trials == 0:
            return 0, 0
        
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - confidence) / 2)
        p = successes / trials
        
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        return max(0, center - margin), min(1, center + margin)
    
    def generate_summary_report(self, df_results: pd.DataFrame):
        """Generate human-readable summary report."""
        print("\nGenerating summary report...")
        
        report = []
        report.append("=" * 80)
        report.append("ALGORITHM VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("Test Design:")
        report.append("  - 5 dwelling configurations")
        report.append("  - Python: 1000 Monte Carlo iterations")
        report.append("  - VBA: 20 runs")
        report.append("  - Data points per dwelling per metric: ~28,800")
        report.append("")
        report.append("Expected Result if Algorithms Match:")
        report.append("  - ~50% of VBA points should fall in Python's IQR")
        report.append("  - 95% CI should contain 50%")
        report.append("  - p-value > 0.05 (not significantly different from 50%)")
        report.append("")
        report.append("=" * 80)
        report.append("")
        
        # Overall summary
        total_pass = df_results['passes'].sum()
        total_tests = len(df_results)
        
        report.append(f"OVERALL RESULT: {total_pass}/{total_tests} tests PASS")
        report.append("")
        
        if total_pass == total_tests:
            report.append("✓✓✓ ALGORITHMS MATCH ✓✓✓")
            report.append("All metrics for all dwellings show VBA falling within expected")
            report.append("distribution. Python port is statistically identical to VBA.")
        else:
            report.append("✗✗✗ ALGORITHMS DIFFER ✗✗✗")
            report.append(f"{total_tests - total_pass} metrics show significant deviation.")
            report.append("There are algorithmic differences between Python and VBA.")
        
        report.append("")
        report.append("=" * 80)
        report.append("")
        
        # Per-dwelling breakdown
        for dwelling in range(1, 6):
            d = df_results[df_results['dwelling'] == dwelling]
            if len(d) == 0:
                continue
            
            report.append(f"{DWELLING_DESC[dwelling]}")
            report.append("-" * 80)
            
            for _, row in d.iterrows():
                status = "✓ PASS" if row['passes'] else "✗ FAIL"
                report.append(f"  {row['metric']}:")
                report.append(f"    VBA in IQR: {row['pct_in_iqr']:.1f}%  (95% CI: [{row['ci_low']:.1f}%, {row['ci_high']:.1f}%])")
                report.append(f"    p-value: {row['p_value']:.4f}")
                report.append(f"    Status: {status}")
                
                if not row['passes']:
                    if row['pct_in_iqr'] < 45:
                        report.append(f"    → VBA systematically LOWER than Python")
                    elif row['pct_in_iqr'] > 55:
                        report.append(f"    → VBA systematically HIGHER than Python")
                
                report.append("")
            
            report.append("")
        
        report.append("=" * 80)
        
        # Save report
        report_file = self.output_dir / 'validation_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"  Saved: {report_file}")
        
        # Print to console
        print("\n" + '\n'.join(report))
    
    def plot_iqr_match_rates(self, df_results: pd.DataFrame):
        """Visualize IQR match rates."""
        print("\nGenerating plots...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by dwelling
        for dwelling in range(1, 6):
            d = df_results[df_results['dwelling'] == dwelling]
            if len(d) == 0:
                continue
            
            x = range(len(d))
            y = d['pct_in_iqr'].values
            ci_low = d['ci_low'].values
            ci_high = d['ci_high'].values
            
            ax.errorbar(x, y, 
                       yerr=[y - ci_low, ci_high - y],
                       marker='o', 
                       label=DWELLING_DESC[dwelling],
                       capsize=5)
        
        # Add expected 50% line
        ax.axhline(50, color='black', linestyle='--', linewidth=2, 
                  label='Expected (50%)', alpha=0.7)
        
        # Add acceptable range (45-55%)
        ax.axhspan(45, 55, alpha=0.1, color='green', label='Acceptable range')
        
        ax.set_ylabel('% of VBA Points in Python IQR')
        ax.set_xlabel('Metric Index')
        ax.set_title('Algorithm Validation: VBA vs Python IQR Match Rates')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'iqr_match_rates.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_file}")
    
    def run_validation(self):
        """Run complete validation analysis."""
        print("=" * 80)
        print("STARTING VALIDATION ANALYSIS")
        print("=" * 80)
        
        df_results = self.compute_iqr_match_rate()
        self.plot_iqr_match_rates(df_results)
        self.generate_summary_report(df_results)
        
        print("\n" + "=" * 80)
        print(f"VALIDATION COMPLETE - Results in {self.output_dir}")
        print("=" * 80)


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python validate_algorithm.py <python_monte_carlo.parquet> <vba_runs_dir>")
        print()
        print("Example:")
        print("  python validate_algorithm.py monte_carlo_minute.parquet vba_20_runs/")
        sys.exit(1)
    
    python_file = sys.argv[1]
    vba_dir = sys.argv[2]
    
    analyzer = ValidationAnalyzer(python_file, vba_dir)
    analyzer.run_validation()


if __name__ == '__main__':
    main()
