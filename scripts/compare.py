#!/usr/bin/env python3
"""
CREST Comprehensive Comparison Tool

Combines and improves upon quick_compare.py, compare_simulations.py, and compare_results.py
with enhanced debugging features for tracking down discrepancies.

Usage:
    python compare.py <vba_dir> <python_dir> [--output report.md] [--detailed]
"""

import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, statistical tests will be skipped")


class CRESTComparison:
    """Comprehensive CREST simulation comparison tool."""

    def __init__(self, vba_dir: Path, python_dir: Path, detailed: bool = False):
        self.vba_dir = Path(vba_dir)
        self.python_dir = Path(python_dir)
        self.detailed = detailed
        self.results = {}

    def load_data(self) -> bool:
        """Load all available data files."""
        print("="*80)
        print("LOADING DATA")
        print("="*80)

        try:
            # Load daily summaries (required)
            print("\n1. Loading daily summaries...")
            self.vba_daily = self._load_csv(self.vba_dir / "results_daily_summary.csv")
            self.python_daily = self._load_csv(self.python_dir / "results_daily_summary.csv")

            if self.vba_daily is None or self.python_daily is None:
                print("  ❌ Failed to load daily summaries")
                return False
            print(f"  ✅ VBA: {len(self.vba_daily)} dwellings")
            print(f"  ✅ Python: {len(self.python_daily)} dwellings")

            # Load dwelling configs (optional)
            print("\n2. Loading dwelling configurations...")
            self.vba_config = self._load_csv(self.vba_dir / "dwellings_config.csv")
            self.python_config = self._load_csv(self.python_dir / "dwellings_config.csv")

            if self.vba_config is not None and self.python_config is not None:
                print(f"  ✅ VBA: {len(self.vba_config)} configs")
                print(f"  ✅ Python: {len(self.python_config)} configs")
            else:
                print("  ⚠️  Dwelling configs not available")

            # Load climate data (optional)
            print("\n3. Loading climate data...")
            self.vba_climate = self._load_csv(self.vba_dir / "global_climate.csv")
            self.python_climate = self._load_csv(self.python_dir / "global_climate.csv")

            if self.vba_climate is not None and self.python_climate is not None:
                print(f"  ✅ VBA: {len(self.vba_climate)} timesteps")
                print(f"  ✅ Python: {len(self.python_climate)} timesteps")
            else:
                print("  ⚠️  Climate data not available")

            # Load minute-level data only if detailed mode
            if self.detailed:
                print("\n4. Loading minute-level data (detailed mode)...")
                self.vba_minute = self._load_csv(self.vba_dir / "results_minute_level.csv", nrows=10000)
                self.python_minute = self._load_csv(self.python_dir / "results_minute_level.csv", nrows=10000)

                if self.vba_minute is not None and self.python_minute is not None:
                    print(f"  ✅ Loaded first {len(self.vba_minute)} rows")
                else:
                    print("  ⚠️  Minute-level data not available")
            else:
                self.vba_minute = None
                self.python_minute = None

            return True

        except Exception as e:
            print(f"  ❌ Error loading data: {e}")
            return False

    def _load_csv(self, filepath: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load CSV file with error handling."""
        try:
            if not filepath.exists():
                return None

            df = pd.read_csv(filepath, **kwargs)
            df = df.dropna(how='all')

            # Remove unnamed columns
            unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
            for col in unnamed_cols:
                if df[col].isna().all():
                    df = df.drop(columns=[col])

            return df
        except Exception as e:
            return None

    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find first matching column name from list of possibilities."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def compare_dwelling_by_dwelling(self, n_dwellings: int = 20):
        """Detailed dwelling-by-dwelling comparison table."""
        print("\n" + "="*80)
        print(f"DWELLING-BY-DWELLING COMPARISON (First {n_dwellings})")
        print("="*80)

        # Map column names
        elec_col_vba = self._find_column(self.vba_daily, ['Total_Electricity_kWh', 'Total Electricity'])
        elec_col_py = self._find_column(self.python_daily, ['Total_Electricity_kWh', 'Total Electricity'])

        gas_col_vba = self._find_column(self.vba_daily, ['Total_Gas_m3', 'Total Gas', 'Total_Gas'])
        gas_col_py = self._find_column(self.python_daily, ['Total_Gas_m3', 'Total Gas', 'Total_Gas'])

        temp_col_vba = self._find_column(self.vba_daily, ['Mean_Internal_Temp_C', 'Mean Internal Temp'])
        temp_col_py = self._find_column(self.python_daily, ['Mean_Internal_Temp_C', 'Mean Internal Temp'])

        if not all([elec_col_vba, elec_col_py]):
            print("❌ Cannot find electricity columns")
            return

        print(f"\n{'Dwell':<6} {'VBA Elec':<10} {'Py Elec':<10} {'Diff':<10} {'Diff%':<8} "
              f"{'VBA Gas':<9} {'Py Gas':<9} {'VBA Temp':<9} {'Py Temp':<9}")
        print("-"*95)

        comparisons = []
        n = min(n_dwellings, len(self.vba_daily), len(self.python_daily))

        for i in range(n):
            vba_elec = self.vba_daily.iloc[i][elec_col_vba]
            py_elec = self.python_daily.iloc[i][elec_col_py]
            diff = vba_elec - py_elec
            diff_pct = (diff / vba_elec * 100) if vba_elec != 0 else 0

            vba_gas = self.vba_daily.iloc[i][gas_col_vba] if gas_col_vba else 0
            py_gas = self.python_daily.iloc[i][gas_col_py] if gas_col_py else 0

            vba_temp = self.vba_daily.iloc[i][temp_col_vba] if temp_col_vba else 0
            py_temp = self.python_daily.iloc[i][temp_col_py] if temp_col_py else 0

            status = "✓" if abs(diff_pct) < 1 else "⚠" if abs(diff_pct) < 10 else "✗"

            print(f"{i+1:<6} {vba_elec:<10.2f} {py_elec:<10.2f} {diff:<10.2f} "
                  f"{status} {diff_pct:>5.1f}% {vba_gas:<9.2f} {py_gas:<9.2f} "
                  f"{vba_temp:<9.2f} {py_temp:<9.2f}")

            comparisons.append({
                'dwelling': i+1,
                'vba_elec': vba_elec,
                'py_elec': py_elec,
                'diff': diff,
                'diff_pct': abs(diff_pct)
            })

        return comparisons

    def analyze_outliers(self, comparisons: List[Dict]):
        """Identify and analyze outliers."""
        print("\n" + "="*80)
        print("OUTLIER ANALYSIS")
        print("="*80)

        if not comparisons:
            return

        diffs = [c['diff_pct'] for c in comparisons]

        # Sort by difference
        sorted_comps = sorted(comparisons, key=lambda x: x['diff_pct'], reverse=True)

        print("\n📊 Top 10 WORST Matches (largest discrepancies):")
        print(f"{'Rank':<6} {'Dwelling':<10} {'VBA':<12} {'Python':<12} {'Diff%':<10}")
        print("-"*60)
        for rank, comp in enumerate(sorted_comps[:10], 1):
            print(f"{rank:<6} {comp['dwelling']:<10} {comp['vba_elec']:<12.2f} "
                  f"{comp['py_elec']:<12.2f} {comp['diff_pct']:<10.1f}%")

        print("\n📊 Top 10 BEST Matches (smallest discrepancies):")
        print(f"{'Rank':<6} {'Dwelling':<10} {'VBA':<12} {'Python':<12} {'Diff%':<10}")
        print("-"*60)
        for rank, comp in enumerate(sorted_comps[-10:][::-1], 1):
            print(f"{rank:<6} {comp['dwelling']:<10} {comp['vba_elec']:<12.2f} "
                  f"{comp['py_elec']:<12.2f} {comp['diff_pct']:<10.1f}%")

        # Percentile analysis
        print(f"\n📊 Difference Distribution:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"{'Percentile':<12} {'Difference %':<12}")
        print("-"*24)
        for p in percentiles:
            val = np.percentile(diffs, p)
            print(f"{p}th{'':<9} {val:<12.2f}%")

    def compare_aggregate_stats(self):
        """Compare aggregate statistics with enhanced analysis."""
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)

        metrics = {
            'electricity': (['Total_Electricity_kWh', 'Total Electricity'], 'Electricity (kWh)'),
            'gas': (['Total_Gas_m3', 'Total Gas', 'Total_Gas'], 'Gas (m³)'),
            'water': (['Total_Hot_Water_L', 'Total Hot Water'], 'Hot Water (L)'),
            'temp': (['Mean_Internal_Temp_C', 'Mean Internal Temp'], 'Temperature (°C)')
        }

        results = {}

        for metric_key, (possible_names, display_name) in metrics.items():
            vba_col = self._find_column(self.vba_daily, possible_names)
            py_col = self._find_column(self.python_daily, possible_names)

            if not vba_col or not py_col:
                continue

            vba_data = self.vba_daily[vba_col].dropna()
            py_data = self.python_daily[py_col].dropna()

            print(f"\n{display_name}:")
            print(f"  {'Statistic':<15} {'VBA':<12} {'Python':<12} {'Diff':<12} {'Diff%':<10}")
            print("  " + "-"*65)

            stats_dict = {
                'Mean': (vba_data.mean(), py_data.mean()),
                'Median': (vba_data.median(), py_data.median()),
                'Std Dev': (vba_data.std(), py_data.std()),
                'Min': (vba_data.min(), py_data.min()),
                'Max': (vba_data.max(), py_data.max()),
                'Q1 (25%)': (vba_data.quantile(0.25), py_data.quantile(0.25)),
                'Q3 (75%)': (vba_data.quantile(0.75), py_data.quantile(0.75)),
            }

            for stat_name, (vba_val, py_val) in stats_dict.items():
                diff = vba_val - py_val
                diff_pct = (diff / vba_val * 100) if vba_val != 0 else 0
                print(f"  {stat_name:<15} {vba_val:<12.2f} {py_val:<12.2f} "
                      f"{diff:<12.2f} {diff_pct:<10.1f}%")

            # Statistical tests if scipy available
            if HAS_SCIPY:
                ks_stat, ks_pval = stats.ks_2samp(vba_data, py_data)
                t_stat, t_pval = stats.ttest_ind(vba_data, py_data)

                print(f"\n  Statistical Tests:")
                print(f"    KS Test:  statistic={ks_stat:.4f}, p-value={ks_pval:.4f} "
                      f"({'SAME' if ks_pval > 0.05 else 'DIFFERENT'} distributions)")
                print(f"    t-test:   statistic={t_stat:.4f}, p-value={t_pval:.4f} "
                      f"({'SAME' if t_pval > 0.05 else 'DIFFERENT'} means)")

            results[metric_key] = {
                'vba': vba_data,
                'python': py_data,
                'stats': stats_dict
            }

        return results

    def compare_component_breakdown(self, n_dwellings: int = 10):
        """Compare component-level breakdown (appliances, lighting, etc)."""
        print("\n" + "="*80)
        print(f"COMPONENT BREAKDOWN ANALYSIS (First {n_dwellings})")
        print("="*80)

        # Check what component columns are available
        component_mappings = {
            'appliances': (['Appliance_kWh', 'Appliance demand'], 'Appliances'),
            'lighting': (['Lighting_kWh', 'Lighting demand'], 'Lighting'),
            'heating_gas': (['Gas_m3', 'Total_Gas_m3'], 'Heating Gas'),
        }

        available_components = []
        for key, (possible_names, display_name) in component_mappings.items():
            vba_col = self._find_column(self.vba_daily, possible_names)
            py_col = self._find_column(self.python_daily, possible_names)
            if vba_col and py_col:
                available_components.append((key, vba_col, py_col, display_name))

        if not available_components:
            print("⚠️  No component-level data available in both datasets")
            return

        n = min(n_dwellings, len(self.vba_daily), len(self.python_daily))

        for key, vba_col, py_col, display_name in available_components:
            print(f"\n{display_name}:")
            print(f"  {'Dwelling':<10} {'VBA':<12} {'Python':<12} {'Diff':<12} {'Diff%':<10}")
            print("  " + "-"*60)

            for i in range(n):
                vba_val = self.vba_daily.iloc[i][vba_col]
                py_val = self.python_daily.iloc[i][py_col]
                diff = vba_val - py_val
                diff_pct = (diff / vba_val * 100) if vba_val != 0 else 0

                status = "✓" if abs(diff_pct) < 5 else "⚠" if abs(diff_pct) < 20 else "✗"
                print(f"  {i+1:<10} {vba_val:<12.2f} {py_val:<12.2f} "
                      f"{diff:<12.2f} {status} {diff_pct:>5.1f}%")

    def generate_text_histogram(self, data, title, bins=20, width=50):
        """Generate ASCII histogram for visualizing distributions."""
        print(f"\n{title}:")
        hist, bin_edges = np.histogram(data, bins=bins)
        max_count = max(hist)

        for i in range(bins):
            bar_length = int((hist[i] / max_count) * width) if max_count > 0 else 0
            bar = '█' * bar_length
            print(f"  {bin_edges[i]:6.1f} - {bin_edges[i+1]:6.1f} | {bar} {hist[i]}")

    def compare_distributions(self):
        """Visual comparison of distributions."""
        print("\n" + "="*80)
        print("DISTRIBUTION VISUALIZATION")
        print("="*80)

        elec_col_vba = self._find_column(self.vba_daily, ['Total_Electricity_kWh'])
        elec_col_py = self._find_column(self.python_daily, ['Total_Electricity_kWh'])

        if elec_col_vba and elec_col_py:
            vba_data = self.vba_daily[elec_col_vba].dropna()
            py_data = self.python_daily[elec_col_py].dropna()

            self.generate_text_histogram(vba_data, "VBA Electricity Distribution (kWh)")
            self.generate_text_histogram(py_data, "Python Electricity Distribution (kWh)")

    def generate_summary(self):
        """Generate executive summary."""
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)

        elec_col_vba = self._find_column(self.vba_daily, ['Total_Electricity_kWh'])
        elec_col_py = self._find_column(self.python_daily, ['Total_Electricity_kWh'])

        if elec_col_vba and elec_col_py:
            vba_mean = self.vba_daily[elec_col_vba].mean()
            py_mean = self.python_daily[elec_col_py].mean()
            diff = vba_mean - py_mean
            diff_pct = (diff / vba_mean * 100) if vba_mean != 0 else 0

            print(f"\nOverall Electricity Comparison:")
            print(f"  VBA Mean:         {vba_mean:.2f} kWh")
            print(f"  Python Mean:      {py_mean:.2f} kWh")
            print(f"  Absolute Diff:    {diff:.2f} kWh")
            print(f"  Relative Diff:    {diff_pct:.1f}%")

            if abs(diff_pct) < 1:
                print(f"\n  ✅ EXCELLENT match (<1% difference)")
            elif abs(diff_pct) < 5:
                print(f"\n  ✓ GOOD match (<5% difference)")
            elif abs(diff_pct) < 10:
                print(f"\n  ⚠ FAIR match (<10% difference)")
            else:
                print(f"\n  ✗ POOR match (>{diff_pct:.1f}% difference)")
                print(f"     Further investigation needed!")

    def run(self):
        """Run complete comparison analysis."""
        if not self.load_data():
            print("\n❌ Failed to load data")
            return False

        # Run all comparisons
        comparisons = self.compare_dwelling_by_dwelling(n_dwellings=20)

        if comparisons:
            self.analyze_outliers(comparisons)

        self.compare_aggregate_stats()
        self.compare_component_breakdown(n_dwellings=10)

        if self.detailed:
            self.compare_distributions()

        self.generate_summary()

        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive CREST simulation comparison'
    )
    parser.add_argument('vba_dir', help='VBA/Excel output directory')
    parser.add_argument('python_dir', help='Python output directory')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Run detailed analysis (slower)')

    args = parser.parse_args()

    comparison = CRESTComparison(args.vba_dir, args.python_dir, args.detailed)
    success = comparison.run()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
