#!/usr/bin/env python3
"""Comprehensive Monte Carlo analysis with quartile-based statistics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Dwelling mapping
DWELLING_MAP = {1: 27, 2: 8, 3: 30, 4: 37, 5: 7}
DWELLING_DESC = {
    1: "D27: 2res, 14app (minimal)",
    2: "D8: 1res, 25app (maximal)",
    3: "D30: 5res, 17app (high occ)",
    4: "D37: 5res, 22app (high occ+app)",
    5: "D7: 2res, 21app (typical)"
}

# Metrics to analyze - using actual column names from CSV
METRICS = {
    'Total_Electricity_W': 'Total Electricity (W)',
    'Lighting_W': 'Lighting (W)',
    'Appliances_W': 'Appliances (W)',
    'Gas_Consumption_m3_per_min': 'Gas (m³/min)',
    'Internal_Temp_C': 'Indoor Temp (°C)',
    'Hot_Water_Demand_L_per_min': 'Hot Water (L/min)'
}

class MonteCarloAnalyzer:
    def __init__(self, minute_parquet: str, daily_csv: str, vba_dir: str = None):
        """Load Monte Carlo results and VBA data."""
        print("Loading Monte Carlo data...")
        self.df_minute = pd.read_parquet(minute_parquet)
        self.df_daily = pd.read_csv(daily_csv)
        
        print(f"  Loaded {len(self.df_minute):,} minute records")
        print(f"  {len(self.df_daily)} daily records")
        
        # Load VBA data if provided
        self.vba_minute = {}
        self.vba_daily = {}
        if vba_dir:
            self._load_vba_data(vba_dir)
        
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_vba_data(self, vba_dir: str):
        """Load VBA comparison data."""
        print(f"Loading VBA data from {vba_dir}...")
        
        # Load VBA daily totals - try both naming conventions
        for filename in ['results_daily_summary.csv', 'daily_totals.csv']:
            vba_daily_path = Path(vba_dir) / filename
            if vba_daily_path.exists():
                self.vba_daily = pd.read_csv(vba_daily_path)
                # Normalize dwelling column name
                for col in ['Dwelling', 'dwelling_index', 'dwelling']:
                    if col in self.vba_daily.columns:
                        self.vba_daily = self.vba_daily.rename(columns={col: 'dwelling_index'})
                        break
                print(f"  Loaded VBA daily totals from {filename}")
                break
        
        # Load VBA minute-level data - single combined file
        for filename in ['results_minute_level.csv', 'detailed_results.csv']:
            vba_minute_path = Path(vba_dir) / filename
            if vba_minute_path.exists():
                df_all = pd.read_csv(vba_minute_path)
                
                # Find dwelling column
                dwelling_col = None
                for col in ['Dwelling', 'dwelling_index', 'dwelling']:
                    if col in df_all.columns:
                        dwelling_col = col
                        break
                
                if dwelling_col:
                    # Split by dwelling - map original dwelling IDs to 1-5
                    for dwelling_idx in range(1, 6):
                        orig_id = DWELLING_MAP[dwelling_idx]
                        dwelling_data = df_all[df_all[dwelling_col] == orig_id].copy()
                        if len(dwelling_data) > 0:
                            self.vba_minute[dwelling_idx] = dwelling_data
                    print(f"  Loaded VBA minute data from {filename} ({len(self.vba_minute)} dwellings)")
                break
    
    def compute_minute_statistics(self) -> pd.DataFrame:
        """Compute quartile statistics for each minute, dwelling, metric."""
        print("\nComputing minute-level statistics...")
        
        # Detect time column name
        time_col = None
        for col in ['Minute', 'minute', 'time', 'timestep', 'time_minute']:
            if col in self.df_minute.columns:
                time_col = col
                break
        
        if time_col is None:
            print("ERROR: Could not find time column in minute data")
            print(f"Available columns: {list(self.df_minute.columns)}")
            return pd.DataFrame()
        
        print(f"  Using time column: '{time_col}'")
        
        stats_list = []
        
        for dwelling in sorted(self.df_minute['dwelling'].unique()):
            print(f"  Dwelling {dwelling}...")
            d = self.df_minute[self.df_minute['dwelling'] == dwelling]
            
            for minute in range(1, 1441):
                m = d[d[time_col] == minute]
                
                if len(m) == 0:
                    continue
                
                row = {'dwelling': dwelling, 'minute': minute}
                
                for col, label in METRICS.items():
                    if col in m.columns:
                        values = m[col].values
                        row[f'{col}_median'] = np.median(values)
                        row[f'{col}_q1'] = np.percentile(values, 25)
                        row[f'{col}_q3'] = np.percentile(values, 75)
                        row[f'{col}_mean'] = np.mean(values)
                        row[f'{col}_std'] = np.std(values)
                        row[f'{col}_min'] = np.min(values)
                        row[f'{col}_max'] = np.max(values)
                        row[f'{col}_iqr'] = row[f'{col}_q3'] - row[f'{col}_q1']
                
                stats_list.append(row)
        
        df_stats = pd.DataFrame(stats_list)
        
        # Save to parquet
        stats_file = self.output_dir / 'minute_statistics.parquet'
        df_stats.to_parquet(stats_file, index=False)
        print(f"  Saved: {stats_file}")
        
        return df_stats
    
    def compare_to_vba(self, df_stats: pd.DataFrame) -> pd.DataFrame:
        """Compare VBA single run to Python quartile distributions."""
        print("\nComparing to VBA...")
        
        comparison_results = []
        
        for dwelling in sorted(df_stats['dwelling'].unique()):
            if dwelling not in self.vba_minute:
                print(f"  Dwelling {dwelling}: No VBA data")
                continue
            
            print(f"  Dwelling {dwelling}...")
            
            stats = df_stats[df_stats['dwelling'] == dwelling]
            vba = self.vba_minute[dwelling]
            
            for col in METRICS.keys():
                if col not in vba.columns:
                    continue
                
                # Find time column in VBA data
                vba_time_col = None
                for tc in ['Minute', 'minute', 'time']:
                    if tc in vba.columns:
                        vba_time_col = tc
                        break
                
                if vba_time_col is None:
                    continue
                
                # Convert both to same type for merging
                stats_merge = stats.copy()
                vba_merge = vba[[vba_time_col, col]].copy()
                
                # Ensure minute columns are same type
                stats_merge['minute'] = stats_merge['minute'].astype(int)
                
                # Parse VBA time column - might be string like "00:01:00" or integer
                if vba_merge[vba_time_col].dtype == 'object':
                    # Try parsing as time string HH:MM:SS
                    try:
                        # Convert to minute of day (0-1439)
                        def parse_time_to_minute(t):
                            if pd.isna(t):
                                return None
                            parts = str(t).split(':')
                            if len(parts) >= 2:
                                return int(parts[0]) * 60 + int(parts[1]) + 1  # 1-based
                            return int(t)
                        
                        vba_merge[vba_time_col] = vba_merge[vba_time_col].apply(parse_time_to_minute)
                    except:
                        # Just try converting to int
                        vba_merge[vba_time_col] = pd.to_numeric(vba_merge[vba_time_col], errors='coerce')
                else:
                    vba_merge[vba_time_col] = vba_merge[vba_time_col].astype(int)
                
                # Drop any NaN minutes
                vba_merge = vba_merge.dropna(subset=[vba_time_col])
                vba_merge[vba_time_col] = vba_merge[vba_time_col].astype(int)
                
                # Merge VBA with Python stats on minute
                merged = stats_merge.merge(
                    vba_merge, 
                    left_on='minute',
                    right_on=vba_time_col,
                    how='inner'
                )
                
                if len(merged) == 0:
                    continue
                
                # Clean data - remove NaN/inf
                clean = merged[[col, f'{col}_median', f'{col}_q1', f'{col}_q3']].copy()
                clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(clean) < 10:  # Need enough points
                    print(f"    Skipping {col}: insufficient valid data ({len(clean)} points)")
                    continue
                
                # Calculate comparison metrics
                in_iqr = ((clean[col] >= clean[f'{col}_q1']) & 
                          (clean[col] <= clean[f'{col}_q3'])).mean()
                
                # Correlation and errors (with error handling)
                try:
                    corr = np.corrcoef(clean[col], clean[f'{col}_median'])[0, 1]
                except:
                    corr = 0.0
                
                rmse = np.sqrt(np.mean((clean[col] - clean[f'{col}_median'])**2))
                bias = np.mean(clean[col] - clean[f'{col}_median'])
                
                # Scaling (linear regression with error handling)
                try:
                    slope, intercept = np.polyfit(clean[f'{col}_median'], clean[col], 1)
                except:
                    # Fall back to simple ratio if polyfit fails
                    slope = np.mean(clean[col]) / np.mean(clean[f'{col}_median']) if np.mean(clean[f'{col}_median']) != 0 else 1.0
                    intercept = 0.0
                
                comparison_results.append({
                    'dwelling': dwelling,
                    'original_id': DWELLING_MAP[dwelling],
                    'metric': col,
                    'in_iqr_pct': in_iqr * 100,
                    'correlation': corr,
                    'rmse': rmse,
                    'bias': bias,
                    'slope': slope,
                    'intercept': intercept
                })
        
        df_comp = pd.DataFrame(comparison_results)
        
        # Save comparison
        comp_file = self.output_dir / 'vba_comparison_metrics.csv'
        df_comp.to_csv(comp_file, index=False)
        print(f"  Saved: {comp_file}")
        
        return df_comp
    
    def analyze_daily_distributions(self):
        """Analyze daily aggregate distributions."""
        print("\nAnalyzing daily distributions...")
        
        daily_stats = []
        
        for dwelling in sorted(self.df_daily['dwelling'].unique()):
            d = self.df_daily[self.df_daily['dwelling'] == dwelling]
            
            for metric in ['electricity_kwh', 'gas_m3', 'water_L', 'temp_C']:
                if metric not in d.columns:
                    continue
                
                values = d[metric].values
                
                stats_row = {
                    'dwelling': dwelling,
                    'original_id': DWELLING_MAP[dwelling],
                    'metric': metric,
                    'median': np.median(values),
                    'q1': np.percentile(values, 25),
                    'q3': np.percentile(values, 75),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'iqr': np.percentile(values, 75) - np.percentile(values, 25)
                }
                
                # Add VBA comparison if available
                if len(self.vba_daily) > 0:
                    orig_id = DWELLING_MAP[dwelling]
                    vba_row = self.vba_daily[self.vba_daily['dwelling_index'] == orig_id]
                    if len(vba_row) > 0:
                        # Map metric names to VBA column names
                        vba_metric_map = {
                            'electricity_kwh': 'Total_Electricity_kWh',
                            'gas_m3': 'Total_Gas_m3',
                            'water_L': 'Total_Hot_Water_L',
                            'temp_C': 'Mean_Internal_Temp_C'
                        }
                        vba_col = vba_metric_map.get(metric, metric)
                        
                        if vba_col in vba_row.columns:
                            vba_val = vba_row[vba_col].values[0]
                            stats_row['vba_value'] = vba_val
                            stats_row['vba_ratio'] = vba_val / stats_row['median'] if stats_row['median'] > 0 else 0
                            stats_row['vba_in_iqr'] = (vba_val >= stats_row['q1']) and (vba_val <= stats_row['q3'])
                
                daily_stats.append(stats_row)
        
        df_daily_stats = pd.DataFrame(daily_stats)
        
        # Save
        daily_file = self.output_dir / 'daily_statistics.csv'
        df_daily_stats.to_csv(daily_file, index=False)
        print(f"  Saved: {daily_file}")
        
        return df_daily_stats
    
    def plot_time_series_with_iqr(self, df_stats: pd.DataFrame, metric: str):
        """Plot time-series with IQR bands and VBA overlay."""
        print(f"\nPlotting time-series: {METRICS.get(metric, metric)}...")
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'{METRICS.get(metric, metric)} - Python IQR vs VBA', fontsize=14)
        
        for idx, dwelling in enumerate(sorted(df_stats['dwelling'].unique())):
            ax = axes[idx]
            
            # Python stats
            d = df_stats[df_stats['dwelling'] == dwelling].sort_values('minute')
            minutes = d['minute'].values
            
            median = d[f'{metric}_median'].values
            q1 = d[f'{metric}_q1'].values
            q3 = d[f'{metric}_q3'].values
            
            # Plot IQR band
            ax.fill_between(minutes, q1, q3, alpha=0.3, color='blue', label='Python Q1-Q3')
            ax.plot(minutes, median, 'b-', linewidth=1, label='Python Median')
            
            # VBA overlay
            if dwelling in self.vba_minute and metric in self.vba_minute[dwelling].columns:
                vba_data = self.vba_minute[dwelling].copy()
                vba_time_col = 'Minute' if 'Minute' in vba_data.columns else 'minute'
                
                # Parse time column
                if vba_data[vba_time_col].dtype == 'object':
                    def parse_time_to_minute(t):
                        if pd.isna(t):
                            return None
                        parts = str(t).split(':')
                        if len(parts) >= 2:
                            return int(parts[0]) * 60 + int(parts[1]) + 1
                        return float(t)
                    
                    vba_data[vba_time_col] = vba_data[vba_time_col].apply(parse_time_to_minute)
                else:
                    vba_data[vba_time_col] = pd.to_numeric(vba_data[vba_time_col], errors='coerce')
                
                vba_data = vba_data.dropna(subset=[vba_time_col]).sort_values(vba_time_col)
                
                ax.plot(vba_data[vba_time_col].values, vba_data[metric].values, 'r-', 
                       linewidth=0.8, alpha=0.7, label='VBA')
            
            ax.set_ylabel(METRICS.get(metric, metric))
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title(DWELLING_DESC[dwelling], fontsize=10)
        
        axes[-1].set_xlabel('Minute of Day')
        plt.tight_layout()
        
        plot_file = self.output_dir / f'timeseries_{metric}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_file}")
    
    def plot_scatter_vba_vs_python(self, df_stats: pd.DataFrame, metric: str):
        """Scatter plot: VBA vs Python median for each minute."""
        print(f"\nPlotting scatter: {METRICS.get(metric, metric)}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, dwelling in enumerate(sorted(df_stats['dwelling'].unique())):
            if dwelling not in self.vba_minute or metric not in self.vba_minute[dwelling].columns:
                axes[idx].text(0.5, 0.5, 'No VBA data', ha='center', va='center')
                axes[idx].set_title(DWELLING_DESC[dwelling])
                continue
            
            # Merge data
            stats = df_stats[df_stats['dwelling'] == dwelling]
            vba = self.vba_minute[dwelling]
            
            # Find time columns
            time_col_vba = None
            for tc in ['Minute', 'minute', 'time']:
                if tc in vba.columns:
                    time_col_vba = tc
                    break
            
            if time_col_vba is None:
                axes[idx].text(0.5, 0.5, 'No time column in VBA', ha='center', va='center')
                axes[idx].set_title(DWELLING_DESC[dwelling])
                continue
            
            # Prepare data with correct types
            stats_merge = stats.copy()
            vba_merge = vba[[time_col_vba, metric]].copy()
            stats_merge['minute'] = stats_merge['minute'].astype(int)
            
            # Parse VBA time column
            if vba_merge[time_col_vba].dtype == 'object':
                def parse_time_to_minute(t):
                    if pd.isna(t):
                        return None
                    parts = str(t).split(':')
                    if len(parts) >= 2:
                        return int(parts[0]) * 60 + int(parts[1]) + 1
                    return int(t)
                
                vba_merge[time_col_vba] = vba_merge[time_col_vba].apply(parse_time_to_minute)
            
            vba_merge = vba_merge.dropna(subset=[time_col_vba])
            vba_merge[time_col_vba] = vba_merge[time_col_vba].astype(int)
            
            merged = stats_merge.merge(vba_merge, 
                                      left_on='minute', 
                                      right_on=time_col_vba, 
                                      how='inner')
            
            if len(merged) == 0:
                axes[idx].text(0.5, 0.5, 'No matching data', ha='center', va='center')
                axes[idx].set_title(DWELLING_DESC[dwelling])
                continue
            
            # Clean data
            clean = merged[[f'{metric}_median', metric]].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean) < 10:
                axes[idx].text(0.5, 0.5, f'Insufficient data ({len(clean)} pts)', ha='center', va='center')
                axes[idx].set_title(DWELLING_DESC[dwelling])
                continue
            
            # Scatter colored by time of day
            scatter = axes[idx].scatter(
                clean[f'{metric}_median'], 
                clean[metric],
                c=clean.index,  # Use index as proxy for time progression
                cmap='viridis',
                alpha=0.6,
                s=10
            )
            
            # 1:1 line
            lim = [
                min(clean[f'{metric}_median'].min(), clean[metric].min()),
                max(clean[f'{metric}_median'].max(), clean[metric].max())
            ]
            axes[idx].plot(lim, lim, 'k--', alpha=0.5, linewidth=1)
            
            # Linear fit with error handling
            try:
                slope, intercept = np.polyfit(clean[f'{metric}_median'], clean[metric], 1)
                fit_line = slope * np.array(lim) + intercept
                axes[idx].plot(lim, fit_line, 'r-', alpha=0.7, linewidth=1.5,
                              label=f'y={slope:.2f}x+{intercept:.1f}')
            except:
                # Fallback: just show mean ratio
                ratio = clean[metric].mean() / clean[f'{metric}_median'].mean()
                axes[idx].text(0.5, 0.9, f'Ratio: {ratio:.2f}', 
                             transform=axes[idx].transAxes, ha='center')
            
            axes[idx].set_xlabel('Python Median')
            axes[idx].set_ylabel('VBA')
            axes[idx].set_title(DWELLING_DESC[dwelling], fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.suptitle(f'{METRICS.get(metric, metric)} - VBA vs Python Median\n(Color = Minute of Day)', 
                    fontsize=14)
        plt.tight_layout()
        
        plot_file = self.output_dir / f'scatter_{metric}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_file}")
    
    def plot_daily_boxplots(self, df_daily_stats: pd.DataFrame):
        """Box plots of daily totals with VBA overlay."""
        print("\nPlotting daily box plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics_daily = [
            ('electricity_kwh', 'Electricity (kWh)'),
            ('gas_m3', 'Gas (m³)'),
            ('water_L', 'Water (L)'),
            ('temp_C', 'Temperature (°C)')
        ]
        
        for ax, (metric, label) in zip(axes.flatten(), metrics_daily):
            data = []
            labels = []
            vba_vals = []
            
            for dwelling in range(1, 6):
                d = self.df_daily[self.df_daily['dwelling'] == dwelling]
                if metric in d.columns:
                    data.append(d[metric].values)
                    labels.append(f"D{DWELLING_MAP[dwelling]}")
                    
                    # VBA value
                    stat_row = df_daily_stats[
                        (df_daily_stats['dwelling'] == dwelling) & 
                        (df_daily_stats['metric'] == metric)
                    ]
                    if len(stat_row) > 0 and 'vba_value' in stat_row.columns:
                        vba_vals.append(stat_row['vba_value'].values[0])
                    else:
                        vba_vals.append(None)
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            # Overlay VBA points
            for i, vba_val in enumerate(vba_vals):
                if vba_val is not None:
                    ax.plot(i + 1, vba_val, 'r*', markersize=15, label='VBA' if i == 0 else '')
            
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(alpha=0.3)
            if any(v is not None for v in vba_vals):
                ax.legend()
        
        plt.suptitle('Daily Totals Distribution (Python) with VBA Comparison', fontsize=14)
        plt.tight_layout()
        
        plot_file = self.output_dir / 'daily_boxplots.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_file}")
    
    def generate_summary_report(self, df_comp: pd.DataFrame, df_daily_stats: pd.DataFrame):
        """Generate text summary report."""
        print("\nGenerating summary report...")
        
        report = []
        report.append("=" * 80)
        report.append("MONTE CARLO ANALYSIS SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        if len(df_comp) > 0:
            report.append("VBA vs PYTHON COMPARISON - MINUTE-LEVEL DATA")
            report.append("-" * 80)
            
            for dwelling in sorted(df_comp['dwelling'].unique()):
                d = df_comp[df_comp['dwelling'] == dwelling]
                orig_id = DWELLING_MAP[dwelling]
                report.append(f"\n{DWELLING_DESC[dwelling]} (Original ID: {orig_id})")
                report.append("")
                
                for _, row in d.iterrows():
                    metric_name = METRICS.get(row['metric'], row['metric'])
                    report.append(f"  {metric_name}:")
                    report.append(f"    VBA in Python IQR:  {row['in_iqr_pct']:.1f}% of time (expect ~50%)")
                    report.append(f"    Correlation:         {row['correlation']:.3f}")
                    report.append(f"    RMSE:                {row['rmse']:.2f}")
                    report.append(f"    Bias (VBA-Python):   {row['bias']:.2f}")
                    report.append(f"    Linear fit:          VBA = {row['slope']:.3f} × Python + {row['intercept']:.2f}")
                    report.append("")
        
        if len(df_daily_stats) > 0:
            report.append("\n")
            report.append("DAILY TOTALS COMPARISON")
            report.append("-" * 80)
            
            for dwelling in range(1, 6):
                d = df_daily_stats[df_daily_stats['dwelling'] == dwelling]
                if len(d) == 0:
                    continue
                
                report.append(f"\n{DWELLING_DESC[dwelling]}")
                report.append("")
                
                for _, row in d.iterrows():
                    if 'vba_value' in row and pd.notna(row['vba_value']):
                        report.append(f"  {row['metric']}:")
                        report.append(f"    Python: {row['median']:.2f} (Q1={row['q1']:.2f}, Q3={row['q3']:.2f})")
                        report.append(f"    VBA:    {row['vba_value']:.2f}")
                        report.append(f"    Ratio:  {row['vba_ratio']:.3f} (VBA/Python)")
                        report.append(f"    In IQR: {'✓ YES' if row['vba_in_iqr'] else '✗ NO'}")
                        report.append("")
        
        report.append("\n")
        report.append("=" * 80)
        
        # Save report
        report_file = self.output_dir / 'summary_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"  Saved: {report_file}")
        
        # Also print to console
        print("\n" + '\n'.join(report))
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("STARTING COMPREHENSIVE MONTE CARLO ANALYSIS")
        print("=" * 80)
        
        # 1. Compute minute statistics
        df_stats = self.compute_minute_statistics()
        
        # 2. Analyze daily distributions
        df_daily_stats = self.analyze_daily_distributions()
        
        # 3. Compare to VBA (if available)
        df_comp = pd.DataFrame()
        if self.vba_minute:
            df_comp = self.compare_to_vba(df_stats)
        
        # 4. Generate plots
        for metric in ['Total_Electricity_W', 'Internal_Temp_C', 'Gas_Consumption_m3_per_min']:
            if f'{metric}_median' in df_stats.columns:
                self.plot_time_series_with_iqr(df_stats, metric)
                if self.vba_minute:
                    self.plot_scatter_vba_vs_python(df_stats, metric)
        
        self.plot_daily_boxplots(df_daily_stats)
        
        # 5. Generate summary report
        self.generate_summary_report(df_comp, df_daily_stats)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 80)


def main():
    import sys
    
    minute_file = sys.argv[1] if len(sys.argv) > 1 else 'monte_carlo_minute.parquet'
    daily_file = sys.argv[2] if len(sys.argv) > 2 else 'monte_carlo_daily.csv'
    vba_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    analyzer = MonteCarloAnalyzer(minute_file, daily_file, vba_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
