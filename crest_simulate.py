#!/usr/bin/env python3
"""
CREST Demand Model - Main Simulation Script

A high-resolution (1-minute) stochastic integrated thermal-electrical
domestic energy demand simulator.
"""

import argparse
import sys
from pathlib import Path
import json

# Add crest package to path
sys.path.insert(0, str(Path(__file__).parent))

from crest.data.loader import CRESTDataLoader
from crest.core.climate import GlobalClimate, ClimateConfig
from crest.simulation.dwelling import Dwelling, DwellingConfig
from crest.output.writer import ResultsWriter, OutputConfig
from crest.utils import random as rng_module
import numpy as np


def load_activity_statistics(data_loader: CRESTDataLoader) -> dict:
    """
    Load activity statistics into a dictionary.

    Parameters
    ----------
    data_loader : CRESTDataLoader
        Data loader instance

    Returns
    -------
    dict
        Activity statistics indexed by key (weekend_activeoccupants_profile)
    """
    activity_stats = {}
    activity_df = data_loader.load_activity_stats()

    # Parse activity statistics (simplified)
    # In production, would parse all activity profiles from CSV
    # For now, create placeholder structure
    for weekend in [0, 1]:
        for occupants in range(7):  # 0-6 active occupants
            for profile in ['Active', 'Washing', 'Cooking', 'Ironing', 'HouseCleaning']:
                key = f"{weekend}_{occupants}_{profile}"
                # Default to low activity probability
                activity_stats[key] = np.full(144, 0.01)

    return activity_stats


def main():
    """Main simulation entry point."""
    parser = argparse.ArgumentParser(
        description="CREST Demand Model - Stochastic domestic energy demand simulator"
    )
    parser.add_argument(
        "--num-dwellings",
        type=int,
        default=1,
        help="Number of dwellings to simulate (default: 1)"
    )
    parser.add_argument(
        "--residents",
        type=int,
        default=2,
        help="Number of residents per dwelling (default: 2)"
    )
    parser.add_argument(
        "--weekend",
        action="store_true",
        help="Simulate weekend day (default: weekday)"
    )
    parser.add_argument(
        "--day",
        type=int,
        default=15,
        help="Day of month (default: 15)"
    )
    parser.add_argument(
        "--month",
        type=int,
        default=6,
        help="Month of year (default: 6 = June)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data directory (default: ./data)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save simulation results (default: None, no output saved)"
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Save minute-level detailed results (large files)"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="JSON file with per-dwelling configurations (overrides --residents and other dwelling params)"
    )

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        rng_module.set_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Load data
    print("Loading data...")
    data_loader = CRESTDataLoader(args.data_dir)

    # Load activity statistics
    print("Loading activity statistics...")
    activity_statistics = load_activity_statistics(data_loader)

    # Create global climate
    print(f"Simulating climate for {args.month}/{args.day}...")
    climate_config = ClimateConfig(
        day_of_month=args.day,
        month_of_year=args.month
    )
    global_climate = GlobalClimate(climate_config, data_loader)
    global_climate.run_all()

    # Initialize output writer if requested
    results_writer = None
    if args.output_dir:
        output_config = OutputConfig(
            output_dir=args.output_dir,
            save_minute_data=args.save_detailed,
            save_daily_summary=True,
            save_global_climate=True
        )
        results_writer = ResultsWriter(output_config)
        print(f"Saving results to: {args.output_dir}")

        # Write global climate data once
        results_writer.write_global_climate(global_climate)

    # Load dwelling configurations if provided
    dwelling_configs_list = None
    if args.config_file:
        print(f"Loading dwelling configurations from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            dwelling_configs_list = json.load(f)
        print(f"  Loaded {len(dwelling_configs_list)} dwelling configurations")
        # Override num_dwellings to match config file
        args.num_dwellings = len(dwelling_configs_list)

    # Simulate dwellings
    results = []
    dwellings = []  # Store dwelling objects for output

    for dwelling_idx in range(args.num_dwellings):
        print(f"\nSimulating dwelling {dwelling_idx + 1}/{args.num_dwellings}...")

        # Configure dwelling - either from config file or command-line args
        if dwelling_configs_list:
            # Load from config file
            cfg = dwelling_configs_list[dwelling_idx]
            dwelling_config = DwellingConfig(
                dwelling_index=dwelling_idx,
                num_residents=cfg['num_residents'],
                building_index=cfg['building_index'],
                heating_system_index=cfg['heating_system_index'],
                cooling_system_index=cfg.get('cooling_system_index', 0),
                is_weekend=args.weekend,
                has_pv=cfg.get('has_pv', False),
                has_solar_thermal=cfg.get('has_solar_thermal', False)
            )
        else:
            # Use command-line defaults
            dwelling_config = DwellingConfig(
                dwelling_index=dwelling_idx,
                num_residents=args.residents,
                building_index=0,  # Simplified: use first building type
                heating_system_index=0,  # Simplified: use first heating system
                is_weekend=args.weekend,
                has_pv=False,
                has_solar_thermal=False
            )

        # Create and run dwelling simulation
        dwelling = Dwelling(
            dwelling_config,
            global_climate,
            data_loader,
            activity_statistics
        )

        print("  Running simulation...")
        dwelling.run_simulation()

        # Store dwelling for output
        dwellings.append(dwelling)

        # Collect results
        print("  Calculating daily totals...")
        daily_electricity = sum(dwelling.get_total_electricity_demand(t) for t in range(1, 1441)) / 60.0  # Wh
        daily_gas = dwelling.heating_system.get_daily_fuel_consumption()  # m³
        daily_hot_water = dwelling.hot_water.get_daily_hot_water_volume()  # litres

        results.append({
            'dwelling': dwelling_idx,
            'electricity_kwh': daily_electricity / 1000.0,
            'gas_m3': daily_gas,
            'hot_water_litres': daily_hot_water
        })

        print(f"  Daily electricity: {daily_electricity/1000.0:.2f} kWh")
        print(f"  Daily gas: {daily_gas:.2f} m³")
        print(f"  Daily hot water: {daily_hot_water:.2f} litres")

    # Write results to files if requested
    if results_writer:
        print("\nWriting results to files...")
        for dwelling_idx, dwelling in enumerate(dwellings):
            print(f"  Writing dwelling {dwelling_idx + 1}/{args.num_dwellings}...")
            if args.save_detailed:
                results_writer.write_minute_data(dwelling_idx, dwelling)
            results_writer.write_daily_summary(dwelling_idx, dwelling)
        results_writer.close()
        print(f"Results saved to: {args.output_dir}")

    # Summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    if args.num_dwellings > 1:
        avg_elec = np.mean([r['electricity_kwh'] for r in results])
        avg_gas = np.mean([r['gas_m3'] for r in results])
        avg_water = np.mean([r['hot_water_litres'] for r in results])
        print(f"Average electricity: {avg_elec:.2f} kWh/day")
        print(f"Average gas: {avg_gas:.2f} m³/day")
        print(f"Average hot water: {avg_water:.2f} litres/day")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
