#!/usr/bin/env python3
"""
CREST Demand Model - Main Simulation Script

A high-resolution (1-minute) stochastic integrated thermal-electrical
domestic energy demand simulator.
"""

import argparse
import sys
from pathlib import Path

# Add crest package to path
sys.path.insert(0, str(Path(__file__).parent))

from crest.data.loader import CRESTDataLoader
from crest.core.climate import GlobalClimate, ClimateConfig
from crest.simulation.dwelling import Dwelling, DwellingConfig
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

    # Simulate dwellings
    results = []

    for dwelling_idx in range(args.num_dwellings):
        print(f"\nSimulating dwelling {dwelling_idx + 1}/{args.num_dwellings}...")

        # Configure dwelling
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
