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
from crest.simulation.config import Country, City, UrbanRural
from crest.output.writer import ResultsWriter, OutputConfig
from crest.utils import random as rng_module
import numpy as np
import pandas as pd


def load_activity_statistics(data_loader: CRESTDataLoader) -> dict:
    """
    Load activity statistics into a dictionary.

    VBA Reference: LoadActivityStatistics (mdlThermalElectricalModel.bas lines 761-801)
    Reads 72 activity profiles from ActivityStats.csv rows 30-101.

    Parameters
    ----------
    data_loader : CRESTDataLoader
        Data loader instance

    Returns
    -------
    dict
        Activity statistics indexed by key "{weekend}_{active_occupants}_{profile_id}"
        Each value is a numpy array of 144 ten-minute modifier values.
    """
    activity_stats = {}
    activity_df = data_loader.load_activity_stats()

    # VBA: For i = 30 To 101 (rows 30-101 in Excel contain the 72 activity profiles)
    # Python: Due to CSV headers, these are at pandas rows 25-96 (0-indexed)
    # Only process rows with valid activity profile IDs (starting with "Act_")
    for row_idx in range(len(activity_df)):
        row = activity_df.iloc[row_idx]

        # Check if this row has valid activity data
        weekend = row.iloc[1]
        occupants = row.iloc[2]
        profile_id = row.iloc[3]

        # Skip rows without valid activity profile data
        if pd.isna(weekend) or pd.isna(occupants) or pd.isna(profile_id):
            continue
        if not isinstance(profile_id, str) or not profile_id.startswith('Act_'):
            continue

        # VBA: objActivityStatsItem.IsWeekend = IIf(wsActivityStats.Range("B" + CStr(i)).Value = 1, True, False)
        # Column B (index 1) = Weekend flag (0 or 1)
        is_weekend = int(weekend)

        # VBA: objActivityStatsItem.ActiveOccupantCount = wsActivityStats.Range("C" + CStr(i)).Value
        # Column C (index 2) = Active occupant count (0-6)
        active_occupants = int(occupants)

        # VBA: objActivityStatsItem.ID = wsActivityStats.Range("D" + CStr(i)).Value
        # Column D (index 3) = Activity ID (e.g., "Act_TV", "Act_Cooking")
        profile_id = str(profile_id)

        # VBA: For j = 0 To 143
        #      objActivityStatsItem.Modifiers(j) = wsActivityStats.Range(strCell).Value
        # Columns E onwards (indices 4-147) = 144 ten-minute modifiers
        modifiers = row.iloc[4:148].values.astype(float)

        # VBA: strKey = IIf(objActivityStatsItem.IsWeekend, "1", "0") + "_" +
        #              CStr(objActivityStatsItem.ActiveOccupantCount) + "_" +
        #              objActivityStatsItem.ID
        key = f"{is_weekend}_{active_occupants}_{profile_id}"

        # VBA: objActivityStatistics.Add Item:=objActivityStatsItem, Key:=strKey
        activity_stats[key] = modifiers

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
    parser.add_argument(
        "--country",
        type=str,
        default="UK",
        choices=["UK", "India"],
        help="Country for appliance ownership and water temperature (default: UK)"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="England",
        choices=["England", "N Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Itanagar"],
        help="City/region for climate temperature profiles (default: England)"
    )
    parser.add_argument(
        "--urban-rural",
        type=str,
        default="Urban",
        choices=["Urban", "Rural"],
        help="Urban or rural location for appliance ownership (default: Urban)"
    )

    args = parser.parse_args()

    # Convert string arguments to enums
    country = Country(args.country)
    city = City(args.city)
    urban_rural = UrbanRural(args.urban_rural)

    # Set random seed if specified
    if args.seed is not None:
        rng_module.set_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    print(f"Location: {city.value}, {country.value} ({urban_rural.value})")

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
        month_of_year=args.month,
        city=city
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
                country=country,  # Use CLI-specified country
                urban_rural=urban_rural,  # Use CLI-specified urban/rural
                cooling_system_index=cfg.get('cooling_system_index', 0),
                pv_system_index=cfg.get('pv_system_index', 0),
                solar_thermal_index=cfg.get('solar_thermal_index', 0),
                is_weekend=args.weekend
            )
        else:
            # Use command-line defaults
            dwelling_config = DwellingConfig(
                dwelling_index=dwelling_idx,
                num_residents=args.residents,
                building_index=1,  # Use building index 1 (1-based indexing)
                heating_system_index=1,  # Use heating system index 1 (1-based indexing)
                country=country,  # Use CLI-specified country
                urban_rural=urban_rural,  # Use CLI-specified urban/rural
                cooling_system_index=0,
                pv_system_index=0,  # No PV by default
                solar_thermal_index=0,  # No solar thermal by default
                is_weekend=args.weekend
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
        # VBA Reference: DailyTotals (mdlThermalElectricalModel.bas lines 1057-1121)
        print("  Calculating daily totals...")

        # VBA line 1081: GetMeanActiveOccupancy
        mean_active_occupancy = dwelling.occupancy.get_mean_active_occupancy()

        # VBA line 1082: GetPrActivelyOccupied
        proportion_actively_occupied = dwelling.occupancy.get_probability_actively_occupied()

        # VBA line 1083: GetDailySumLighting / 60 / 1000 (W·min → kWh)
        lighting_demand_kwh = dwelling.lighting.get_daily_energy() / 60.0 / 1000.0

        # VBA line 1084: GetDailySumApplianceDemand / 60 / 1000 (W·min → kWh)
        appliance_demand_kwh = dwelling.appliances.get_daily_energy() / 60.0 / 1000.0

        # VBA line 1086: Total = Lighting + Appliances
        total_electricity_kwh = lighting_demand_kwh + appliance_demand_kwh

        # VBA line 1087: GetDailySumPvOutput / 60 / 1000 (W·min → kWh)
        if dwelling.pv_system is not None:
            pv_output_kwh = dwelling.pv_system.get_daily_sum_pv_output() / 60.0 / 1000.0
        else:
            pv_output_kwh = 0.0

        # VBA line 1088: GetDailySumP_net / 60 / 1000 (W·min → kWh)
        if dwelling.pv_system is not None:
            net_electricity_kwh = dwelling.pv_system.get_daily_sum_net_demand() / 60.0 / 1000.0
        else:
            net_electricity_kwh = total_electricity_kwh

        # VBA line 1090: GetDailySumP_self / 60 / 1000 (W·min → kWh)
        if dwelling.pv_system is not None:
            self_consumption_kwh = dwelling.pv_system.get_daily_sum_self_consumption() / 60.0 / 1000.0
        else:
            self_consumption_kwh = 0.0

        # VBA line 1092: GetDailySumHotWaterDemand (litres, no conversion)
        hot_water_litres = dwelling.hot_water.get_daily_hot_water_volume()

        # VBA line 1094: GetMeanTheta_i (°C, no conversion)
        average_indoor_temp = dwelling.building.get_mean_theta_i()

        # VBA line 1096: GetDailySumThermalEnergySpace / 60 / 1000 (W·min → kWh)
        thermal_energy_space_kwh = dwelling.heating_system.get_daily_thermal_energy_space() / 60.0 / 1000.0

        # VBA line 1097: GetDailySumThermalEnergyWater / 60 / 1000 (W·min → kWh)
        thermal_energy_water_kwh = dwelling.heating_system.get_daily_thermal_energy_water() / 60.0 / 1000.0

        # VBA line 1099: GetDailySumFuelFlow / 60 (W·min/60 → m³)
        gas_m3 = dwelling.heating_system.get_daily_fuel_consumption() / 60.0

        # VBA line 1118: GetSpaceThermostatSetpoint (°C, no conversion)
        space_setpoint = dwelling.heating_controls.get_space_thermostat_setpoint()

        # VBA line 1119: GetDailySumPhi_s / 60 / 1000 (W·min → kWh)
        if dwelling.solar_thermal is not None:
            solar_thermal_kwh = dwelling.solar_thermal.get_daily_sum_phi_s() / 60.0 / 1000.0
        else:
            solar_thermal_kwh = 0.0

        # Store all 17 metrics matching VBA DailyTotals output (lines 1103-1119)
        results.append({
            'dwelling_index': dwelling_idx,  # Column 1
            'mean_active_occupancy': mean_active_occupancy,  # Column 3
            'proportion_actively_occupied': proportion_actively_occupied,  # Column 4
            'lighting_kwh': lighting_demand_kwh,  # Column 5
            'appliances_kwh': appliance_demand_kwh,  # Column 6
            'pv_output_kwh': pv_output_kwh,  # Column 7
            'total_electricity_kwh': total_electricity_kwh,  # Column 8
            'self_consumption_kwh': self_consumption_kwh,  # Column 9
            'net_electricity_kwh': net_electricity_kwh,  # Column 10
            'hot_water_litres': hot_water_litres,  # Column 11
            'average_indoor_temp': average_indoor_temp,  # Column 12
            'thermal_energy_space_kwh': thermal_energy_space_kwh,  # Column 13
            'thermal_energy_water_kwh': thermal_energy_water_kwh,  # Column 14
            'gas_m3': gas_m3,  # Column 15
            'space_setpoint': space_setpoint,  # Column 16
            'solar_thermal_kwh': solar_thermal_kwh  # Column 17
        })

        print(f"  Daily electricity: {total_electricity_kwh:.2f} kWh (lighting: {lighting_demand_kwh:.2f}, appliances: {appliance_demand_kwh:.2f})")
        print(f"  Daily gas: {gas_m3:.2f} m³")
        print(f"  Daily hot water: {hot_water_litres:.2f} litres")
        if dwelling.pv_system is not None:
            print(f"  PV output: {pv_output_kwh:.2f} kWh, Net demand: {net_electricity_kwh:.2f} kWh")
        print(f"  Indoor temp: {average_indoor_temp:.1f}°C")

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
        # Calculate averages across all dwellings
        avg_elec = np.mean([r['total_electricity_kwh'] for r in results])
        avg_lighting = np.mean([r['lighting_kwh'] for r in results])
        avg_appliances = np.mean([r['appliances_kwh'] for r in results])
        avg_pv = np.mean([r['pv_output_kwh'] for r in results])
        avg_gas = np.mean([r['gas_m3'] for r in results])
        avg_water = np.mean([r['hot_water_litres'] for r in results])
        avg_temp = np.mean([r['average_indoor_temp'] for r in results])

        print(f"Average total electricity: {avg_elec:.2f} kWh/day")
        print(f"  - Lighting: {avg_lighting:.2f} kWh/day")
        print(f"  - Appliances: {avg_appliances:.2f} kWh/day")
        if avg_pv > 0:
            print(f"Average PV output: {avg_pv:.2f} kWh/day")
        print(f"Average gas: {avg_gas:.2f} m³/day")
        print(f"Average hot water: {avg_water:.2f} litres/day")
        print(f"Average indoor temperature: {avg_temp:.1f}°C")
    else:
        # Single dwelling - show all metrics
        r = results[0]
        print(f"Total electricity: {r['total_electricity_kwh']:.2f} kWh/day")
        print(f"  - Lighting: {r['lighting_kwh']:.2f} kWh/day")
        print(f"  - Appliances: {r['appliances_kwh']:.2f} kWh/day")
        if r['pv_output_kwh'] > 0:
            print(f"PV output: {r['pv_output_kwh']:.2f} kWh/day")
            print(f"Net demand: {r['net_electricity_kwh']:.2f} kWh/day")
            print(f"Self-consumption: {r['self_consumption_kwh']:.2f} kWh/day")
        print(f"Gas consumption: {r['gas_m3']:.2f} m³/day")
        print(f"Hot water: {r['hot_water_litres']:.2f} litres/day")
        print(f"Indoor temperature: {r['average_indoor_temp']:.1f}°C")
        print(f"Space heating: {r['thermal_energy_space_kwh']:.2f} kWh/day")
        print(f"Water heating: {r['thermal_energy_water_kwh']:.2f} kWh/day")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
