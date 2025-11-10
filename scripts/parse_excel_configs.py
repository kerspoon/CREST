#!/usr/bin/env python3
"""
Parse Excel dwelling configurations into JSON format for Python simulation.
"""

import pandas as pd
import json
import sys
from pathlib import Path

def parse_dwelling_configs(excel_config_path: Path, output_json: Path):
    """
    Parse Excel dwelling configurations.

    Excel columns (after header rows):
    0: Dwelling index
    1: Number of residents
    2: Building index
    3: Primary heating system index
    4: PV system index
    5: Solar thermal collector index
    6: Cooling system index
    7: (empty)
    8+: Appliance ownership flags
    """

    # Read CSV, skip first 3 rows (headers), use row 4 as header
    df = pd.read_csv(excel_config_path, skiprows=3, header=0)

    # Get first 100 dwellings
    df = df.head(100)

    # Parse into dwelling configs
    configs = []

    for idx, row in df.iterrows():
        # Column names from Excel
        dwelling_idx = int(row.iloc[0])
        num_residents = int(row.iloc[1])
        building_index = int(row.iloc[2])
        heating_system_index = int(row.iloc[3])
        pv_system_index = int(row.iloc[4])
        solar_thermal_index = int(row.iloc[5])
        cooling_system_index = int(row.iloc[6])

        config = {
            'dwelling_index': dwelling_idx,
            'num_residents': num_residents,
            'building_index': building_index,
            'heating_system_index': heating_system_index,
            'pv_system_index': pv_system_index,
            'solar_thermal_index': solar_thermal_index,
            'cooling_system_index': cooling_system_index,
            'has_pv': pv_system_index > 0,
            'has_solar_thermal': solar_thermal_index > 0,
            'has_cooling': cooling_system_index > 0
        }

        configs.append(config)

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(configs, f, indent=2)

    print(f"Parsed {len(configs)} dwelling configurations")
    print(f"Saved to: {output_json}")

    # Print summary statistics
    residents = [c['num_residents'] for c in configs]
    buildings = [c['building_index'] for c in configs]
    heating = [c['heating_system_index'] for c in configs]

    print("\nConfiguration Summary:")
    print(f"  Residents: min={min(residents)}, max={max(residents)}, mean={sum(residents)/len(residents):.2f}")
    print(f"  Building types: {sorted(set(buildings))}")
    print(f"  Heating systems: {sorted(set(heating))}")
    print(f"  PV systems: {sum(1 for c in configs if c['has_pv'])} dwellings")
    print(f"  Solar thermal: {sum(1 for c in configs if c['has_solar_thermal'])} dwellings")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_excel_configs.py <excel_config.csv> [output.json]")
        sys.exit(1)

    excel_config = Path(sys.argv[1])
    output_json = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dwelling_configs.json")

    if not excel_config.exists():
        print(f"Error: Config file not found: {excel_config}")
        sys.exit(1)

    parse_dwelling_configs(excel_config, output_json)
