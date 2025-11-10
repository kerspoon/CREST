#!/usr/bin/env python3
"""
Excel Reader for CREST 100-House Validation

Extracts data from the Excel/VBA model's 100-house simulation file
and converts it to CSV format matching the Python output structure.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Extract data from CREST Excel 100-house simulation file"
    )
    parser.add_argument(
        "excel_file",
        type=Path,
        help="Path to Excel file (CREST_Demand_Model_v2.3.3__100_houses.xlsm)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/excel_100houses"),
        help="Output directory for extracted CSVs (default: results/excel_100houses)"
    )

    args = parser.parse_args()

    if not args.excel_file.exists():
        print(f"Error: Excel file not found: {args.excel_file}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Excel file: {args.excel_file}")
    print(f"This may take a few minutes...")

    # Read Excel file sheets
    try:
        # Read results - disaggregated (minute-level data)
        # Excel structure: Row 0-2 are headers/description, Row 3 has column names, Row 4 has units
        print("\nReading 'Results - disaggregated' sheet...")
        df_minute = pd.read_excel(
            args.excel_file,
            sheet_name='Results - disaggregated',
            engine='openpyxl',
            skiprows=3,  # Skip first 3 rows, use row 4 as header
            header=0
        )

        # Skip the units row (first data row after header)
        df_minute = df_minute.iloc[1:]

        # Map Excel column names to Python output format
        minute_column_mapping = {
            'Dwelling index': 'Dwelling',
            'Time': 'Minute',
            'Occupancy': 'At_Home',
            'Activity': 'Active',
            'Lighting demand': 'Lighting_W',
            'Appliance demand': 'Appliances_W',
            'Net dwelling electricity demand': 'Total_Electricity_W',
            'Outdoor temperature': 'Outdoor_Temp_C',
            'Outdoor global radiation (horizontal)': 'Irradiance_Wm2',
            'Internal building node temperature': 'Internal_Temp_C',
            'External building node temperature': 'External_Building_Temp_C',
            'Hot water demand (litres)': 'Hot_Water_Demand_L_per_min',
            'Hot water temperature in hot water tank': 'Cylinder_Temp_C',
            'Emitter temperature': 'Emitter_Temp_C',
            'Cooler Emitter temperature': 'Cooling_Emitter_Temp_C',
            'Primary heating system thermal output': 'Total_Heat_Output_W',
            'Heat output from primary heating system to space': 'Space_Heating_W',
            'Heat output from primary heating system to hot water': 'Water_Heating_W',
            'Fuel flow rate (gas)': 'Gas_Consumption_m3_per_min',
            'Passive solar gains': 'Passive_Solar_Gains_W',
            'Casual thermal gains from occupants, lighting and appliances': 'Casual_Gains_W',
            'Electricity used by heating system': 'Heating_Electricity_W',
            'PV output': 'PV_Output_W',
            'Electricity used by cooling system': 'Cooling_Electricity_W'
        }

        df_minute = df_minute.rename(columns=minute_column_mapping)

        # Keep only columns that match Python output
        python_columns = ['Dwelling', 'Minute', 'At_Home', 'Active', 'Lighting_W',
                         'Appliances_W', 'Total_Electricity_W', 'Outdoor_Temp_C',
                         'Irradiance_Wm2', 'Internal_Temp_C', 'External_Building_Temp_C',
                         'Hot_Water_Demand_L_per_min', 'Cylinder_Temp_C', 'Emitter_Temp_C',
                         'Cooling_Emitter_Temp_C', 'Total_Heat_Output_W', 'Space_Heating_W',
                         'Water_Heating_W', 'Gas_Consumption_m3_per_min',
                         'Passive_Solar_Gains_W', 'Casual_Gains_W', 'Heating_Electricity_W',
                         'PV_Output_W', 'Cooling_Electricity_W']

        df_minute = df_minute[[col for col in python_columns if col in df_minute.columns]]

        # Convert data types
        for col in df_minute.columns:
            if col not in ['Dwelling', 'Minute']:
                df_minute[col] = pd.to_numeric(df_minute[col], errors='coerce')

        # Save to CSV
        output_file = args.output_dir / "results_minute_level.csv"
        df_minute.to_csv(output_file, index=False)
        print(f"  Saved {len(df_minute)} rows to {output_file}")
        print(f"  Columns: {list(df_minute.columns[:10])}...")

    except Exception as e:
        print(f"  Warning: Could not read 'Results - disaggregated': {e}")

    try:
        # Read results - daily totals
        print("\nReading 'Results - daily totals' sheet...")
        df_daily = pd.read_excel(
            args.excel_file,
            sheet_name='Results - daily totals',
            engine='openpyxl',
            skiprows=1  # Skip first header row
        )

        # Rename columns to match Python output
        column_mapping = {
            'Dwelling index': 'Dwelling',
            'Total dwelling electricity demand': 'Total_Electricity_kWh',
            'Gas demand': 'Total_Gas_m3',
            'Hot water demand (litres)': 'Total_Hot_Water_L',
            'Average indoor air temperature': 'Mean_Internal_Temp_C'
        }
        df_daily = df_daily.rename(columns=column_mapping)

        # Convert electricity from Wh to kWh if needed
        if 'Total_Electricity_kWh' in df_daily.columns:
            # Check if values are in Wh (typically > 100) vs kWh (typically < 10)
            if df_daily['Total_Electricity_kWh'].mean() > 10:
                df_daily['Total_Electricity_kWh'] = df_daily['Total_Electricity_kWh'] / 1000.0

        # Select only columns that exist in both datasets
        keep_columns = ['Dwelling', 'Total_Electricity_kWh', 'Total_Gas_m3',
                       'Total_Hot_Water_L', 'Mean_Internal_Temp_C']
        df_daily = df_daily[[col for col in keep_columns if col in df_daily.columns]]

        # Save to CSV
        output_file = args.output_dir / "results_daily_summary.csv"
        df_daily.to_csv(output_file, index=False)
        print(f"  Saved {len(df_daily)} rows to {output_file}")
        print(f"  Columns: {list(df_daily.columns)}")

    except Exception as e:
        print(f"  Warning: Could not read 'Results - daily totals': {e}")

    try:
        # Read global climate
        print("\nReading 'GlobalClimate' sheet...")
        df_climate = pd.read_excel(
            args.excel_file,
            sheet_name='GlobalClimate',
            engine='openpyxl'
        )

        # Save to CSV
        output_file = args.output_dir / "global_climate.csv"
        df_climate.to_csv(output_file, index=False)
        print(f"  Saved {len(df_climate)} rows to {output_file}")

    except Exception as e:
        print(f"  Warning: Could not read 'GlobalClimate': {e}")

    try:
        # Read dwellings configuration
        print("\nReading 'Dwellings' sheet...")
        df_dwellings = pd.read_excel(
            args.excel_file,
            sheet_name='Dwellings',
            engine='openpyxl'
        )

        # Save to CSV
        output_file = args.output_dir / "dwellings_config.csv"
        df_dwellings.to_csv(output_file, index=False)
        print(f"  Saved {len(df_dwellings)} rows to {output_file}")

    except Exception as e:
        print(f"  Warning: Could not read 'Dwellings': {e}")

    print(f"\n✓ Excel data extracted to: {args.output_dir}")


if __name__ == "__main__":
    main()
