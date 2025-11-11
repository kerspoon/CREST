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


def _select_from_distribution(proportions: np.ndarray, rng: np.random.Generator) -> int:
    """
    Inverse transform sampling from discrete probability distribution.

    VBA Reference: Inverse transform logic in AssignDwellingParameters (lines 1194-1275)

    The VBA algorithm:
    1. Generate random number: dblRand = Rnd()
    2. Loop through proportions, accumulating cumulative probability
    3. Return first index where dblRand < dblCumulativeP

    Parameters
    ----------
    proportions : np.ndarray
        Array of probabilities (must sum to ~1.0)
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    int
        Selected index (0-based)

    Examples
    --------
    >>> proportions = np.array([0.2, 0.3, 0.5])
    >>> index = _select_from_distribution(proportions, rng)
    >>> # Returns 0, 1, or 2 with probabilities 20%, 30%, 50%
    """
    # VBA lines 1195-1204: Generate random and find first cumulative match
    cumulative = np.cumsum(proportions)
    rand_value = rng.random()

    # VBA: If dblRand < dblCumulativeP Then ... Exit For
    # np.searchsorted finds the first index where rand_value < cumulative[index]
    index = int(np.searchsorted(cumulative, rand_value))

    # Clamp to valid range (in case of floating point errors where sum != 1.0)
    return min(index, len(proportions) - 1)


def assign_dwelling_parameters(
    data_loader: CRESTDataLoader,
    dwelling_index: int,
    rng: np.random.Generator
) -> DwellingConfig:
    """
    Stochastically assign all parameters for one dwelling.

    VBA Reference: AssignDwellingParameters (mdlThermalElectricalModel.bas lines 1130-1288)
    Uses inverse transform sampling on cumulative probability distributions.

    Parameters
    ----------
    data_loader : CRESTDataLoader
        Data loader with proportion arrays
    dwelling_index : int
        Dwelling identifier (0-based)
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    DwellingConfig
        Stochastically generated dwelling configuration

    Notes
    -----
    Special logic: Combi boiler (type 2) precludes solar thermal system
    VBA lines 1248-1262: If heating type = 2, solar_thermal_index = 0
    """
    # VBA lines 1197-1205: Determine number of residents (1-5)
    # VBA: For intRow = 1 To intMaxNumberResidents (1-based)
    # Python: Returns 0-4, add 1 to get 1-5 residents
    resident_props = data_loader.load_resident_proportions()
    num_residents_index = _select_from_distribution(resident_props, rng)
    num_residents = num_residents_index + 1  # Convert 0-based to 1-5 range

    # VBA lines 1207-1218: Determine building index
    # VBA: lngDwellingBuildingIndex = lngRow (1-based)
    # Python: Returns 0-based index, add 1 for 1-based
    building_props = data_loader.load_building_proportions()
    building_index_0based = _select_from_distribution(building_props, rng)
    building_index = building_index_0based + 1  # 1-based indexing

    # VBA lines 1220-1231: Determine primary heating system index
    # VBA: lngDwellingPrimaryHeatingSystemIndex = lngRow (1-based)
    heating_props = data_loader.load_heating_proportions()
    heating_index_0based = _select_from_distribution(heating_props, rng)
    heating_index = heating_index_0based + 1  # 1-based indexing

    # VBA lines 1233-1244: Determine PV system index
    # VBA: lngDwellingPvSystemIndex = lngRow
    # Note: Index 0 = no PV, Index 1+ = PV system types
    pv_props = data_loader.load_pv_proportions()
    pv_index = _select_from_distribution(pv_props, rng)
    # Already 0-based where 0 = no PV

    # VBA lines 1246-1262: Determine solar thermal index
    # SPECIAL LOGIC: Combi boiler (type 2) precludes solar thermal
    # VBA line 1248: If wsPrimaryHeatingSystems.Cells(4 + lngDwellingPrimaryHeatingSystemIndex, 4) = 2
    heating_type = data_loader.get_heating_type(heating_index)
    if heating_type == 2:  # Combi boiler
        # VBA line 1249: lngDwellingSolarThermalIndex = 0
        solar_thermal_index = 0
    else:
        # VBA lines 1251-1261: Select stochastically
        solar_thermal_props = data_loader.load_solar_thermal_proportions()
        solar_thermal_index = _select_from_distribution(solar_thermal_props, rng)
        # Already 0-based where 0 = no solar thermal

    # VBA lines 1264-1275: Determine cooling system index
    # Note: Index 0 = no cooling, Index 1+ = cooling system types
    cooling_props = data_loader.load_cooling_proportions()
    cooling_index = _select_from_distribution(cooling_props, rng)
    # Already 0-based where 0 = no cooling

    # VBA lines 1278-1286: Write parameters (we return DwellingConfig instead)
    return DwellingConfig(
        dwelling_index=dwelling_index,
        num_residents=num_residents,
        building_index=building_index,
        heating_system_index=heating_index,
        pv_system_index=pv_index,
        solar_thermal_index=solar_thermal_index,
        cooling_system_index=cooling_index,
        country=Country.UK,  # Will be set by caller based on CLI args
        urban_rural=UrbanRural.URBAN,  # Will be set by caller based on CLI args
        is_weekend=False  # Will be set by caller based on CLI args
    )


def aggregate_results(dwellings: list) -> dict:
    """
    Aggregate time series results across all dwellings.

    VBA Reference: AggregateResults (mdlThermalElectricalModel.bas lines 810-1048)

    Aggregates 22 time series variables (1440 timesteps each) across all dwellings.
    Normalizes by population (occupancy), dwelling count (temperatures/states),
    or sums totals (power, fuel).

    Parameters
    ----------
    dwellings : list
        List of Dwelling objects

    Returns
    -------
    dict
        Dictionary with 22 aggregated time series arrays (each 1440 timesteps)
    """
    num_dwellings = len(dwellings)
    if num_dwellings == 0:
        return {}

    # VBA line 940: Get total population
    total_population = sum(d.config.num_residents for d in dwellings)

    # Initialize aggregated arrays (VBA lines 891-933)
    # All arrays are 1440 timesteps (minutes in a day)
    aggregated = {
        'occupancy': np.zeros(1440),  # Normalized by population
        'activity': np.zeros(1440),  # Normalized by population
        'P_e': np.zeros(1440),  # Total electricity (kW)
        'P_pv': np.zeros(1440),  # Total PV output (kW)
        'P_net': np.zeros(1440),  # Net demand (kW)
        'Phi_hSpace': np.zeros(1440),  # Space heating (kW)
        'Phi_hWater': np.zeros(1440),  # Water heating (kW)
        'Phi_s': np.zeros(1440),  # Solar thermal (kW)
        'theta_i': np.zeros(1440),  # Indoor temp (averaged)
        'V_dhw': np.zeros(1440),  # Hot water volume total (litres)
        'theta_cyl': np.zeros(1440),  # Cylinder temp (averaged)
        'space_timer': np.zeros(1440),  # Space timer state (averaged)
        'water_timer': np.zeros(1440),  # Water timer state (averaged)
        'heating_on': np.zeros(1440),  # Heating on/off (averaged)
        'water_heating_on': np.zeros(1440),  # Water heating on/off (averaged)
        'M_fuel': np.zeros(1440),  # Fuel flow total (m³)
        'Phi_collector': np.zeros(1440),  # Solar collector (kW)
        'P_self': np.zeros(1440),  # Self-consumption (kW)
        'mean_V_dhw': np.zeros(1440),  # Hot water per dwelling (litres)
        'mean_V_gas': np.zeros(1440),  # Fuel per dwelling (m³)
        'cooling_timer': np.zeros(1440),  # Cooling timer (averaged)
        'cooling_on': np.zeros(1440),  # Cooling on/off (averaged)
    }

    # VBA lines 969-995: Loop through dwellings and aggregate
    for dwelling in dwellings:
        for t in range(1, 1441):  # VBA uses 1-based timesteps
            # VBA lines 975-976: Occupancy and activity
            occupancy_state = dwelling.occupancy.get_active_occupancy_at_timestep((t - 1) // 10)
            activity_state = dwelling.occupancy.get_combined_states_1min()[t - 1]
            aggregated['occupancy'][t - 1] += occupancy_state
            # Activity: 1 if active (odd combined state), 0 if dormant (even combined state)
            aggregated['activity'][t - 1] += (int(activity_state) % 2)

            # VBA line 977: Total electricity (lighting + appliances)
            P_lighting = dwelling.lighting.get_total_demand(t)
            P_appliances = dwelling.appliances.get_total_demand(t)
            aggregated['P_e'][t - 1] += (P_lighting + P_appliances)

            # VBA line 978: PV output
            if dwelling.pv_system:
                aggregated['P_pv'][t - 1] += dwelling.pv_system.P_pv[t - 1]
                aggregated['P_self'][t - 1] += dwelling.pv_system.P_self[t - 1]

            # VBA lines 979-980: Heating
            aggregated['Phi_hSpace'][t - 1] += dwelling.heating_system.get_heat_to_space(t)
            aggregated['Phi_hWater'][t - 1] += dwelling.heating_system.get_heat_to_hot_water(t)

            # VBA line 981: Solar thermal (Phi_collector is same as Phi_s - both refer to phi_s array)
            if dwelling.solar_thermal:
                aggregated['Phi_s'][t - 1] += dwelling.solar_thermal.phi_s[t - 1]
                aggregated['Phi_collector'][t - 1] += dwelling.solar_thermal.phi_s[t - 1]

            # VBA lines 982-984: Temperatures and hot water volume
            aggregated['theta_i'][t - 1] += dwelling.building.get_internal_temperature(t)
            aggregated['V_dhw'][t - 1] += dwelling.hot_water.hot_water_demand[t - 1]
            aggregated['theta_cyl'][t - 1] += dwelling.building.get_cylinder_temperature(t)

            # VBA lines 985-988: Control states (direct array access for efficiency)
            aggregated['space_timer'][t - 1] += int(dwelling.heating_controls.get_space_timer_state(t))
            aggregated['water_timer'][t - 1] += int(dwelling.heating_controls.hot_water_timer[t - 1])
            aggregated['heating_on'][t - 1] += int(dwelling.heating_controls.get_space_thermostat_state(t))
            aggregated['water_heating_on'][t - 1] += int(dwelling.heating_controls.hot_water_thermostat[t - 1])

            # VBA line 989: Fuel flow
            aggregated['M_fuel'][t - 1] += dwelling.heating_system.m_fuel[t - 1]

            # VBA lines 992-993: Cooling (control states are in heating_controls)
            if dwelling.cooling_system:
                aggregated['cooling_timer'][t - 1] += int(dwelling.heating_controls.space_cooling_timer[t - 1])
                aggregated['cooling_on'][t - 1] += int(dwelling.heating_controls.space_cooling_thermostat[t - 1])

    # VBA lines 998-1019: Normalize and convert units
    for t in range(1440):
        # Normalize by population (VBA lines 998-999)
        if total_population > 0:
            aggregated['occupancy'][t] /= total_population
            aggregated['activity'][t] /= total_population

        # Convert W → kW (VBA lines 1000-1005, 1014-1015)
        aggregated['P_e'][t] /= 1000.0
        aggregated['P_pv'][t] /= 1000.0
        aggregated['P_net'][t] = aggregated['P_e'][t] - aggregated['P_pv'][t]  # VBA line 1002
        aggregated['Phi_hSpace'][t] /= 1000.0
        aggregated['Phi_hWater'][t] /= 1000.0
        aggregated['Phi_s'][t] /= 1000.0
        aggregated['Phi_collector'][t] /= 1000.0
        aggregated['P_self'][t] /= 1000.0

        # Average by dwelling count (VBA lines 1006, 1008-1012, 1018-1019)
        aggregated['theta_i'][t] /= num_dwellings
        aggregated['theta_cyl'][t] /= num_dwellings
        aggregated['space_timer'][t] /= num_dwellings
        aggregated['water_timer'][t] /= num_dwellings
        aggregated['heating_on'][t] /= num_dwellings
        aggregated['water_heating_on'][t] /= num_dwellings
        aggregated['cooling_timer'][t] /= num_dwellings
        aggregated['cooling_on'][t] /= num_dwellings

        # Per-dwelling averages (VBA lines 1016-1017)
        aggregated['mean_V_dhw'][t] = aggregated['V_dhw'][t] / num_dwellings
        aggregated['mean_V_gas'][t] = aggregated['M_fuel'][t] / num_dwellings

        # V_dhw and M_fuel kept as totals (VBA lines 1007, 1013)

    return aggregated


def save_dwelling_configs_to_csv(configs: list, output_file: Path):
    """
    Save dwelling configurations to CSV file in Excel 'Dwellings' sheet format.

    Parameters
    ----------
    configs : list[DwellingConfig]
        List of dwelling configurations to save
    output_file : Path
        Path to output CSV file
    """
    # Create header rows matching Excel format
    rows = []

    # Row 1: Header description
    rows.append(['Dwelling parameters', '', '', '', '', '', '', '', 'Appliance and water fixtures owned'])

    # Row 2: Column names
    rows.append(['Dwelling index', 'Number of residents', 'Building index',
                 'Primary heating system index', 'PV system index',
                 'Solar thermal collector index', 'Cooling system index', '', 'Chest freezer'])

    # Row 3-4: Empty rows
    rows.append(['', '', '', '', '', '', '', '', ''])
    rows.append(['', '', '', '', '', '', '', '', ''])

    # Data rows
    for config in configs:
        rows.append([
            config.dwelling_index + 1,  # Convert back to 1-based
            config.num_residents,
            config.building_index,
            config.heating_system_index,
            config.pv_system_index,
            config.solar_thermal_index,
            config.cooling_system_index,
            '',
            ''  # Appliance data not included in simplified format
        ])

    # Write to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, header=False)
    print(f"  Saved dwelling configurations to: {output_file}")


def load_dwelling_configs_from_csv(config_file: Path, country: Country, urban_rural: UrbanRural, is_weekend: bool) -> list:
    """
    Load dwelling configurations from CSV file.

    CSV format matches Excel 'Dwellings' sheet:
    - Row 1: Header description
    - Row 2: Column names (Dwelling index, Number of residents, Building index, etc.)
    - Row 3-4: Empty rows
    - Row 5+: Dwelling data

    Columns used (0-based indexing):
    - Column 0: Dwelling index (1-based in CSV)
    - Column 1: Number of residents (1-6)
    - Column 2: Building index (1-based)
    - Column 3: Primary heating system index (1-based)
    - Column 4: PV system index (0=none, 1+=system type)
    - Column 5: Solar thermal collector index (0=none, 1+=system type)
    - Column 6: Cooling system index (0=none, 1+=system type)

    Parameters
    ----------
    config_file : Path
        Path to CSV file with dwelling configurations
    country : Country
        Country enum (UK or India)
    urban_rural : UrbanRural
        Urban/Rural enum
    is_weekend : bool
        Weekend flag

    Returns
    -------
    list[DwellingConfig]
        List of dwelling configurations
    """
    df = pd.read_csv(config_file)

    # Find the data start row (skip header rows)
    # Look for first row with valid dwelling index (numeric)
    data_start_row = 0
    for i, row in df.iterrows():
        if pd.notna(row.iloc[0]) and str(row.iloc[0]).isdigit():
            data_start_row = i
            break

    configs = []
    for i in range(data_start_row, len(df)):
        row = df.iloc[i]

        # Column indices (0-based): dwelling_index=0, residents=1, building=2, heating=3, pv=4, solar=5, cooling=6
        if pd.isna(row.iloc[0]):  # No more valid data
            break

        dwelling_index = int(row.iloc[0]) - 1  # Convert to 0-based
        num_residents = int(row.iloc[1])
        building_index = int(row.iloc[2])
        heating_system_index = int(row.iloc[3])
        pv_system_index = int(row.iloc[4]) if pd.notna(row.iloc[4]) else 0
        solar_thermal_index = int(row.iloc[5]) if pd.notna(row.iloc[5]) else 0
        cooling_system_index = int(row.iloc[6]) if pd.notna(row.iloc[6]) else 0

        # Load appliance ownership from columns 8-38 (31 appliances)
        # Column 7 is empty separator in VBA format
        appliance_ownership = None
        if len(row) > 8:  # Has appliance columns
            # Extract columns 8-38 (31 appliances)
            appliance_ownership = []
            for col_idx in range(8, min(39, len(row))):
                val = row.iloc[col_idx]
                if pd.notna(val):
                    # Convert to bool: 1 or -1 means True, 0 means False
                    appliance_ownership.append(bool(int(float(val))))
                else:
                    appliance_ownership.append(False)

            # Ensure we have exactly 31 values (pad with False if needed)
            while len(appliance_ownership) < 31:
                appliance_ownership.append(False)
            appliance_ownership = appliance_ownership[:31]  # Truncate if too many

        config = DwellingConfig(
            dwelling_index=dwelling_index,
            num_residents=num_residents,
            building_index=building_index,
            heating_system_index=heating_system_index,
            pv_system_index=pv_system_index,
            solar_thermal_index=solar_thermal_index,
            cooling_system_index=cooling_system_index,
            country=country,
            urban_rural=urban_rural,
            is_weekend=is_weekend,
            appliance_ownership=appliance_ownership
        )
        configs.append(config)

    return configs


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
        default=None,
        help="Number of residents for single dwelling simulations (default: stochastic)"
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
    parser.add_argument(
        "--year",
        type=int,
        default=2006,
        help="Year for India database interpolation (2006-2031, default: 2006). UK ignores this parameter."
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="CSV file with dwelling configurations (default: None, use stochastic assignment). "
             "Format: same as Excel 'Dwellings' sheet with columns for dwelling index, residents, "
             "building, heating, PV, solar thermal, and cooling system indices."
    )
    parser.add_argument(
        "--save-dwelling-config",
        action="store_true",
        help="Save stochastically generated dwelling configurations to CSV in output directory"
    )

    args = parser.parse_args()

    # Convert string arguments to enums
    country = Country(args.country)
    city = City(args.city)
    urban_rural = UrbanRural(args.urban_rural)

    # Database selection for proportions
    # VBA Reference: SetApplianceDatabase, SetBuildingProportions, etc. (lines 548-754)
    # NOTE: The VBA functions select appropriate database columns based on country/urban-rural/year.
    # For UK: Uses column B directly (already in CSVs)
    # For India: Interpolates between year columns (2006-2031) based on --year parameter
    # Current CSV files have UK data in column B and zeros for India columns.
    # The load_*_proportions() methods in loader.py read from column B, which works correctly for UK.
    # India interpolation would require implementing year-based column selection in the loader.
    if country != Country.UK:
        print("ERROR: The year-based interpolation for India simulations (2006-2031) is not implemented.")
        print("       This affects appliance ownership, building types, heating systems, and cooling systems")
        print("       for India simulations only. UK simulations are fully functional.")
        sys.exit(1)

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
    # Pass the global RNG for reproducible climate generation
    global_climate = GlobalClimate(climate_config, data_loader, rng_module.get_rng())
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

    # Simulate dwellings
    # VBA Reference: RunThermalElectricalDemandModel main loop (lines 282-488)
    results = []
    dwellings = []  # Store dwelling objects for output

    # Load dwelling configurations from file or generate stochastically
    dwelling_configs = []
    if args.config_file:
        # Load configurations from CSV file
        if not args.config_file.exists():
            print(f"ERROR: Config file not found: {args.config_file}")
            sys.exit(1)

        print(f"Loading dwelling configurations from: {args.config_file}")
        dwelling_configs = load_dwelling_configs_from_csv(
            args.config_file,
            country,
            urban_rural,
            args.weekend
        )
        print(f"  Loaded {len(dwelling_configs)} dwelling configurations")

        # Override num_dwellings if config file specifies different number
        if args.num_dwellings != len(dwelling_configs):
            print(f"  Note: --num-dwellings ({args.num_dwellings}) overridden by config file ({len(dwelling_configs)} dwellings)")
            args.num_dwellings = len(dwelling_configs)
    else:
        # Generate dwelling configurations stochastically
        # Create RNG for stochastic parameter assignment
        param_rng = rng_module.get_rng()

        for dwelling_idx in range(args.num_dwellings):
            # VBA Reference: AssignDwellingParameters (lines 1130-1288)
            # VBA line 296: If wsMain.Shapes("objAssignDwellingParameters").ControlFormat.Value = 1
            if args.num_dwellings == 1 and args.residents is not None:
                # Single dwelling with --residents specified: Use manual configuration
                dwelling_config = DwellingConfig(
                    dwelling_index=dwelling_idx,
                    num_residents=args.residents,
                    building_index=1,  # Use building index 1 (1-based indexing)
                    heating_system_index=1,  # Use heating system index 1 (1-based indexing)
                    country=country,
                    urban_rural=urban_rural,
                    cooling_system_index=0,
                    pv_system_index=0,  # No PV by default
                    solar_thermal_index=0,  # No solar thermal by default
                    is_weekend=args.weekend
                )
            else:
                # Multi-dwelling or no --residents specified: Use stochastic generation
                # VBA line 298: AssignDwellingParameters intDwellingIndex
                dwelling_config = assign_dwelling_parameters(
                    data_loader=data_loader,
                    dwelling_index=dwelling_idx,
                    rng=param_rng
                )
                # Override country/urban_rural/weekend from CLI args
                dwelling_config.country = country
                dwelling_config.urban_rural = urban_rural
                dwelling_config.is_weekend = args.weekend

            dwelling_configs.append(dwelling_config)

        # Save dwelling configs if requested
        if args.save_dwelling_config and args.output_dir:
            config_output_file = args.output_dir / "dwellings_config.csv"
            save_dwelling_configs_to_csv(dwelling_configs, config_output_file)

    # Simulate all dwellings
    for dwelling_idx, dwelling_config in enumerate(dwelling_configs):
        print(f"\nSimulating dwelling {dwelling_idx + 1}/{args.num_dwellings}...")

        print(f"  Config: {dwelling_config.num_residents} residents, "
              f"building {dwelling_config.building_index}, "
              f"heating {dwelling_config.heating_system_index}, "
              f"PV {dwelling_config.pv_system_index}, "
              f"solar thermal {dwelling_config.solar_thermal_index}, "
              f"cooling {dwelling_config.cooling_system_index}")

        # Create and run dwelling simulation
        # Pass the global RNG so all dwellings share the same seeded random sequence
        dwelling = Dwelling(
            dwelling_config,
            global_climate,
            data_loader,
            activity_statistics,
            rng_module.get_rng()
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
        proportion_actively_occupied = dwelling.occupancy.get_proportion_actively_occupied()

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

    # Aggregate results across all dwellings
    # VBA Reference: AggregateResults (lines 810-1048)
    if args.num_dwellings > 1:
        print("\nAggregating results across all dwellings...")
        aggregated = aggregate_results(dwellings)
        print(f"  Aggregated {len(aggregated)} time series variables (1440 timesteps each)")
        print(f"  Total population: {sum(d.config.num_residents for d in dwellings)} residents")
        print(f"  Peak total demand: {np.max(aggregated['P_e']):.2f} kW")
        print(f"  Average indoor temperature: {np.mean(aggregated['theta_i']):.1f}°C")

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
