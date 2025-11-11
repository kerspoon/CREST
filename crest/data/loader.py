"""
Data Loading Utilities

Loads all CSV data files for the CREST Demand Model including:
- Transition probability matrices (TPMs)
- Building and system configurations
- Activity statistics and climate data
- Dwelling specifications
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class CRESTDataLoader:
    """
    Central data loader for all CREST model CSV files.

    Handles loading and caching of all configuration files, transition probability
    matrices, and reference data needed for the simulation.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.

        Parameters
        ----------
        data_dir : Path, optional
            Directory containing CSV data files. If None, uses default location.
        """
        if data_dir is None:
            # Default to data directory relative to this file
            package_root = Path(__file__).parent.parent.parent
            data_dir = package_root / "data"

        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Cache for loaded data
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file with caching.

        Parameters
        ----------
        filename : str
            Name of the CSV file to load
        **kwargs
            Additional arguments passed to pd.read_csv()

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        if filename not in self._cache:
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            self._cache[filename] = pd.read_csv(filepath, **kwargs)

        return self._cache[filename]

    # ===============================================================================================
    # OCCUPANCY DATA
    # ===============================================================================================

    def load_occupancy_tpm(self, num_residents: int, is_weekend: bool) -> pd.DataFrame:
        """
        Load transition probability matrix for occupancy model.

        Parameters
        ----------
        num_residents : int
            Number of residents (1-6)
        is_weekend : bool
            True for weekend, False for weekday

        Returns
        -------
        pd.DataFrame
            Transition probability matrix with state labels in row 0
        """
        if not 1 <= num_residents <= 6:
            raise ValueError(f"num_residents must be 1-6, got {num_residents}")

        day_type = "we" if is_weekend else "wd"
        filename = f"tpm{num_residents}_{day_type}.csv"
        # Skip 9 header/description rows (lines 1-9), keep state labels (line 10) as row 0
        # Use dtype=str to preserve state labels as strings (e.g., "00", "01", "10")
        df = self._load_csv(filename, skiprows=9, header=None, dtype=str)

        # Convert data rows (all except row 0) back to float for probabilities
        for i in range(1, len(df)):
            df.iloc[i, :] = pd.to_numeric(df.iloc[i, :], errors='coerce')

        # Fix state labels in row 0: convert "1.0" → "01", "2.0" → "02", etc.
        for col in range(2, len(df.columns)):
            label = df.iloc[0, col]
            if label and label != 'nan':
                # Convert float strings to proper 2-digit format
                try:
                    val = float(label)
                    df.iloc[0, col] = f"{int(val):02d}"
                except:
                    pass

        return df

    def load_starting_states(self) -> pd.DataFrame:
        """Load initial occupancy state probabilities."""
        # Skip 4 header rows, use row 5 as column headers
        return self._load_csv("Starting_states.csv", skiprows=4, header=0)

    def load_24hr_occupancy(self) -> pd.DataFrame:
        """Load 24-hour occupancy correction factors."""
        # Skip 2 header rows, use row 3 as column headers
        return self._load_csv("24hr_occupancy.csv", skiprows=2, header=0)

    # ===============================================================================================
    # ACTIVITY AND APPLIANCE DATA
    # ===============================================================================================

    def load_activity_stats(self) -> pd.DataFrame:
        """Load time-use survey activity probability profiles."""
        # Skip 3 header rows, use row 4 as column headers
        return self._load_csv("ActivityStats.csv", skiprows=3, header=0)

    def load_appliances_and_fixtures(self) -> pd.DataFrame:
        """Load appliance and water fixture specifications."""
        # Skip 3 header rows, use row 4 as column headers
        return self._load_csv("AppliancesAndWaterFixtures.csv", skiprows=3, header=0)

    # ===============================================================================================
    # BUILDING AND HEATING SYSTEM DATA
    # ===============================================================================================

    def load_buildings(self) -> pd.DataFrame:
        """Load building thermal parameter specifications."""
        # Skip 2 title rows, use row 3 (symbol names) as column headers, skip row 4 (units)
        df = self._load_csv("Buildings.csv", skiprows=[0, 1, 3], header=0)

        # Rename columns to match Python naming convention (add underscores)
        # VBA uses: Hob, Hbi, Cb, Ci, As, Hv, Hem, Cem, etc.
        # Python expects: H_ob, H_bi, C_b, C_i, A_s, H_v, H_em, C_em, etc.
        #
        # Note: CSV has unnamed columns for cooling system (columns 17-19):
        # - Column 17: θcool (Unnamed: 17) - nominal temperature of coolers
        # - Column 18: Hemcool (appears as Hem.1) - heat transfer coefficient for cooling
        # - Column 19: Cemcool (Unnamed: 19) - thermal capacitance of cooling emitters
        rename_map = {
            'Hob': 'H_ob', 'Hbi': 'H_bi', 'Cb': 'C_b', 'Ci': 'C_i',
            'As': 'A_s', 'Hv': 'H_v', 'Hem': 'H_em', 'Cem': 'C_em',
            'mem': 'm_em', 'Hem.1': 'H_emcool'
        }
        df = df.rename(columns=rename_map)

        # Manually name the cooling capacitance column (column 19)
        # Find the unnamed column after H_emcool
        cols = list(df.columns)
        for i, col in enumerate(cols):
            if col == 'H_emcool' and i + 1 < len(cols) and 'Unnamed' in str(cols[i + 1]):
                df = df.rename(columns={cols[i + 1]: 'C_emcool'})
                break

        # Also rename the theta columns for emitters
        df = df.rename(columns={'θem': 'theta_em'})

        # Find and rename the cooling emitter nominal temperature (column 17, appears before H_emcool)
        cols = list(df.columns)
        for i, col in enumerate(cols):
            if col == 'H_emcool' and i > 0 and 'Unnamed' in str(cols[i - 1]):
                df = df.rename(columns={cols[i - 1]: 'theta_cool'})
                break

        return df

    def load_primary_heating_systems(self) -> pd.DataFrame:
        """Load heating system specifications (boilers, etc.)."""
        # Skip title, long descriptions, and units rows; use symbols row as header
        # Row 0: Title
        # Row 1: Long column descriptions
        # Row 2: Short symbols (use as header)
        # Row 3: Units
        # Row 4+: Data
        df = self._load_csv("PrimaryHeatingSystems.csv", skiprows=[0, 1, 3], header=0)

        # Rename columns for consistency
        rename_map = {
            'Vcyl': 'V_cyl',
            'Hloss': 'H_loss'
        }
        df = df.rename(columns=rename_map)
        return df

    def load_cooling_systems(self) -> pd.DataFrame:
        """Load cooling system specifications."""
        # Skip 4 header rows, use row 5 as column headers
        return self._load_csv("CoolingSystems.csv", skiprows=4, header=0)

    def load_heating_controls(self) -> pd.DataFrame:
        """Load heating control specifications (thermostats, timers)."""
        # CSV structure:
        # Rows 0-2: Titles
        # Row 3: "Demand temperature,Percentage of homes" (space heating header)
        # Rows 4-18: Space heating data (15 rows)
        # Row 19: blank
        # Rows 20-22: Hot water titles
        # Row 23: "Hot water delivery temperature,Percentage of homes" (hot water header)
        # Rows 24-35: Hot water data (12 rows)
        # Rows 36-40: Cooling offset info

        # Load without header to get raw data
        df = self._load_csv("HeatingControls.csv", header=None)

        # Extract space heating setpoints (rows 4-18, columns 0-1)
        space_heating = df.iloc[4:19, 0:2].copy()
        space_heating.columns = ['temperature', 'probability']
        space_heating = space_heating.reset_index(drop=True)

        # Extract hot water setpoints (rows 24-35, columns 0-1)
        hot_water = df.iloc[24:36, 0:2].copy()
        hot_water.columns = ['temperature', 'probability']
        hot_water = hot_water.reset_index(drop=True)

        # Extract cooling offset (row 37, column 2)
        # Row 37 (0-based) = "Adapted higher than the heating settings, shifted by:",,5.0,degrees
        cooling_offset = float(df.iloc[37, 2])

        # Store in dict for easy access
        result = pd.DataFrame({
            'space_heating_temps': [space_heating['temperature'].values],
            'space_heating_probs': [space_heating['probability'].values],
            'hot_water_temps': [hot_water['temperature'].values],
            'hot_water_probs': [hot_water['probability'].values],
            'cooling_offset': [cooling_offset]
        })

        return result

    def load_heating_controls_tpm(self) -> pd.DataFrame:
        """Load transition probability matrix for heating timer states."""
        # Skip 7 header rows, don't use any row as header (all numeric data)
        return self._load_csv("HeatingControlsTPM.csv", skiprows=7, header=None)

    # ===============================================================================================
    # CLIMATE DATA
    # ===============================================================================================

    def load_global_climate(self) -> pd.DataFrame:
        """Load historical climate data (temperature, irradiance)."""
        # Skip 4 header rows, use row 5 as column headers
        return self._load_csv("GlobalClimate.csv", skiprows=4, header=0)

    def load_irradiance(self) -> pd.DataFrame:
        """Load solar irradiance data."""
        # Skip 2 header rows, use row 3 as column headers
        return self._load_csv("Irradiance.csv", skiprows=2, header=0)

    def load_clearness_index_tpm(self) -> pd.DataFrame:
        """Load transition probability matrix for clearness index."""
        # Skip 9 header rows (lines 1-9), data starts at line 10
        return self._load_csv("ClearnessIndexTPM.csv", skiprows=9, header=None)

    def load_climate_data_and_cooling_tech(self) -> pd.DataFrame:
        """Load regional climate data and cooling technology info."""
        # Skip 3 header rows, use row 4 as column headers
        return self._load_csv("ClimateDataandCoolingTech.csv", skiprows=3, header=0)

    # ===============================================================================================
    # LIGHTING DATA
    # ===============================================================================================

    def load_lighting_config(self) -> pd.DataFrame:
        """Load lighting configuration parameters."""
        # Skip 2 header rows, use row 3 as column headers
        return self._load_csv("light_config.csv", skiprows=2, header=0)

    def load_bulbs(self) -> pd.DataFrame:
        """Load example bulb configurations for dwellings."""
        # Skip 10 header/description rows (rows 1-10 in 1-indexed counting)
        # Row 10 has "Number,count,..." which are sub-labels, skip it too
        # Data starts at row 11 (1-indexed) = row 10 (0-indexed)
        # With skiprows=10, row 10 (file) becomes index 0 in DataFrame
        return self._load_csv("bulbs.csv", skiprows=10, header=None)

    # ===============================================================================================
    # HOT WATER DATA
    # ===============================================================================================

    def load_water_usage(self) -> pd.DataFrame:
        """Load hot water volume probability distributions."""
        # Skip 5 header rows, use row 6 as column headers
        return self._load_csv("WaterUsage.csv", skiprows=5, header=0)

    # ===============================================================================================
    # RENEWABLE SYSTEMS DATA
    # ===============================================================================================

    def load_pv_systems(self) -> pd.DataFrame:
        """Load PV system specifications."""
        # Skip 4 header rows, use row 5 as column headers
        return self._load_csv("PV_systems.csv", skiprows=4, header=0)

    def load_solar_thermal_systems(self) -> pd.DataFrame:
        """Load solar thermal system specifications."""
        # Skip 3 header rows, use row 4 as column headers
        return self._load_csv("SolarThermalSystems.csv", skiprows=3, header=0)

    # ===============================================================================================
    # DWELLING CONFIGURATION
    # ===============================================================================================

    def load_dwellings(self) -> pd.DataFrame:
        """Load dwelling configuration and assignments."""
        # Skip 3 header rows, use row 4 as column headers
        return self._load_csv("Dwellings.csv", skiprows=3, header=0)

    # ===============================================================================================
    # UTILITY FUNCTIONS
    # ===============================================================================================

    def get_numpy_array(self, filename: str, **kwargs) -> np.ndarray:
        """
        Load a CSV file and return as numpy array.

        Parameters
        ----------
        filename : str
            Name of the CSV file
        **kwargs
            Additional arguments passed to pd.read_csv()

        Returns
        -------
        np.ndarray
            Data as numpy array
        """
        df = self._load_csv(filename, **kwargs)
        return df.values

    def clear_cache(self):
        """Clear the data cache to free memory."""
        self._cache.clear()

    def preload_all(self):
        """
        Preload all data files into cache.

        Useful for batch simulations to avoid repeated file I/O.
        """
        # Occupancy TPMs (12 files: 6 residents × 2 day types)
        for n_res in range(1, 7):
            for is_weekend in [False, True]:
                self.load_occupancy_tpm(n_res, is_weekend)

        # Other data files
        self.load_starting_states()
        self.load_24hr_occupancy()
        self.load_activity_stats()
        self.load_appliances_and_fixtures()
        self.load_buildings()
        self.load_primary_heating_systems()
        self.load_cooling_systems()
        self.load_heating_controls()
        self.load_heating_controls_tpm()
        self.load_global_climate()
        self.load_irradiance()
        self.load_clearness_index_tpm()
        self.load_climate_data_and_cooling_tech()
        self.load_lighting_config()
        self.load_bulbs()
        self.load_water_usage()
        self.load_pv_systems()
        self.load_solar_thermal_systems()
        self.load_dwellings()


# Singleton instance for convenience
_default_loader: Optional[CRESTDataLoader] = None


def get_default_loader(data_dir: Optional[Path] = None) -> CRESTDataLoader:
    """
    Get or create the default data loader instance.

    Parameters
    ----------
    data_dir : Path, optional
        Data directory path. Only used on first call.

    Returns
    -------
    CRESTDataLoader
        The default loader instance
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = CRESTDataLoader(data_dir)
    return _default_loader
