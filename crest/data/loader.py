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
            Transition probability matrix
        """
        if not 1 <= num_residents <= 6:
            raise ValueError(f"num_residents must be 1-6, got {num_residents}")

        day_type = "we" if is_weekend else "wd"
        filename = f"tpm{num_residents}_{day_type}.csv"
        return self._load_csv(filename, header=None)

    def load_starting_states(self) -> pd.DataFrame:
        """Load initial occupancy state probabilities."""
        return self._load_csv("Starting_states.csv")

    def load_24hr_occupancy(self) -> pd.DataFrame:
        """Load 24-hour occupancy correction factors."""
        return self._load_csv("24hr_occupancy.csv")

    # ===============================================================================================
    # ACTIVITY AND APPLIANCE DATA
    # ===============================================================================================

    def load_activity_stats(self) -> pd.DataFrame:
        """Load time-use survey activity probability profiles."""
        return self._load_csv("ActivityStats.csv")

    def load_appliances_and_fixtures(self) -> pd.DataFrame:
        """Load appliance and water fixture specifications."""
        return self._load_csv("AppliancesAndWaterFixtures.csv")

    # ===============================================================================================
    # BUILDING AND HEATING SYSTEM DATA
    # ===============================================================================================

    def load_buildings(self) -> pd.DataFrame:
        """Load building thermal parameter specifications."""
        return self._load_csv("Buildings.csv")

    def load_primary_heating_systems(self) -> pd.DataFrame:
        """Load heating system specifications (boilers, etc.)."""
        return self._load_csv("PrimaryHeatingSystems.csv")

    def load_cooling_systems(self) -> pd.DataFrame:
        """Load cooling system specifications."""
        return self._load_csv("CoolingSystems.csv")

    def load_heating_controls(self) -> pd.DataFrame:
        """Load heating control specifications (thermostats, timers)."""
        return self._load_csv("HeatingControls.csv")

    def load_heating_controls_tpm(self) -> pd.DataFrame:
        """Load transition probability matrix for heating timer states."""
        return self._load_csv("HeatingControlsTPM.csv", header=None)

    # ===============================================================================================
    # CLIMATE DATA
    # ===============================================================================================

    def load_global_climate(self) -> pd.DataFrame:
        """Load historical climate data (temperature, irradiance)."""
        return self._load_csv("GlobalClimate.csv")

    def load_irradiance(self) -> pd.DataFrame:
        """Load solar irradiance data."""
        return self._load_csv("Irradiance.csv")

    def load_clearness_index_tpm(self) -> pd.DataFrame:
        """Load transition probability matrix for clearness index."""
        return self._load_csv("ClearnessIndexTPM.csv", header=None)

    def load_climate_data_and_cooling_tech(self) -> pd.DataFrame:
        """Load regional climate data and cooling technology info."""
        return self._load_csv("ClimateDataandCoolingTech.csv")

    # ===============================================================================================
    # LIGHTING DATA
    # ===============================================================================================

    def load_lighting_config(self) -> pd.DataFrame:
        """Load lighting configuration parameters."""
        return self._load_csv("light_config.csv")

    def load_bulbs(self) -> pd.DataFrame:
        """Load example bulb configurations for dwellings."""
        return self._load_csv("bulbs.csv")

    # ===============================================================================================
    # HOT WATER DATA
    # ===============================================================================================

    def load_water_usage(self) -> pd.DataFrame:
        """Load hot water volume probability distributions."""
        return self._load_csv("WaterUsage.csv")

    # ===============================================================================================
    # RENEWABLE SYSTEMS DATA
    # ===============================================================================================

    def load_pv_systems(self) -> pd.DataFrame:
        """Load PV system specifications."""
        return self._load_csv("PV_systems.csv")

    def load_solar_thermal_systems(self) -> pd.DataFrame:
        """Load solar thermal system specifications."""
        return self._load_csv("SolarThermalSystems.csv")

    # ===============================================================================================
    # DWELLING CONFIGURATION
    # ===============================================================================================

    def load_dwellings(self) -> pd.DataFrame:
        """Load dwelling configuration and assignments."""
        return self._load_csv("Dwellings.csv")

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
