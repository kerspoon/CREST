"""
CREST Demand Model - Photovoltaic System

Photovoltaic (PV) system model for electricity generation from solar radiation.

VBA Source: original/clsPVSystem.cls (398 lines)
CSV Data: data/PV_systems.csv (3 PV system configurations)

Key Features:
- Solar geometry calculations for incident radiation on tilted panels
- PV electricity generation (P = G × A × η)
- Net demand calculation (appliances + lighting - PV)
- Self-consumption calculation (min of generation and demand)

Execution: Runs ONCE per day in pre-simulation phase (not in thermal loop)
"""

import numpy as np
from typing import TYPE_CHECKING

from crest.core.solar import SolarGeometry

if TYPE_CHECKING:
    from crest.data.loader import CRESTDataLoader
    from crest.utils.random import RandomGenerator
    from crest.core.climate import LocalClimate
    from crest.core.appliances import Appliances
    from crest.core.lighting import Lighting


class PVSystem:
    """
    Photovoltaic system model for solar electricity generation.

    Calculates incident solar radiation on a tilted PV array and converts
    it to electrical power output. Also calculates net dwelling electricity
    demand and self-consumption of PV generation.

    VBA Source: clsPVSystem.cls (398 lines)
    """

    def __init__(self, data_loader: 'CRESTDataLoader', random_gen: 'RandomGenerator'):
        """
        Initialize PV system model.

        Args:
            data_loader: CREST data loader for CSV files
            random_gen: Random number generator (not used but standard interface)

        VBA Source: Class initialization (lines 30-60)
        """
        self.data_loader = data_loader
        self.random_gen = random_gen

        # Index variables (VBA lines 31-34)
        self.dwelling_index: int = 0
        self.pv_system_index: int = 0
        self.run_number: int = 0

        # PV system parameters (VBA lines 36-47)
        self.A_array: float = 0.0  # Array size (m²)
        self.eta_pv: float = 0.0  # PV efficiency (0-1)
        self.slope: float = 0.0  # Panel tilt angle (degrees)
        self.azimuth: float = 0.0  # Panel azimuth angle (degrees, 0=South)

        # Output arrays - 1440 timesteps (VBA lines 50-60)
        # Note: VBA uses 1-based arrays (1 To 1440), Python uses 0-based
        self.G_i = np.zeros(1440)  # Net radiation incident on PV array (W/m²)
        self.P_pv = np.zeros(1440)  # Electricity output of PV system (W)
        self.P_net = np.zeros(1440)  # Net dwelling electricity demand (W)
        self.P_self = np.zeros(1440)  # Dwelling self-consumption (W)

        # Component references (set by initialize)
        self.climate: 'LocalClimate' = None
        self.appliances: 'Appliances' = None
        self.lighting: 'Lighting' = None

        # Location parameters (set by initialize from dwelling config)
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.meridian: float = 0.0
        self.day_of_year: int = 0

    def initialize(self,
                   dwelling_index: int,
                   run_number: int,
                   climate: 'LocalClimate',
                   appliances: 'Appliances',
                   lighting: 'Lighting',
                   pv_system_index: int = None,
                   latitude: float = 52.2,
                   longitude: float = -0.9,
                   meridian: float = 0.0,
                   day_of_year: int = None,
                   month: int = 6,
                   day: int = 15) -> None:
        """
        Initialize PV system with configuration and component references.

        Args:
            dwelling_index: Dwelling index (0-based in Python)
            run_number: Simulation run number
            climate: GlobalClimate instance
            appliances: Appliances instance
            lighting: Lighting instance
            pv_system_index: PV system configuration index (0=no PV, 1-3=PV types)
                           If None, will be loaded from CSV in future
            latitude: Latitude in degrees (default: 52.2° N - Loughborough, UK)
            longitude: Longitude in degrees (default: -0.9° W)
            meridian: Standard meridian for timezone (default: 0° - Greenwich)
            day_of_year: Day of year (1-365). If None, calculated from month/day
            month: Month of year (1-12) - used if day_of_year is None
            day: Day of month (1-31) - used if day_of_year is None

        VBA Source: InitialisePVSystem (lines 86-113)
        """
        # Store indexes (VBA lines 91-92)
        self.dwelling_index = dwelling_index
        self.run_number = run_number

        # Store component references
        self.climate = climate
        self.appliances = appliances
        self.lighting = lighting

        # Store location parameters (VBA lines 203-205)
        self.latitude = latitude
        self.longitude = longitude
        self.meridian = meridian

        # Calculate day of year if not provided (VBA lines 208-211)
        if day_of_year is None:
            # Simple approximation: month_start_day + day
            month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            self.day_of_year = month_days[month - 1] + day
        else:
            self.day_of_year = day_of_year

        # Determine PV system index (VBA lines 94-98)
        # If pv_system_index provided, use it; otherwise default to 0 (no PV)
        if pv_system_index is not None:
            self.pv_system_index = pv_system_index
        else:
            # Future: Load from dwelling configuration CSV
            self.pv_system_index = 0

        # Load PV system parameters if PV is installed (VBA lines 101-111)
        if self.pv_system_index > 0:
            self._load_pv_system_config()

    def _load_pv_system_config(self) -> None:
        """
        Load PV system configuration from CSV file.

        Reads array size, efficiency, azimuth, and slope from PV_systems.csv.

        VBA Source: InitialisePVSystem (lines 104-109)
        CSV: data/PV_systems.csv
        Header rows: 4 (rows 0-3)
        Data starts: row 4
        Columns: PV system index, Proportion, Array area (m²), Efficiency, Slope (°), Azimuth (°)
        """
        pv_systems = self.data_loader.load_pv_systems()

        # VBA uses offset=4 (4 header rows), then indexes with intPVSystemIndex
        # CSV structure: row 4 = system 1, row 5 = system 2, etc.
        # Python 0-based: pv_system_index 1 → row 0 of data (after skipping headers)
        system_row = self.pv_system_index - 1  # Convert 1-based to 0-based

        if system_row < 0 or system_row >= len(pv_systems):
            raise ValueError(f"Invalid PV system index: {self.pv_system_index}")

        # Extract PV system parameters (VBA lines 105-108)
        # CSV columns: 0=index, 1=proportion, 2=A_array, 3=eta_pv, 4=slope, 5=azimuth
        self.A_array = float(pv_systems.iloc[system_row]['Array area'])
        self.eta_pv = float(pv_systems.iloc[system_row]['System efficiency'])
        self.azimuth = float(pv_systems.iloc[system_row]['Azimuth of panel'])
        self.slope = float(pv_systems.iloc[system_row]['Slope of panel'])

    def calculate_pv_output(self) -> None:
        """
        Calculate PV output for all 1440 timesteps of the day.

        This method runs ONCE per day in the pre-simulation phase (before thermal loop).

        Calculates:
        1. Solar geometry (altitude, azimuth, incident angle)
        2. Incident radiation on tilted panel (direct, diffuse, reflected)
        3. PV electrical output: P_pv = G_i × A_array × eta_pv

        VBA Source: CalculatePVOutput (lines 122-337)
        """
        # Check if dwelling has PV (VBA lines 214-218)
        if self.pv_system_index == 0:
            # No PV system - set all outputs to zero
            self.P_pv[:] = 0.0
            self.G_i[:] = 0.0
            return

        # Create solar geometry calculator (VBA lines 221-231)
        solar_geom = SolarGeometry(
            day_of_year=self.day_of_year,
            latitude=self.latitude,
            longitude=self.longitude,
            meridian=self.meridian,
            enable_daylight_saving=True  # VBA line 241
        )

        # Get clearsky radiation and clearness index from climate
        # VBA lines 297, 320: aLocalClimate(intRunNumber).GetG_oClearsky(intCount)
        G_o_clearsky_array = np.array([
            self.climate.get_clearsky_irradiance(timestep)
            for timestep in range(1, 1441)  # VBA 1-based timesteps
        ])

        clearness_index_array = np.array([
            self.climate.get_clearness_index(timestep)
            for timestep in range(1, 1441)  # VBA 1-based timesteps
        ])

        # Calculate incident radiation for all timesteps (VBA lines 237-335)
        radiation = solar_geom.calculate_all_day_radiation(
            slope=self.slope,
            azimuth=self.azimuth,
            G_o_clearsky_array=G_o_clearsky_array,
            clearness_index_array=clearness_index_array
        )

        # Store incident radiation (VBA line 324)
        self.G_i = radiation['G_incident']

        # Calculate PV output: P = G × A × η (VBA line 328)
        self.P_pv = self.G_i * self.A_array * self.eta_pv

    def calculate_net_demand(self) -> None:
        """
        Calculate net dwelling electricity demand.

        Net demand = Appliances + Lighting - PV generation

        Positive values: net import from grid
        Negative values: net export to grid

        VBA Source: CalculateNetDemand (lines 346-363)
        """
        # Loop through all 1440 minutes (VBA lines 353-361)
        for timestep_0based in range(1440):
            timestep_1based = timestep_0based + 1

            # Get appliance and lighting demand (VBA lines 356-357)
            total_appliance_demand = self.appliances.get_total_demand(timestep_1based)
            total_lighting_demand = self.lighting.get_total_demand(timestep_1based)

            # Calculate net demand (VBA line 359)
            self.P_net[timestep_0based] = (
                total_appliance_demand +
                total_lighting_demand -
                self.P_pv[timestep_0based]
            )

    def calculate_self_consumption(self) -> None:
        """
        Calculate PV self-consumption.

        Self-consumption = min(PV generation, Appliances + Lighting)

        This represents the portion of PV generation consumed on-site
        rather than exported to the grid.

        VBA Source: CalculateSelfConsumption (lines 372-383)
        """
        # Loop through all 1440 minutes (VBA lines 375-382)
        for timestep_0based in range(1440):
            timestep_1based = timestep_0based + 1

            # Get appliance and lighting demand (VBA line 378)
            total_demand = (
                self.lighting.get_total_demand(timestep_1based) +
                self.appliances.get_total_demand(timestep_1based)
            )

            # Self-consumption is minimum of demand and generation (VBA lines 376-380)
            self.P_self[timestep_0based] = min(total_demand, self.P_pv[timestep_0based])

    # ===============================================================================================
    # Getter Methods (VBA Property Get)
    # ===============================================================================================

    def get_pv_output(self, timestep: int) -> float:
        """
        Get PV output at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            PV power output (W)

        Note: Not in VBA, added for consistency with other components
        """
        return self.P_pv[timestep - 1]

    def get_incident_radiation(self, timestep: int) -> float:
        """
        Get incident radiation at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Incident radiation (W/m²)

        Note: Not in VBA, added for consistency with other components
        """
        return self.G_i[timestep - 1]

    def get_net_demand(self, timestep: int) -> float:
        """
        Get net electricity demand at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Net electricity demand (W)

        Note: Not in VBA, added for consistency with other components
        """
        return self.P_net[timestep - 1]

    def get_self_consumption(self, timestep: int) -> float:
        """
        Get self-consumption at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Self-consumption (W)

        Note: Not in VBA, added for consistency with other components
        """
        return self.P_self[timestep - 1]

    def get_daily_sum_pv_output(self) -> float:
        """
        Get daily sum of PV output.

        Returns:
            Total daily PV generation (W·min or Wh/60)

        VBA Source: Property GetDailySumPvOutput (lines 66-68)
        """
        return np.sum(self.P_pv)

    def get_daily_sum_net_demand(self) -> float:
        """
        Get daily sum of net electricity demand.

        Returns:
            Total daily net demand (W·min or Wh/60)

        VBA Source: Property GetDailySumP_net (lines 70-72)
        """
        return np.sum(self.P_net)

    def get_daily_sum_self_consumption(self) -> float:
        """
        Get daily sum of self-consumption.

        Returns:
            Total daily self-consumption (W·min or Wh/60)

        VBA Source: Property GetDailySumP_self (lines 74-76)
        """
        return np.sum(self.P_self)
