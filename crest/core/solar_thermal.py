"""
CREST Demand Model - Solar Thermal Collector

Solar thermal collector model for hot water heating from solar radiation.

VBA Source: original/clsSolarThermal.cls (501 lines)
CSV Data: data/SolarThermalSystems.csv (2 system types: flat plate, evacuated tube)

Key Features:
- Solar geometry calculations for incident radiation on tilted collectors
- Thermal dynamics: collector temperature evolution using Euler integration
- Control logic: hysteresis thermostat for pump control
- Heat transfer to hot water cylinder
- Pump electricity demand

Execution: Runs 1440 TIMES per day in thermal loop (coupled to building physics)

Critical Note: VBA has a bug at line 428 (Tan(Declination)/Tan(Declination) should
               be Tan(Declination)/Tan(Latitude)). We use the corrected SolarGeometry
               class instead of duplicating the buggy code.
"""

import numpy as np
from typing import TYPE_CHECKING

from crest.core.solar import SolarGeometry
from crest.simulation.config import SPECIFIC_HEAT_CAPACITY_WATER

if TYPE_CHECKING:
    from crest.data.loader import CRESTDataLoader
    from crest.utils.random import RandomGenerator
    from crest.core.climate import GlobalClimate
    from crest.core.building import Building


class SolarThermal:
    """
    Solar thermal collector model for hot water heating.

    Models a solar thermal collector with thermal mass, pump control logic,
    and heat transfer to hot water cylinder. Temperature evolution is calculated
    using explicit Euler integration with 60-second timesteps.

    VBA Source: clsSolarThermal.cls (501 lines)
    """

    def __init__(self, data_loader: 'CRESTDataLoader', random_gen: 'RandomGenerator'):
        """
        Initialize solar thermal collector model.

        Args:
            data_loader: CREST data loader for CSV files
            random_gen: Random number generator (not used but standard interface)

        VBA Source: Class initialization (lines 30-97)
        """
        self.data_loader = data_loader
        self.random_gen = random_gen

        # Index variables (VBA lines 31-34)
        self.dwelling_index: int = 0
        self.solar_thermal_index: int = 0
        self.run_number: int = 0

        # Collector parameters (VBA lines 36-81)
        self.N: float = 0.0  # Number of collectors
        self.A_coll_absorb: float = 0.0  # Absorber area (m²)
        self.A_coll_aperture: float = 0.0  # Aperture area (m²)
        self.A_coll_gross: float = 0.0  # Gross area (m²)
        self.eta_zero: float = 0.0  # Efficiency curve intercept
        self.k1: float = 0.0  # Efficiency curve slope coefficient (W/K/m²)
        self.k2: float = 0.0  # Efficiency curve curvature coefficient (W/K²/m²)
        self.F_prime: float = 0.0  # Collector efficiency factor
        self.U_L_lin: float = 0.0  # Total loss coefficient linear (W/K/m²)
        self.m_pump: float = 0.0  # Pump mass flow rate (kg/s)
        self.P_pump_solar: float = 0.0  # Pump electricity demand (W)
        self.C_collector: float = 0.0  # Collector heat capacitance (J/K)
        self.tau_alpha: float = 0.0  # Transmission-absorption product
        self.slope: float = 0.0  # Collector tilt angle (degrees)
        self.azimuth: float = 0.0  # Collector azimuth angle (degrees)

        # Output arrays - 1440 timesteps (VBA lines 84-97)
        # Note: VBA uses 1-based arrays (1 To 1440), Python uses 0-based
        self.theta_collector = np.zeros(1440)  # Collector temperature (°C)
        self.phi_s = np.zeros(1440)  # Useful heat to cylinder (W)
        self.P_pump_solar_array = np.zeros(1440)  # Pump electricity (W)
        self.P_incident = np.zeros(1440)  # Incident solar power (W)
        self.solar_thermal_on_off = np.zeros(1440)  # Pump control state (0/1)

        # Component references (set by initialize)
        self.climate: 'GlobalClimate' = None
        self.building: 'Building' = None

        # Location and time parameters (set by initialize)
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.meridian: float = 0.0
        self.day_of_year: int = 0

        # Solar geometry calculator (created in initialize)
        self.solar_geom: SolarGeometry = None

    def initialize(self,
                   dwelling_index: int,
                   run_number: int,
                   climate: 'GlobalClimate',
                   building: 'Building',
                   solar_thermal_index: int = None,
                   latitude: float = 52.2,
                   longitude: float = -0.9,
                   meridian: float = 0.0,
                   day_of_year: int = None,
                   month: int = 6,
                   day: int = 15) -> None:
        """
        Initialize solar thermal system with configuration and component references.

        Args:
            dwelling_index: Dwelling index (0-based in Python)
            run_number: Simulation run number
            climate: GlobalClimate instance
            building: Building instance (for cylinder temperature feedback)
            solar_thermal_index: Solar thermal system index (0=none, 1-2=system types)
                                If None, will be loaded from CSV in future
            latitude: Latitude in degrees (default: 52.2° N - Loughborough, UK)
            longitude: Longitude in degrees (default: -0.9° W)
            meridian: Standard meridian for timezone (default: 0° - Greenwich)
            day_of_year: Day of year (1-365). If None, calculated from month/day
            month: Month of year (1-12) - used if day_of_year is None
            day: Day of month (1-31) - used if day_of_year is None

        VBA Source: InitialiseSolarThermal (lines 126-162)
        """
        # Store indexes (VBA lines 130-131)
        self.dwelling_index = dwelling_index
        self.run_number = run_number

        # Store component references
        self.climate = climate
        self.building = building

        # Store location parameters (VBA lines 379-381 in GetIncidentRadiation)
        # Note: VBA bug - dblDayOfYear used but never set. We fix this.
        self.latitude = latitude
        self.longitude = longitude
        self.meridian = meridian

        # Calculate day of year if not provided
        if day_of_year is None:
            # Simple approximation: month_start_day + day
            month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            self.day_of_year = month_days[month - 1] + day
        else:
            self.day_of_year = day_of_year

        # Create solar geometry calculator (used in calculate_solar_thermal_output)
        self.solar_geom = SolarGeometry(
            day_of_year=self.day_of_year,
            latitude=self.latitude,
            longitude=self.longitude,
            meridian=self.meridian,
            enable_daylight_saving=True
        )

        # Determine solar thermal system index (VBA line 132)
        if solar_thermal_index is not None:
            self.solar_thermal_index = solar_thermal_index
        else:
            # Future: Load from dwelling configuration CSV
            self.solar_thermal_index = 0

        # Load solar thermal system parameters if installed (VBA lines 135-158)
        if self.solar_thermal_index > 0:
            self._load_solar_thermal_config()

        # Set initial temperature of collector to outdoor air temperature (VBA lines 161-162)
        # This is done here because we need climate to be set
        self.theta_collector[0] = self.climate.get_temperature(1)  # VBA timestep 1 → Python index 0

    def _load_solar_thermal_config(self) -> None:
        """
        Load solar thermal system configuration from CSV file.

        Reads 18 parameters from SolarThermalSystems.csv including number of collectors,
        areas, efficiency curve parameters, pump specs, and orientation.

        VBA Source: InitialiseSolarThermal (lines 138-156)
        CSV: data/SolarThermalSystems.csv
        Header rows: 4 (rows 0-3)
        Data starts: row 4
        Columns: 18 parameters (see VBA lines 140-154)
        """
        solar_thermal_systems = self.data_loader.load_solar_thermal_systems()

        # VBA uses offset=4 (4 header rows), then indexes with intSolarThermalIndex
        # CSV structure: row 4 = system 1, row 5 = system 2, etc.
        # Python 0-based: solar_thermal_index 1 → row 0 of data (after skipping headers)
        system_row = self.solar_thermal_index - 1  # Convert 1-based to 0-based

        if system_row < 0 or system_row >= len(solar_thermal_systems):
            raise ValueError(f"Invalid solar thermal system index: {self.solar_thermal_index}")

        # Extract solar thermal system parameters (VBA lines 140-154)
        # CSV columns match VBA .Cells(offset+index, column) access
        # Column 3 = Number of collectors, Column 4 = Aperture area, etc.
        self.N = float(solar_thermal_systems.iloc[system_row]['Number of collectors'])
        self.A_coll_aperture = float(solar_thermal_systems.iloc[system_row]['Collector aperture area'])
        self.A_coll_absorb = float(solar_thermal_systems.iloc[system_row]['Collector absorber area'])
        self.A_coll_gross = float(solar_thermal_systems.iloc[system_row]['Collector gross area'])
        self.eta_zero = float(solar_thermal_systems.iloc[system_row]['Collector efficiency intercept'])
        self.k1 = float(solar_thermal_systems.iloc[system_row]['Collector efficiency slope coefficient '])
        self.k2 = float(solar_thermal_systems.iloc[system_row]['Collector efficiency curvature coefficient '])
        self.F_prime = float(solar_thermal_systems.iloc[system_row]['Collector efficiency factor'])
        self.U_L_lin = float(solar_thermal_systems.iloc[system_row]['Total loss coefficient, linear'])
        self.m_pump = float(solar_thermal_systems.iloc[system_row]['Pump mass flow rate'])
        self.P_pump_solar = float(solar_thermal_systems.iloc[system_row]['Pump power'])
        self.C_collector = float(solar_thermal_systems.iloc[system_row]['Collector heat capacitance'])
        self.tau_alpha = float(solar_thermal_systems.iloc[system_row]['Transmission absorption product'])
        self.slope = float(solar_thermal_systems.iloc[system_row]['Slope of collector'])
        self.azimuth = float(solar_thermal_systems.iloc[system_row]['Azimuth of collector'])

    def calculate_solar_thermal_output(self, current_timestep: int) -> None:
        """
        Calculate solar thermal output for current timestep.

        This method is called 1440 TIMES per day (once per minute) in the thermal loop.

        Calculates:
        1. Incident radiation on collector using SolarGeometry
        2. Pump control state (hysteresis thermostat)
        3. Useful heat transfer to cylinder
        4. Collector temperature evolution (Euler integration)
        5. Pump electricity demand

        Args:
            current_timestep: Current timestep (1-1440, VBA 1-based)

        VBA Source: CalculateSolarThermalOutput (lines 171-280)
        """
        # Convert to 0-based index for Python arrays
        timestep_0based = current_timestep - 1

        # Constants (VBA lines 209-215)
        time_interval = 60  # seconds (VBA line 209)
        theta_max = 70.0  # Maximum collector temperature for pump shutoff (°C) (VBA line 212)
        theta_control = 2.0  # Temperature difference to start pump (°C) (VBA line 215)

        # Check if dwelling has solar thermal (VBA lines 218-224)
        if self.solar_thermal_index == 0:
            # No solar thermal system - set all outputs to zero
            self.phi_s[timestep_0based] = 0.0
            self.solar_thermal_on_off[timestep_0based] = 0.0
            self.P_pump_solar_array[timestep_0based] = 0.0
            self.P_incident[timestep_0based] = 0.0
            return

        # Get incident radiation using SolarGeometry (VBA line 228)
        # This replaces the buggy GetIncidentRadiation private function
        G_o_clearsky = self.climate.get_clearsky_irradiance(current_timestep)
        clearness_index = self.climate.get_clearness_index(current_timestep)

        radiation = self.solar_geom.calculate_incident_radiation(
            timestep=current_timestep,
            slope=self.slope,
            azimuth=self.azimuth,
            G_o_clearsky=G_o_clearsky,
            clearness_index=clearness_index
        )

        G_i = radiation['G_incident']

        # Calculate incident solar power on collector area (VBA line 231)
        self.P_incident[timestep_0based] = self.A_coll_aperture * G_i * self.N

        # Get temperatures from previous timestep (VBA lines 233-245)
        if current_timestep == 1:
            # First timestep - use current values (VBA lines 236-239)
            theta_cyl = self.building.get_cylinder_temperature(current_timestep)
            theta_collector = self.theta_collector[timestep_0based]
            theta_o = self.climate.get_temperature(current_timestep)
        else:
            # Use previous timestep values (VBA lines 241-243)
            theta_cyl = self.building.get_cylinder_temperature(current_timestep - 1)
            theta_collector = self.theta_collector[timestep_0based - 1]
            theta_o = self.climate.get_temperature(current_timestep - 1)

        # Determine pump control state (hysteresis thermostat) (VBA lines 248-255)
        # Pump ON if: (T_collector - T_cylinder > 2°C) AND (T_cylinder < 70°C)
        if ((theta_collector - theta_cyl) > theta_control) and (theta_cyl < theta_max):
            self.solar_thermal_on_off[timestep_0based] = 1.0
        else:
            self.solar_thermal_on_off[timestep_0based] = 0.0

        # Calculate useful heat flow to cylinder (VBA lines 258-259)
        # Phi_s = N × pump_on × m_pump × Cp_water × (T_collector - T_cylinder)
        self.phi_s[timestep_0based] = (
            self.N *
            self.solar_thermal_on_off[timestep_0based] *
            self.m_pump *
            SPECIFIC_HEAT_CAPACITY_WATER *
            (theta_collector - theta_cyl)
        )

        # Calculate total loss coefficient of collector (VBA line 262)
        # U_L = (k1 + k2 × (T_collector - T_outdoor)) / F'
        U_L = (self.k1 + self.k2 * (theta_collector - theta_o)) / self.F_prime

        # Calculate temperature change of collector using Euler method (VBA lines 265-270)
        # Heat balance: C_collector × dT/dt = Q_solar - Q_loss - Q_extracted
        # where:
        #   Q_solar = τα × G_i × A_absorb (solar heat gain)
        #   Q_loss = A_aperture × U_L × (T_collector - T_outdoor) (heat loss)
        #   Q_extracted = Phi_s / N (heat extracted by pump per collector)
        delta_theta_collector = (time_interval / self.C_collector) * (
            self.tau_alpha * G_i * self.A_coll_absorb -
            self.A_coll_aperture * U_L * (theta_collector - theta_o) -
            (self.phi_s[timestep_0based] / self.N)
        )

        # Update collector temperature (VBA line 272)
        self.theta_collector[timestep_0based] = theta_collector + delta_theta_collector

        # Assign pump power based on control state (VBA line 275)
        self.P_pump_solar_array[timestep_0based] = (
            self.P_pump_solar * self.solar_thermal_on_off[timestep_0based]
        )

    # ===============================================================================================
    # Getter Methods (VBA Property Get)
    # ===============================================================================================

    def get_phi_s(self, timestep: int) -> float:
        """
        Get useful heat delivered to cylinder at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Useful heat to cylinder (W)

        VBA Source: Property GetPhi_s (lines 108-110)
        """
        return self.phi_s[timestep - 1]

    def get_P_pumpsolar(self, timestep: int) -> float:
        """
        Get pump electricity demand at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Pump electricity demand (W)

        VBA Source: Property GetP_pumpsolar (lines 112-114)
        """
        return self.P_pump_solar_array[timestep - 1]

    def get_collector_temperature(self, timestep: int) -> float:
        """
        Get collector temperature at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Collector temperature (°C)

        Note: Not in VBA, added for consistency
        """
        return self.theta_collector[timestep - 1]

    def get_incident_power(self, timestep: int) -> float:
        """
        Get incident solar power at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Incident solar power (W)

        Note: Not in VBA, added for consistency
        """
        return self.P_incident[timestep - 1]

    def get_pump_state(self, timestep: int) -> float:
        """
        Get pump control state at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Pump state (0=off, 1=on)

        Note: Not in VBA, added for consistency
        """
        return self.solar_thermal_on_off[timestep - 1]

    def get_daily_sum_phi_s(self) -> float:
        """
        Get daily sum of useful heat delivered to cylinder.

        Returns:
            Total daily heat delivered (W·min or Wh/60)

        VBA Source: Property GetDailySumPhi_s (lines 104-106)
        """
        return np.sum(self.phi_s)
