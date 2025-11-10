"""
Global and Local Climate Models

Implements stochastic climate simulation including:
- Clearness index (cloud cover) using Markov chains
- Solar irradiance calculations
- Outdoor temperature generation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    PI,
    DAY_SUMMER_TIME_STARTS,
    DAY_SUMMER_TIME_END,
    City
)
from ..utils import markov
from ..utils.random import RandomGenerator
from ..data.loader import CRESTDataLoader


@dataclass
class ClimateConfig:
    """Configuration for climate model."""
    day_of_month: int
    month_of_year: int
    city: City = City.ENGLAND  # City/region for temperature profiles
    longitude: float = -1.26  # Loughborough, UK
    latitude: float = 52.77   # Loughborough, UK
    meridian: float = 0.0     # Greenwich meridian
    use_daylight_saving: bool = True


class GlobalClimate:
    """
    Global climate model providing shared climate data for all dwellings.

    Generates:
    - Clearness index (1-minute resolution)
    - Solar irradiance (1-minute resolution)
    - Outdoor temperature (1-minute resolution)
    """

    def __init__(
        self,
        config: ClimateConfig,
        data_loader: CRESTDataLoader,
        rng: Optional[RandomGenerator] = None
    ):
        """
        Initialize the global climate model.

        Parameters
        ----------
        config : ClimateConfig
            Climate configuration
        data_loader : CRESTDataLoader
            Data loader for TPMs and climate data
        rng : RandomGenerator, optional
            Random number generator
        """
        self.config = config
        self.data_loader = data_loader
        self.rng = rng if rng is not None else RandomGenerator()

        # Storage arrays (1-minute resolution, 1440 timesteps)
        self.clearness_index = np.zeros(TIMESTEPS_PER_DAY_1MIN)
        self.g_o_clearsky = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Clear sky beam radiation (W/m²)
        self.g_o = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Global horizontal irradiance (W/m²)
        self.theta_o = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Outside temperature (°C)

        # Temperature array for full year (used for daily average temperature model)
        self.temp_array = np.zeros((365, 7))

        # Calculate day of year
        date_str = f"{config.day_of_month}/{config.month_of_year}/2015"
        self.day_of_year = datetime.strptime(date_str, "%d/%m/%Y").timetuple().tm_yday

        # Load clearness index TPM
        self.clearness_tpm = self.data_loader.load_clearness_index_tpm().values

    def simulate_clearness_index(self):
        """
        Simulate clearness index using Markov chain.

        Clearness index k represents cloud cover (k=1 is clear sky, k=0 is fully cloudy).
        Uses 101 bins where bin 101 represents k=1.0 (clear sky).
        """
        # Start with clear sky (bin 101)
        current_bin = 100  # 0-based indexing: bin 101 in VBA = index 100

        # Set initial clearness index to 1.0 (clear sky)
        self.clearness_index[0] = 1.0

        # Simulate transitions for each minute
        for timestep in range(1, TIMESTEPS_PER_DAY_1MIN):
            # Get transition probabilities for current bin
            # TPM structure: rows are current states (bins), columns are next states
            # Skip first 2 columns which are labels (like occupancy TPM)
            transition_probs = self.clearness_tpm[current_bin, 2:].astype(float)

            # Normalize probabilities
            transition_probs = markov.normalize_probabilities(transition_probs)

            # Select next bin
            rng_value = self.rng.random()
            next_bin = markov.select_next_state(transition_probs, rng_value)
            current_bin = next_bin

            # Convert bin to clearness index value
            if current_bin == 100:  # Bin 101 in 1-based = index 100 in 0-based
                k_value = 1.0
            else:
                k_value = ((current_bin + 1) / 100.0) - 0.01

            self.clearness_index[timestep] = k_value

    def calculate_global_irradiance(self):
        """
        Calculate global horizontal irradiance for each minute of the day.

        Uses solar geometry and clearness index to compute irradiance values.
        """
        # Calculate solar geometry parameters
        b_param = 360.0 * (self.day_of_year - 81) / 364.0

        # Equation of time (minutes)
        eot = (9.87 * np.sin(2 * b_param * PI / 180.0) -
               7.53 * np.cos(b_param * PI / 180.0) -
               1.5 * np.sin(b_param * PI / 180.0))

        # Time correction factor (minutes)
        tcf = 4.0 * (self.config.longitude - self.config.meridian) + eot

        # Declination (degrees)
        declination = 23.45 * np.sin(2 * PI * (284 + self.day_of_year) / 365.25)

        # Optical depth
        optical_depth = 0.174 + 0.035 * np.sin(2 * PI * (self.day_of_year - 100) / 365.0)

        # Extraterrestrial radiation (W/m²)
        g_et = 1367.0 * (1.0 + 0.034 * np.cos(2 * PI * self.day_of_year / 365.25))

        minute_count = 0

        # Loop through each hour and minute
        for hour in range(24):
            # Apply daylight saving time if needed
            if self.config.use_daylight_saving:
                if DAY_SUMMER_TIME_STARTS <= self.day_of_year < DAY_SUMMER_TIME_END:
                    local_hour = hour - 1
                else:
                    local_hour = hour
            else:
                local_hour = hour

            for minute in range(1, 61):
                # Hours before solar noon
                hours_before_noon = 12.0 - (local_hour + minute / 60.0 + tcf / 60.0)

                # Hour angle (degrees)
                hour_angle = 15.0 * hours_before_noon

                # Solar altitude angle (degrees)
                sin_altitude = (
                    np.cos(self.config.latitude * PI / 180.0) *
                    np.cos(declination * PI / 180.0) *
                    np.cos(hour_angle * PI / 180.0) +
                    np.sin(self.config.latitude * PI / 180.0) *
                    np.sin(declination * PI / 180.0)
                )
                solar_altitude = np.arcsin(np.clip(sin_altitude, -1, 1)) * 180.0 / PI

                # Clear sky beam radiation (plane tracking sun)
                if solar_altitude > 0:
                    g_clearsky = g_et * np.exp(-optical_depth / np.sin(solar_altitude * PI / 180.0))
                else:
                    g_clearsky = 0.0

                # Store clear sky irradiance
                self.g_o_clearsky[minute_count] = g_clearsky

                # Get clearness index for this timestep
                k = self.clearness_index[minute_count]

                # Calculate global horizontal irradiance
                if solar_altitude > 0:
                    self.g_o[minute_count] = g_clearsky * k * np.sin(solar_altitude * PI / 180.0)
                else:
                    self.g_o[minute_count] = 0.0

                minute_count += 1

    def run_temperature_model(self):
        """
        Generate stochastic outdoor temperature profile.

        Uses a combination of:
        - Sinusoidal annual variation
        - Autoregressive moving average (ARMA) model for daily variation
        - Diurnal temperature profile correlated with solar irradiance
        """
        # First generate daily average temperatures for the year
        self._generate_daily_temperatures()

        # Then generate minute-by-minute temperatures for the simulation day
        self._generate_minute_temperatures()

    def _get_month_from_day(self, day: int) -> int:
        """
        Get month (1-12) from day of year (0-364).

        Matches VBA logic (clsGlobalClimate.cls lines 653-677).

        Parameters
        ----------
        day : int
            Day of year (0-based: 0-364)

        Returns
        -------
        int
            Month number (1-12)
        """
        day_1based = day + 1  # Convert to 1-based for VBA compatibility

        if day_1based < 32:
            return 1
        elif day_1based < 60:
            return 2
        elif day_1based < 91:
            return 3
        elif day_1based < 121:
            return 4
        elif day_1based < 152:
            return 5
        elif day_1based < 182:
            return 6
        elif day_1based < 213:
            return 7
        elif day_1based < 244:
            return 8
        elif day_1based < 274:
            return 9
        elif day_1based < 305:
            return 10
        elif day_1based < 335:
            return 11
        else:
            return 12

    def _generate_daily_temperatures(self):
        """
        Generate daily average temperatures for a full year using ARMA model.

        Matches VBA Td_model (clsGlobalClimate.cls lines 574-706).
        Stores results in self.temp_array for later use.
        """
        # Load monthly temperature data from CSV
        climate_data = self.data_loader.load_climate_data_and_cooling_tech()

        # Extract England monthly temps (Mean, Min, Max)
        # CSV structure after skiprows=3, header=0:
        # Row 2 onwards contains month data, columns 1-3 are England Mean/Min/Max
        monthly_mean = []
        monthly_min = []
        monthly_max = []

        # Extract 12 months of data for England (rows 2-13, columns 1-3)
        for month_idx in range(2, 14):  # Rows 2-13 in loaded DataFrame
            if month_idx < len(climate_data):
                monthly_mean.append(float(climate_data.iloc[month_idx, 1]))
                monthly_min.append(float(climate_data.iloc[month_idx, 2]))
                monthly_max.append(float(climate_data.iloc[month_idx, 3]))
            else:
                # Fallback defaults
                monthly_mean.append(10.0)
                monthly_min.append(5.0)
                monthly_max.append(15.0)

        # Temperature model parameters (England, VBA lines 640-645)
        ar_coef = 0.81    # Autoregressive coefficient (AR)
        ma_coef = 0.62    # Moving average coefficient (MA)
        sd_factor = 0.1   # SD scaling factor

        # VBA lines 652-682: Determine daily temps and random noise
        for day in range(365):
            # Get month for this day (VBA lines 653-677)
            month = self._get_month_from_day(day)
            month_idx = month - 1  # Convert to 0-based for array access

            # Daily average from monthly mean (VBA line 678)
            self.temp_array[day, 0] = monthly_mean[month_idx]

            # Calculate SD for random noise (VBA line 679)
            sd = (monthly_max[month_idx] - monthly_min[month_idx]) * sd_factor

            # Generate random noise for this day (VBA line 681)
            # VBA: NormInv(Rnd(), 0, SD)
            self.temp_array[day, 1] = self.rng.normal(0, sd)

        # Initialize ARMA components (VBA lines 684-687)
        self.temp_array[0, 2] = 0  # AR component
        self.temp_array[0, 3] = 0  # MA component
        self.temp_array[0, 4] = 0  # ARMA component

        # VBA lines 691-697: ARMA model
        for day in range(1, 365):
            # AR part: AR(t) = AR * AR(t-1) + E(t)
            self.temp_array[day, 2] = (
                self.temp_array[day - 1, 2] * ar_coef + self.temp_array[day, 1]
            )

            # MA part: MA(t) = E(t) + MA * E(t-1)
            self.temp_array[day, 3] = (
                self.temp_array[day, 1] + self.temp_array[day - 1, 1] * ma_coef
            )

            # ARMA part: ARMA(t) = AR * AR(t-1) + MA * E(t-1) + E(t)
            self.temp_array[day, 4] = (
                self.temp_array[day - 1, 2] * ar_coef +
                self.temp_array[day - 1, 1] * ma_coef +
                self.temp_array[day, 1]
            )

        # Store final daily average temp with ARMA variation
        # VBA uses column 7 for final output (would be set in later code)
        for day in range(365):
            # Use monthly mean + ARMA component
            self.temp_array[day, 6] = self.temp_array[day, 0] + self.temp_array[day, 4]

    def _generate_minute_temperatures(self):
        """
        Generate minute-by-minute temperature profile for the simulation day.

        Matches VBA RunTemperatureModel (clsGlobalClimate.cls lines 331-560).
        Uses cumulative irradiance ratios to determine temperature timing.
        """
        # Get daily average temperature for this day (VBA line 422)
        day_idx = self.day_of_year - 1
        td = self.temp_array[day_idx, 6] if day_idx < 365 else 10.0  # Column 6 has final ARMA result

        # Solar constant with Earth-Sun distance correction (VBA line 415)
        solar_constant = 1367 * (1 + 0.034 * np.cos(2 * PI * self.day_of_year / 365.25))

        # Cloud cooling rate parameter (VBA line 425)
        cloud_cooling_rate_base = 0.1 / 60.0

        # Initialize cumulative radiation array (VBA lines 405, 428-430)
        # Columns: [cum_extraterrestrial, cum_global, ratio, temperature]
        cum_rad = np.zeros((1440, 4))

        # Variables for tracking max ratio and timing
        kx_max = 0.0  # Maximum cumulative irradiance ratio
        kx_max_i = 0   # Minute when max ratio occurs
        arctic_night = False  # Flag for polar regions with no sunrise

        # VBA lines 433-457: Calculate cumulative irradiances and find max ratio
        for minute in range(1, 1440):
            # Cumulative global horizontal irradiance
            cum_rad[minute, 1] = cum_rad[minute - 1, 1] + self.g_o[minute]

            # If daylight, calculate cumulative extraterrestrial and ratio
            if self.g_o[minute] > 0:
                cum_rad[minute, 0] = cum_rad[minute - 1, 0] + solar_constant
                if cum_rad[minute, 0] > 0:
                    cum_rad[minute, 2] = cum_rad[minute, 1] / cum_rad[minute, 0]
            else:
                cum_rad[minute, 0] = 0
                cum_rad[minute, 2] = 0  # No ratio at night

            # Find maximum ratio (VBA lines 447-450)
            if cum_rad[minute, 2] > kx_max:
                kx_max = cum_rad[minute, 2]
                kx_max_i = minute

        # Total daily irradiation (VBA lines 453-460)
        irradiation = cum_rad[-1, 1] * (60.0 / 3600.0) / 1000.0  # Convert to kWh/m²

        # Temperature standard deviation (VBA line 463)
        dtd_sd = 1.0

        # Daily temperature range based on irradiation (VBA lines 466-468)
        dtd = 20 * np.log10(irradiation + 2.5) - 7
        dtd += self.rng.normal(0, dtd_sd)

        # Min and max temperatures for the day (VBA lines 471-472)
        td_min = td - 0.5 * dtd
        td_max = td + 0.5 * dtd

        # Workaround for Arctic night (no sunrise) (VBA lines 475-479)
        if kx_max == 0:
            kx_max = 1
            kx_max_i = 720  # Noon
            td_min_i = 1
            arctic_night = True

        # Temperature slopes before/after max (VBA lines 483-484)
        slope_before = (td_max - td_min) / kx_max if kx_max > 0 else 0
        slope_after = slope_before * 1.7  # Faster cooling after peak

        # VBA lines 487-499: Calculate temps up to max temperature
        td_min_i = 0  # Track when minimum temp occurs
        for minute in range(1, kx_max_i + 1):
            if self.g_o[minute] > 0:
                # Daylight: temperature based on cumulative ratio
                cum_rad[minute, 3] = td_min + slope_before * cum_rad[minute, 2]
                self.theta_o[minute] = cum_rad[minute, 3]
            elif arctic_night:
                # Arctic night: linear increase to noon
                cum_rad[minute, 3] = td_min + slope_before * minute / kx_max_i
                self.theta_o[minute] = cum_rad[minute, 3]
            else:
                # Before sunrise: will be filled in later
                cum_rad[minute, 3] = 0
                td_min_i = minute + 1

        # VBA lines 502-522: Calculate temps after max temperature
        td_sunset_i = 0  # Track when sunset occurs
        td_sunset = td_min  # Temperature at sunset

        if arctic_night:
            # Arctic night: linear decrease after noon (VBA lines 504-508)
            td_sunset_i = 1080
            for minute in range(kx_max_i, td_sunset_i + 1):
                cum_rad[minute, 3] = td_max - slope_before * (minute - kx_max_i) / kx_max_i
                self.theta_o[minute] = cum_rad[minute, 3]
                td_sunset = cum_rad[minute, 3]
        else:
            # Normal day: temps after max temp (VBA lines 511-521)
            for minute in range(kx_max_i + 1, 1439):
                if self.g_o[minute] > 0:
                    # Daylight: faster cooling with slope_after
                    cum_rad[minute, 3] = td_max - slope_after * (kx_max - cum_rad[minute, 2])
                    self.theta_o[minute] = cum_rad[minute, 3]
                    td_sunset = cum_rad[minute, 3]
                    td_sunset_i = minute
                else:
                    # After sunset: will be filled in later
                    cum_rad[minute, 3] = 0

        # VBA lines 525-531: Calculate overnight cooling rate
        di = td_min_i + 1440 - td_sunset_i  # Minutes of darkness
        dt = td_sunset - td_min  # Temperature drop needed
        cooling_rate = dt / di if di > 0 else 0
        cloud_cooling_rate = 0.025

        # VBA lines 534-541: Calculate overnight mean clearness index
        overnight_mean_clearness = 0
        count = 0
        for minute in range(0, td_min_i + 1):
            overnight_mean_clearness += self.clearness_index[minute]
            count += 1
        for minute in range(td_sunset_i, 1440):
            overnight_mean_clearness += self.clearness_index[minute]
            count += 1
        if count > 0:
            overnight_mean_clearness /= count

        # VBA lines 544-548: Calculate overnight temps after sunset
        for minute in range(td_sunset_i, 1440):
            if minute == 0:
                prev_temp = td_sunset
            else:
                prev_temp = cum_rad[minute - 1, 3] if cum_rad[minute - 1, 3] > 0 else td_sunset

            cloud_adjustment = cloud_cooling_rate * (overnight_mean_clearness - self.clearness_index[minute])
            cum_rad[minute, 3] = prev_temp - (cooling_rate - cloud_adjustment)
            self.theta_o[minute] = cum_rad[minute, 3]

        # VBA lines 550-556: Calculate overnight temps before sunrise
        cum_rad[0, 3] = cum_rad[1439, 3] if cum_rad[1439, 3] > 0 else td_min
        self.theta_o[0] = cum_rad[0, 3]

        for minute in range(1, td_min_i + 1):
            prev_temp = cum_rad[minute - 1, 3]
            cloud_adjustment = cloud_cooling_rate * (overnight_mean_clearness - self.clearness_index[minute])
            cum_rad[minute, 3] = prev_temp - (cooling_rate - cloud_adjustment)
            self.theta_o[minute] = cum_rad[minute, 3]

    def run_all(self):
        """
        Run complete climate simulation: clearness index, irradiance, and temperature.
        """
        self.simulate_clearness_index()
        self.calculate_global_irradiance()
        self.run_temperature_model()


class LocalClimate:
    """
    Local climate model for individual dwellings.

    Provides access to global climate data with potential for dwelling-specific variations.
    """

    def __init__(self, global_climate: GlobalClimate, dwelling_index: int = 0):
        """
        Initialize local climate model.

        Parameters
        ----------
        global_climate : GlobalClimate
            Reference to global climate model
        dwelling_index : int, optional
            Index of the dwelling (for future dwelling-specific variations)
        """
        self.global_climate = global_climate
        self.dwelling_index = dwelling_index

    def get_irradiance(self, minute: int) -> float:
        """Get global horizontal irradiance at specified minute (W/m²)."""
        return self.global_climate.g_o[minute]

    def get_clearsky_irradiance(self, minute: int) -> float:
        """Get clear sky irradiance at specified minute (W/m²)."""
        return self.global_climate.g_o_clearsky[minute]

    def get_temperature(self, minute: int) -> float:
        """Get outdoor temperature at specified minute (°C)."""
        return self.global_climate.theta_o[minute]

    def get_clearness_index(self, minute: int) -> float:
        """Get clearness index at specified minute (0-1)."""
        return self.global_climate.clearness_index[minute]
