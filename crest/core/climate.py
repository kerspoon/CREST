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
    DAY_SUMMER_TIME_END
)
from ..utils import markov
from ..utils.random import RandomGenerator
from ..data.loader import CRESTDataLoader


@dataclass
class ClimateConfig:
    """Configuration for climate model."""
    day_of_month: int
    month_of_year: int
    longitude: float = -1.26  # Loughborough, UK
    latitude: float = 52.77   # Loughborough, UK
    meridian: float = 0.0     # Greenwich meridian
    city: str = "England"
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

    def _generate_daily_temperatures(self):
        """
        Generate daily average temperatures for a full year using ARMA model.

        Stores results in self.temp_array for later use.
        """
        # Load monthly temperature data for selected city
        climate_data = self.data_loader.load_climate_data_and_cooling_tech()

        # Temperature model parameters (England defaults)
        ty_mean = 9.3     # Mean annual temperature (°C)
        ty_sd = 6.5       # Standard deviation (°C)
        ty_shift = -115   # Day offset for sinusoid
        ar_coef = 0.81    # Autoregressive coefficient
        ma_coef = 0.62    # Moving average coefficient
        sd_factor = 0.1   # SD scaling factor

        # Monthly temperature data (placeholder - would load from climate_data)
        # For now use England defaults
        monthly_temps = [5.0, 5.0, 7.0, 9.0, 12.0, 15.0, 17.0, 17.0, 15.0, 11.0, 8.0, 5.0]

        # Generate daily temperatures using sinusoidal approximation + stochastic variation
        for day in range(365):
            # Determine which month this day falls in
            month_idx = min(day // 30, 11)  # Simplified month assignment

            # Sinusoidal approximation to daily average
            sinusoid = ty_mean + ty_sd * np.cos(2 * PI * (day + ty_shift) / 365.0)

            # Add ARMA stochastic variation (simplified version)
            if day == 0:
                stochastic_component = 0.0
            else:
                stochastic_component = self.rng.normal(0, sd_factor * ty_sd)

            # Combine sinusoid with monthly adjustment
            daily_avg = (sinusoid + monthly_temps[month_idx]) / 2.0 + stochastic_component

            # Store in temp array (multiple columns for different temperature metrics)
            self.temp_array[day, 0] = daily_avg
            # Other columns could store min/max/variation etc.

    def _generate_minute_temperatures(self):
        """
        Generate minute-by-minute temperature profile for the simulation day.

        Uses correlation with solar irradiance to create diurnal variation.
        """
        # Get daily average temperature for this day
        day_idx = self.day_of_year - 1
        daily_avg = self.temp_array[day_idx, 0] if day_idx < 365 else 10.0

        # Create diurnal variation correlated with irradiance
        max_irradiance = np.max(self.g_o)

        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            # Normalized irradiance (0-1)
            if max_irradiance > 0:
                norm_irradiance = self.g_o[minute] / max_irradiance
            else:
                norm_irradiance = 0.0

            # Temperature variation around daily average (±5°C variation)
            # Warmer during day (high irradiance), cooler at night
            temp_variation = 5.0 * (2.0 * norm_irradiance - 1.0)

            # Add small random variation
            random_variation = self.rng.normal(0, 0.5)

            self.theta_o[minute] = daily_avg + temp_variation + random_variation

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
