"""
Lighting Model

Simulates domestic lighting demand with irradiance-based switching.
Handles up to 60 bulbs per dwelling with occupancy-dependent operation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    MAX_BULBS_PER_DWELLING,
    Country
)
from ..utils.random import RandomGenerator
from ..data.loader import CRESTDataLoader


@dataclass
class LightingConfig:
    """Configuration for lighting model."""
    dwelling_index: int
    country: Country = Country.UK  # Country for lighting behavior
    run_number: int = 0


class Lighting:
    """
    Domestic lighting model.

    Simulates lighting demand based on:
    - Solar irradiance (lights more likely when dark)
    - Occupancy (lights only on when occupied)
    - Stochastic switching behavior
    """

    def __init__(
        self,
        config: LightingConfig,
        data_loader: CRESTDataLoader,
        rng: Optional[RandomGenerator] = None
    ):
        """
        Initialize lighting model.

        Parameters
        ----------
        config : LightingConfig
            Configuration parameters
        data_loader : CRESTDataLoader
            Data loader for lighting specs
        rng : RandomGenerator, optional
            Random number generator
        """
        self.config = config
        self.data_loader = data_loader
        self.rng = rng if rng is not None else RandomGenerator()

        # Load lighting configuration
        self._load_lighting_config()

        # Storage arrays
        self.total_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W
        self.thermal_gains = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W
        self.bulb_states = np.zeros((TIMESTEPS_PER_DAY_1MIN, MAX_BULBS_PER_DWELLING), dtype=bool)

        # References (set externally)
        self.occupancy = None
        self.local_climate = None

    def _load_lighting_config(self):
        """Load lighting configuration."""
        lighting_config = self.data_loader.load_lighting_config()
        bulbs_data = self.data_loader.load_bulbs()

        # Get dwelling-specific bulb configuration
        # Simplified: use typical values
        self.num_bulbs = 20  # Typical UK dwelling
        self.bulb_powers = np.random.choice([5, 11, 15, 20, 60], size=self.num_bulbs)  # W, mix of LED/CFL/incandescent

        # Lighting parameters
        self.irradiance_threshold = 60.0  # W/m² - lights more likely below this
        self.calibration_scalar = 1.0  # Calibration factor for demand matching

    def set_occupancy(self, occupancy):
        """Set reference to occupancy model."""
        self.occupancy = occupancy

    def set_local_climate(self, local_climate):
        """Set reference to local climate model."""
        self.local_climate = local_climate

    def run_simulation(self):
        """
        Run lighting simulation.

        Generates stochastic bulb switching based on irradiance and occupancy.
        """
        if self.occupancy is None or self.local_climate is None:
            raise RuntimeError("Occupancy and climate must be set before running simulation")

        # Get active occupancy (10-minute resolution)
        active_occupancy_10min = self.occupancy.active_occupancy

        # Simulate each minute
        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            # Get irradiance
            irradiance = self.local_climate.get_irradiance(minute)

            # Get active occupants (from 10-minute periods)
            ten_min_idx = minute // 10
            active_occupants = active_occupancy_10min[ten_min_idx]

            # Calculate switching probability based on irradiance
            # Low irradiance → high probability of lights being on
            irradiance_factor = max(0.0, 1.0 - (irradiance / self.irradiance_threshold))

            # Only switch lights if someone is actively home
            if active_occupants > 0:
                # Each bulb has independent switching probability
                for bulb in range(self.num_bulbs):
                    # Base probability modified by irradiance
                    switch_prob = irradiance_factor * self.calibration_scalar * 0.1

                    # Stochastic switching with duration
                    if self.rng.random() < switch_prob:
                        self.bulb_states[minute, bulb] = True
                    else:
                        self.bulb_states[minute, bulb] = False
            else:
                # No one home - all lights off
                self.bulb_states[minute, :] = False

        # Calculate demand
        self._calculate_demand()

    def _calculate_demand(self):
        """Calculate total lighting demand and thermal gains."""
        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            # Sum power of all lit bulbs
            total_power = 0.0
            for bulb in range(self.num_bulbs):
                if self.bulb_states[minute, bulb]:
                    total_power += self.bulb_powers[bulb]

            self.total_demand[minute] = total_power

            # Thermal gains (assume 90% of light energy becomes heat)
            thermal_conversion = 0.9
            self.thermal_gains[minute] = total_power * thermal_conversion

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_total_demand(self, timestep: int) -> float:
        """Get total lighting demand at specified timestep (1-based) in Watts."""
        return self.total_demand[timestep - 1]

    def get_thermal_gain(self, timestep: int) -> float:
        """Get thermal gains at specified timestep (1-based) in Watts."""
        return self.thermal_gains[timestep - 1]

    def get_daily_energy(self) -> float:
        """Get total daily lighting energy in Wh."""
        return np.sum(self.total_demand) / 60.0  # Convert W·min to Wh
