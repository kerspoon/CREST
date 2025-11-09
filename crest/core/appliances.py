"""
Appliances Model

Simulates 31 different domestic appliance types with stochastic switching
based on activity patterns and occupancy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    MAX_APPLIANCE_TYPES
)
from ..utils.random import RandomGenerator
from ..data.loader import CRESTDataLoader


@dataclass
class ApplianceSpec:
    """Specification for an appliance."""
    name: str
    rated_power: float  # Watts
    standby_power: float  # Watts
    cycle_length: int  # Minutes
    restart_delay: int  # Minutes
    ownership_prob: float
    use_profile: str  # Activity profile name
    prob_switch_on: float
    calibration_scalar: float = 1.0


@dataclass
class AppliancesConfig:
    """Configuration for appliances model."""
    dwelling_index: int
    run_number: int = 0


class Appliances:
    """
    Domestic appliances model.

    Simulates electrical demand from 31 appliance types using activity-based
    stochastic switching.
    """

    def __init__(
        self,
        config: AppliancesConfig,
        data_loader: CRESTDataLoader,
        activity_statistics: Dict,
        is_weekend: bool,
        rng: Optional[RandomGenerator] = None
    ):
        """
        Initialize appliances model.

        Parameters
        ----------
        config : AppliancesConfig
            Configuration parameters
        data_loader : CRESTDataLoader
            Data loader for appliance specs
        activity_statistics : dict
            Activity probability profiles
        is_weekend : bool
            Weekend flag for activity profiles
        rng : RandomGenerator, optional
            Random number generator
        """
        self.config = config
        self.data_loader = data_loader
        self.activity_statistics = activity_statistics
        self.is_weekend = is_weekend
        self.rng = rng if rng is not None else RandomGenerator()

        # Load appliance specifications
        self._load_appliance_specs()

        # Storage arrays
        self.total_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W
        self.thermal_gains = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W
        self.appliance_demands = np.zeros((TIMESTEPS_PER_DAY_1MIN, MAX_APPLIANCE_TYPES))  # W

        # Reference to occupancy model (set externally)
        self.occupancy = None

    def _load_appliance_specs(self):
        """Load appliance specifications from data."""
        appliances_data = self.data_loader.load_appliances_and_fixtures()

        # Create appliance specs (simplified - actual data parsing would be more complex)
        # In production, parse all 31 appliances from CSV
        self.appliances = []
        for i in range(min(MAX_APPLIANCE_TYPES, len(appliances_data))):
            if i < len(appliances_data):
                app_data = appliances_data.iloc[i]
                self.appliances.append(ApplianceSpec(
                    name=app_data.get('Name', f'Appliance{i}'),
                    rated_power=app_data.get('RatedPower', 100.0),
                    standby_power=app_data.get('StandbyPower', 0.0),
                    cycle_length=int(app_data.get('CycleLength', 60)),
                    restart_delay=int(app_data.get('RestartDelay', 0)),
                    ownership_prob=app_data.get('Ownership', 0.5),
                    use_profile=app_data.get('UseProfile', 'Active'),
                    prob_switch_on=app_data.get('ProbSwitchOn', 0.01),
                    calibration_scalar=app_data.get('CalibrationScalar', 1.0)
                ))

        # Determine which appliances dwelling owns (stochastic)
        self.has_appliance = [
            self.rng.random() < app.ownership_prob
            for app in self.appliances
        ]

    def set_occupancy(self, occupancy):
        """Set reference to occupancy model."""
        self.occupancy = occupancy

    def run_simulation(self):
        """
        Run appliance demand simulation.

        Generates stochastic appliance usage events based on occupancy and activities.
        """
        if self.occupancy is None:
            raise RuntimeError("Occupancy model must be set before running simulation")

        # Get active occupancy array (10-minute resolution)
        active_occupancy = self.occupancy.active_occupancy

        # Simulate each appliance
        for app_idx, appliance in enumerate(self.appliances):
            if not self.has_appliance[app_idx]:
                # Dwelling doesn't own this appliance
                continue

            # Initialize appliance state
            cycle_time_left = 0
            restart_delay_left = 0
            current_power = appliance.standby_power

            # Simulate each minute
            for minute in range(TIMESTEPS_PER_DAY_1MIN):
                # Get 10-minute period index
                ten_min_idx = minute // 10

                # Get active occupants for this period
                active_occupants = active_occupancy[ten_min_idx]

                # Default power is standby
                power = appliance.standby_power

                # Handle appliance state
                if cycle_time_left <= 0 and restart_delay_left > 0:
                    # Appliance off, waiting for restart delay
                    restart_delay_left -= 1
                    power = appliance.standby_power

                elif cycle_time_left <= 0:
                    # Appliance can start
                    if active_occupants > 0:
                        # Check if appliance starts
                        if self._check_appliance_start(appliance, active_occupants, ten_min_idx):
                            # Start appliance cycle
                            cycle_time_left = appliance.cycle_length
                            restart_delay_left = appliance.restart_delay
                            power = appliance.rated_power
                            cycle_time_left -= 1
                        else:
                            power = appliance.standby_power
                    else:
                        power = appliance.standby_power

                else:
                    # Appliance is running
                    if active_occupants == 0:
                        # Pause if no active occupants (will resume when they return)
                        power = appliance.standby_power
                    else:
                        # Continue running
                        power = appliance.rated_power
                        cycle_time_left -= 1

                # Store power demand
                self.appliance_demands[minute, app_idx] = power

        # Calculate totals
        self._calculate_totals()

    def _check_appliance_start(self, appliance: ApplianceSpec, active_occupants: int, ten_min_idx: int) -> bool:
        """
        Check if appliance should start based on activity probability.

        Parameters
        ----------
        appliance : ApplianceSpec
            Appliance specification
        active_occupants : int
            Number of active occupants
        ten_min_idx : int
            10-minute period index (0-143)

        Returns
        -------
        bool
            True if appliance should start
        """
        # Create key for activity statistics lookup
        weekend_flag = "1" if self.is_weekend else "0"
        key = f"{weekend_flag}_{active_occupants}_{appliance.use_profile}"

        # Get activity probability for this time period
        if key in self.activity_statistics:
            activity_prob = self.activity_statistics[key][ten_min_idx]
        else:
            # Default low probability if profile not found
            activity_prob = 0.001

        # Check if appliance starts
        combined_prob = activity_prob * appliance.prob_switch_on * appliance.calibration_scalar
        return self.rng.random() < combined_prob

    def _calculate_totals(self):
        """Calculate total electrical demand and thermal gains."""
        # Sum all appliance demands
        self.total_demand = np.sum(self.appliance_demands, axis=1)

        # Thermal gains (assume 80% of electrical energy becomes heat)
        thermal_conversion = 0.8
        self.thermal_gains = self.total_demand * thermal_conversion

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_total_demand(self, timestep: int) -> float:
        """Get total appliance demand at specified timestep (1-based) in Watts."""
        return self.total_demand[timestep - 1]

    def get_thermal_gain(self, timestep: int) -> float:
        """Get thermal gains at specified timestep (1-based) in Watts."""
        return self.thermal_gains[timestep - 1]

    def get_daily_energy(self) -> float:
        """Get total daily appliance energy in Wh."""
        return np.sum(self.total_demand) / 60.0  # Convert W·min to Wh
