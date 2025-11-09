"""
Heating and Cooling Controls

Implements thermostats and timers for heating and cooling systems:
- Hysteresis thermostats with deadbands
- Timer schedules using Markov chains
- Control signals for heating/cooling systems
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    THERMOSTAT_DEADBAND_SPACE,
    THERMOSTAT_DEADBAND_WATER,
    THERMOSTAT_DEADBAND_EMITTER,
    TIMER_RANDOM_SHIFT_MINUTES
)
from ..utils import markov
from ..utils.random import RandomGenerator
from ..data.loader import CRESTDataLoader


@dataclass
class HeatingControlsConfig:
    """Configuration for heating controls."""
    dwelling_index: int
    building_index: int
    heating_system_index: int
    cooling_system_index: int = 0
    run_number: int = 0


class HeatingControls:
    """
    Heating and cooling controls model.

    Manages thermostats and timers for:
    - Space heating
    - Hot water
    - Space cooling
    - Heating/cooling emitters
    """

    def __init__(
        self,
        config: HeatingControlsConfig,
        data_loader: CRESTDataLoader,
        rng: Optional[RandomGenerator] = None
    ):
        """
        Initialize heating controls.

        Parameters
        ----------
        config : HeatingControlsConfig
            Configuration parameters
        data_loader : CRESTDataLoader
            Data loader for control settings
        rng : RandomGenerator, optional
            Random number generator
        """
        self.config = config
        self.data_loader = data_loader
        self.rng = rng if rng is not None else RandomGenerator()

        # Load heating system type (determines hot water control logic)
        heating_systems = data_loader.load_primary_heating_systems()
        if config.heating_system_index < len(heating_systems):
            heating_params = heating_systems.iloc[config.heating_system_index]
            self.heating_system_type = int(heating_params.get('HeatingSystemType', 1))
        else:
            self.heating_system_type = 1  # Default to regular boiler

        # Load cooling system type
        cooling_systems = data_loader.load_cooling_systems()
        if config.cooling_system_index < len(cooling_systems):
            cooling_params = cooling_systems.iloc[config.cooling_system_index]
            self.cooling_system_type = int(cooling_params.get('CoolingSystemType', 0))
        else:
            self.cooling_system_type = 0  # No cooling

        # Thermostat setpoints (will be assigned stochastically)
        self._assign_thermostat_setpoints()

        # Deadbands
        self.space_heating_deadband = THERMOSTAT_DEADBAND_SPACE
        self.space_cooling_deadband = THERMOSTAT_DEADBAND_SPACE
        self.hot_water_deadband = THERMOSTAT_DEADBAND_WATER
        self.emitter_deadband = THERMOSTAT_DEADBAND_EMITTER
        self.cooler_emitter_deadband = THERMOSTAT_DEADBAND_EMITTER

        # Generate timer schedules
        self._generate_timer_schedules()

        # Thermostat state arrays (1440 timesteps)
        self.space_heating_thermostat = np.zeros(TIMESTEPS_PER_DAY_1MIN, dtype=bool)
        self.space_cooling_thermostat = np.zeros(TIMESTEPS_PER_DAY_1MIN, dtype=bool)
        self.hot_water_thermostat = np.zeros(TIMESTEPS_PER_DAY_1MIN, dtype=bool)
        self.emitter_thermostat = np.zeros(TIMESTEPS_PER_DAY_1MIN, dtype=bool)
        self.cooler_emitter_thermostat = np.zeros(TIMESTEPS_PER_DAY_1MIN, dtype=bool)

        # Control signal arrays
        self.heater_on_off = np.zeros(TIMESTEPS_PER_DAY_1MIN, dtype=bool)
        self.heat_water_on_off = np.zeros(TIMESTEPS_PER_DAY_1MIN, dtype=bool)

        # References to other components (set externally)
        self.building = None
        self.hot_water = None

    def _assign_thermostat_setpoints(self):
        """Assign thermostat setpoints stochastically from distributions."""
        controls_data = self.data_loader.load_heating_controls()

        # Space heating thermostat setpoint
        # Draw from distribution in HeatingControls.csv
        # Simplified: use typical UK value
        space_setpoints = [18.0, 19.0, 20.0, 21.0, 22.0]
        space_probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        setpoint_idx = markov.select_next_state(np.array(space_probs), self.rng.random())
        self.space_heating_setpoint = space_setpoints[setpoint_idx]

        # Space cooling setpoint (offset from heating setpoint)
        cooling_offset = 5.0  # Cooling starts 5°C above heating setpoint
        self.space_cooling_setpoint = self.space_heating_setpoint + cooling_offset

        # Hot water setpoint
        water_setpoints = [55.0, 60.0, 65.0]
        water_probs = [0.2, 0.6, 0.2]
        setpoint_idx = markov.select_next_state(np.array(water_probs), self.rng.random())
        self.hot_water_setpoint = water_setpoints[setpoint_idx]

        # Emitter setpoint (typically higher than room setpoint)
        self.emitter_setpoint = self.space_heating_setpoint + 10.0

        # Cooler emitter setpoint
        self.cooler_emitter_setpoint = self.space_cooling_setpoint - 5.0

    def _generate_timer_schedules(self):
        """Generate timer schedules using Markov chain transitions."""
        # Load timer TPM
        timer_tpm = self.data_loader.load_heating_controls_tpm().values

        # Generate 30-minute resolution timer schedule (48 periods)
        timer_schedule_30min = np.zeros(48, dtype=int)

        # Start in OFF state (state 0)
        current_state = 0

        for period in range(48):
            # Get transition probabilities for current state
            # TPM has 2 states: 0 (OFF), 1 (ON)
            row_idx = current_state
            transition_probs = timer_tpm[row_idx, 2:].astype(float)  # Skip label columns

            # Normalize
            transition_probs = markov.normalize_probabilities(transition_probs)

            # Select next state
            next_state = markov.select_next_state(transition_probs, self.rng.random())
            timer_schedule_30min[period] = next_state
            current_state = next_state

        # Convert to 1-minute resolution
        self.space_heating_timer = self._expand_to_1min(timer_schedule_30min)
        self.hot_water_timer = self._expand_to_1min(timer_schedule_30min)  # Same schedule

        # Apply random time shift for diversity (±15 minutes)
        self.space_heating_timer = self._time_shift_schedule(self.space_heating_timer)
        self.hot_water_timer = self._time_shift_schedule(self.hot_water_timer)

        # Cooling timer (typically off in UK, can use reverse schedule)
        self.space_cooling_timer = np.zeros(TIMESTEPS_PER_DAY_1MIN, dtype=bool)

    def _expand_to_1min(self, schedule_30min: np.ndarray) -> np.ndarray:
        """
        Expand 30-minute resolution schedule to 1-minute resolution.

        Parameters
        ----------
        schedule_30min : np.ndarray
            48-period schedule (30-minute intervals)

        Returns
        -------
        np.ndarray
            1440-period schedule (1-minute resolution)
        """
        # Each 30-minute period → 30 one-minute periods
        schedule_1min = np.repeat(schedule_30min, 30)
        return schedule_1min.astype(bool)

    def _time_shift_schedule(self, schedule: np.ndarray) -> np.ndarray:
        """
        Apply random time shift to schedule for diversity.

        Parameters
        ----------
        schedule : np.ndarray
            1440-period boolean schedule

        Returns
        -------
        np.ndarray
            Time-shifted schedule
        """
        # Random shift within ±15 minutes
        shift = int(self.rng.uniform(-TIMER_RANDOM_SHIFT_MINUTES, TIMER_RANDOM_SHIFT_MINUTES))

        # Circular shift
        shifted = np.roll(schedule, shift)
        return shifted

    def initialize_thermostat_states(self, initial_temp_i: float, initial_temp_cyl: float,
                                    initial_temp_em: float, initial_temp_cool: float):
        """
        Initialize thermostat states based on initial temperatures.

        Parameters
        ----------
        initial_temp_i : float
            Initial internal air temperature (°C)
        initial_temp_cyl : float
            Initial hot water cylinder temperature (°C)
        initial_temp_em : float
            Initial emitter temperature (°C)
        initial_temp_cool : float
            Initial cooler emitter temperature (°C)
        """
        # Hot water thermostat
        self.hot_water_thermostat[0] = initial_temp_cyl <= self.hot_water_setpoint

        # Space heating thermostat
        self.space_heating_thermostat[0] = initial_temp_i < self.space_heating_setpoint

        # Space cooling thermostat
        self.space_cooling_thermostat[0] = initial_temp_i > self.space_cooling_setpoint

        # Emitter thermostats
        self.emitter_thermostat[0] = initial_temp_em < self.emitter_setpoint
        self.cooler_emitter_thermostat[0] = initial_temp_cool > self.cooler_emitter_setpoint

        # Initial control signals
        self._calculate_control_signals(1)

    def set_building(self, building):
        """Set reference to building model."""
        self.building = building

    def set_hot_water(self, hot_water):
        """Set reference to hot water model."""
        self.hot_water = hot_water

    def calculate_control_states(self, timestep: int):
        """
        Calculate thermostat and timer states for current timestep.

        Parameters
        ----------
        timestep : int
            Current timestep (1-based, 1-1440)
        """
        if timestep == 1:
            # States already initialized
            return

        if self.building is None:
            raise RuntimeError("Building must be set before calculating control states")

        # Get temperatures from previous timestep
        idx = timestep - 1
        prev_idx = idx - 1

        theta_i = self.building.get_internal_temperature(timestep - 1)
        theta_cyl = self.building.get_cylinder_temperature(timestep - 1)
        theta_em = self.building.get_emitter_temperature(timestep - 1)
        theta_cool = self.building.theta_cool[prev_idx]  # Direct access for cooling

        # Hot water thermostat (hysteresis logic)
        if self.hot_water_thermostat[prev_idx]:
            # Was ON: stay ON if temp < setpoint + deadband
            self.hot_water_thermostat[idx] = theta_cyl < (self.hot_water_setpoint + self.hot_water_deadband)
        else:
            # Was OFF: turn ON if temp <= setpoint - deadband
            self.hot_water_thermostat[idx] = theta_cyl <= (self.hot_water_setpoint - self.hot_water_deadband)

        # Space heating thermostat
        if self.space_heating_thermostat[prev_idx]:
            # Was ON: stay ON if temp < setpoint + deadband
            self.space_heating_thermostat[idx] = theta_i < (self.space_heating_setpoint + self.space_heating_deadband)
        else:
            # Was OFF: turn ON if temp < setpoint - deadband
            self.space_heating_thermostat[idx] = theta_i < (self.space_heating_setpoint - self.space_heating_deadband)

        # Space cooling thermostat (reverse logic)
        if self.space_cooling_thermostat[prev_idx]:
            # Was ON: stay ON if temp > setpoint - deadband
            self.space_cooling_thermostat[idx] = theta_i > (self.space_cooling_setpoint - self.space_cooling_deadband)
        else:
            # Was OFF: turn ON if temp > setpoint + deadband
            self.space_cooling_thermostat[idx] = theta_i > (self.space_cooling_setpoint + self.space_cooling_deadband)

        # Emitter thermostat
        if self.emitter_thermostat[prev_idx]:
            self.emitter_thermostat[idx] = theta_em < (self.emitter_setpoint + self.emitter_deadband)
        else:
            self.emitter_thermostat[idx] = theta_em < (self.emitter_setpoint - self.emitter_deadband)

        # Cooler emitter thermostat
        if self.cooler_emitter_thermostat[prev_idx]:
            self.cooler_emitter_thermostat[idx] = theta_cool > (self.cooler_emitter_setpoint - self.cooler_emitter_deadband)
        else:
            self.cooler_emitter_thermostat[idx] = theta_cool > (self.cooler_emitter_setpoint + self.cooler_emitter_deadband)

        # Calculate control signals
        self._calculate_control_signals(timestep)

    def _calculate_control_signals(self, timestep: int):
        """
        Calculate control signals to send to heating/cooling systems.

        Parameters
        ----------
        timestep : int
            Current timestep (1-based)
        """
        idx = timestep - 1

        # Hot water control signal
        if self.heating_system_type == 2:
            # Combi boiler: instantaneous demand
            if self.hot_water is not None:
                h_demand = self.hot_water.get_h_demand(timestep)
                self.heat_water_on_off[idx] = h_demand > 0
            else:
                self.heat_water_on_off[idx] = False
        else:
            # Regular or system boiler: timer AND thermostat
            self.heat_water_on_off[idx] = (
                self.hot_water_timer[idx] and
                self.hot_water_thermostat[idx]
            )

        # Main heater control signal
        # Heater is ON if hot water OR space heating is needed
        space_heating_needed = (
            self.space_heating_timer[idx] and
            self.space_heating_thermostat[idx] and
            self.emitter_thermostat[idx]
        )

        self.heater_on_off[idx] = self.heat_water_on_off[idx] or space_heating_needed

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_heat_water_on_off(self, timestep: int) -> bool:
        """Get hot water control signal at specified timestep (1-based)."""
        return bool(self.heat_water_on_off[timestep - 1])

    def get_heater_on_off(self, timestep: int) -> bool:
        """Get heater control signal at specified timestep (1-based)."""
        return bool(self.heater_on_off[timestep - 1])

    def get_hot_water_thermostat_setpoint(self) -> float:
        """Get hot water thermostat setpoint (°C)."""
        return self.hot_water_setpoint

    def get_space_thermostat_state(self, timestep: int) -> bool:
        """Get space heating thermostat state at specified timestep (1-based)."""
        return bool(self.space_heating_thermostat[timestep - 1])

    def get_space_timer_state(self, timestep: int) -> bool:
        """Get space heating timer state at specified timestep (1-based)."""
        return bool(self.space_heating_timer[timestep - 1])

    def get_emitter_thermostat_state(self, timestep: int) -> bool:
        """Get emitter thermostat state at specified timestep (1-based)."""
        return bool(self.emitter_thermostat[timestep - 1])

    def get_cooler_emitter_state(self, timestep: int) -> bool:
        """Get cooler emitter state at specified timestep (1-based)."""
        return bool(self.cooler_emitter_thermostat[timestep - 1])

    def get_space_thermostat_setpoint(self) -> float:
        """Get space heating thermostat setpoint (°C)."""
        return self.space_heating_setpoint

    def get_space_cooling_thermostat_state(self, timestep: int) -> bool:
        """Get space cooling thermostat state at specified timestep (1-based)."""
        return bool(self.space_cooling_thermostat[timestep - 1])

    def get_space_cooling_timer_state(self, timestep: int) -> bool:
        """Get space cooling timer state at specified timestep (1-based)."""
        return bool(self.space_cooling_timer[timestep - 1])
