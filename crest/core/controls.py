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
    is_weekend: bool = False
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

        # Load heating system type (determines hot water control logic - strict mode)
        heating_systems = data_loader.load_primary_heating_systems()
        if config.heating_system_index < len(heating_systems):
            heating_params = heating_systems.iloc[config.heating_system_index]
            try:
                # Column name from CSV row 2 symbols
                self.heating_system_type = int(heating_params['1 = regular, 2 = combi'])
            except KeyError as e:
                raise KeyError(
                    f"Missing required column '1 = regular, 2 = combi' in PrimaryHeatingSystems.csv "
                    f"for index {config.heating_system_index}. Available columns: {list(heating_params.index)}"
                )
        else:
            self.heating_system_type = 1  # Default to regular boiler

        # Load cooling system type (strict mode)
        # Negative index means no cooling system
        if config.cooling_system_index >= 0:
            cooling_systems = data_loader.load_cooling_systems()
            if config.cooling_system_index < len(cooling_systems):
                cooling_params = cooling_systems.iloc[config.cooling_system_index]
                try:
                    # Column name from CSV - cooling systems use 'Type of system'
                    self.cooling_system_type = int(cooling_params['Type of system'])
                except KeyError as e:
                    raise KeyError(
                        f"Missing required column in CoolingSystems.csv "
                        f"for index {config.cooling_system_index}. Available columns: {list(cooling_params.index)}"
                    )
            else:
                self.cooling_system_type = 0  # No cooling
        else:
            self.cooling_system_type = 0  # No cooling (negative index)

        # Deadbands
        self.space_heating_deadband = THERMOSTAT_DEADBAND_SPACE
        self.space_cooling_deadband = THERMOSTAT_DEADBAND_SPACE
        self.hot_water_deadband = THERMOSTAT_DEADBAND_WATER
        self.emitter_deadband = THERMOSTAT_DEADBAND_EMITTER
        self.cooler_emitter_deadband = THERMOSTAT_DEADBAND_EMITTER

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

        # Lazy initialization state
        self._initialized = False

    def initialize(self):
        """
        Initialize heating control setpoints and schedules (~50+ RNG calls).

        VBA Reference: clsHeatingControls lines 207-330
        - Assign thermostat setpoints (lines 207-239)
        - Generate timer schedules (lines 209-330)

        Must be called before calculate_control_states().
        """
        if self._initialized:
            return

        # Thermostat setpoints (will be assigned stochastically)
        self._assign_thermostat_setpoints()

        # Generate timer schedules
        self._generate_timer_schedules()

        self._initialized = True

    def _assign_thermostat_setpoints(self):
        """
        Assign thermostat setpoints stochastically from distributions.

        Matches VBA InitialiseHeatingControls (lines 205-259).
        """
        controls_data = self.data_loader.load_heating_controls()

        # Space heating thermostat setpoint (VBA lines 205-224)
        # Draw from distribution in HeatingControls.csv using cumulative probability
        space_temps = controls_data['space_heating_temps'].iloc[0].astype(float)
        space_probs = controls_data['space_heating_probs'].iloc[0].astype(float)

        # Use cumulative probability method (VBA-style)
        rand_val = self.rng.random()
        cumulative_p = 0.0
        for i in range(len(space_temps)):
            cumulative_p += space_probs[i]
            if rand_val < cumulative_p:
                self.space_heating_setpoint = float(space_temps[i])
                break
        else:
            # Fallback to last value if probabilities don't sum to 1.0
            self.space_heating_setpoint = float(space_temps[-1])

        # Space cooling setpoint (VBA line 228)
        # Offset from heating setpoint to avoid "thermostat fights"
        cooling_offset = controls_data['cooling_offset'].iloc[0]
        self.space_cooling_setpoint = self.space_heating_setpoint + cooling_offset

        # Check if heating should be disabled (VBA lines 230-235)
        # For heating_system_type > 3 (no gas heating), set to -99
        if self.heating_system_type > 3:
            self.space_heating_setpoint = -99.0

        # Hot water setpoint (VBA lines 237-255)
        water_temps = controls_data['hot_water_temps'].iloc[0].astype(float)
        water_probs = controls_data['hot_water_probs'].iloc[0].astype(float)

        rand_val = self.rng.random()
        cumulative_p = 0.0
        for i in range(len(water_temps)):
            cumulative_p += water_probs[i]
            if rand_val < cumulative_p:
                self.hot_water_setpoint = float(water_temps[i])
                break
        else:
            self.hot_water_setpoint = float(water_temps[-1])

        # Emitter setpoints from Buildings.csv (VBA lines 197, 200)
        buildings_data = self.data_loader.load_buildings()
        if self.config.building_index < len(buildings_data):
            building_params = buildings_data.iloc[self.config.building_index]
            self.emitter_setpoint = float(building_params['theta_em'])
            self.cooler_emitter_setpoint = float(building_params['theta_cool'])
        else:
            # Fallback values
            self.emitter_setpoint = 50.0
            self.cooler_emitter_setpoint = 20.0

    def _generate_timer_schedules(self):
        """
        Generate timer schedules using Markov chain transitions.

        Matches VBA InitialiseHeatingControls (lines 307-404).
        """
        # Load timer TPM (VBA lines 315-317)
        # TPM range C8:F103 = columns 2-5 (0-based), rows 7-102 (0-based after skiprows=7)
        # Columns: Period (0), State (1), Weekday→0 (2), Weekday→1 (3), Weekend→0 (4), Weekend→1 (5)
        timer_tpm = self.data_loader.load_heating_controls_tpm().values

        # Generate space heating 30-minute schedule (48 periods)
        # VBA lines 329-355
        space_schedule_30min = np.zeros(48, dtype=int)

        # Determine initial state probabilistically (VBA lines 329-335)
        # Weekday: 9% chance of starting ON, Weekend: 10% chance
        rand_val = self.rng.random()
        if self.config.is_weekend:
            current_state = 1 if rand_val < 0.10 else 0
        else:
            current_state = 1 if rand_val < 0.09 else 0

        space_schedule_30min[0] = current_state

        # Generate remaining periods using Markov chain (VBA lines 339-355)
        for period in range(1, 48):  # periods 1-47 (VBA intHH = 1 To 47)
            # Row index in TPM (VBA line 344)
            # intRow = (intHH - 1) * 2 + intCurrentState + 1
            # For Python 0-based: row = (period - 1) * 2 + current_state
            row_idx = (period - 1) * 2 + current_state

            # Select column based on weekday/weekend (VBA line 320, 350)
            # VBA: intColumn = IIf(blnWeekend, 3, 1) → 1-based columns
            # TPM columns: 0=period, 1=state, 2=wd→0, 3=wd→1, 4=we→0, 5=we→1
            if self.config.is_weekend:
                # Weekend: column 3 in VBA (1-based) = column 2 in Python (0-based after accounting for labels)
                # Actually VBA uses aTPM(intRow, intColumn) where intColumn=3 for weekend
                # aTPM = wsHeatingTPM.Range("C8:F103"), so column 1 of aTPM = column C = column 2 (0-based)
                # intColumn=1 → column C, intColumn=3 → column E
                prob_state_0 = timer_tpm[row_idx, 4]  # Weekend→0
            else:
                # Weekday: column 1 in VBA → column C in Excel → column 2 (0-based)
                prob_state_0 = timer_tpm[row_idx, 2]  # Weekday→0

            # Determine next state (VBA line 350)
            # intNextState = IIf(dblRand < aTPM(intRow, intColumn), 0, 1)
            rand_val = self.rng.random()
            next_state = 0 if rand_val < prob_state_0 else 1

            space_schedule_30min[period] = next_state
            current_state = next_state

        # Generate space cooling schedule (VBA lines 358-385)
        # Only for cooling_system_type > 1
        cooling_schedule_30min = np.zeros(48, dtype=int)
        if self.cooling_system_type > 1:
            # Initial state: 90% chance of starting ON (VBA lines 359-364)
            rand_val = self.rng.random()
            current_state = 1 if rand_val < 0.90 else 0
            cooling_schedule_30min[0] = current_state

            # Generate using cooling TPM (columns 10-13 in Excel = K-N)
            # After skiprows=7, these are columns 10-13 (0-based)
            for period in range(1, 48):
                row_idx = (period - 1) * 2 + current_state
                if self.config.is_weekend:
                    prob_state_0 = timer_tpm[row_idx, 12]  # Cooling weekend→0
                else:
                    prob_state_0 = timer_tpm[row_idx, 10]  # Cooling weekday→0

                rand_val = self.rng.random()
                next_state = 0 if rand_val < prob_state_0 else 1
                cooling_schedule_30min[period] = next_state
                current_state = next_state

        # Generate hot water schedule (VBA lines 389-395)
        # First period OFF (for diversity), rest ON
        hot_water_schedule_30min = np.ones(48, dtype=int)
        hot_water_schedule_30min[0] = 0

        # Expand to 1-minute resolution (VBA line 398-400)
        space_1min = self._expand_to_1min(space_schedule_30min)
        cooling_1min = self._expand_to_1min(cooling_schedule_30min)
        hot_water_1min = self._expand_to_1min(hot_water_schedule_30min)

        # Apply time shift (VBA lines 402-404)
        self.space_heating_timer = self._time_shift_schedule(space_1min)
        self.space_cooling_timer = self._time_shift_schedule(cooling_1min)
        self.hot_water_timer = self._time_shift_schedule(hot_water_1min)

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

        Matches VBA TimeShiftVector (lines 633-688).

        Parameters
        ----------
        schedule : np.ndarray
            1440-period boolean schedule

        Returns
        -------
        np.ndarray
            Time-shifted schedule
        """
        # VBA line 654: intShift = Round((Rnd() * intShiftInterval) - (intShiftInterval / 2), 0)
        # intShiftInterval = 30, so shift range is [-15, 15]
        # Python: round(uniform(0, 1) * 30 - 15) = round(uniform(-15, 15))
        shift_interval = 30
        shift = round(self.rng.random() * shift_interval - (shift_interval / 2))

        # Circular shift (VBA lines 656-682)
        # np.roll() is equivalent to VBA's wraparound logic
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
        theta_cool = self.building.get_theta_cool(timestep - 1)  # VBA line 463

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
