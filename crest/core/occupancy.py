"""
Four-state Domestic Occupancy Model

Implements a Markov chain-based occupancy model with four states:
- At home / Away
- Active / Asleep

Based on UK Time Use Survey 2000 data.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..simulation.config import (
    TIMESTEPS_PER_DAY_10MIN,
    TIMESTEPS_PER_DAY_1MIN,
    OCCUPANT_THERMAL_GAIN_ACTIVE,
    OCCUPANT_THERMAL_GAIN_DORMANT
)
from ..utils import markov
from ..utils.random import RandomGenerator
from ..data.loader import CRESTDataLoader


@dataclass
class OccupancyConfig:
    """Configuration parameters for the occupancy model."""
    num_residents: int
    is_weekend: bool
    dwelling_index: int = 0


class Occupancy:
    """
    Four-state domestic occupancy model using Markov chains.

    Simulates occupancy patterns for a dwelling based on the number of residents
    and day type (weekend vs. weekday), using transition probability matrices
    derived from UK time-use survey data.

    The model operates at 10-minute resolution (144 timesteps per day).
    """

    def __init__(
        self,
        config: OccupancyConfig,
        data_loader: CRESTDataLoader,
        rng: Optional[RandomGenerator] = None
    ):
        """
        Initialize the occupancy model.

        Parameters
        ----------
        config : OccupancyConfig
            Configuration parameters
        data_loader : CRESTDataLoader
            Data loader for accessing TPMs and starting states
        rng : RandomGenerator, optional
            Random number generator. If None, creates a new one.
        """
        # Validate inputs
        if not 1 <= config.num_residents <= 6:
            raise ValueError(f"Number of residents must be 1-6, got {config.num_residents}")

        self.config = config
        self.data_loader = data_loader
        self.rng = rng if rng is not None else RandomGenerator()

        # Number of possible states: (residents + 1) * (residents + 1)
        # States are encoded as strings like "10", "11", "00", etc.
        # First digit: number at home (0 to residents)
        # Second digit: number active (0 to residents)
        self.num_possible_states = (config.num_residents + 1) ** 2

        # Storage arrays (10-minute resolution)
        self.combined_states = np.empty(TIMESTEPS_PER_DAY_10MIN, dtype='U2')  # String array for states like "11"
        self.active_occupancy = np.zeros(TIMESTEPS_PER_DAY_10MIN, dtype=int)  # Number actively occupied
        self.thermal_gains = np.zeros(TIMESTEPS_PER_DAY_10MIN, dtype=float)  # Thermal gains in Watts

        # Load TPM for this dwelling
        self.tpm = self.data_loader.load_occupancy_tpm(
            config.num_residents,
            config.is_weekend
        ).values

        # Load starting states
        self.starting_states_df = self.data_loader.load_starting_states()

        # Load 24-hour occupancy data
        self.occupancy_24hr_df = self.data_loader.load_24hr_occupancy()

        # Lazy initialization state
        self._initialized = False
        self._initial_state = None
        self._is_24hr_occupancy_dwelling = False

    def initialize(self):
        """
        Initialize occupancy state (2 RNG calls).

        VBA Reference: clsOccupancy lines 204-238
        - Select initial state at 00:00
        - Determine 24-hour occupancy flag

        Must be called before run_simulation().
        """
        if self._initialized:
            return

        # Determine initial state at 00:00 (1 RNG call)
        # VBA: clsOccupancy:204
        self._initial_state = self._get_initial_state()
        self.combined_states[0] = self._initial_state
        self.active_occupancy[0] = self._extract_active_occupancy(self._initial_state)

        # Determine if this is a 24-hour occupancy dwelling (1 RNG call)
        # VBA: clsOccupancy:238
        self._is_24hr_occupancy_dwelling = self._determine_24hr_occupancy()

        self._initialized = True

    def run_simulation(self):
        """
        Run the four-state occupancy simulation for a full day (143 RNG calls).

        Generates occupancy patterns using a Markov chain model with transition
        probabilities based on time of day, current state, and number of residents.

        VBA Reference: clsOccupancy:298 (143 state transitions)

        Must call initialize() first.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() before run_simulation()")

        current_state = self._initial_state

        # Simulate each subsequent 10-minute timestep (143 transitions)
        # VBA: clsOccupancy:298
        for timestep in range(1, TIMESTEPS_PER_DAY_10MIN):
            # Get transition probabilities for current state and time
            transition_probs = self._get_transition_probabilities(
                timestep,
                current_state,
                self._is_24hr_occupancy_dwelling
            )

            # Normalize and handle dead-end states
            transition_probs = markov.normalize_probabilities(transition_probs)

            # Select next state using inverse transform method
            rng_value = self.rng.random()
            next_state_idx = markov.select_next_state(transition_probs, rng_value)

            # Get state label from TPM header row (row 0, columns starting at index 2)
            current_state = self.tpm[0, next_state_idx + 2]

            # Store results
            self.combined_states[timestep] = current_state
            self.active_occupancy[timestep] = self._extract_active_occupancy(current_state)

        # Calculate thermal gains for all timesteps
        self._calculate_thermal_gains()

    def _get_initial_state(self) -> str:
        """
        Determine the initial occupancy state at 00:00.

        Uses the starting states distribution to randomly select an initial state.

        State encoding (based on four-state occupancy model):
        -------------------------------------------------------
        The model tracks four states per resident:
        1. At home and active
        2. At home and asleep
        3. Away from home and active
        4. Away from home and asleep

        These are aggregated into a two-digit state string "XY":
        - X = number of residents AT HOME (active or asleep)
        - Y = number of residents ACTIVE (at home or away)

        Examples:
        - "31" = 3 at home, 1 active (could be 1 at-home+active, 2 at-home+asleep)
        - "01" = 0 at home, 1 active (1 away+active, others away+asleep) - VALID state
        - "52" = 5 at home, 2 active (could be 2 at-home+active, 3 at-home+asleep)

        The VBA comment "they are not necessarily at home" (line 345-349) refers to
        the fact that active residents can be away from home (states 3 and 4).

        Returns
        -------
        str
            Initial state (e.g., "10", "11", "31")
        """
        # Column index for this number of residents (1-indexed in CSV)
        col_idx = self.config.num_residents  # Assuming 0-based, column 1 is for 1 resident

        # Row offset for weekend vs weekday
        # VBA: rows 7-54 weekday (1-based), 61-108 weekend (1-based)
        # After removing "Combined state" row from DataFrame:
        # Weekday data starts at row 0 (first data row "00")
        # Weekend data starts 54 rows later (49 weekday states + 5 header rows in original CSV)
        # But we've removed the "Combined state" row, so weekend is at row 53
        row_offset = 53 if self.config.is_weekend else 0

        # Get probabilities for all states
        state_probs = []
        state_labels = []
        for i in range(49):  # VBA loops 0 to 48 (49 states)
            row_idx = row_offset + i
            if row_idx < len(self.starting_states_df):
                prob = self.starting_states_df.iloc[row_idx, col_idx]
                state = self.starting_states_df.iloc[row_idx, 0]
                state_probs.append(prob)
                state_labels.append(state)

        # Select initial state using cumulative probability
        state_probs = np.array(state_probs)
        state_probs = state_probs / np.sum(state_probs)  # Normalize just in case

        rng_value = self.rng.random()
        state_idx = markov.select_next_state(state_probs, rng_value)
        selected_state = str(state_labels[state_idx])

        return selected_state

    def _determine_24hr_occupancy(self) -> bool:
        """
        Determine if this dwelling should have 24-hour occupancy.

        Uses uplift factors to ensure the correct proportion of dwellings have
        someone always at home, matching UK time-use survey statistics.

        Returns
        -------
        bool
            True if this dwelling should have 24-hour occupancy
        """
        # Get uplift factor from 24hr occupancy data
        # VBA: ws24hrOccupancy.Cells(intResidents + 3, IIf(blnWeekend = False, 6, 7))
        # VBA row numbering (1-based, with 3 header rows):
        #   Row 4 = resident 1, Row 5 = resident 2, ..., Row 9 = resident 6
        # Python after skiprows=2: Row 0 = header, Row 1-6 = residents 1-6
        # So: VBA row (residents + 3) → Python iloc[residents - 1]
        row_idx = self.config.num_residents - 1

        # VBA columns 6 (weekday) and 7 (weekend) are the uplift factors (1-based)
        # Python iloc columns: 5 (weekday) and 6 (weekend) (0-based)
        col_idx = 5 if not self.config.is_weekend else 6

        uplift_factor = self.occupancy_24hr_df.iloc[row_idx, col_idx]

        # Random decision based on uplift factor
        return self.rng.random() < uplift_factor

    def _get_transition_probabilities(
        self,
        timestep: int,
        current_state: str,
        is_24hr_occupancy: bool
    ) -> np.ndarray:
        """
        Get transition probabilities for the current timestep and state.

        Parameters
        ----------
        timestep : int
            Current timestep (0-based, 0-143)
        current_state : str
            Current combined state (e.g., "10")
        is_24hr_occupancy : bool
            Whether to modify probabilities for 24-hour occupancy

        Returns
        -------
        np.ndarray
            Array of transition probabilities
        """
        # Calculate row index in TPM using VBA-compatible formula
        # VBA uses 1-based indexing, so we need to convert
        row_idx = markov.calculate_tpm_row_index(
            timestep + 1,  # Convert to 1-based
            current_state,
            self.config.num_residents,
            self.num_possible_states,
            vba_compatible=True
        )

        # Extract row of transition probabilities (skip first 2 columns which are labels)
        # VBA indexing: row_idx is 1-based, convert to 0-based
        transition_probs = self.tpm[row_idx - 1, 2:].astype(float).copy()

        # Modify for 24-hour occupancy if needed
        if is_24hr_occupancy:
            transition_probs = markov.modify_24hr_occupancy_probabilities(
                transition_probs,
                self.config.num_residents
            )

        return transition_probs

    def _extract_active_occupancy(self, state: str) -> int:
        """
        Extract the number of actively occupied residents from a state string.

        Active occupancy is the minimum of (people at home, people active).

        Parameters
        ----------
        state : str
            Combined state string (e.g., "10" means 1 at home, 0 active)

        Returns
        -------
        int
            Number of active occupants
        """
        at_home = int(state[0])
        active = int(state[1])
        return min(at_home, active)

    def _calculate_thermal_gains(self):
        """
        Calculate thermal gains from occupants for all timesteps.

        Active occupants generate 147W, dormant occupants generate 84W.
        """
        for i in range(TIMESTEPS_PER_DAY_10MIN):
            state = self.combined_states[i]
            at_home = int(state[0])
            active = int(state[1])
            active_occupants = self.active_occupancy[i]

            # Dormant occupants = (at home) - (active)
            dormant_occupants = max(0, at_home - active)

            # Total thermal gain
            self.thermal_gains[i] = (
                OCCUPANT_THERMAL_GAIN_DORMANT * dormant_occupants +
                OCCUPANT_THERMAL_GAIN_ACTIVE * active_occupants
            )

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_active_occupancy_at_timestep(self, ten_minute_step: int) -> int:
        """
        Get active occupancy at a specific 10-minute timestep.

        Parameters
        ----------
        ten_minute_step : int
            Timestep index (0-143)

        Returns
        -------
        int
            Number of active occupants
        """
        return self.active_occupancy[ten_minute_step]

    def get_thermal_gain_at_timestep(self, ten_minute_step: int) -> float:
        """
        Get thermal gain at a specific 10-minute timestep.

        Parameters
        ----------
        ten_minute_step : int
            Timestep index (0-143)

        Returns
        -------
        float
            Thermal gain in Watts
        """
        return self.thermal_gains[ten_minute_step]

    def get_mean_active_occupancy(self) -> float:
        """
        Calculate mean active occupancy over the day.

        Returns
        -------
        float
            Mean number of active occupants
        """
        return np.mean(self.active_occupancy)

    def get_proportion_actively_occupied(self) -> float:
        """
        Calculate proportion that dwelling is actively occupied.

        Returns
        -------
        float
            Proportion of timesteps with at least one active occupant
        """
        occupied_steps = np.sum(self.active_occupancy >= 1)
        return occupied_steps / TIMESTEPS_PER_DAY_10MIN

    def get_active_occupancy_1min(self) -> np.ndarray:
        """
        Get active occupancy at 1-minute resolution.

        Expands 10-minute data by repeating each value 10 times.

        Returns
        -------
        np.ndarray
            Active occupancy for each minute (length 1440)
        """
        return np.repeat(self.active_occupancy, 10)

    def get_combined_states_1min(self) -> np.ndarray:
        """
        Get combined states at 1-minute resolution.

        Expands 10-minute data by repeating each value 10 times.

        Returns
        -------
        np.ndarray
            Combined state strings for each minute (length 1440)
        """
        return np.repeat(self.combined_states, 10)
