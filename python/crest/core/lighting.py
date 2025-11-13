"""
Lighting Model

Simulates domestic lighting demand with irradiance-based switching.
Handles up to 60 bulbs per dwelling with occupancy-dependent operation.

AUDIT STATUS: ✅ COMPLETE - Full VBA implementation (clsLighting.cls)
"""

import numpy as np
import pandas as pd
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
    - Stochastic switching behavior with duration persistence

    VBA Reference: clsLighting.cls (292 lines)
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

        # VBA: intDwellingIndex, intRunNumber (lines 19-20)
        self.dwelling_index = config.dwelling_index
        self.run_number = config.run_number

        # VBA: aSimulationArray(1 To 1443, 1 To 60) (line 31)
        # Row 1: Bulb numbers, Row 2: Ratings, Row 3: Relative use, Rows 4-1443: Demand
        # Python: Store separately for clarity
        self.bulb_demands = np.zeros((TIMESTEPS_PER_DAY_1MIN, MAX_BULBS_PER_DWELLING))  # W

        # VBA: aTotalLightingDemand(1 To 1440, 1 To 1) (line 42)
        self.total_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W

        # VBA: Note - thermal gains = lighting demand (100% conversion) (line 55)
        self.thermal_gains = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W

        # References (set externally)
        self.occupancy = None
        self.local_climate = None

        # BUG FIX #1: Bulb relative use will be generated in run_simulation()
        # to match VBA's RNG sequence (generated per-bulb, interleaved with simulation)
        self.bulb_relative_use = None

        # Lazy initialization state
        self._initialized = False

    def initialize(self):
        """
        Initialize lighting configuration with RNG calls (2 RNG calls).

        VBA Reference: clsLighting lines 79-87
        - Line 83: Determine irradiance threshold (normal distribution) - 1 RNG call
        - Line 87: Select bulb configuration (1-100) - 1 RNG call

        Must be called before run_simulation().
        """
        if self._initialized:
            return

        # Load lighting configuration (includes 2 RNG calls)
        self._load_lighting_config()

        self._initialized = True

    def _load_lighting_config(self):
        """
        Load lighting configuration from CSV files.

        VBA Reference: InitialiseLighting (lines 74-102)
        """
        # Load raw CSV for parameter extraction
        # BUG FIX #5: Load from CSV instead of hardcoding
        light_config_raw = pd.read_csv(
            self.data_loader.data_dir / "light_config.csv",
            header=None
        )

        # VBA: Determine irradiance threshold using Monte Carlo (lines 81-84)
        # BUG FIX #5: Load from CSV row 4 (0-indexed: 3), columns F-G (5-6)
        irradiance_mean = float(light_config_raw.iloc[3, 5])  # Row 4, Col F
        irradiance_sd = float(light_config_raw.iloc[3, 6])     # Row 4, Col G

        self.irradiance_threshold = self._get_monte_carlo_normal_dist_guess(
            irradiance_mean,
            irradiance_sd
        )

        print(f"  Irradiance threshold params: mean={irradiance_mean}, sd={irradiance_sd}, th={self.irradiance_threshold}")

        # VBA: Choose a random house from 100 provided (lines 86-87)
        # intBulbConfiguration = Int((100 * Rnd) + 1)  ' 1 to 100
        bulb_config_idx = int(self.rng.random() * 100) + 1  # 1-100

        # VBA: Get the bulb data (line 90)
        # aBulbArray = wsBulbs.Range("A" + CStr(intBulbConfiguration + 10) + ":BI" + CStr(intBulbConfiguration + 10))
        # Excel row = bulb_config_idx + 10 (e.g., config 1 → row 11)
        # CSV line = Excel row (1-indexed)
        # Pandas iloc = CSV line - 1 (0-indexed)
        bulbs_data = self.data_loader.load_bulbs()

        # Skip header rows (10 rows) and get the specific configuration
        # bulb_config_idx ranges from 1-100, so we need row index (bulb_config_idx - 1) after skipping headers
        bulb_row_idx = bulb_config_idx - 1

        if bulb_row_idx >= len(bulbs_data):
            # Fallback if index out of range
            bulb_row_idx = 0

        bulb_row = bulbs_data.iloc[bulb_row_idx]

        # VBA: Get the number of bulbs (line 93)
        # intNumBulbs = aBulbArray(1, 2)
        self.num_bulbs = int(bulb_row.iloc[1])  # Column 2 (0-indexed: column 1)

        # VBA: Get bulb ratings from columns 3+ (line 134)
        # intRating = aBulbArray(1, i + 2)  where i = 1 to intNumBulbs
        self.bulb_powers = np.zeros(self.num_bulbs)

        # BUG FIX #5: Load India scaling factor from CSV row 24 (0-indexed: 23), column G (6)
        india_scaling_factor = float(light_config_raw.iloc[23, 6])  # Row 24, Col G = 0.275

        for i in range(self.num_bulbs):
            # Column index: i + 2 (VBA) = i + 2 (0-indexed)
            rating = float(bulb_row.iloc[i + 2])

            # VBA: Scale down bulb power for India (lines 136-140)
            if self.config.country == Country.INDIA:
                rating = rating * india_scaling_factor

            self.bulb_powers[i] = rating

        # VBA: Get calibration scalar (lines 99-100)
        # dblCalibrationScalar = wsLightConfig.Range("F24").Value
        # BUG FIX #5: Load from CSV row 24 (0-indexed: 23), column F (5)
        self.calibration_scalar = float(light_config_raw.iloc[23, 5])  # Row 24, Col F

        # BUG FIX #1: DO NOT generate relative use weightings here!
        # VBA generates them in RunLightingSimulation (line 151), interleaved with bulb simulation
        # We'll generate them in run_simulation() to match VBA's RNG sequence

        # BUG FIX #5: Load effective occupancy from CSV rows 38-42 (0-indexed: 37-41), column E (4)
        self.effective_occupancy = np.zeros(6)  # 0-5 occupants
        self.effective_occupancy[0] = 0.0  # 0 occupants (row 37 is header, no data)
        for occ in range(1, 6):
            row_idx = 37 + occ - 1  # Row 38-42 (0-indexed: 37-41)
            self.effective_occupancy[occ] = float(light_config_raw.iloc[row_idx, 4])

        print(f"DWELLING {self.dwelling_index}:")
        print(f"  Calibration scalar: {self.calibration_scalar}")
        print(f"  Bulb config: {bulb_config_idx}, Num bulbs: {self.num_bulbs}")
        print(f"  Bulb powers: {self.bulb_powers[:5]}")  # First 5
        print(f"  Effective occ: {self.effective_occupancy}")

        # BUG FIX #5: Load duration ranges from CSV rows 55-63 (0-indexed: 54-62)
        self.duration_ranges = []
        for range_idx in range(9):  # 9 ranges
            row_idx = 54 + range_idx  # Rows 55-63 (0-indexed: 54-62)
            lower_dur = int(float(light_config_raw.iloc[row_idx, 2]))  # Col C
            upper_dur = int(float(light_config_raw.iloc[row_idx, 3]))  # Col D
            cumulative_prob = float(light_config_raw.iloc[row_idx, 4])  # Col E
            self.duration_ranges.append((lower_dur, upper_dur, cumulative_prob))

    def _get_monte_carlo_normal_dist_guess(self, mean: float, std_dev: float) -> float:
        """
        Generate a normally distributed random value.

        VBA Reference: GetMonteCarloNormalDistGuess function
        """
        return self.rng.normal(mean, std_dev)

    def set_occupancy(self, occupancy):
        """Set reference to occupancy model."""
        self.occupancy = occupancy

    def set_local_climate(self, local_climate):
        """Set reference to local climate model."""
        self.local_climate = local_climate

    def run_simulation(self):
        """
        Run lighting simulation.

        VBA Reference: RunLightingSimulation (lines 111-244)

        Generates stochastic bulb switching based on irradiance and occupancy,
        with duration persistence.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() before run_simulation()")

        if self.occupancy is None or self.local_climate is None:
            raise RuntimeError("Occupancy and climate must be set before running simulation")

        # VBA: Load active occupancy array (line 97)
        # intActiveOccupancy = aOccupancy(intRunNumber).GetActiveOccupancy
        active_occupancy_10min = self.occupancy.active_occupancy

        # BUG FIX #1: Generate bulb relative use weightings HERE, not in __init__
        # VBA generates them in RunLightingSimulation (line 151), one per bulb,
        # interleaved with simulation logic. This ensures correct RNG sequence.
        self.bulb_relative_use = np.zeros(self.num_bulbs)

        # VBA: For each bulb (line 131)
        # For i = 1 To intNumBulbs
        for bulb_idx in range(self.num_bulbs):

            # VBA: Get the bulb rating (already loaded in __init__)
            bulb_rating = self.bulb_powers[bulb_idx]

            # VBA: Assign random bulb use weighting (lines 149-152)
            # THIS HAPPENS HERE, not in __init__!
            # dblCalibratedRelativeUseWeighting = -dblCalibrationScalar * Application.WorksheetFunction.Ln(Rnd())
            calibrated_relative_use = -self.calibration_scalar * np.log(self.rng.random())
            self.bulb_relative_use[bulb_idx] = calibrated_relative_use

            # VBA: Calculate the bulb usage at each minute of the day (lines 154-240)
            # intTime = 1, Do While (intTime <= 1440)
            minute = 0
            while minute < TIMESTEPS_PER_DAY_1MIN:

                # VBA: Get the irradiance for this minute (lines 162-163)
                # intIrradiance = aLocalClimate(intRunNumber).GetG_o(intTime)
                # Note: VBA intTime is 1-based, GetG_o expects 1-based
                irradiance = self.local_climate.get_irradiance(minute + 1)

                # VBA: Get active occupants (lines 165-167)
                # intActiveOccupants = intActiveOccupancy(((intTime - 1) \ 10), 0)
                ten_min_idx = minute // 10
                active_occupants = active_occupancy_10min[ten_min_idx]

                # BUG FIX #4: Use assert instead of clamping to match VBA exactly
                # VBA has no bounds checking - it just accesses the array
                # If >5 occupants exist, VBA would read beyond array (undefined behavior)
                # We assert this never happens to catch bugs while matching VBA behavior
                assert 0 <= active_occupants <= 5, \
                    f"Active occupants ({active_occupants}) out of range [0,5]. " \
                    f"Occupancy model should never generate >5 active occupants."

                # VBA: Determine if bulb switch-on condition is passed (lines 169-173)
                # blnLowIrradiance = ((intIrradiance < intIrradianceThreshold) Or (Rnd() < 0.05))
                # CRITICAL: VBA's Or operator does NOT short-circuit - it always evaluates both sides!
                # Python's 'or' DOES short-circuit, so we must evaluate random() first to match VBA
                rand_5pct = self.rng.random() < 0.05
                low_irradiance = (irradiance < self.irradiance_threshold) or rand_5pct

                # VBA: Get effective occupancy for sharing (line 176)
                # dblEffectiveOccupancy = wsLightConfig.Range("E" + CStr(37 + intActiveOccupants)).Value
                effective_occ = self.effective_occupancy[active_occupants]

                # VBA: Check probability of switch on (line 179)
                # If (blnLowIrradiance And (Rnd() < (dblEffectiveOccupancy * dblCalibratedRelativeUseWeighting))) Then
                # CRITICAL: VBA's And operator does NOT short-circuit - it always evaluates both sides!
                # Python's 'and' DOES short-circuit, so we must evaluate random() first to match VBA
                rand_switch = self.rng.random() < (effective_occ * calibrated_relative_use)
                if low_irradiance and rand_switch:

                    # VBA: This is a switch on event (line 181)

                    # VBA: Determine how long this bulb is on for (lines 183-209)
                    r1 = self.rng.random()

                    # Find duration range using cumulative probability
                    light_duration = 1  # Default
                    for lower_dur, upper_dur, cumulative_prob in self.duration_ranges:
                        if r1 < cumulative_prob:
                            # VBA: Get another random number and pick duration in range (lines 200-203)
                            r2 = self.rng.random()
                            # BUG FIX #2: Use round() instead of int()
                            # VBA assigns Double to Integer, which ROUNDS to nearest integer
                            # Python int() truncates (floors), which is different behavior
                            light_duration = round(r2 * (upper_dur - lower_dur) + lower_dur)
                            break

                    # VBA: Light stays on for duration (lines 211-228)
                    # For j = 1 To intLightDuration
                    for j in range(light_duration):

                        # VBA: Range check (line 214)
                        if minute >= TIMESTEPS_PER_DAY_1MIN:
                            break

                        # VBA: Get active occupants for this minute (line 217)
                        ten_min_idx = minute // 10
                        active_occupants = active_occupancy_10min[ten_min_idx]

                        # VBA: If no active occupants, turn off (lines 219-220)
                        if active_occupants == 0:
                            break

                        # VBA: Store the demand (line 223)
                        # aSimulationArray(3 + intTime, i) = intRating
                        self.bulb_demands[minute, bulb_idx] = bulb_rating

                        # VBA: Increment the time (line 226)
                        minute += 1

                else:
                    # VBA: The bulb remains off (lines 231-236)
                    # aSimulationArray(3 + intTime, i) = 0
                    self.bulb_demands[minute, bulb_idx] = 0
                    minute += 1

        # VBA: TotalLightingDemand() called separately (lines 252-272)
        self._calculate_total_demand()

    def _calculate_total_demand(self):
        """
        Calculate total lighting demand.

        VBA Reference: TotalLightingDemand (lines 252-272)
        """
        # VBA: Sum all bulbs for each minute (lines 257-269)
        # For intRow = 1 To 1440
        #     For intCol = 1 To 60
        #         dblRowSum = dblRowSum + aSimulationArray(intRow + 3, intCol)
        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            row_sum = np.sum(self.bulb_demands[minute, :])
            self.total_demand[minute] = row_sum

            # VBA: Thermal gains = lighting demand (100% conversion) (line 55-56)
            # ThermalGains = aTotalLightingDemand()
            self.thermal_gains[minute] = row_sum

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_total_demand(self, timestep: int) -> float:
        """
        Get total lighting demand at specified timestep (1-based) in Watts.

        VBA Reference: GetTotalLightingDemand property (lines 50-52)
        """
        # VBA: GetTotalLightingDemand = aTotalLightingDemand(timestep, 1)
        return self.total_demand[timestep - 1]

    def get_thermal_gain(self, timestep: int) -> float:
        """
        Get thermal gains at specified timestep (1-based) in Watts.

        VBA Reference: GetPhi_cLighting property (lines 60-62)
        """
        # VBA: GetPhi_cLighting = aTotalLightingDemand(timestep, 1)
        # Note: Thermal gains = lighting demand for lighting (100% conversion)
        return self.thermal_gains[timestep - 1]

    def get_daily_energy(self) -> float:
        """
        Get total daily lighting sum (matches VBA units exactly).

        VBA Reference: GetDailySumLighting property (lines 63-65)

        Note: VBA returns raw sum without unit conversion. Units are technically
        W summed over 1440 minutes.
        """
        # VBA: GetDailySumLighting = WorksheetFunction.Sum(aTotalLightingDemand)
        # BUG FIX #3: Removed /60.0 conversion - VBA does NOT convert units
        return np.sum(self.total_demand)
