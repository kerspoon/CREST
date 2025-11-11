"""
Appliances Model

Simulates 31 different domestic appliance types with stochastic switching
based on activity patterns and occupancy.

AUDIT STATUS: ✅ COMPLETE - Full VBA implementation (clsAppliances.cls)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    MAX_APPLIANCE_TYPES,
    Country,
    UrbanRural
)
from ..utils.random import RandomGenerator
from ..data.loader import CRESTDataLoader


@dataclass
class ApplianceSpec:
    """Specification for an appliance (from CSV)."""
    name: str
    rated_power: int  # Watts (will be varied with Monte Carlo)
    standby_power: int  # Watts
    cycle_length: int  # Minutes
    restart_delay: int  # Minutes
    ownership_prob: float
    use_profile: str  # Activity profile name
    prob_switch_on: float
    heat_gains_ratio: float  # Fraction of power that becomes heat


@dataclass
class AppliancesConfig:
    """Configuration for appliances model."""
    dwelling_index: int
    country: Country = Country.UK  # Country for appliance ownership
    urban_rural: UrbanRural = UrbanRural.URBAN  # Urban/Rural for appliance ownership
    run_number: int = 0


class Appliances:
    """
    Domestic appliances model.

    Simulates electrical demand from 31 appliance types using activity-based
    stochastic switching.

    VBA Reference: clsAppliances.cls (481 lines)
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
            Activity probability profiles (key format: "weekend_activeOcc_profile")
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
        # VBA: aSimulationArray(1 To 1442, 1 To 31) - rows 1-2 headers, 3-1442 data
        # Python: Just store the power data (1440 minutes x 31 appliances)
        self.appliance_demands = np.zeros((TIMESTEPS_PER_DAY_1MIN, MAX_APPLIANCE_TYPES))  # W

        # VBA: aTotalApplianceDemand(1 To 1440, 1 To 1) - total across appliances
        self.total_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W

        # VBA: aApplianceThermalGains(1 To 1440, 1 To 1)
        self.thermal_gains = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W

        # Reference to occupancy model (set externally)
        self.occupancy = None

        # References to other systems for total demand calculation (set externally)
        self.heating_system = None
        self.solar_thermal = None
        self.cooling_system = None

    def _load_appliance_specs(self):
        """
        Load appliance specifications from CSV.

        VBA Reference: Lines 162-176 (loads from wsAppliancesAndWaterFixtures)
        CSV columns (VBA Excel references -> 0-based indices after skiprows=3):
        - E: Short name (column 4)
        - F: Proportion/Ownership (column 5)
        - G: Activity use profile (column 6)
        - P: Mean cycle power (column 15)
        - R: Mean cycle length (column 17)
        - S: Delay after cycle (column 18)
        - T: Standby power (column 19)
        - V: Target demand (column 21)
        - X: Cycles per year (column 23)
        - AD: Probability of switch on (column 29)
        - AF: Heat gains ratio (column 32)
        """
        appliances_data = self.data_loader.load_appliances_and_fixtures()

        # VBA loads 31 appliances (intAppliance = 1 To 31)
        self.appliances = []
        self.has_appliance = []

        # Note: CSV has 4 header rows when loaded with skiprows=3, header=0:
        # iloc[0-3] = additional headers, iloc[4] = first data row (CHEST_FREEZER)
        for i in range(min(MAX_APPLIANCE_TYPES, len(appliances_data) - 4)):
            row_idx = i + 4  # Skip 4 header rows to get to actual data
            if row_idx < len(appliances_data):
                row = appliances_data.iloc[row_idx]

                # Extract data matching VBA column references
                # CSV columns (1-based) → pandas iloc (0-based after skiprows=3):
                # - E (column 5, 1-based) = iloc[4] (0-based)
                # - F (column 6) = iloc[5]
                # - G (column 7) = iloc[6]
                # - P (column 16) = iloc[15]
                # - R (column 18) = iloc[17]
                # - S (column 19) = iloc[18]
                # - T (column 20) = iloc[19]
                # - AD (column 30) = iloc[29]
                # - AF (column 32, Excel 1-based) = iloc[31] (pandas 0-based)
                appliance_name = str(row.iloc[4]) if len(row) > 4 else f'Appliance{i}'
                ownership = float(row.iloc[5]) if len(row) > 5 else 0.5
                use_profile = str(row.iloc[6]) if len(row) > 6 else 'ACTIVE_OCC'
                rated_power = int(float(row.iloc[15])) if len(row) > 15 else 100
                cycle_length = int(float(row.iloc[17])) if len(row) > 17 else 60
                restart_delay = int(float(row.iloc[18])) if len(row) > 18 else 0
                standby_power = int(float(row.iloc[19])) if len(row) > 19 else 0
                prob_switch_on = float(row.iloc[29]) if len(row) > 29 else 0.01
                # BUG FIX: Excel column AF (32nd, 1-based) = pandas iloc[31] (0-based), not iloc[32]
                heat_gains_ratio = float(row.iloc[31]) if len(row) > 31 else 0.8

                self.appliances.append(ApplianceSpec(
                    name=appliance_name,
                    rated_power=rated_power,
                    standby_power=standby_power,
                    cycle_length=cycle_length,
                    restart_delay=restart_delay,
                    ownership_prob=ownership,
                    use_profile=use_profile,
                    prob_switch_on=prob_switch_on,
                    heat_gains_ratio=heat_gains_ratio
                ))

                # VBA: aApplianceConfiguration(i, 1) = IIf(dblRan < dblProportion, True, False)
                # VBA line 125
                self.has_appliance.append(self.rng.random() < ownership)

    def set_occupancy(self, occupancy):
        """Set reference to occupancy model."""
        self.occupancy = occupancy

    def set_heating_system(self, heating_system):
        """Set reference to heating system for total demand calculation."""
        self.heating_system = heating_system

    def set_solar_thermal(self, solar_thermal):
        """Set reference to solar thermal for total demand calculation."""
        self.solar_thermal = solar_thermal

    def set_cooling_system(self, cooling_system):
        """Set reference to cooling system for total demand calculation."""
        self.cooling_system = cooling_system

    def run_simulation(self):
        """
        Run appliance demand simulation.

        VBA Reference: RunApplianceSimulation (lines 137-283)

        Generates stochastic appliance usage events based on occupancy and activities.
        """
        if self.occupancy is None:
            raise RuntimeError("Occupancy model must be set before running simulation")

        # VBA: aActiveOccupancy = aOccupancy(intRunNumber).GetActiveOccupancy
        # Get active occupancy array (10-minute resolution, 144 periods)
        active_occupancy = self.occupancy.active_occupancy

        # VBA: For intAppliance = 1 To 31 (line 156)
        for app_idx, appliance in enumerate(self.appliances):

            # VBA: Check if dwelling has this appliance (lines 179-190)
            if not self.has_appliance[app_idx]:
                # Dwelling doesn't own this appliance - write zeros
                self.appliance_demands[:, app_idx] = 0
                continue

            # VBA: Initialisation (line 159)
            cycle_time_left = 0
            restart_delay_time_left = 0

            # VBA: Randomly delay the start of appliances with restart delay (line 195)
            # intRestartDelayTimeLeft = Rnd() * intRestartDelay * 2
            restart_delay_time_left = int(self.rng.random() * appliance.restart_delay * 2)

            # VBA: Make the rated power variable over a normal distribution (line 197)
            # intRatedPower = GetMonteCarloNormalDistGuess(Val(intRatedPower), intRatedPower / 10)
            rated_power = self._get_monte_carlo_normal_dist_guess(
                appliance.rated_power,
                appliance.rated_power / 10
            )

            # VBA: Loop through each minute of the day (line 203-276)
            # intMinute = 1, Do While (intMinute <= 1440)
            for minute in range(TIMESTEPS_PER_DAY_1MIN):

                # VBA: Set default (standby) power (line 207)
                power = appliance.standby_power

                # VBA: Get the ten minute period count (line 210)
                # intTenMinuteCount = ((intMinute - 1) \ 10)
                # Note: VBA minute is 1-based, Python is 0-based, so formula matches
                ten_min_idx = minute // 10

                # VBA: Get active occupants for this minute (line 213)
                # intActiveOccupants = aActiveOccupancy(intTenMinuteCount, 0)
                active_occupants = active_occupancy[ten_min_idx]

                # VBA: If appliance is off waiting for restart delay (lines 218-222)
                if cycle_time_left <= 0 and restart_delay_time_left > 0:
                    # Decrement the restart delay
                    restart_delay_time_left -= 1
                    power = appliance.standby_power

                # VBA: Else if this appliance is off (lines 225-252)
                elif cycle_time_left <= 0:

                    # VBA: There must be active occupants, or profile must not depend on occupancy (line 228)
                    # If (intActiveOccupants > 0 And strUseProfile <> "CUSTOM") Or (strUseProfile = "LEVEL") Then
                    can_start = (
                        (active_occupants > 0 and appliance.use_profile != "CUSTOM") or
                        (appliance.use_profile == "LEVEL")
                    )

                    if can_start:
                        # VBA: Default activity probability to 1 (line 231)
                        activity_probability = 1.0

                        # VBA: For appliances that depend on activity profiles (lines 234-242)
                        # If (strUseProfile <> "LEVEL") And (strUseProfile <> "ACTIVE_OCC") And (strUseProfile <> "CUSTOM")
                        if (appliance.use_profile != "LEVEL" and
                            appliance.use_profile != "ACTIVE_OCC" and
                            appliance.use_profile != "CUSTOM"):

                            # VBA: Get activity probability for this profile (lines 237-240)
                            # strKey = IIf(blnWeekend, "1", "0") + "_" + CStr(intActiveOccupants) + "_" + strUseProfile
                            weekend_flag = "1" if self.is_weekend else "0"
                            key = f"{weekend_flag}_{active_occupants}_{appliance.use_profile}"

                            # Get activity probability from statistics
                            if key in self.activity_statistics:
                                # VBA: dblActivityProbability = objActivityStatistics(strKey).Modifiers(intTenMinuteCount)
                                # Note: Modifiers array is 0-based in our implementation
                                activity_probability = self.activity_statistics[key][ten_min_idx]
                            else:
                                # If profile not found, use very low probability
                                activity_probability = 0.001

                        # VBA: Check probability and switch on (line 245)
                        # If (Rnd() < (dblActivityProbability * dblProbSwitchOn)) Then
                        if self.rng.random() < (activity_probability * appliance.prob_switch_on):
                            # VBA: StartAppliance (line 248)
                            cycle_time_left, power = self._start_appliance(appliance, rated_power)

                            # BUG FIX: VBA line 392 sets intRestartDelayTimeLeft = intRestartDelay
                            # This must be reset EVERY time the appliance starts, not just at initialization
                            restart_delay_time_left = appliance.restart_delay

                            # VBA decrements cycle_time_left in StartAppliance (line 398)
                            cycle_time_left -= 1
                        else:
                            power = appliance.standby_power
                    else:
                        power = appliance.standby_power

                # VBA: Else appliance is on (lines 253-268)
                else:
                    # VBA: If occupants become inactive, check if we should continue (line 255)
                    # If (intActiveOccupants = 0) And (strUseProfile <> "LEVEL") And (strUseProfile <> "ACT_LAUNDRY") And (strUseProfile <> "CUSTOM")
                    should_pause = (
                        active_occupants == 0 and
                        appliance.use_profile != "LEVEL" and
                        appliance.use_profile != "ACT_LAUNDRY" and
                        appliance.use_profile != "CUSTOM"
                    )

                    if should_pause:
                        # VBA: Do nothing - activity will complete upon return (lines 257-259)
                        # Don't decrement cycle_time_left, don't set power (keep standby)
                        power = appliance.standby_power
                    else:
                        # VBA: Set the power and decrement (lines 263-266)
                        power = self._get_power_usage(appliance, rated_power, cycle_time_left)
                        cycle_time_left -= 1

                # VBA: Set the appliance power at this time step (line 272)
                # aSimulationArray(2 + intMinute, intAppliance) = intPower
                # Note: VBA minute is 1-based (1-1440), array offset by 2 for headers
                # Python minute is 0-based (0-1439)
                self.appliance_demands[minute, app_idx] = power

        # After all appliances simulated, calculate totals
        # VBA: TotalApplianceDemand() called separately (lines 292-314)
        self._calculate_total_demand()

        # VBA: CalculateThermalGains() called separately (lines 349-377)
        self._calculate_thermal_gains()

    def _start_appliance(self, appliance: ApplianceSpec, rated_power: int) -> tuple:
        """
        Start a cycle for the current appliance.

        VBA Reference: StartAppliance (lines 386-400)

        Returns
        -------
        tuple
            (cycle_time_left, power) after starting
        """
        # VBA: Determine how long this appliance is going to be on for (line 389)
        # intCycleTimeLeft = CycleLength()
        cycle_time_left = self._cycle_length(appliance)

        # VBA: Determine if this appliance has a delay (line 392)
        # intRestartDelayTimeLeft = intRestartDelay
        # Note: This is set in the state machine, not returned here

        # VBA: Set the power (line 395)
        power = self._get_power_usage(appliance, rated_power, cycle_time_left)

        # VBA: Decrement the cycle time left (line 398)
        # intCycleTimeLeft = intCycleTimeLeft - 1
        # Note: We return the value before decrement, caller will decrement

        return cycle_time_left, power

    def _cycle_length(self, appliance: ApplianceSpec) -> int:
        """
        Determine cycle length for an appliance.

        VBA Reference: CycleLength() (lines 412-431)

        Returns
        -------
        int
            Cycle length in minutes
        """
        # VBA: Set the value to that provided in configuration (line 415)
        cycle_length = appliance.cycle_length

        # VBA: Use the TV watching length data approximation (lines 418-422)
        # If (strApplianceType = "TV1") Or (strApplianceType = "TV2") Or (strApplianceType = "TV3") Then
        if appliance.name in ["TV1", "TV2", "TV3"]:
            # VBA: CycleLength = CInt(70 * ((0 - Log(1 - Rnd())) ^ 1.1))
            # Note: VBA Log is natural log (ln), Python np.log is also natural log
            # Average viewing time is approximately 73 minutes
            cycle_length = int(70 * ((0 - np.log(1 - self.rng.random())) ** 1.1))

        # VBA: Heating appliances get variation (lines 424-428)
        elif appliance.name in ["STORAGE_HEATER", "ELEC_SPACE_HEATING"]:
            # VBA: CycleLength = GetMonteCarloNormalDistGuess(CDbl(intMeanCycleLength), intMeanCycleLength / 10)
            cycle_length = self._get_monte_carlo_normal_dist_guess(
                float(appliance.cycle_length),
                appliance.cycle_length / 10
            )

        return cycle_length

    def _get_power_usage(self, appliance: ApplianceSpec, rated_power: int, cycle_time_left: int) -> int:
        """
        Get power usage for appliance at current point in cycle.

        VBA Reference: GetPowerUsage() (lines 440-480)

        Parameters
        ----------
        appliance : ApplianceSpec
            Appliance specification
        rated_power : int
            Rated power (possibly varied from spec)
        cycle_time_left : int
            Minutes left in cycle

        Returns
        -------
        int
            Power in Watts
        """
        # VBA: Set the return power to the rated power (line 443)
        power = rated_power

        # VBA: Some appliances have a custom power profile (line 446)
        # Case "WASHING_MACHINE", "WASHER_DRYER":
        if appliance.name in ["WASHING_MACHINE", "WASHER_DRYER"]:

            # VBA: Calculate the total cycle time (lines 451-455)
            if appliance.name == "WASHING_MACHINE":
                total_cycle_time = 138
            else:  # WASHER_DRYER
                total_cycle_time = 198

            # VBA: Power profile based on minutes elapsed (lines 459-476)
            # Select Case (iTotalCycleTime - intCycleTimeLeft + 1)
            minutes_elapsed = total_cycle_time - cycle_time_left + 1

            # VBA power profile (exact values from lines 461-473)
            if 1 <= minutes_elapsed <= 8:
                power = 73  # Start-up and fill
            elif 9 <= minutes_elapsed <= 29:
                power = 2056  # Heating
            elif 30 <= minutes_elapsed <= 81:
                power = 73  # Wash and drain
            elif 82 <= minutes_elapsed <= 92:
                power = 73  # Spin
            elif 93 <= minutes_elapsed <= 94:
                power = 250  # Rinse
            elif 95 <= minutes_elapsed <= 105:
                power = 73  # Spin
            elif 106 <= minutes_elapsed <= 107:
                power = 250  # Rinse
            elif 108 <= minutes_elapsed <= 118:
                power = 73  # Spin
            elif 119 <= minutes_elapsed <= 120:
                power = 250  # Rinse
            elif 121 <= minutes_elapsed <= 131:
                power = 73  # Spin
            elif 132 <= minutes_elapsed <= 133:
                power = 250  # Rinse
            elif 134 <= minutes_elapsed <= 138:
                power = 568  # Fast spin
            elif 139 <= minutes_elapsed <= 198:
                power = 2500  # Drying cycle (washer dryer only)
            else:
                power = appliance.standby_power  # Case Else

        return power

    def _get_monte_carlo_normal_dist_guess(self, mean: float, std_dev: float) -> int:
        """
        Generate a normally distributed random value.

        VBA Reference: GetMonteCarloNormalDistGuess function (in mdlThermalElectricalModel.bas)

        Parameters
        ----------
        mean : float
            Mean value
        std_dev : float
            Standard deviation

        Returns
        -------
        int
            Random value from normal distribution
        """
        # VBA uses WorksheetFunction.NormInv(Rnd(), mean, std_dev)
        # Python equivalent: use Box-Muller or inverse CDF
        value = self.rng.normal(mean, std_dev)
        return int(value)

    def _calculate_total_demand(self):
        """
        Calculate total appliance demand including other systems.

        VBA Reference: TotalApplianceDemand (lines 292-314)
        """
        # VBA: Sum across all appliances for each minute (lines 297-305)
        # For intRow = 1 To 1440
        #     For intCol = 1 To 31
        #         dblRowSum = dblRowSum + aSimulationArray(intRow + 2, intCol)
        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            row_sum = np.sum(self.appliance_demands[minute, :])

            # VBA: Add heating system, solar thermal pump, and cooling system (lines 307-310)
            # aTotalApplianceDemand(intRow, 1) = dblRowSum _
            #     + aPrimaryHeatingSystem(intRunNumber).GetHeatingSystemPowerDemand(intRow) _
            #     + aSolarThermal(intRunNumber).GetP_pumpsolar(intRow) _
            #     + aCoolingSystem(intRunNumber).GetCoolingSystemPowerDemand(intRow)

            # Add heating system demand if available
            if self.heating_system is not None:
                row_sum += self.heating_system.get_heating_system_power_demand(minute + 1)  # 1-based API

            # Add solar thermal pump demand if available
            if self.solar_thermal is not None:
                row_sum += self.solar_thermal.get_P_pumpsolar(minute + 1)  # 1-based API

            # Add cooling system demand if available
            if self.cooling_system is not None:
                row_sum += self.cooling_system.get_cooling_system_power_demand(minute + 1)  # 1-based API

            self.total_demand[minute] = row_sum

    def _calculate_thermal_gains(self):
        """
        Calculate thermal gains from appliances.

        VBA Reference: CalculateThermalGains (lines 349-377)
        """
        # VBA: Load heat gains ratio array (line 357)
        # vntHeatGainsRatio = wsAppliancesAndWaterFixtures.Range("rHeatGainsRatio").Value
        # Note: We loaded this in _load_appliance_specs as heat_gains_ratio

        # VBA: For each minute (lines 359-375)
        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            thermal_sum = 0.0

            # VBA: For each appliance (lines 363-371)
            for app_idx in range(len(self.appliances)):
                # VBA: Only if dwelling has this appliance (line 364)
                if self.has_appliance[app_idx]:
                    # VBA: Get appliance power and multiply by heat gains ratio (lines 366-368)
                    # dblAppliancePower = aSimulationArray(intMinute + 2, intApplianceIndex)
                    # dblSum = dblSum + dblAppliancePower * vntHeatGainsRatio(intApplianceIndex, 1)
                    appliance_power = self.appliance_demands[minute, app_idx]
                    thermal_sum += appliance_power * self.appliances[app_idx].heat_gains_ratio

            # VBA: Store thermal gains (line 373)
            self.thermal_gains[minute] = thermal_sum

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_total_demand(self, timestep: int) -> float:
        """
        Get total appliance demand at specified timestep (1-based) in Watts.

        VBA Reference: GetTotalApplianceDemand property (lines 67-69)
        """
        # VBA: GetTotalApplianceDemand = aTotalApplianceDemand(timestep, 1)
        return self.total_demand[timestep - 1]

    def get_thermal_gain(self, timestep: int) -> float:
        """
        Get thermal gains at specified timestep (1-based) in Watts.

        VBA Reference: GetPhi_cAppliances property (lines 77-79)
        """
        # VBA: GetPhi_cAppliances = aApplianceThermalGains(timestep, 1)
        return self.thermal_gains[timestep - 1]

    def get_daily_energy(self) -> float:
        """
        Get total daily appliance sum (matches VBA units exactly).

        VBA Reference: GetDailySumApplianceDemand property (lines 81-83)

        Note: VBA returns raw sum without unit conversion. For exact VBA matching,
        we return the same. Units are technically W summed over 1440 minutes.
        """
        # VBA: GetDailySumApplianceDemand = WorksheetFunction.Sum(aTotalApplianceDemand)
        # BUG FIX: Removed /60.0 conversion - VBA does NOT convert units
        return np.sum(self.total_demand)

    def get_appliance_demand(self, timestep: int, appliance_idx: int) -> float:
        """
        Get power demand for a specific appliance at a timestep.

        Parameters
        ----------
        timestep : int
            Timestep (1-based, 1-1440)
        appliance_idx : int
            Appliance index (0-based, 0-30)

        Returns
        -------
        float
            Power demand in Watts
        """
        return self.appliance_demands[timestep - 1, appliance_idx]
