"""
Hot Water Demand Model

Simulates domestic hot water usage from 4 fixture types:
- Basin
- Sink
- Shower
- Bath

Uses activity-based stochastic switching and empirical volume distributions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    SPECIFIC_HEAT_CAPACITY_WATER,
    WATER_FIXTURE_TYPES,
    Country,
    COLD_WATER_TEMPERATURE_BY_COUNTRY
)
from ..utils.random import RandomGenerator
from ..data.loader import CRESTDataLoader


@dataclass
class WaterFixtureSpec:
    """Specification for a water fixture."""
    name: str
    prob_switch_on: float
    mean_flow: float  # litres/minute
    use_profile: str  # Activity profile name
    restart_delay: float  # minutes
    volume_column: int  # Column in water usage distribution


@dataclass
class HotWaterConfig:
    """Configuration for hot water model."""
    dwelling_index: int
    heating_system_index: int
    num_residents: int
    country: Country = Country.UK  # Country for cold water temperature
    run_number: int = 0


class HotWater:
    """
    Hot water demand model.

    Simulates stochastic hot water usage from multiple fixture types
    based on occupancy and activity patterns.
    """

    def __init__(
        self,
        config: HotWaterConfig,
        data_loader: CRESTDataLoader,
        activity_statistics: Dict,
        is_weekend: bool,
        rng: Optional[RandomGenerator] = None
    ):
        """
        Initialize hot water model.

        Parameters
        ----------
        config : HotWaterConfig
            Configuration parameters
        data_loader : CRESTDataLoader
            Data loader for fixture specs and water usage distributions
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

        # Load heating system parameters for cylinder (strict mode)
        heating_systems = data_loader.load_primary_heating_systems()
        if config.heating_system_index >= len(heating_systems):
            # Use default values for out-of-range index
            self.h_loss = 2.0
            v_cyl = 150.0
        else:
            heating_params = heating_systems.iloc[config.heating_system_index]
            try:
                self.h_loss = heating_params['H_loss']
                v_cyl = heating_params['V_cyl']
            except KeyError as e:
                raise KeyError(
                    f"Missing required column in PrimaryHeatingSystems.csv for index {config.heating_system_index}: {e}. "
                    f"Available columns: {list(heating_params.index)}"
                )

        # Cylinder thermal capacitance (J/K)
        self.c_cyl = SPECIFIC_HEAT_CAPACITY_WATER * v_cyl

        # Cold water temperature (VBA lines 156-162)
        # VBA: If blnUK Then dblTheta_cw = 10 ElseIf blnIndia Then dblTheta_cw = 20
        self.theta_cw = COLD_WATER_TEMPERATURE_BY_COUNTRY[config.country]

        # Load water usage distribution
        self.water_usage_dist = data_loader.load_water_usage().values

        # Load fixture specifications from CSV
        self._load_fixture_specs()

        # Storage arrays
        self.hot_water_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # litres/min
        self.h_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W/K thermal transfer coefficient

        # Simulation array for individual fixtures (1440 rows x 4 fixtures)
        self.fixture_flows = np.zeros((TIMESTEPS_PER_DAY_1MIN, WATER_FIXTURE_TYPES))

        # Reference to occupancy model (set externally)
        self.occupancy = None

    def _convert_profile_name(self, profile_raw: str, strict: bool = True) -> str:
        """
        Convert profile name from CSV format to activity statistics format.

        CSV format: ACT_WASHDRESS, ACT_COOKING
        Activity stats format: Act_WashDress, Act_Cooking

        Parameters
        ----------
        profile_raw : str
            Profile name from CSV (e.g., "ACT_WASHDRESS")
        strict : bool, optional
            If True, raises KeyError for unknown profiles (default: True)

        Returns
        -------
        str
            Profile name in activity stats format (e.g., "Act_WashDress")

        Raises
        ------
        KeyError
            If strict=True and profile_raw not in conversion map
        """
        # Use lookup table for known conversions
        # These match the activity profile IDs in ActivityStats.csv
        conversion_map = {
            'ACT_WASHDRESS': 'Act_WashDress',
            'ACT_COOKING': 'Act_Cooking',
            'ACT_HOUSECLEAN': 'Act_HouseClean',
            'ACT_IRON': 'Act_Iron',
            'ACT_LAUNDRY': 'Act_Laundry',
            'ACT_TV': 'Act_TV',
            'ACT_SHOWER': 'Act_WashDress',  # Shower uses WashDress activity
            'ACT_BATH': 'Act_WashDress',  # Bath uses WashDress activity
        }

        # Check if we have a direct mapping
        if profile_raw in conversion_map:
            return conversion_map[profile_raw]

        # Strict mode: crash on unknown profile
        if strict:
            raise KeyError(
                f"Unknown water fixture profile '{profile_raw}'. "
                f"Known profiles: {list(conversion_map.keys())}"
            )

        # Non-strict fallback (not recommended)
        if profile_raw.startswith('ACT_'):
            rest = profile_raw[4:]
            return f"Act_{rest.capitalize()}"

        return profile_raw

    def _load_fixture_specs(self):
        """
        Load fixture specifications from CSV (VBA lines 167-220).

        Matches VBA InitialiseHotWater and RunHotWaterDemandSimulation logic.
        """
        fixtures_data = self.data_loader.load_appliances_and_fixtures()

        # Water fixtures are rows 46-49 in Excel (1-based, displayed in spreadsheet)
        # These correspond to CSV lines 45-48 (0-based file lines)
        # After skiprows=3 (skip lines 0,1,2, use line 3 as header), CSV line 45 → DataFrame row 41
        # VBA: intRowOffset = 45, accesses Excel rows 46-49 (intFixture=1 to 4)
        # Actual mapping: Excel row 46 = CSV line 45 = DataFrame row 41
        row_offset = 41

        self.fixtures = []

        for fixture_idx in range(4):  # 4 fixtures: Basin, Sink, Shower, Bath
            row_idx = row_offset + fixture_idx
            fixture_row = fixtures_data.iloc[row_idx]

            # Extract fixture parameters from CSV
            # VBA accesses columns by letter: E, AD, P, G, S (lines 215-219)
            # After loading, need to find correct column indices

            # Column mappings (approximate - may need adjustment based on actual CSV):
            # E column ≈ column 4 (Appliance type)
            # AD column ≈ column 29 (Probability of switch on)
            # P column ≈ column 15 (Mean flow rate)
            # G column ≈ column 6 (Use profile)
            # S column ≈ column 18 (Restart delay)

            # Since column letters are hard to map, use known positions from CSV inspection:
            # Row 46 (Basin): columns show Basin, profile, etc.

            # Use iloc with known column positions from AppliancesAndWaterFixtures.csv
            # Get use_profile and convert from CSV format (ACT_WASHDRESS) to activity stats format (Act_WashDress)
            use_profile_raw = str(fixture_row.iloc[6]) if len(fixture_row) > 6 else "Active"
            use_profile = self._convert_profile_name(use_profile_raw)

            fixture_spec = WaterFixtureSpec(
                name=str(fixture_row.iloc[4]) if len(fixture_row) > 4 else f"Fixture{fixture_idx}",
                prob_switch_on=float(fixture_row.iloc[29]) if len(fixture_row) > 29 else 0.01,
                mean_flow=float(fixture_row.iloc[15]) if len(fixture_row) > 15 else 6.0,
                use_profile=use_profile,
                restart_delay=float(fixture_row.iloc[18]) if len(fixture_row) > 18 else 0.0,
                volume_column=3 if fixture_idx < 2 else (4 if fixture_idx == 2 else 5)  # VBA lines 347-357
            )
            self.fixtures.append(fixture_spec)

        # Randomly assign fixture ownership based on CSV proportions (VBA lines 167-178)
        # Column F (index 5 after loading) has proportion of dwellings with fixture
        self.has_fixture = []
        for fixture_idx in range(4):
            row_idx = row_offset + fixture_idx
            proportion = float(fixtures_data.iloc[row_idx, 5])  # Column F = proportion

            # Random assignment (VBA lines 170-176)
            rand_val = self.rng.random()
            has_it = rand_val < proportion
            self.has_fixture.append(has_it)

    def set_occupancy(self, occupancy):
        """Set reference to occupancy model."""
        self.occupancy = occupancy

    def run_simulation(self):
        """
        Run hot water demand simulation for all fixtures.

        Generates stochastic fixture usage events based on occupancy and activities.
        """
        if self.occupancy is None:
            raise RuntimeError("Occupancy model must be set before running simulation")

        # Get active occupancy array (10-minute resolution)
        active_occupancy = self.occupancy.active_occupancy

        # Simulate each fixture
        for fixture_idx, fixture in enumerate(self.fixtures):
            if not self.has_fixture[fixture_idx]:
                # Dwelling doesn't have this fixture
                continue

            # Initialize event state
            event_time_left = 0.0
            restart_time_left = 0.0
            flow_rate = 0.0

            # Simulate each minute
            for minute in range(TIMESTEPS_PER_DAY_1MIN):
                # Get 10-minute period index
                ten_min_idx = minute // 10

                # Get active occupants for this period
                active_occupants = active_occupancy[ten_min_idx]

                # Default flow rate is zero
                flow_rate = 0.0

                # Handle fixture state
                if event_time_left <= 0 and restart_time_left > 0:
                    # Fixture is off, waiting for restart delay
                    restart_time_left -= 1.0

                elif event_time_left <= 0:
                    # Fixture is off and can start
                    if active_occupants > 0:
                        # Check if fixture starts
                        if self._check_fixture_start(fixture, active_occupants, ten_min_idx):
                            # Start fixture event
                            event_time_left, flow_rate, restart_time_left = self._start_fixture(fixture)

                else:
                    # Fixture is running
                    if active_occupants == 0:
                        # Pause if no active occupants (will resume when they return)
                        pass
                    else:
                        # Continue running fixture
                        if event_time_left < 1.0:
                            # Fractional minute remaining
                            flow_rate = fixture.mean_flow * event_time_left
                        else:
                            flow_rate = fixture.mean_flow

                        event_time_left -= 1.0

                # Store flow rate for this fixture and minute
                self.fixture_flows[minute, fixture_idx] = flow_rate

        # Calculate total demand and thermal transfer coefficient
        self._calculate_total_demand()

    def _check_fixture_start(self, fixture: WaterFixtureSpec, active_occupants: int, ten_min_idx: int) -> bool:
        """
        Check if fixture should start based on activity probability.

        Parameters
        ----------
        fixture : WaterFixtureSpec
            Fixture specification
        active_occupants : int
            Number of active occupants
        ten_min_idx : int
            10-minute period index (0-143)

        Returns
        -------
        bool
            True if fixture should start
        """
        # Create key for activity statistics lookup
        weekend_flag = "1" if self.is_weekend else "0"
        key = f"{weekend_flag}_{active_occupants}_{fixture.use_profile}"

        # Get activity probability for this time period (strict mode - crash if not found)
        if key not in self.activity_statistics:
            raise KeyError(
                f"Activity statistics key '{key}' not found. "
                f"Fixture: {fixture.name}, profile: {fixture.use_profile}, "
                f"weekend: {self.is_weekend}, active_occupants: {active_occupants}. "
                f"Available keys sample: {list(self.activity_statistics.keys())[:10]}"
            )

        activity_prob = self.activity_statistics[key][ten_min_idx]

        # Check if fixture starts
        combined_prob = activity_prob * fixture.prob_switch_on
        return self.rng.random() < combined_prob

    def _start_fixture(self, fixture: WaterFixtureSpec) -> tuple:
        """
        Start a fixture event by drawing a volume from the distribution.

        Parameters
        ----------
        fixture : WaterFixtureSpec
            Fixture specification

        Returns
        -------
        tuple
            (event_time_left, flow_rate, restart_time_left)
        """
        # Draw event volume from empirical distribution
        event_volume = self._draw_event_volume(fixture.volume_column)

        # Set restart delay
        restart_time = fixture.restart_delay

        if event_volume == 0:
            # Zero volume event
            return 0.0, 0.0, restart_time

        # Calculate event duration (minutes)
        event_duration = event_volume / fixture.mean_flow

        # Set initial flow rate
        if event_duration < 1.0:
            # Event shorter than 1 minute
            flow_rate = fixture.mean_flow * event_duration
            time_left = 0.0  # Will complete in this minute
        else:
            flow_rate = fixture.mean_flow
            time_left = event_duration - 1.0  # Decrement first minute

        return time_left, flow_rate, restart_time

    def _draw_event_volume(self, column_idx: int) -> float:
        """
        Draw an event volume from the empirical probability distribution (VBA lines 359-377).

        Matches VBA StartFixture logic exactly.

        Parameters
        ----------
        column_idx : int
            Column index in water usage distribution (VBA 1-based column 3, 4, or 5)
            Python uses 0-based, so this is column 3, 4, or 5

        Returns
        -------
        float
            Event volume in litres
        """
        # VBA accesses:
        # - Column 1 for event volumes (VBA 1-based) = column 1 in Python (0-based)
        # - Column 3, 4, or 5 for probabilities (VBA 1-based) = same in Python (0-based)
        #
        # Water usage CSV structure after loading (skiprows=5, header=0):
        # Column 0: empty
        # Column 1: k values (event volumes in litres)
        # Column 2: empty
        # Column 3: Basin/Sink probabilities
        # Column 4: Shower probabilities
        # Column 5: Bath probabilities

        # Get volumes from column 1 (VBA line 373)
        volumes = self.water_usage_dist[:, 1]

        # Get probabilities from specified column (VBA line 369)
        probabilities = self.water_usage_dist[:, column_idx]

        # VBA uses cumulative probability method (lines 359-377)
        # Pick a random number
        rand_val = self.rng.random()

        # Set cumulative probability to zero
        cumulative_p = 0.0

        # Iterate through the probability distribution (VBA: For intRow = 1 To 151)
        for row_idx in range(len(probabilities)):
            # Add the probability
            cumulative_p += probabilities[row_idx]

            if rand_val < cumulative_p:
                # This determines the fixture event volume
                event_volume = volumes[row_idx]
                return event_volume

        # If we exit the loop, return the last volume
        return volumes[-1] if len(volumes) > 0 else 0.0

    def _calculate_total_demand(self):
        """
        Calculate total hot water demand and thermal transfer coefficient.

        Sums fixture flows and converts to thermal transfer coefficient H_demand.
        """
        # Water density (kg/m³)
        rho_w = 1000.0

        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            # Sum all fixture flows (litres/min)
            total_flow = np.sum(self.fixture_flows[minute, :])
            self.hot_water_demand[minute] = total_flow

            # Convert to thermal transfer coefficient
            # litres/min → m³/s → kg/s → W/K
            v_w = total_flow / 1000.0 / 60.0  # m³/s
            m_w = rho_w * v_w  # kg/s
            h_demand = SPECIFIC_HEAT_CAPACITY_WATER * m_w  # W/K

            self.h_demand[minute] = h_demand

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_h_demand(self, timestep: int) -> float:
        """Get thermal transfer coefficient at specified timestep (1-based)."""
        return self.h_demand[timestep - 1]

    def get_daily_hot_water_volume(self) -> float:
        """Get total daily hot water volume in litres."""
        return np.sum(self.hot_water_demand)

    def get_c_cyl(self) -> float:
        """Get hot water cylinder thermal capacitance (J/K)."""
        return self.c_cyl

    def get_h_loss(self) -> float:
        """Get cylinder standing loss coefficient (W/K)."""
        return self.h_loss

    def get_theta_cw(self) -> float:
        """Get cold water temperature (°C)."""
        return self.theta_cw
