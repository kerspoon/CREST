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
    COLD_WATER_TEMPERATURE,
    WATER_FIXTURE_TYPES
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

        # Load heating system parameters for cylinder
        heating_systems = data_loader.load_primary_heating_systems()
        if config.heating_system_index >= len(heating_systems):
            # Use default values
            self.h_loss = 2.0
            v_cyl = 150.0
        else:
            heating_params = heating_systems.iloc[config.heating_system_index]
            self.h_loss = heating_params.get('H_loss', 2.0)
            v_cyl = heating_params.get('V_cyl', 150.0)

        # Cylinder thermal capacitance (J/K)
        self.c_cyl = SPECIFIC_HEAT_CAPACITY_WATER * v_cyl

        # Cold water temperature
        self.theta_cw = COLD_WATER_TEMPERATURE

        # Load fixture specifications
        self._load_fixture_specs()

        # Load water usage distribution
        self.water_usage_dist = data_loader.load_water_usage().values

        # Storage arrays
        self.hot_water_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # litres/min
        self.h_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # W/K thermal transfer coefficient

        # Simulation array for individual fixtures (1440 rows x 4 fixtures)
        self.fixture_flows = np.zeros((TIMESTEPS_PER_DAY_1MIN, WATER_FIXTURE_TYPES))

        # Reference to occupancy model (set externally)
        self.occupancy = None

    def _load_fixture_specs(self):
        """Load fixture specifications from data."""
        fixtures_data = self.data_loader.load_appliances_and_fixtures()

        # Water fixtures are typically rows 46-49 in the Excel sheet (after appliances)
        # Row offset 45 in VBA, so rows 45-48 in 0-based indexing
        # For now, create default fixture specs
        # TODO: Parse actual CSV if fixture data is included

        self.fixtures = [
            WaterFixtureSpec(
                name="Basin",
                prob_switch_on=0.015,
                mean_flow=6.0,
                use_profile="Washing",
                restart_delay=1.0,
                volume_column=2  # Column index in water usage distribution
            ),
            WaterFixtureSpec(
                name="Sink",
                prob_switch_on=0.01,
                mean_flow=6.0,
                use_profile="Cooking",
                restart_delay=1.0,
                volume_column=2  # Same distribution as basin
            ),
            WaterFixtureSpec(
                name="Shower",
                prob_switch_on=0.008,
                mean_flow=8.0,
                use_profile="Washing",
                restart_delay=5.0,
                volume_column=3
            ),
            WaterFixtureSpec(
                name="Bath",
                prob_switch_on=0.004,
                mean_flow=12.0,
                use_profile="Washing",
                restart_delay=5.0,
                volume_column=4
            )
        ]

        # Determine which fixtures dwelling has (simplified: all fixtures)
        self.has_fixture = [True] * WATER_FIXTURE_TYPES

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

        # Get activity probability for this time period
        if key in self.activity_statistics:
            activity_prob = self.activity_statistics[key][ten_min_idx]
        else:
            # Default low probability if profile not found
            activity_prob = 0.001

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
        Draw an event volume from the empirical probability distribution.

        Parameters
        ----------
        column_idx : int
            Column index in water usage distribution (0-based)

        Returns
        -------
        float
            Event volume in litres
        """
        # Get probability distribution for this fixture type
        # Water usage CSV has volumes in column 0, probabilities in other columns
        volumes = self.water_usage_dist[:, 0]
        probabilities = self.water_usage_dist[:, column_idx]

        # Normalize probabilities
        prob_sum = np.sum(probabilities)
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        else:
            # Uniform distribution if no data
            probabilities = np.ones_like(probabilities) / len(probabilities)

        # Draw random sample using inverse transform
        rng_value = self.rng.random()
        cumulative_prob = np.cumsum(probabilities)

        # Find first bin where cumulative probability exceeds random value
        idx = np.searchsorted(cumulative_prob, rng_value)
        idx = min(idx, len(volumes) - 1)

        return volumes[idx]

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
