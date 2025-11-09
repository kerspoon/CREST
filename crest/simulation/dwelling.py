"""
Dwelling Orchestrator

Manages all components of a single dwelling and coordinates the simulation.
"""

from dataclasses import dataclass
from typing import Optional

from ..core.occupancy import Occupancy, OccupancyConfig
from ..core.climate import LocalClimate
from ..core.building import Building, BuildingConfig
from ..core.water import HotWater, HotWaterConfig
from ..core.heating import HeatingSystem, HeatingSystemConfig
from ..core.controls import HeatingControls, HeatingControlsConfig
from ..core.appliances import Appliances, AppliancesConfig
from ..core.lighting import Lighting, LightingConfig
from ..core.renewables import PVSystem, SolarThermal, CoolingSystem, PVConfig, SolarThermalConfig, CoolingConfig
from ..data.loader import CRESTDataLoader
from ..utils.random import RandomGenerator
from ..simulation.config import TIMESTEPS_PER_DAY_1MIN


@dataclass
class DwellingConfig:
    """Configuration for a dwelling."""
    dwelling_index: int
    num_residents: int
    building_index: int
    heating_system_index: int
    cooling_system_index: int = 0
    is_weekend: bool = False
    has_pv: bool = False
    has_solar_thermal: bool = False


class Dwelling:
    """
    Dwelling orchestrator.

    Manages all subsystems for a single dwelling and coordinates the simulation.
    """

    def __init__(
        self,
        config: DwellingConfig,
        global_climate,
        data_loader: CRESTDataLoader,
        activity_statistics: dict,
        rng: Optional[RandomGenerator] = None
    ):
        """
        Initialize dwelling.

        Parameters
        ----------
        config : DwellingConfig
            Dwelling configuration
        global_climate : GlobalClimate
            Global climate model instance
        data_loader : CRESTDataLoader
            Data loader
        activity_statistics : dict
            Activity probability profiles
        rng : RandomGenerator, optional
            Random number generator
        """
        self.config = config
        self.global_climate = global_climate
        self.data_loader = data_loader
        self.activity_statistics = activity_statistics
        self.rng = rng if rng is not None else RandomGenerator()

        # Create local climate
        self.local_climate = LocalClimate(global_climate, config.dwelling_index)

        # Create occupancy model
        occ_config = OccupancyConfig(
            num_residents=config.num_residents,
            is_weekend=config.is_weekend,
            dwelling_index=config.dwelling_index
        )
        self.occupancy = Occupancy(occ_config, data_loader, self.rng)

        # Create building thermal model
        building_config = BuildingConfig(
            building_index=config.building_index,
            dwelling_index=config.dwelling_index
        )
        self.building = Building(building_config, data_loader)
        self.building.set_local_climate(self.local_climate)
        self.building.set_occupancy(self.occupancy)

        # Create hot water model
        hw_config = HotWaterConfig(
            dwelling_index=config.dwelling_index,
            heating_system_index=config.heating_system_index,
            num_residents=config.num_residents
        )
        self.hot_water = HotWater(hw_config, data_loader, activity_statistics, config.is_weekend, self.rng)
        self.hot_water.set_occupancy(self.occupancy)
        self.building.set_hot_water(self.hot_water)

        # Create heating system
        heating_config = HeatingSystemConfig(
            heating_system_index=config.heating_system_index,
            dwelling_index=config.dwelling_index
        )
        self.heating_system = HeatingSystem(heating_config, data_loader)
        self.heating_system.set_building(self.building)
        self.building.set_heating_system(self.heating_system)

        # Create heating controls
        controls_config = HeatingControlsConfig(
            dwelling_index=config.dwelling_index,
            building_index=config.building_index,
            heating_system_index=config.heating_system_index,
            cooling_system_index=config.cooling_system_index,
            is_weekend=config.is_weekend
        )
        self.heating_controls = HeatingControls(controls_config, data_loader, self.rng)
        self.heating_controls.set_building(self.building)
        self.heating_controls.set_hot_water(self.hot_water)
        self.heating_system.set_heating_controls(self.heating_controls)
        self.building.set_heating_controls(self.heating_controls)

        # Create appliances
        app_config = AppliancesConfig(dwelling_index=config.dwelling_index)
        self.appliances = Appliances(app_config, data_loader, activity_statistics, config.is_weekend, self.rng)
        self.appliances.set_occupancy(self.occupancy)
        self.building.set_appliances(self.appliances)

        # Create lighting
        light_config = LightingConfig(dwelling_index=config.dwelling_index)
        self.lighting = Lighting(light_config, data_loader, self.rng)
        self.lighting.set_occupancy(self.occupancy)
        self.lighting.set_local_climate(self.local_climate)
        self.building.set_lighting(self.lighting)

        # Create renewable systems
        if config.has_pv:
            self.pv_system = PVSystem(PVConfig())
            self.pv_system.set_local_climate(self.local_climate)
        else:
            self.pv_system = None

        if config.has_solar_thermal:
            self.solar_thermal = SolarThermal(SolarThermalConfig())
            self.solar_thermal.set_local_climate(self.local_climate)
            self.building.set_solar_thermal(self.solar_thermal)
        else:
            self.solar_thermal = None

        # Create cooling system
        if config.cooling_system_index > 0:
            self.cooling_system = CoolingSystem(CoolingConfig())
            self.cooling_system.set_heating_controls(self.heating_controls)
            self.cooling_system.set_building(self.building)
            self.building.set_cooling_system(self.cooling_system)
        else:
            self.cooling_system = None

    def run_simulation(self):
        """
        Run complete simulation for this dwelling.

        Executes all models in the correct order.
        """
        # 1. Run occupancy simulation (generates occupancy patterns)
        self.occupancy.run_simulation()

        # 2. Run hot water simulation (generates demand events)
        self.hot_water.run_simulation()

        # 3. Run appliances simulation (generates electrical demand)
        self.appliances.run_simulation()

        # 4. Run lighting simulation (generates electrical demand)
        self.lighting.run_simulation()

        # 5. Run renewable systems
        if self.pv_system:
            self.pv_system.calculate_output()

        if self.solar_thermal:
            self.solar_thermal.calculate_output()

        # 6. Initialize building temperatures
        initial_outdoor_temp = self.local_climate.get_temperature(0)
        self.building.initialize_temperatures(initial_outdoor_temp)

        # 7. Initialize heating controls
        self.heating_controls.initialize_thermostat_states(
            self.building.theta_i[0],
            self.building.theta_cyl[0],
            self.building.theta_em[0],
            self.building.theta_cool[0]
        )

        # 8. Main simulation loop (minute by minute)
        for timestep in range(1, TIMESTEPS_PER_DAY_1MIN + 1):
            # Update heating controls
            self.heating_controls.calculate_control_states(timestep)

            # Calculate heating system output
            self.heating_system.calculate_heat_output(timestep)

            # Calculate cooling system output
            if self.cooling_system:
                self.cooling_system.calculate_cooling(timestep)

            # Calculate building thermal response
            self.building.calculate_temperature_change(timestep)

    def get_total_electricity_demand(self, timestep: int) -> float:
        """
        Get total electricity demand at timestep (W).

        Includes appliances, lighting, heating pump/standby, and cooling.
        """
        total = 0.0
        total += self.appliances.get_total_demand(timestep)
        total += self.lighting.get_total_demand(timestep)
        total += self.heating_system.get_heating_system_power_demand(timestep)

        if self.cooling_system:
            total += self.cooling_system.get_electricity_demand(timestep)

        # Subtract PV generation
        if self.pv_system:
            total -= self.pv_system.get_power_output(timestep)

        return total
