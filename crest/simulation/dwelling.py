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
from ..core.pv import PVSystem
from ..core.solar_thermal import SolarThermal
from ..core.cooling import CoolingSystem
from ..data.loader import CRESTDataLoader
from ..utils.random import RandomGenerator
from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    Country,
    UrbanRural
)


@dataclass
class DwellingConfig:
    """Configuration for a dwelling."""
    dwelling_index: int
    num_residents: int
    building_index: int
    heating_system_index: int
    country: Country = Country.UK  # Country for appliance/water/lighting behavior
    urban_rural: UrbanRural = UrbanRural.URBAN  # Urban/Rural for appliance ownership
    cooling_system_index: int = 0
    pv_system_index: int = 0  # PV system index (0 = no PV)
    solar_thermal_index: int = 0  # Solar thermal system index (0 = no solar thermal)
    is_weekend: bool = False


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
            heating_system_index=config.heating_system_index,
            dwelling_index=config.dwelling_index
        )
        self.building = Building(building_config, data_loader)
        self.building.set_local_climate(self.local_climate)
        self.building.set_occupancy(self.occupancy)

        # Create hot water model
        hw_config = HotWaterConfig(
            dwelling_index=config.dwelling_index,
            heating_system_index=config.heating_system_index,
            num_residents=config.num_residents,
            country=config.country
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
        app_config = AppliancesConfig(
            dwelling_index=config.dwelling_index,
            country=config.country,
            urban_rural=config.urban_rural
        )
        self.appliances = Appliances(app_config, data_loader, activity_statistics, config.is_weekend, self.rng)
        self.appliances.set_occupancy(self.occupancy)
        self.building.set_appliances(self.appliances)

        # Create lighting
        light_config = LightingConfig(
            dwelling_index=config.dwelling_index,
            country=config.country
        )
        self.lighting = Lighting(light_config, data_loader, self.rng)
        self.lighting.set_occupancy(self.occupancy)
        self.lighting.set_local_climate(self.local_climate)
        self.building.set_lighting(self.lighting)

        # Create renewable systems
        # VBA Reference: clsPVSystem initialization (mdlThermalElectricalModel.bas lines 356-365)
        # VBA: intPVSystemIndex from clsDwelling.InitialiseDwelling (line 42)
        if config.pv_system_index > 0:
            self.pv_system = PVSystem(data_loader, self.rng)
            # Use run_number=1 for single dwelling simulation
            self.pv_system.initialize(
                dwelling_index=config.dwelling_index,
                run_number=1,
                climate=self.local_climate,
                appliances=self.appliances,
                lighting=self.lighting,
                pv_system_index=config.pv_system_index
            )
        else:
            self.pv_system = None

        # VBA Reference: clsSolarThermal initialization (mdlThermalElectricalModel.bas lines 423-429)
        # VBA: intSolarThermalIndex from wsDwellings (clsSolarThermal.cls line 77)
        if config.solar_thermal_index > 0:
            self.solar_thermal = SolarThermal(data_loader, self.rng)
            # Use run_number=1 for single dwelling simulation
            self.solar_thermal.initialize(
                dwelling_index=config.dwelling_index,
                run_number=1,
                climate=self.local_climate,
                building=self.building,
                solar_thermal_index=config.solar_thermal_index
            )
            self.building.set_solar_thermal(self.solar_thermal)
        else:
            self.solar_thermal = None

        # Create cooling system
        # VBA Reference: clsCoolingSystem initialization (mdlThermalElectricalModel.bas lines 384-390)
        # VBA: intCoolingSystemIndex from wsDwellings (clsCoolingSystem.cls line 80)
        if config.cooling_system_index > 0:
            self.cooling_system = CoolingSystem(data_loader, self.rng)
            # Use run_number=1 for single dwelling simulation
            self.cooling_system.initialize(
                dwelling_index=config.dwelling_index,
                run_number=1,
                controls=self.heating_controls,
                building=self.building,
                cooling_system_index=config.cooling_system_index
            )
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

        # 5. Run PV system - ONLY calculate PV output (pre-simulation)
        # Net demand and self-consumption calculated AFTER thermal loop
        # VBA Reference: mdlThermalElectricalModel.bas lines 436-437
        if self.pv_system:
            self.pv_system.calculate_pv_output()

        # Note: Solar thermal runs in thermal loop (timestep-by-timestep), not here

        # 6. Initialize building temperatures
        initial_outdoor_temp = self.local_climate.get_temperature(0)
        self.building.initialize_temperatures(initial_outdoor_temp, random_gen=self.rng)

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

            # Calculate solar thermal output (runs in thermal loop)
            if self.solar_thermal:
                self.solar_thermal.calculate_solar_thermal_output(timestep)

            # Calculate heating system output
            self.heating_system.calculate_heat_output(timestep)

            # Calculate cooling system output
            if self.cooling_system:
                self.cooling_system.calculate_cooling_output(timestep)

            # Calculate building thermal response
            self.building.calculate_temperature_change(timestep)

        # 9. After thermal loop - calculate appliance totals including heating/cooling electricity
        # VBA Reference: mdlThermalElectricalModel.bas lines 449-451
        self.appliances.calculate_total_demand()

        # 10. After thermal loop - calculate PV net demand and self-consumption
        # CRITICAL: Must be AFTER appliances.calculate_total_demand() because
        # PV calculations need total demand which includes heating/cooling electricity
        # VBA Reference: mdlThermalElectricalModel.bas lines 454-461
        if self.pv_system:
            self.pv_system.calculate_net_demand()
            self.pv_system.calculate_self_consumption()

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
            total += self.cooling_system.get_cooling_system_power_demand(timestep)

        # Add solar thermal pump electricity
        if self.solar_thermal:
            total += self.solar_thermal.get_P_pumpsolar(timestep)

        # Subtract PV generation
        if self.pv_system:
            total -= self.pv_system.get_pv_output(timestep)

        return total

    def write_dwelling_index(self, current_date, dwelling_index_row_offset: int = 0):
        """
        Write dwelling index, date, and time columns for output.

        In the VBA version, this writes to Excel worksheet. In Python, we return
        a list of tuples (dwelling_index, date, time) for each of 1440 timesteps.

        Args:
            current_date: Current date (datetime.date or similar)
            dwelling_index_row_offset: Row offset for multi-dwelling output (not used in Python)

        Returns:
            List of tuples (dwelling_index, date, time_str) for each timestep

        VBA Reference: clsDwelling.WriteDwellingIndex (lines 54-70)
        """
        import datetime

        result = []
        start_time = datetime.datetime.strptime("00:00", "%H:%M")

        for minute in range(1, 1441):  # VBA 1-based: 1 To 1440
            # VBA line 65-67: Write dwelling index, date, time
            # .Range("A").Offset = intDwellingIndex
            # .Range("B").Offset = currentDate
            # .Range("C").Offset = DateAdd("n", intMinute - 1, strStartDate)

            time_offset = datetime.timedelta(minutes=minute - 1)
            current_time = start_time + time_offset
            time_str = current_time.strftime("%H:%M")

            result.append((self.config.dwelling_index, current_date, time_str))

        return result

    def get_daily_totals(self):
        """
        Calculate daily totals for all energy flows.

        Returns a dictionary with daily aggregated values:
        - mean_active_occupancy: Mean number of active occupants
        - proportion_actively_occupied: Proportion of time actively occupied
        - lighting_demand: Daily lighting demand (kWh)
        - appliance_demand: Daily appliance demand (kWh)
        - pv_output: Daily PV generation (kWh)
        - total_electricity_demand: Daily total electricity (kWh)
        - self_consumption: Daily PV self-consumption (kWh)
        - net_electricity_demand: Daily net demand after PV (kWh)
        - hot_water_demand: Daily hot water demand (litres)
        - average_indoor_temperature: Mean internal temperature (°C)
        - thermal_energy_space: Daily space heating energy (kWh)
        - thermal_energy_water: Daily water heating energy (kWh)
        - gas_demand: Daily gas/fuel demand (kWh)
        - space_thermostat_setpoint: Space heating setpoint (°C)
        - solar_thermal_output: Daily solar thermal output (kWh)

        VBA Reference: mdlThermalElectricalModel.DailyTotals (lines 1057-1121)
        """
        # VBA lines 1081-1082: Occupancy metrics
        mean_active_occupancy = self.occupancy.get_mean_active_occupancy()
        proportion_actively_occupied = self.occupancy.get_proportion_actively_occupied()

        # VBA lines 1083-1084: Lighting and appliances (convert W·min to kWh)
        # VBA: / 60 / 1000 (minutes to hours, W to kW)
        lighting_demand = self.lighting.get_daily_energy() / 60 / 1000
        appliance_demand = self.appliances.get_daily_energy() / 60 / 1000

        # VBA line 1086: Total electricity before PV
        total_electricity_demand = lighting_demand + appliance_demand

        # VBA lines 1087-1090: PV metrics
        if self.pv_system:
            pv_output = self.pv_system.get_daily_sum_pv_output() / 60 / 1000
            net_electricity_demand = self.pv_system.get_daily_sum_net_demand() / 60 / 1000
            self_consumption = self.pv_system.get_daily_sum_self_consumption() / 60 / 1000
        else:
            pv_output = 0.0
            net_electricity_demand = total_electricity_demand
            self_consumption = 0.0

        # VBA line 1092: Hot water demand (already in litres)
        hot_water_demand = self.hot_water.get_daily_hot_water_volume()

        # VBA line 1094: Average indoor temperature
        average_indoor_temperature = self.building.get_mean_theta_i()

        # VBA lines 1096-1097: Heating system thermal energy
        thermal_energy_space = self.heating_system.get_daily_thermal_energy_space() / 60 / 1000
        thermal_energy_water = self.heating_system.get_daily_thermal_energy_water() / 60 / 1000

        # VBA line 1099: Gas/fuel demand (convert W·min to kWh)
        # VBA: / 60 (minutes to hours, already in kW units)
        gas_demand = self.heating_system.get_daily_fuel_consumption() / 60

        # VBA line 1118: Space thermostat setpoint
        space_thermostat_setpoint = self.heating_controls.get_space_thermostat_setpoint()

        # VBA line 1119: Solar thermal output
        if self.solar_thermal:
            solar_thermal_output = self.solar_thermal.get_daily_sum_phi_s() / 60 / 1000
        else:
            solar_thermal_output = 0.0

        return {
            'mean_active_occupancy': mean_active_occupancy,
            'proportion_actively_occupied': proportion_actively_occupied,
            'lighting_demand': lighting_demand,
            'appliance_demand': appliance_demand,
            'pv_output': pv_output,
            'total_electricity_demand': total_electricity_demand,
            'self_consumption': self_consumption,
            'net_electricity_demand': net_electricity_demand,
            'hot_water_demand': hot_water_demand,
            'average_indoor_temperature': average_indoor_temperature,
            'thermal_energy_space': thermal_energy_space,
            'thermal_energy_water': thermal_energy_water,
            'gas_demand': gas_demand,
            'space_thermostat_setpoint': space_thermostat_setpoint,
            'solar_thermal_output': solar_thermal_output
        }
