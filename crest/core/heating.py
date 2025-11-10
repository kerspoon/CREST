"""
Heating System Model

Models primary heating systems including:
- Gas boilers (regular, combi, system)
- Electric heating
- Heat distribution between space and hot water
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    BOILER_THERMAL_EFFICIENCY
)
from ..data.loader import CRESTDataLoader


@dataclass
class HeatingSystemConfig:
    """Configuration for heating system."""
    heating_system_index: int
    dwelling_index: int = 0
    run_number: int = 0


class HeatingSystem:
    """
    Primary heating system model.

    Handles heat distribution to space heating and hot water,
    fuel/electricity consumption, and pump operation.
    """

    def __init__(
        self,
        config: HeatingSystemConfig,
        data_loader: CRESTDataLoader
    ):
        """
        Initialize heating system.

        Parameters
        ----------
        config : HeatingSystemConfig
            Configuration parameters
        data_loader : CRESTDataLoader
            Data loader for heating system specs
        """
        self.config = config
        self.data_loader = data_loader

        # Load heating system parameters
        heating_systems = data_loader.load_primary_heating_systems()
        if config.heating_system_index >= len(heating_systems):
            raise ValueError(f"Heating system index {config.heating_system_index} out of range")

        heating_params = heating_systems.iloc[config.heating_system_index]

        # System parameters
        self.heating_type = heating_params.get('HeatingType', 'Boiler')
        self.heating_system_type = int(heating_params.get('HeatingSystemType', 1))
        self.fuel_type = heating_params.get('FuelType', 'Gas')
        self.fuel_flow_rate = heating_params.get('FuelFlowRate', 0.5)  # m³/min for gas, kW for electric
        self.phi_h_max = heating_params.get('Phi_h', 12000.0)  # W, max heat output
        self.p_standby = heating_params.get('P_standby', 5.0)  # W, standby power
        self.p_pump = heating_params.get('P_pump', 60.0)  # W, pump power
        self.eta_h = heating_params.get('Eta_h', BOILER_THERMAL_EFFICIENCY)  # Thermal efficiency

        # Storage arrays (1440 timesteps)
        self.phi_h_output = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Total heat output (W)
        self.phi_h_water = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Heat to hot water (W)
        self.phi_h_space = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Heat to space (W)
        self.m_fuel = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Fuel flow rate (m³/min for gas)
        self.heating_electricity = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Electricity for heating (W)
        self.p_h = np.zeros(TIMESTEPS_PER_DAY_1MIN)  # Pump/standby electricity (W)

        # References to other components (set externally)
        self.heating_controls = None
        self.building = None

    def set_heating_controls(self, heating_controls):
        """Set reference to heating controls."""
        self.heating_controls = heating_controls

    def set_building(self, building):
        """Set reference to building model."""
        self.building = building

    def calculate_heat_output(self, timestep: int):
        """
        Calculate heat output for space and hot water.

        Parameters
        ----------
        timestep : int
            Current timestep (1-based, 1-1440)
        """
        if self.heating_controls is None or self.building is None:
            raise RuntimeError("Heating controls and building must be set before calculating heat output")

        # Get control signals from heating controls
        heater_on_off = self.heating_controls.get_heater_on_off(timestep)
        heat_water_on_off = self.heating_controls.get_heat_water_on_off(timestep)

        # Space heating requires thermostat AND timer AND emitter thermostat
        space_thermostat = self.heating_controls.get_space_thermostat_state(timestep)
        space_timer = self.heating_controls.get_space_timer_state(timestep)
        emitter_thermostat = self.heating_controls.get_emitter_thermostat_state(timestep)
        space_heating_on_off = space_thermostat and space_timer and emitter_thermostat

        # Initialize outputs
        idx = timestep - 1
        self.phi_h_water[idx] = 0.0
        self.phi_h_space[idx] = 0.0
        self.phi_h_output[idx] = 0.0
        self.m_fuel[idx] = 0.0
        self.heating_electricity[idx] = 0.0

        # Pump operates if space heating thermostat and timer are on (even if emitter thermostat is off)
        if space_thermostat and space_timer:
            self.p_h[idx] = self.p_pump
        else:
            self.p_h[idx] = self.p_standby

        # Calculate heat outputs if heater is on
        if heater_on_off:
            phi_h_water = 0.0
            phi_h_space = 0.0

            # Hot water has priority (for regular boilers)
            if heat_water_on_off:
                # Get target heat for hot water from building
                phi_h_water_target = self.building.get_target_heat_water(timestep)

                # Allocate heat to water (bounded by max output)
                phi_h_water = max(0.0, min(self.phi_h_max, phi_h_water_target))
                self.phi_h_water[idx] = phi_h_water

                # If space heating is also required, allocate remaining capacity
                if space_heating_on_off:
                    phi_h_space_target = self.building.get_target_heat_space(timestep)

                    # Allocate remaining capacity to space
                    phi_h_space = max(0.0, min(self.phi_h_max - phi_h_water, phi_h_space_target))
                    self.phi_h_space[idx] = phi_h_space

            else:
                # Only space heating required
                if space_heating_on_off:
                    phi_h_space_target = self.building.get_target_heat_space(timestep)

                    # Allocate heat to space (bounded by max output)
                    phi_h_space = max(0.0, min(self.phi_h_max, phi_h_space_target))
                    self.phi_h_space[idx] = phi_h_space

            # Total heat output
            phi_h_total = self.phi_h_space[idx] + self.phi_h_water[idx]
            self.phi_h_output[idx] = phi_h_total

            # When heater is firing, pump always runs at full power (VBA line 243)
            self.p_h[idx] = self.p_pump

            # Calculate fuel/electricity consumption
            if self.phi_h_max > 0:
                utilization = phi_h_total / self.phi_h_max

                # Heating systems 1-3 are gas boilers (use fuel)
                # Heating systems 4+ are electric (use electricity)
                if self.config.heating_system_index < 3:
                    # Gas boiler: fuel flow rate in m³/min
                    self.m_fuel[idx] = self.fuel_flow_rate * utilization
                else:
                    # Electric heating: electricity in W
                    self.heating_electricity[idx] = self.fuel_flow_rate * 1000.0 * utilization

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_phi_h(self, timestep: int) -> float:
        """Get total heat output at specified timestep (1-based)."""
        return self.phi_h_output[timestep - 1]

    def get_heat_to_space(self, timestep: int) -> float:
        """Get heat to space at specified timestep (1-based)."""
        return self.phi_h_space[timestep - 1]

    def get_heat_to_hot_water(self, timestep: int) -> float:
        """Get heat to hot water at specified timestep (1-based)."""
        return self.phi_h_water[timestep - 1]

    def get_heating_system_power_demand(self, timestep: int) -> float:
        """Get total electricity demand (pump + heating) at specified timestep (1-based)."""
        idx = timestep - 1
        return self.p_h[idx] + self.heating_electricity[idx]

    def get_heating_system_type(self) -> int:
        """Get heating system type (1=regular, 2=combi, 3=system)."""
        return self.heating_system_type

    def get_daily_thermal_energy_space(self) -> float:
        """Get total daily thermal energy for space heating (W·min, VBA units)."""
        return np.sum(self.phi_h_space)

    def get_daily_thermal_energy_water(self) -> float:
        """Get total daily thermal energy for hot water (W·min, VBA units)."""
        return np.sum(self.phi_h_water)

    def get_daily_fuel_consumption(self) -> float:
        """Get total daily fuel consumption (m³ for gas)."""
        return np.sum(self.m_fuel)

    def get_daily_heating_electricity(self) -> float:
        """Get total daily heating electricity (W·min, VBA units)."""
        return np.sum(self.heating_electricity)
