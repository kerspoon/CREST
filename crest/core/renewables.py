"""
Renewable Energy Systems

Implements:
- Photovoltaic (PV) systems
- Solar thermal collectors
- Cooling systems
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..simulation.config import TIMESTEPS_PER_DAY_1MIN, PI


@dataclass
class PVConfig:
    """PV system configuration."""
    array_size: float = 4.0  # m²
    efficiency: float = 0.15  # 15%
    tilt: float = 35.0  # degrees
    azimuth: float = 180.0  # degrees (south-facing)


class PVSystem:
    """Photovoltaic system model."""

    def __init__(self, config: PVConfig):
        self.config = config
        self.power_output = np.zeros(TIMESTEPS_PER_DAY_1MIN)

    def set_local_climate(self, local_climate):
        self.local_climate = local_climate

    def calculate_output(self):
        """Calculate PV output for all timesteps."""
        if self.local_climate is None:
            return

        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            irradiance = self.local_climate.get_irradiance(minute)
            # Simplified: use horizontal irradiance with efficiency
            self.power_output[minute] = (
                irradiance * self.config.array_size * self.config.efficiency
            )

    def get_power_output(self, timestep: int) -> float:
        """Get PV power output at timestep (1-based) in Watts."""
        return self.power_output[timestep - 1]


@dataclass
class SolarThermalConfig:
    """Solar thermal system configuration."""
    collector_area: float = 3.0  # m²
    efficiency: float = 0.6  # 60%
    tank_volume: float = 150.0  # litres


class SolarThermal:
    """Solar thermal collector model."""

    def __init__(self, config: SolarThermalConfig):
        self.config = config
        self.collector_heat = np.zeros(TIMESTEPS_PER_DAY_1MIN)

    def set_local_climate(self, local_climate):
        self.local_climate = local_climate

    def calculate_output(self):
        """Calculate solar thermal output for all timesteps."""
        if self.local_climate is None:
            return

        for minute in range(TIMESTEPS_PER_DAY_1MIN):
            irradiance = self.local_climate.get_irradiance(minute)
            # Heat output = irradiance × area × efficiency
            self.collector_heat[minute] = (
                irradiance * self.config.collector_area * self.config.efficiency
            )

    def get_collector_heat(self, timestep: int) -> float:
        """Get collector heat output at timestep (1-based) in Watts."""
        return self.collector_heat[timestep - 1]


@dataclass
class CoolingConfig:
    """Cooling system configuration."""
    cooling_capacity: float = 3000.0  # W
    cop: float = 3.0  # Coefficient of performance


class CoolingSystem:
    """Space cooling system model."""

    def __init__(self, config: CoolingConfig):
        self.config = config
        self.cooling_output = np.zeros(TIMESTEPS_PER_DAY_1MIN)
        self.electricity_demand = np.zeros(TIMESTEPS_PER_DAY_1MIN)

    def set_heating_controls(self, heating_controls):
        self.heating_controls = heating_controls

    def set_building(self, building):
        self.building = building

    def calculate_cooling(self, timestep: int):
        """Calculate cooling output for current timestep."""
        if self.heating_controls is None:
            return

        idx = timestep - 1

        # Check if cooling is needed
        cooling_needed = (
            self.heating_controls.get_space_cooling_thermostat_state(timestep) and
            self.heating_controls.get_space_cooling_timer_state(timestep) and
            self.heating_controls.get_cooler_emitter_state(timestep)
        )

        if cooling_needed:
            # Provide cooling (negative heat)
            self.cooling_output[idx] = -self.config.cooling_capacity
            self.electricity_demand[idx] = self.config.cooling_capacity / self.config.cop
        else:
            self.cooling_output[idx] = 0.0
            self.electricity_demand[idx] = 0.0

    def get_cooling_to_space(self, timestep: int) -> float:
        """Get cooling output at timestep (1-based) in Watts (negative for cooling)."""
        return self.cooling_output[timestep - 1]

    def get_electricity_demand(self, timestep: int) -> float:
        """Get electricity demand at timestep (1-based) in Watts."""
        return self.electricity_demand[timestep - 1]
