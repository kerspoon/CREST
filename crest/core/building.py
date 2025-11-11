"""
Low-order Building Thermal Model

Implements a 5-node thermal capacitance network model:
- External building node
- Internal building node
- Heating emitters (radiators)
- Cooling emitters (coils)
- Hot water cylinder

Uses Euler's method to solve coupled differential equations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..simulation.config import (
    TIMESTEPS_PER_DAY_1MIN,
    THERMAL_TIMESTEP_SECONDS,
    COLD_WATER_TEMPERATURE
)
from ..data.loader import CRESTDataLoader


@dataclass
class BuildingConfig:
    """Configuration for building thermal model."""
    building_index: int
    heating_system_index: int
    dwelling_index: int = 0
    run_number: int = 0


class Building:
    """
    Low-order building thermal model.

    Models heat transfer between:
    - Outside air
    - Building fabric (external and internal nodes)
    - Heating/cooling emitters
    - Hot water cylinder
    """

    def __init__(
        self,
        config: BuildingConfig,
        data_loader: CRESTDataLoader
    ):
        """
        Initialize building thermal model.

        Parameters
        ----------
        config : BuildingConfig
            Building configuration
        data_loader : CRESTDataLoader
            Data loader for building parameters
        """
        self.config = config
        self.data_loader = data_loader

        # Load building thermal parameters
        buildings_data = data_loader.load_buildings()
        if config.building_index >= len(buildings_data):
            raise ValueError(f"Building index {config.building_index} out of range")

        building_params = buildings_data.iloc[config.building_index]

        # Thermal transfer coefficients (W/K)
        self.h_ob = building_params['H_ob']      # Outside to building fabric
        self.h_bi = building_params['H_bi']      # Building fabric to internal air
        self.h_v = building_params['H_v']        # Ventilation losses
        self.h_em = building_params['H_em']      # Internal air to heating emitters
        self.h_cool = building_params['H_emcool']  # Internal air to cooling emitters

        # Thermal capacitances (J/K)
        self.c_b = building_params['C_b']        # External building node
        self.c_i = building_params['C_i']        # Internal building node
        self.c_em = building_params['C_em']      # Heating emitters
        self.c_cool = building_params['C_emcool']  # Cooling emitters

        # Solar aperture (m²)
        self.a_s = building_params['A_s']

        # Nominal emitter temperatures (°C)
        self.theta_em_nominal = building_params['theta_em']   # Heating emitters nominal temp
        self.theta_cool_nominal = building_params['theta_cool']  # Cooling emitters nominal temp

        # Load heating system parameters from PrimaryHeatingSystems.csv
        heating_systems_data = data_loader.load_primary_heating_systems()
        if config.heating_system_index >= len(heating_systems_data):
            raise ValueError(f"Heating system index {config.heating_system_index} out of range")

        heating_params = heating_systems_data.iloc[config.heating_system_index]

        # Cylinder standing losses (W/K)
        self.h_loss = heating_params['H_loss']

        # Hot water cylinder volume (litres) and thermal capacitance (J/K)
        v_cyl = heating_params['V_cyl']
        from ..simulation.config import SPECIFIC_HEAT_CAPACITY_WATER
        self.c_cyl = SPECIFIC_HEAT_CAPACITY_WATER * v_cyl

        # Cold water temperature (°C)
        self.theta_cw = COLD_WATER_TEMPERATURE

        # Time step (seconds)
        self.timestep_seconds = THERMAL_TIMESTEP_SECONDS

        # Temperature arrays (1-minute resolution, 1440 timesteps)
        self.theta_b = np.zeros(TIMESTEPS_PER_DAY_1MIN)      # External building node (°C)
        self.theta_i = np.zeros(TIMESTEPS_PER_DAY_1MIN)      # Internal building node (°C)
        self.theta_em = np.zeros(TIMESTEPS_PER_DAY_1MIN)     # Heating emitters (°C)
        self.theta_cool = np.zeros(TIMESTEPS_PER_DAY_1MIN)   # Cooling emitters (°C)
        self.theta_cyl = np.zeros(TIMESTEPS_PER_DAY_1MIN)    # Hot water cylinder (°C)

        # Thermal gains arrays (W)
        self.phi_s = np.zeros(TIMESTEPS_PER_DAY_1MIN)        # Passive solar gains
        self.phi_c = np.zeros(TIMESTEPS_PER_DAY_1MIN)        # Casual gains (occupants, lighting, appliances)

        # References to other system components (set externally)
        self.local_climate = None
        self.heating_system = None
        self.heating_controls = None
        self.cooling_system = None
        self.occupancy = None
        self.lighting = None
        self.appliances = None
        self.hot_water = None
        self.solar_thermal = None

    def initialize_temperatures(self, initial_outdoor_temp: float, random_gen=None):
        """
        Initialize building temperatures using VBA-matched logic.

        Matches VBA InitialiseBuilding (clsBuilding.cls lines 287-297).

        Parameters
        ----------
        initial_outdoor_temp : float
            Initial outdoor temperature from climate model (°C)
        random_gen : Optional
            Random number generator for reproducibility (uses numpy.random if None)
        """
        import numpy as np

        # Use provided RNG or default numpy random
        if random_gen is None:
            rnd = np.random.random
        else:
            rnd = random_gen.random

        # VBA line 291: aTheta_b(1, 1) = Rnd * 2 + WorksheetFunction.Max(16, dblTheta_o)
        self.theta_b[0] = rnd() * 2 + max(16, initial_outdoor_temp)

        # VBA line 292: aTheta_i(1, 1) = Rnd * 2 + WorksheetFunction.Min(WorksheetFunction.Max(19, dblTheta_o), 25)
        self.theta_i[0] = rnd() * 2 + min(max(19, initial_outdoor_temp), 25)

        # VBA line 293: aTheta_em(1, 1) = aTheta_i(1, 1)
        self.theta_em[0] = self.theta_i[0]

        # VBA line 294: aTheta_cool(1, 1) = aTheta_i(1, 1)
        self.theta_cool[0] = self.theta_i[0]

        # VBA line 297: aTheta_cyl(1, 1) = 60 + Rnd() * 2
        self.theta_cyl[0] = 60 + rnd() * 2

    def calculate_temperature_change(self, timestep: int):
        """
        Calculate temperature changes for all thermal nodes using Euler's method.

        Solves coupled differential equations for the 5-node thermal network.

        Parameters
        ----------
        timestep : int
            Current timestep (1-based, 1-1440)
        """
        # Convert to 0-based index
        idx = timestep - 1

        # Get previous temperatures (or initial values if timestep == 1)
        if timestep == 1:
            theta_b_prev = self.theta_b[0]
            theta_i_prev = self.theta_i[0]
            theta_em_prev = self.theta_em[0]
            theta_cool_prev = self.theta_cool[0]
            theta_cyl_prev = self.theta_cyl[0]
        else:
            theta_b_prev = self.theta_b[idx - 1]
            theta_i_prev = self.theta_i[idx - 1]
            theta_em_prev = self.theta_em[idx - 1]
            theta_cool_prev = self.theta_cool[idx - 1]
            theta_cyl_prev = self.theta_cyl[idx - 1]

        # Get external conditions
        theta_o = self.local_climate.get_temperature(idx) if self.local_climate else 10.0
        g_o = self.local_climate.get_irradiance(idx) if self.local_climate else 0.0

        # Get thermal gains from various sources
        phi_h_space = self.heating_system.get_heat_to_space(timestep) if self.heating_system else 0.0
        phi_h_cooling = self.cooling_system.get_cooling_to_space(timestep) if self.cooling_system else 0.0
        phi_h_water = self.heating_system.get_heat_to_hot_water(timestep) if self.heating_system else 0.0

        # Passive solar gains
        phi_s = g_o * self.a_s
        self.phi_s[idx] = phi_s

        # Casual gains from occupants, lighting, appliances
        # Occupancy is at 10-minute resolution, so use integer division
        phi_c_occupancy = self.occupancy.get_thermal_gain_at_timestep(idx // 10) if self.occupancy else 0.0
        phi_c_lighting = self.lighting.get_thermal_gain(timestep) if self.lighting else 0.0
        phi_c_appliances = self.appliances.get_thermal_gain(timestep) if self.appliances else 0.0

        phi_c = phi_c_occupancy + phi_c_lighting + phi_c_appliances
        self.phi_c[idx] = phi_c

        # Solar thermal collector gains (if any)
        phi_collector = self.solar_thermal.get_phi_s(timestep) if self.solar_thermal else 0.0

        # Hot water demand heat transfer coefficient (variable)
        h_dhw = self.hot_water.get_h_demand(timestep) if self.hot_water else 0.0

        # Calculate temperature changes using Euler's method
        # dt/dt = (timestep / capacitance) * (heat flows)

        # External building node
        delta_theta_b = (self.timestep_seconds / self.c_b) * (
            -(self.h_ob + self.h_bi) * theta_b_prev +
            self.h_bi * theta_i_prev +
            self.h_ob * theta_o
        )

        # Internal building node
        delta_theta_i = (self.timestep_seconds / self.c_i) * (
            self.h_bi * theta_b_prev -
            (self.h_v + self.h_bi + self.h_em + self.h_cool + self.h_loss) * theta_i_prev +
            self.h_v * theta_o +
            self.h_em * theta_em_prev +
            self.h_cool * theta_cool_prev +
            self.h_loss * theta_cyl_prev +
            phi_s + phi_c
        )

        # Heating emitters (radiators)
        delta_theta_em = (self.timestep_seconds / self.c_em) * (
            self.h_em * theta_i_prev -
            self.h_em * theta_em_prev +
            phi_h_space
        )

        # Cooling emitters
        delta_theta_cool = (self.timestep_seconds / self.c_cool) * (
            self.h_cool * theta_i_prev -
            self.h_cool * theta_cool_prev +
            phi_h_cooling
        )

        # Hot water cylinder
        delta_theta_cyl = (self.timestep_seconds / self.c_cyl) * (
            self.h_loss * theta_i_prev -
            (self.h_loss + h_dhw) * theta_cyl_prev +
            h_dhw * self.theta_cw +
            phi_h_water +
            phi_collector
        )

        # Update temperatures for current timestep
        self.theta_b[idx] = theta_b_prev + delta_theta_b
        self.theta_i[idx] = theta_i_prev + delta_theta_i
        self.theta_em[idx] = theta_em_prev + delta_theta_em
        self.theta_cool[idx] = theta_cool_prev + delta_theta_cool
        self.theta_cyl[idx] = theta_cyl_prev + delta_theta_cyl

    def set_local_climate(self, local_climate):
        """Set reference to local climate model."""
        self.local_climate = local_climate

    def set_heating_system(self, heating_system):
        """Set reference to heating system."""
        self.heating_system = heating_system

    def set_heating_controls(self, heating_controls):
        """Set reference to heating controls."""
        self.heating_controls = heating_controls

    def set_cooling_system(self, cooling_system):
        """Set reference to cooling system."""
        self.cooling_system = cooling_system

    def set_occupancy(self, occupancy):
        """Set reference to occupancy model."""
        self.occupancy = occupancy

    def set_lighting(self, lighting):
        """Set reference to lighting model."""
        self.lighting = lighting

    def set_appliances(self, appliances):
        """Set reference to appliances model."""
        self.appliances = appliances

    def set_hot_water(self, hot_water):
        """Set reference to hot water model."""
        self.hot_water = hot_water

    def set_solar_thermal(self, solar_thermal):
        """Set reference to solar thermal system."""
        self.solar_thermal = solar_thermal

    # ===============================================================================================
    # PROPERTIES AND ACCESSORS
    # ===============================================================================================

    def get_internal_temperature(self, timestep: int) -> float:
        """Get internal building temperature at specified timestep (1-based)."""
        return self.theta_i[timestep - 1]

    def get_cylinder_temperature(self, timestep: int) -> float:
        """Get hot water cylinder temperature at specified timestep (1-based)."""
        return self.theta_cyl[timestep - 1]

    def get_emitter_temperature(self, timestep: int) -> float:
        """Get heating emitter temperature at specified timestep (1-based)."""
        return self.theta_em[timestep - 1]

    def get_external_temperature(self, timestep: int) -> float:
        """Get external building node temperature at specified timestep (1-based)."""
        return self.theta_b[timestep - 1]

    def get_theta_cool(self, timestep: int) -> float:
        """
        Get cooling emitter temperature at specified timestep (1-based).

        VBA: GetTheta_cool property (clsBuilding.cls lines 115-117)
        """
        return self.theta_cool[timestep - 1]

    def get_mean_theta_i(self) -> float:
        """
        Get mean internal temperature over the entire day.

        VBA: GetMeanTheta_i property (clsBuilding.cls lines 119-121)
        """
        return float(np.mean(self.theta_i))

    def get_target_heat_space(self, timestep: int) -> float:
        """
        Calculate target heat demand for space heating (W).

        Matches VBA GetPhi_hSpace property (clsBuilding.cls lines 158-194).
        Calculates heat needed for emitters to reach target temperature.

        Parameters
        ----------
        timestep : int
            Current timestep (1-based, 1-1440)

        Returns
        -------
        float
            Target heat demand in Watts
        """
        if self.heating_controls is None:
            return 0.0

        # Get previous timestep temperatures (or initial if timestep=1)
        if timestep == 1:
            theta_em = self.theta_em[0]
            theta_i = self.theta_i[0]
        else:
            theta_em = self.theta_em[timestep - 2]
            theta_i = self.theta_i[timestep - 2]

        # Emitter deadband and target (VBA lines 175-177)
        # VBA: dblTheta_emTarget = dblEmitterDeadband + Application.index(wsBuildings.Range("rTheta_em"), intOffset + intBuildingIndex).Value
        emitter_deadband = 5.0
        theta_em_target = emitter_deadband + self.theta_em_nominal

        # Calculate target heat delivery to emitters (VBA lines 189-190)
        # Capacitance term + heat transfer to room
        phi_h_space_target = (
            (self.c_em / self.timestep_seconds) * (theta_em_target - theta_em) +
            self.h_em * (theta_em - theta_i)
        )

        return phi_h_space_target

    def get_target_heat_water(self, timestep: int) -> float:
        """
        Calculate target heat demand for hot water heating (W).

        Matches VBA GetPhi_hWater property (clsBuilding.cls lines 124-155).
        Calculates heat needed to maintain cylinder at setpoint.

        Parameters
        ----------
        timestep : int
            Current timestep (1-based, 1-1440)

        Returns
        -------
        float
            Target heat demand in Watts
        """
        if self.heating_controls is None or self.hot_water is None:
            return 0.0

        # Get hot water thermostat setpoint (VBA line 132)
        theta_target = self.heating_controls.get_hot_water_thermostat_setpoint()

        # Get previous timestep temperatures (or initial if timestep=1) (VBA lines 137-143)
        if timestep == 1:
            theta_cyl = self.theta_cyl[0]
            theta_i = self.theta_i[0]
        else:
            theta_cyl = self.theta_cyl[timestep - 2]
            theta_i = self.theta_i[timestep - 2]

        # Get variable hot water demand heat transfer coefficient (VBA line 145)
        h_dhw = self.hot_water.get_h_demand(timestep)

        # Calculate target heat input (VBA lines 150-152)
        # Capacitance term + hot water draw losses + standing losses
        phi_target = (
            (self.c_cyl / self.timestep_seconds) * (theta_target - theta_cyl) +
            h_dhw * (theta_cyl - self.theta_cw) +
            self.h_loss * (theta_cyl - theta_i)
        )

        return phi_target

    def get_target_cooling(self, timestep: int) -> float:
        """
        Calculate target cooling demand for space cooling (W).

        Matches VBA GetPhi_hCooling property (clsBuilding.cls lines 197-232).
        Calculates cooling needed for emitters to reach target temperature.

        Parameters
        ----------
        timestep : int
            Current timestep (1-based, 1-1440)

        Returns
        -------
        float
            Target cooling demand in Watts (negative for cooling)
        """
        if self.heating_controls is None:
            return 0.0

        # Get previous timestep temperatures (or initial if timestep=1)
        if timestep == 1:
            theta_cool = self.theta_cool[0]
            theta_i = self.theta_i[0]
        else:
            theta_cool = self.theta_cool[timestep - 2]
            theta_i = self.theta_i[timestep - 2]

        # Emitter deadband and target (VBA lines 214-217)
        # VBA: dblTheta_coolTarget = Application.index(wsBuildings.Range("rTheta_cool"), intOffset + intBuildingIndex).Value - dblEmitterDeadband
        emitter_deadband = 5.0
        theta_cool_target = self.theta_cool_nominal - emitter_deadband

        # Calculate target cooling delivery to emitters (VBA lines 227-228)
        # Capacitance term + heat transfer to room
        phi_h_cooling_target = (
            (self.c_cool / self.timestep_seconds) * (theta_cool_target - theta_cool) +
            self.h_cool * (theta_cool - theta_i)
        )

        return phi_h_cooling_target
