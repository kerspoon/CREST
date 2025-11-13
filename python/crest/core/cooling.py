"""
CREST Demand Model - Cooling System

Space cooling system model for air conditioning, air coolers, and fans.

VBA Source: original/clsCoolingSystem.cls (158 lines)
CSV Data: data/CoolingSystems.csv (4 system types)

System Types:
1. No cooling (default)
2. Fans only (20W electrical, minimal cooling)
3. Air cooler (1000W cooling, 100W electrical, COP=10)
4. Air conditioning (5000W cooling, 1250W electrical, COP=4)

Key Features:
- Simple control logic: on/off based on thermostat, timer, and emitter states
- Capacity limiting: cooling provided = min(0, max(capacity, target))
- Electricity demand: pump power when active, standby power when idle

Execution: Runs 1440 TIMES per day in thermal loop (coupled to building and controls)

Note: Despite the name "clsCoolingSystem", this is part of the HVAC system,
      not renewable energy. It consumes electricity rather than generating it.
"""

import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from crest.data.loader import CRESTDataLoader
    from crest.utils.random import RandomGenerator
    from crest.core.controls import HeatingControls
    from crest.core.building import Building


class CoolingSystem:
    """
    Space cooling system model for air conditioning and fans.

    Provides space cooling based on thermostat, timer, and emitter control signals
    from HeatingControls. Calculates cooling delivered to space and electricity
    consumption.

    VBA Source: clsCoolingSystem.cls (158 lines)
    """

    def __init__(self, data_loader: 'CRESTDataLoader', random_gen: 'RandomGenerator'):
        """
        Initialize cooling system model.

        Args:
            data_loader: CREST data loader for CSV files
            random_gen: Random number generator (not used but standard interface)

        VBA Source: Class initialization (lines 20-47)
        """
        self.data_loader = data_loader
        self.random_gen = random_gen

        # Index variables (VBA lines 20-23)
        self.dwelling_index: int = 0
        self.run_number: int = 0
        self.cooling_system_index: int = 0

        # Cooling system parameters (VBA lines 25-41)
        self.cooling_type: str = ""  # Type description (e.g., "Air Conditioning")
        self.cooling_system_type: int = 0  # System type index (1-4)
        self.phi_h_cool: float = 0.0  # Maximum cooling capacity (W, negative)
        self.P_standby_cool: float = 0.0  # Standby power (W)
        self.P_pump_cool: float = 0.0  # Pump/fan power (W)
        self.eta_h_cool: float = 0.0  # Thermal efficiency / COP

        # Output arrays - 1440 timesteps (VBA lines 43-47)
        # Note: VBA uses 1-based arrays (1 To 1440), Python uses 0-based
        self.phi_h_cooling = np.zeros(1440)  # Cooling to space (W, negative)
        self.phi_cooling = np.zeros(1440)  # Electricity demand (W)

        # Component references (set by initialize)
        self.controls: 'HeatingControls' = None
        self.building: 'Building' = None

    def initialize(self,
                   dwelling_index: int,
                   run_number: int,
                   controls: 'HeatingControls',
                   building: 'Building',
                   cooling_system_index: Optional[int] = None) -> None:
        """
        Initialize cooling system with configuration and component references.

        Args:
            dwelling_index: Dwelling index (0-based in Python)
            run_number: Simulation run number
            controls: HeatingControls instance (for thermostat/timer signals)
            building: Building instance (for required cooling load)
            cooling_system_index: Cooling system index (1-4)
                                 1=no cooling, 2=fans, 3=air cooler, 4=AC
                                 If None, will be loaded from CSV in future

        VBA Source: InitialiseCoolingSystem (lines 77-99)
        """
        # Store indexes (VBA lines 84-86)
        self.dwelling_index = dwelling_index
        self.run_number = run_number

        # Store component references
        self.controls = controls
        self.building = building

        # Determine cooling system index (VBA line 86)
        if cooling_system_index is not None:
            self.cooling_system_index = cooling_system_index
        else:
            # Future: Load from dwelling configuration CSV
            self.cooling_system_index = 1  # Default: no cooling

        # Load cooling system parameters (VBA lines 88-97)
        self._load_cooling_system_config()

    def _load_cooling_system_config(self) -> None:
        """
        Load cooling system configuration from CSV file.

        Reads system type, capacity, power consumption, and efficiency from
        CoolingSystems.csv.

        VBA Source: InitialiseCoolingSystem (lines 88-97)
        CSV: data/CoolingSystems.csv
        Header rows: 4 (rows 0-3)
        Data starts: row 4
        Columns: System index, Proportion, Type, System type, ... (30 columns)
        """
        cooling_systems = self.data_loader.load_cooling_systems()

        # VBA uses offset=4 (4 header rows), then indexes with intCoolingSystemIndex
        # CSV structure: row 4 = system 1, row 5 = system 2, etc.
        # Python 0-based: cooling_system_index 1 → row 0 of data (after skipping headers)
        system_row = self.cooling_system_index - 1  # Convert 1-based to 0-based

        if system_row < 0 or system_row >= len(cooling_systems):
            raise ValueError(f"Invalid cooling system index: {self.cooling_system_index}")

        # Extract cooling system parameters (VBA lines 90-95)
        # CSV columns: 0=index, 1=proportion, 2=type, 3=system_type, ...
        #              8=efficiency, 9=heat_output, 10=standby, 11=pump
        self.cooling_type = str(cooling_systems.iloc[system_row]['Type of heating unit'])
        self.cooling_system_type = int(cooling_systems.iloc[system_row]['Type of system'])
        self.phi_h_cool = float(cooling_systems.iloc[system_row]['Heat output of unit'])
        self.P_standby_cool = float(cooling_systems.iloc[system_row]['Standby power'])
        self.P_pump_cool = float(cooling_systems.iloc[system_row]['Pump power'])
        self.eta_h_cool = float(cooling_systems.iloc[system_row]['Thermal efficiency'])

    def calculate_cooling_output(self, current_timestep: int) -> None:
        """
        Calculate cooling output for current timestep.

        This method is called 1440 TIMES per day (once per minute) in the thermal loop.

        Control logic:
        1. Get control signals from HeatingControls (thermostat, timer, emitter)
        2. If all three signals are ON: provide cooling up to capacity
        3. Calculate electricity: pump power if (thermostat AND timer), else standby

        Args:
            current_timestep: Current timestep (1-1440, VBA 1-based)

        VBA Source: CalculateCoolingOutput (lines 107-144)
        """
        # Convert to 0-based index for Python arrays
        timestep_0based = current_timestep - 1

        # Get control signals from heating controller (VBA lines 120-122)
        # This is a 3-way AND gate: thermostat × timer × emitter
        space_cooling_thermostat = self.controls.get_space_cooling_thermostat_state(current_timestep)
        space_cooling_timer = self.controls.get_space_cooling_timer_state(current_timestep)
        cooler_emitter = self.controls.get_cooler_emitter_state(current_timestep)

        space_cooling_on_off = (
            space_cooling_thermostat *
            space_cooling_timer *
            cooler_emitter
        )

        # Set default: no cooling (VBA line 126)
        self.phi_h_cooling[timestep_0based] = 0.0

        # If cooling is needed (VBA lines 129-142)
        if space_cooling_on_off:
            # Get target cooling load from building (VBA line 131)
            phi_h_cooling_target = self.building.get_target_cooling(current_timestep)

            # Calculate cooling provided (VBA line 133)
            # Min(0, Max(capacity, target)) ensures:
            #   - Cooling is negative (Min with 0)
            #   - Doesn't exceed capacity (Max with phi_h_cool which is negative)
            # Example: capacity=-5000W, target=-3000W → max(-5000,-3000)=-3000, min(0,-3000)=-3000
            #          capacity=-5000W, target=-8000W → max(-5000,-8000)=-5000, min(0,-5000)=-5000
            phi_h_cooling = min(0.0, max(self.phi_h_cool, phi_h_cooling_target))
            self.phi_h_cooling[timestep_0based] = phi_h_cooling

            # Calculate electricity demand (VBA lines 136-139)
            # CRITICAL: Electricity calculation is INSIDE the space_cooling_on_off block
            # When space_cooling_on_off is False, phi_cooling is NOT SET (remains at previous value, typically 0)
            # Note: Electricity uses 2-way AND (thermostat × timer), NOT 3-way AND with emitter
            if space_cooling_thermostat * space_cooling_timer == 1:
                self.phi_cooling[timestep_0based] = self.P_pump_cool
            else:
                self.phi_cooling[timestep_0based] = self.P_standby_cool

    # ===============================================================================================
    # Getter Methods (VBA Property Get)
    # ===============================================================================================

    def get_cooling_to_space(self, timestep: int) -> float:
        """
        Get cooling delivered to space at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Cooling to space (W, negative value)

        VBA Source: Property GetCoolingToSpace (lines 54-56)
        """
        return self.phi_h_cooling[timestep - 1]

    def get_cooling_system_power_demand(self, timestep: int) -> float:
        """
        Get cooling system electricity demand at specific timestep.

        Args:
            timestep: Timestep (1-1440, VBA 1-based)

        Returns:
            Electricity demand (W)

        VBA Source: Property GetCoolingSystemPowerDemand (lines 58-60)
        """
        return self.phi_cooling[timestep - 1]

    def get_cooling_system_type(self) -> int:
        """
        Get cooling system type index.

        Returns:
            System type (1=none, 2=fans, 3=air cooler, 4=AC)

        VBA Source: Property GetCoolingSystemType (lines 62-64)
        """
        return self.cooling_system_type

    def get_daily_sum_cooling_energy(self) -> float:
        """
        Get daily sum of cooling energy delivered to space.

        Returns:
            Total daily cooling energy (W·min or Wh/60, negative value)

        VBA Source: Property GetDailySumCoolingEnergy (lines 66-68)
        """
        return np.sum(self.phi_h_cooling)
