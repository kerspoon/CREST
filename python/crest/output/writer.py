"""
Results writer for CREST Demand Model outputs.

Provides functionality to save minute-level and summary data to CSV files
in a format compatible with the original Excel/VBA model for validation.
"""

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, TextIO
import numpy as np


@dataclass
class OutputConfig:
    """Configuration for output writer."""
    output_dir: Path
    save_minute_data: bool = True
    save_daily_summary: bool = True
    save_global_climate: bool = True


class ResultsWriter:
    """
    Writes simulation results to CSV files.

    Output format matches the original Excel model's "Results - disaggregated"
    sheet for validation purposes.
    """

    def __init__(self, config: OutputConfig):
        """
        Initialize results writer.

        Parameters
        ----------
        config : OutputConfig
            Output configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File handles
        self._minute_file: Optional[TextIO] = None
        self._minute_writer: Optional[csv.writer] = None
        self._summary_file: Optional[TextIO] = None
        self._summary_writer: Optional[csv.writer] = None
        self._climate_file: Optional[TextIO] = None
        self._climate_writer: Optional[csv.writer] = None

        # Initialize files
        if config.save_minute_data:
            self._init_minute_file()
        if config.save_daily_summary:
            self._init_summary_file()
        if config.save_global_climate:
            self._init_climate_file()

    def _init_minute_file(self):
        """Initialize minute-level data file with headers.

        Format matches Excel VBA export exactly:
        - Row 1: Description (with UTF-8 BOM)
        - Row 2: Column names
        - Row 3: Symbol names (Greek letters, abbreviations)
        - Row 4: Units
        - Row 5+: Data
        """
        filename = self.output_dir / "results_minute_level.csv"
        self._minute_file = open(filename, 'w', newline='', encoding='utf-8-sig')
        self._minute_writer = csv.writer(self._minute_file)

        # Row 1: Description (BOM handled by encoding='utf-8-sig')
        description_row = ['Simulation results - disaggregated. Individual dwelling simulations are listed below in sequential order (dwelling 1 first, then dwelling 2 etc.)'] + [''] * 39
        self._minute_writer.writerow(description_row)

        # Row 2: Column names (40 columns matching Excel exactly)
        column_names = [
            'Dwelling index',
            'Date',
            'Time',
            'Occupancy',
            'Activity',
            'Lighting demand',
            'Appliance demand',
            'Casual thermal gains from occupants, lighting and appliances',
            'Outdoor temperature',
            'Outdoor global radiation (horizontal)',
            'Passive solar gains',
            'Primary heating system thermal output',
            'External building node temperature',
            'Internal building node temperature',
            'Hot water demand (litres)',
            'Hot water temperature in hot water tank',
            'Space heating timer settings',
            'Hot water heating timer settings',
            'Heating system switched on',
            'Hot water heating required',
            'Emitter temperature',
            'Radiation incident on PV array',
            'PV output',
            'Net dwelling electricity demand',
            'Heat output from primary heating system to space',
            'Heat output from primary heating system to hot water',
            'Fuel flow rate (gas)',
            'Solar power incident on collector',
            'Solar thermal collector control state',
            'Solar thermal collector temperature',
            'Heat gains to cylinder from solar thermal collector',
            'Dwelling self-consumption',
            'Space cooling timer settings',
            'Cooling system switched on',
            'Cooling output from cooling system to space',
            'Cooler Emitter temperature',
            'Heating Thermostat Set Point',
            'Cooling Thermostat Set Point',
            'Electricity used by cooling system',
            'Electricity used by heating system'
        ]
        self._minute_writer.writerow(column_names)

        # Row 3: Symbols (Greek letters and abbreviations)
        symbols = [
            '', '', '',  # Dwelling index, Date, Time
            '', '',  # Occupancy, Activity
            'Plight', 'Pa', 'φc',
            'θo', 'Go', 'φs', 'φh', 'θb', 'θi',
            '', 'θcyl',  # Hot water demand, Cylinder temp
            '', '', '', '',  # Timer settings, heating on, HW required
            'θem', 'Gi', 'Ppv', 'Pnet',
            'φh, space', 'φh, water', 'mfuel',
            '', '', 'θcollector', 'φcollector',
            'Pself',
            '', '',  # Cooling timer, cooling on
            'φh, space', 'θem', 'θem', 'θem',
            '', ''  # Cooling/heating electricity
        ]
        self._minute_writer.writerow(symbols)

        # Row 4: Units
        units = [
            '', '', '',  # Dwelling index, Date, Time
            '', '',  # Occupancy, Activity
            'W', 'W', 'W',
            '°C', 'Wm-2', 'W', 'W', '°C', '°C',
            'ltr.min-1', '°C',
            '', '', '', '',  # Timer settings, heating on, HW required
            '°C', 'Wm-2', 'W', 'W',
            'W', 'W', 'm3/h',
            'W', '', '°C', 'W',
            'kWh',
            '', '',  # Cooling timer, cooling on
            'W', '°C', '°C', '°C',
            'W', 'W'
        ]
        self._minute_writer.writerow(units)

    def _init_summary_file(self):
        """Initialize daily summary file with headers.

        VBA Reference: DailyTotals (mdlThermalElectricalModel.bas lines 1057-1121)
        Outputs all 17 columns matching VBA daily totals sheet.

        Format matches Excel VBA export exactly:
        - Row 1: Description (with UTF-8 BOM)
        - Row 2: Column names
        - Row 3: Symbol names
        - Row 4: Units
        - Row 5+: Data
        """
        filename = self.output_dir / "results_daily_summary.csv"
        self._summary_file = open(filename, 'w', newline='', encoding='utf-8-sig')
        self._summary_writer = csv.writer(self._summary_file)

        # Row 1: Description (BOM handled by encoding='utf-8-sig')
        description_row = ['Simulation results - daily sums'] + [''] * 16
        self._summary_writer.writerow(description_row)

        # Row 2: Column names (17 columns matching VBA DailyTotals output lines 1103-1119)
        column_names = [
            'Dwelling index',
            'Date',
            'Mean active occupancy',
            'Proportion of day actively occupied',
            'Lighting demand',
            'Appliance demand',
            'PV output',
            'Total dwelling electricity demand',
            'Total self-consumption',
            'Net dwelling electricity demand',
            'Hot water demand (litres)',
            'Average indoor air temperature',
            'Thermal energy used for space heating',
            'Thermal energy used for hot water heating',
            'Gas demand',
            'Space thermostat set point',
            'Solar thermal collector heat gains'
        ]
        self._summary_writer.writerow(column_names)

        # Row 3: Symbols
        symbols = [
            '', '',  # Dwelling index, Date
            '', '',  # Mean active occupancy, Proportion day active
            'Elight', 'Ea', 'Epv', 'Etotal', 'Eself', 'Enet',
            '',  # Hot water demand
            'θi',
            'Eh, space', 'Eh, water',
            'mfuel',
            '',  # Thermostat setpoint
            ''   # Solar thermal gains
        ]
        self._summary_writer.writerow(symbols)

        # Row 4: Units
        units = [
            '', '',  # Dwelling index, Date
            '', '',  # Mean active occupancy, Proportion day active
            'kWh', 'kWh', 'kWh', 'kWh', 'kWh', 'kWh',
            'litres/day',
            '°C',
            'kWh', 'kWh',
            'm3/day',
            '°C',
            'kWh'
        ]
        self._summary_writer.writerow(units)

    def _init_climate_file(self):
        """Initialize global climate file with headers."""
        filename = self.output_dir / "global_climate.csv"
        self._climate_file = open(filename, 'w', newline='')
        self._climate_writer = csv.writer(self._climate_file)

        # Write header
        header = [
            'Minute',
            'Clearness_Index',
            'Clear_Sky_Irradiance_Wm2',
            'Global_Irradiance_Wm2',
            'Outdoor_Temperature_C'
        ]
        self._climate_writer.writerow(header)

    def write_minute_data(self, dwelling_idx: int, dwelling, date_str: str = "01/01/2015"):
        """
        Write minute-level data for one dwelling - all 40 columns matching Excel format.

        Parameters
        ----------
        dwelling_idx : int
            Dwelling index (0-based)
        dwelling : Dwelling
            Dwelling object with simulation results
        date_str : str
            Date string in DD/MM/YYYY format (default: "01/01/2015")
        """
        if not self.config.save_minute_data or self._minute_writer is None:
            return

        # Get occupancy data (10-minute resolution, expand to 1-minute)
        occupancy_1min = self._expand_10min_to_1min(dwelling.occupancy.active_occupancy)
        at_home_1min = self._expand_10min_to_1min(
            self._calculate_at_home(dwelling.occupancy.combined_states)
        )

        # Get heating controls - MUST exist for all dwellings
        if not hasattr(dwelling, 'heating_controls'):
            raise AttributeError(
                f"CRITICAL: Dwelling {dwelling_idx + 1} has no heating_controls attribute. "
                "All dwellings must have heating controls initialized."
            )
        heating_controls = dwelling.heating_controls

        if not hasattr(heating_controls, 'space_heating_timer'):
            raise AttributeError(
                f"CRITICAL: Dwelling {dwelling_idx + 1} heating_controls has no space_heating_timer. "
                "HeatingControls must be properly initialized with timer schedules."
            )
        heating_timer = heating_controls.space_heating_timer

        if not hasattr(heating_controls, 'hot_water_timer'):
            raise AttributeError(
                f"CRITICAL: Dwelling {dwelling_idx + 1} heating_controls has no hot_water_timer. "
                "HeatingControls must be properly initialized with timer schedules."
            )
        hw_timer = heating_controls.hot_water_timer

        if not hasattr(heating_controls, 'space_heating_setpoint'):
            raise AttributeError(
                f"CRITICAL: Dwelling {dwelling_idx + 1} heating_controls has no space_heating_setpoint. "
                "HeatingControls must be properly initialized with thermostat setpoints."
            )

        if not hasattr(heating_controls, 'space_cooling_setpoint'):
            raise AttributeError(
                f"CRITICAL: Dwelling {dwelling_idx + 1} heating_controls has no space_cooling_setpoint. "
                "HeatingControls must be properly initialized with thermostat setpoints."
            )

        # Calculate PV irradiance if PV system exists
        pv_irradiance = np.zeros(1440)
        if dwelling.pv_system:
            if not hasattr(dwelling.pv_system, 'G_i'):
                raise AttributeError(
                    f"CRITICAL: Dwelling {dwelling_idx + 1} has pv_system but it's missing 'G_i' attribute. "
                    "PV system must be fully initialized."
                )
            pv_irradiance = dwelling.pv_system.G_i

        # Calculate self-consumption if PV system exists
        self_consumption = np.zeros(1440)
        if dwelling.pv_system:
            if not hasattr(dwelling.pv_system, 'P_self'):
                raise AttributeError(
                    f"CRITICAL: Dwelling {dwelling_idx + 1} has pv_system but it's missing 'P_self' attribute. "
                    "PV system must be fully initialized with self-consumption data."
                )
            self_consumption = dwelling.pv_system.P_self / 60.0 / 1000.0  # Convert W-min to kWh

        # Solar thermal data if available
        solar_collector_power = np.zeros(1440)
        solar_collector_state = np.zeros(1440)
        solar_collector_temp = np.zeros(1440)
        solar_collector_gains = np.zeros(1440)
        if dwelling.solar_thermal:
            if not hasattr(dwelling.solar_thermal, 'P_incident'):
                raise AttributeError(
                    f"CRITICAL: Dwelling {dwelling_idx + 1} has solar_thermal but missing 'P_incident'. "
                    "Solar thermal system must be fully initialized."
                )
            if not hasattr(dwelling.solar_thermal, 'solar_thermal_on_off'):
                raise AttributeError(
                    f"CRITICAL: Dwelling {dwelling_idx + 1} has solar_thermal but missing 'solar_thermal_on_off'. "
                    "Solar thermal system must be fully initialized."
                )
            if not hasattr(dwelling.solar_thermal, 'theta_collector'):
                raise AttributeError(
                    f"CRITICAL: Dwelling {dwelling_idx + 1} has solar_thermal but missing 'theta_collector'. "
                    "Solar thermal system must be fully initialized."
                )
            if not hasattr(dwelling.solar_thermal, 'phi_s'):
                raise AttributeError(
                    f"CRITICAL: Dwelling {dwelling_idx + 1} has solar_thermal but missing 'phi_s'. "
                    "Solar thermal system must be fully initialized."
                )
            solar_collector_power = dwelling.solar_thermal.P_incident
            solar_collector_state = dwelling.solar_thermal.solar_thermal_on_off
            solar_collector_temp = dwelling.solar_thermal.theta_collector
            solar_collector_gains = dwelling.solar_thermal.phi_s

        # Cooling system data if available
        cooling_output = np.zeros(1440)
        if dwelling.cooling_system:
            if not hasattr(dwelling.cooling_system, 'phi_h_cooling'):
                raise AttributeError(
                    f"CRITICAL: Dwelling {dwelling_idx + 1} has cooling_system but missing 'phi_h_cooling'. "
                    "Cooling system must be fully initialized."
                )
            cooling_output = dwelling.cooling_system.phi_h_cooling

        # Write data for each minute
        for minute in range(1, 1441):
            idx = minute - 1  # 0-based index for arrays

            # Format time as HH:MM:SS AM/PM
            time_str = self._format_time_12hr(minute)

            # Get lighting and appliance demand
            lighting_w = dwelling.lighting.get_total_demand(minute)
            appliance_w = dwelling.appliances.get_total_demand(minute)

            # Calculate net electricity demand
            pv_output_w = dwelling.pv_system.get_pv_output(minute) if dwelling.pv_system else 0.0
            heating_elec_w = dwelling.heating_system.get_heating_system_power_demand(minute)
            cooling_elec_w = dwelling.cooling_system.get_cooling_system_power_demand(minute) if dwelling.cooling_system else 0.0
            net_elec_w = lighting_w + appliance_w + heating_elec_w + cooling_elec_w - pv_output_w

            # Collect all 40 variables (matching Excel column order exactly)
            # Arrays MUST be length 1440 - will crash with IndexError if not
            row = [
                dwelling_idx + 1,                                    # 1. Dwelling index
                date_str,                                            # 2. Date
                time_str,                                            # 3. Time
                int(at_home_1min[idx]),                             # 4. Occupancy
                int(occupancy_1min[idx]),                           # 5. Activity
                lighting_w,                                          # 6. Lighting demand (W)
                appliance_w,                                         # 7. Appliance demand (W)
                dwelling.building.phi_c[idx],                        # 8. Casual gains (W)
                dwelling.local_climate.get_temperature(idx),         # 9. Outdoor temp (°C)
                dwelling.local_climate.get_irradiance(idx),          # 10. Global radiation (W/m²)
                dwelling.building.phi_s[idx],                        # 11. Passive solar gains (W)
                dwelling.heating_system.phi_h_output[idx],           # 12. Primary heating output (W)
                dwelling.building.theta_b[idx],                      # 13. External building temp (°C)
                dwelling.building.theta_i[idx],                      # 14. Internal building temp (°C)
                dwelling.hot_water.hot_water_demand[idx],            # 15. Hot water demand (L/min)
                dwelling.building.theta_cyl[idx],                    # 16. Cylinder temp (°C)
                int(heating_timer[idx]),                             # 17. Space heating timer
                int(hw_timer[idx]),                                  # 18. HW heating timer
                0,                                                   # 19. Heating system switched on
                0,                                                   # 20. HW heating required
                dwelling.building.theta_em[idx],                     # 21. Emitter temp (°C)
                pv_irradiance[idx],                                  # 22. PV irradiance (W/m²)
                pv_output_w,                                         # 23. PV output (W)
                net_elec_w,                                          # 24. Net electricity demand (W)
                dwelling.heating_system.phi_h_space[idx],            # 25. Space heating (W)
                dwelling.heating_system.phi_h_water[idx],            # 26. Water heating (W)
                dwelling.heating_system.m_fuel[idx] * 60.0,          # 27. Gas flow (m³/h, convert from m³/min)
                solar_collector_power[idx],                          # 28. Solar collector power (W)
                int(solar_collector_state[idx]),                     # 29. Solar collector state
                solar_collector_temp[idx],                           # 30. Solar collector temp (°C)
                solar_collector_gains[idx],                          # 31. Solar collector gains (W)
                self_consumption[idx],                               # 32. Self-consumption (kWh)
                0,                                                   # 33. Space cooling timer
                0,                                                   # 34. Cooling system switched on
                cooling_output[idx],                                 # 35. Cooling output (W)
                dwelling.building.theta_cool[idx],                   # 36. Cooler emitter temp (°C)
                heating_controls.space_heating_setpoint,             # 37. Heating setpoint (°C)
                heating_controls.space_cooling_setpoint,             # 38. Cooling setpoint (°C)
                cooling_elec_w,                                      # 39. Cooling electricity (W)
                heating_elec_w                                       # 40. Heating electricity (W)
            ]

            self._minute_writer.writerow(row)

        # Flush to disk periodically
        self._minute_file.flush()

    def _format_time_12hr(self, minute: int) -> str:
        """
        Format minute of day as 12-hour time string (HH:MM:SS AM/PM).

        Parameters
        ----------
        minute : int
            Minute of day (1-1440)

        Returns
        -------
        str
            Time string in format "HH:MM:SS AM" or "HH:MM:SS PM"
        """
        # Convert to 0-based
        minute_0 = minute - 1

        hours = minute_0 // 60
        mins = minute_0 % 60

        # Convert to 12-hour format
        am_pm = "AM" if hours < 12 else "PM"
        hours_12 = hours % 12
        if hours_12 == 0:
            hours_12 = 12

        return f"{hours_12:02d}:{mins:02d}:00 {am_pm}"

    def write_daily_summary(self, dwelling_idx: int, dwelling, date_str: str = "2015-01-01 00:00:00"):
        """
        Write daily summary for one dwelling - all 17 VBA-matching columns.

        VBA Reference: DailyTotals (mdlThermalElectricalModel.bas lines 1057-1121)

        Parameters
        ----------
        dwelling_idx : int
            Dwelling index (0-based)
        dwelling : Dwelling
            Dwelling object with simulation results
        date_str : str
            Date string in "YYYY-MM-DD HH:MM:SS" format (default: "2015-01-01 00:00:00")
        """
        if not self.config.save_daily_summary or self._summary_writer is None:
            return

        # VBA line 1081: dblMeanActiveOccupancy = aOccupancy(intRunNumber).GetMeanActiveOccupancy
        mean_active_occupancy = dwelling.occupancy.get_mean_active_occupancy()

        # VBA line 1082: dblProportionActivelyOccupied = aOccupancy(intRunNumber).GetPrActivelyOccupied
        proportion_actively_occupied = dwelling.occupancy.get_proportion_actively_occupied()

        # VBA line 1083: dblLightingDemand = aLighting(intRunNumber).GetDailySumLighting / 60 / 1000
        lighting_kwh = dwelling.lighting.get_daily_energy() / 60.0 / 1000.0

        # VBA line 1084: dblApplianceDemand = aAppliances(intRunNumber).GetDailySumApplianceDemand / 60 / 1000
        appliances_kwh = dwelling.appliances.get_daily_energy() / 60.0 / 1000.0

        # VBA line 1086: dblTotalElectricityDemand = dblLightingDemand + dblApplianceDemand
        # NOTE: This does NOT include heating/cooling/pump electricity - only lighting + appliances
        total_electricity_kwh = lighting_kwh + appliances_kwh

        # VBA line 1087: dblPVOutput = aPVSystem(intRunNumber).GetDailySumPvOutput / 60 / 1000
        pv_output_kwh = dwelling.pv_system.get_daily_sum_pv_output() / 60.0 / 1000.0 if dwelling.pv_system else 0.0

        # VBA line 1090: dblSelfConsumption = aPVSystem(intRunNumber).GetDailySumP_self / 60 / 1000
        self_consumption_kwh = dwelling.pv_system.get_daily_sum_self_consumption() / 60.0 / 1000.0 if dwelling.pv_system else 0.0

        # VBA line 1088: dblNetElectricityDemand = aPVSystem(intRunNumber).GetDailySumP_net / 60 / 1000
        net_electricity_kwh = dwelling.pv_system.get_daily_sum_net_demand() / 60.0 / 1000.0 if dwelling.pv_system else total_electricity_kwh

        # VBA line 1092: dblHotWaterDemand = aHotWater(intRunNumber).GetDailySumHotWaterDemand
        hot_water_litres = dwelling.hot_water.get_daily_hot_water_volume()

        # VBA line 1094: dblAverageIndoorTemperature = aBuilding(intRunNumber).GetMeanTheta_i
        mean_temp = np.mean(dwelling.building.theta_i)

        # VBA line 1096: dblThermalEnergySpace = aPrimaryHeatingSystem(intRunNumber).GetDailySumThermalEnergySpace / 60 / 1000
        thermal_energy_space_kwh = dwelling.heating_system.get_daily_thermal_energy_space() / 60.0 / 1000.0

        # VBA line 1097: dblThermalEnergyWater = aPrimaryHeatingSystem(intRunNumber).GetDailySumThermalEnergyWater / 60 / 1000
        thermal_energy_water_kwh = dwelling.heating_system.get_daily_thermal_energy_water() / 60.0 / 1000.0

        # VBA line 1099: dblGasDemand = aPrimaryHeatingSystem(intRunNumber).GetDailySumFuelFlow / 60
        gas_m3 = dwelling.heating_system.get_daily_fuel_consumption() / 60.0

        # VBA line 1118: aHeatingControls(intRunNumber).GetSpaceThermostatSetpoint
        if not hasattr(dwelling, 'heating_controls'):
            raise AttributeError(
                f"CRITICAL: Dwelling {dwelling_idx + 1} has no heating_controls for daily summary. "
                "All dwellings must have heating controls initialized."
            )
        thermostat_setpoint = dwelling.heating_controls.get_space_thermostat_setpoint()

        # VBA line 1119: aSolarThermal(intRunNumber).GetDailySumPhi_s / 60 / 1000
        solar_thermal_kwh = dwelling.solar_thermal.get_daily_sum_phi_s() / 60.0 / 1000.0 if dwelling.solar_thermal else 0.0

        # Write all 17 columns matching VBA DailyTotals (lines 1103-1119)
        row = [
            dwelling_idx + 1,              # Column 1: Dwelling index (1-based)
            date_str,                      # Column 2: Date (YYYY-MM-DD HH:MM:SS format)
            mean_active_occupancy,         # Column 3: Mean active occupancy
            proportion_actively_occupied,  # Column 4: Proportion day actively occupied
            lighting_kwh,                  # Column 5: Lighting demand (kWh)
            appliances_kwh,                # Column 6: Appliance demand (kWh)
            pv_output_kwh,                 # Column 7: PV output (kWh)
            total_electricity_kwh,         # Column 8: Total electricity demand (kWh)
            self_consumption_kwh,          # Column 9: Self consumption (kWh)
            net_electricity_kwh,           # Column 10: Net electricity demand (kWh)
            hot_water_litres,              # Column 11: Hot water demand (litres)
            mean_temp,                     # Column 12: Average indoor temperature (°C)
            thermal_energy_space_kwh,      # Column 13: Thermal energy space heating (kWh)
            thermal_energy_water_kwh,      # Column 14: Thermal energy water heating (kWh)
            gas_m3,                        # Column 15: Gas demand (m³/day)
            thermostat_setpoint,           # Column 16: Space thermostat setpoint (°C)
            solar_thermal_kwh              # Column 17: Solar thermal heat gains (kWh)
        ]

        self._summary_writer.writerow(row)
        self._summary_file.flush()

    def write_global_climate(self, global_climate):
        """
        Write global climate data.

        Parameters
        ----------
        global_climate : GlobalClimate
            Global climate model with simulation results
        """
        if not self.config.save_global_climate or self._climate_writer is None:
            return

        for minute in range(1440):
            row = [
                minute + 1,  # 1-based minute
                global_climate.clearness_index[minute],
                global_climate.g_o_clearsky[minute],
                global_climate.g_o[minute],
                global_climate.theta_o[minute]
            ]
            self._climate_writer.writerow(row)

        self._climate_file.flush()

    def _expand_10min_to_1min(self, data_10min: np.ndarray) -> np.ndarray:
        """
        Expand 10-minute resolution data to 1-minute resolution.

        Parameters
        ----------
        data_10min : np.ndarray
            Data at 10-minute resolution (144 values)

        Returns
        -------
        np.ndarray
            Data at 1-minute resolution (1440 values)
        """
        # Each 10-minute value applies to 10 consecutive minutes
        data_1min = np.repeat(data_10min, 10)
        return data_1min

    def _calculate_at_home(self, combined_states: np.ndarray) -> np.ndarray:
        """
        Calculate number of people at home from combined occupancy states.

        Parameters
        ----------
        combined_states : np.ndarray
            Array of combined state strings (e.g., "11", "10", "00")

        Returns
        -------
        np.ndarray
            Number of people at home
        """
        at_home = np.zeros(len(combined_states))
        for i, state in enumerate(combined_states):
            if state and len(str(state)) >= 1:
                at_home[i] = int(str(state)[0])
        return at_home

    def close(self):
        """Close all open files."""
        if self._minute_file:
            self._minute_file.close()
        if self._summary_file:
            self._summary_file.close()
        if self._climate_file:
            self._climate_file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
