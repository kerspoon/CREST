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
        """Initialize minute-level data file with headers."""
        filename = self.output_dir / "results_minute_level.csv"
        self._minute_file = open(filename, 'w', newline='')
        self._minute_writer = csv.writer(self._minute_file)

        # Write header matching Excel model columns
        header = [
            'Dwelling',
            'Minute',
            'At_Home',
            'Active',
            'Lighting_W',
            'Appliances_W',
            'Total_Electricity_W',
            'Outdoor_Temp_C',
            'Irradiance_Wm2',
            'Internal_Temp_C',
            'External_Building_Temp_C',
            'Hot_Water_Demand_L_per_min',
            'Cylinder_Temp_C',
            'Emitter_Temp_C',
            'Cooling_Emitter_Temp_C',
            'Total_Heat_Output_W',
            'Space_Heating_W',
            'Water_Heating_W',
            'Gas_Consumption_m3_per_min',
            'Passive_Solar_Gains_W',
            'Casual_Gains_W',
            'Heating_Electricity_W',
            'PV_Output_W',
            'Cooling_Electricity_W'
        ]
        self._minute_writer.writerow(header)

    def _init_summary_file(self):
        """Initialize daily summary file with headers.

        VBA Reference: DailyTotals (mdlThermalElectricalModel.bas lines 1057-1121)
        Outputs all 17 columns matching VBA daily totals sheet.
        """
        filename = self.output_dir / "results_daily_summary.csv"
        self._summary_file = open(filename, 'w', newline='')
        self._summary_writer = csv.writer(self._summary_file)

        # Write header - 17 columns matching VBA DailyTotals output (lines 1103-1119)
        header = [
            'Dwelling',                           # Column 1: VBA line 1103
            'Date',                               # Column 2: VBA line 1104
            'Mean_Active_Occupancy',              # Column 3: VBA line 1105 - GetMeanActiveOccupancy
            'Proportion_Day_Actively_Occupied',   # Column 4: VBA line 1106 - GetPrActivelyOccupied
            'Lighting_Demand_kWh',                # Column 5: VBA line 1107 - GetDailySumLighting / 60 / 1000
            'Appliance_Demand_kWh',               # Column 6: VBA line 1108 - GetDailySumApplianceDemand / 60 / 1000
            'PV_Output_kWh',                      # Column 7: VBA line 1109 - GetDailySumPvOutput / 60 / 1000
            'Total_Electricity_Demand_kWh',       # Column 8: VBA line 1110 - lighting + appliances
            'Self_Consumption_kWh',               # Column 9: VBA line 1111 - GetDailySumP_self / 60 / 1000
            'Net_Electricity_Demand_kWh',         # Column 10: VBA line 1112 - GetDailySumP_net / 60 / 1000
            'Hot_Water_Demand_L',                 # Column 11: VBA line 1113 - GetDailySumHotWaterDemand
            'Average_Indoor_Temperature_C',       # Column 12: VBA line 1114 - GetMeanTheta_i
            'Thermal_Energy_Space_Heating_kWh',   # Column 13: VBA line 1115 - GetDailySumThermalEnergySpace / 60 / 1000
            'Thermal_Energy_Water_Heating_kWh',   # Column 14: VBA line 1116 - GetDailySumThermalEnergyWater / 60 / 1000
            'Gas_Demand_m3',                      # Column 15: VBA line 1117 - GetDailySumFuelFlow / 60
            'Space_Thermostat_Setpoint_C',        # Column 16: VBA line 1118 - GetSpaceThermostatSetpoint
            'Solar_Thermal_Heat_Gains_kWh'        # Column 17: VBA line 1119 - GetDailySumPhi_s / 60 / 1000
        ]
        self._summary_writer.writerow(header)

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

    def write_minute_data(self, dwelling_idx: int, dwelling):
        """
        Write minute-level data for one dwelling.

        Parameters
        ----------
        dwelling_idx : int
            Dwelling index (0-based)
        dwelling : Dwelling
            Dwelling object with simulation results
        """
        if not self.config.save_minute_data or self._minute_writer is None:
            return

        # Get occupancy data (10-minute resolution, expand to 1-minute)
        occupancy_1min = self._expand_10min_to_1min(dwelling.occupancy.active_occupancy)
        at_home_1min = self._expand_10min_to_1min(
            self._calculate_at_home(dwelling.occupancy.combined_states)
        )

        # Write data for each minute
        for minute in range(1, 1441):
            idx = minute - 1  # 0-based index for arrays

            # Collect all variables
            # Note: Some methods use 1-based timestep, others use 0-based indexing
            row = [
                dwelling_idx + 1,  # 1-based dwelling index
                minute,  # 1-based minute of day
                at_home_1min[idx],
                occupancy_1min[idx],
                dwelling.lighting.get_total_demand(minute),  # Uses 1-based
                dwelling.appliances.get_total_demand(minute),  # Uses 1-based
                dwelling.get_total_electricity_demand(minute),  # Uses 1-based
                dwelling.local_climate.get_temperature(idx),  # Uses 0-based
                dwelling.local_climate.get_irradiance(idx),  # Uses 0-based
                dwelling.building.theta_i[idx],  # Direct array access, 0-based
                dwelling.building.theta_b[idx],
                dwelling.hot_water.hot_water_demand[idx],
                dwelling.building.theta_cyl[idx],
                dwelling.building.theta_em[idx],
                dwelling.building.theta_cool[idx],
                dwelling.heating_system.phi_h_output[idx],
                dwelling.heating_system.phi_h_space[idx],
                dwelling.heating_system.phi_h_water[idx],
                dwelling.heating_system.m_fuel[idx],
                dwelling.building.phi_s[idx],
                dwelling.building.phi_c[idx],
                dwelling.heating_system.get_heating_system_power_demand(minute),  # Uses 1-based
                dwelling.pv_system.get_pv_output(minute) if dwelling.pv_system else 0.0,  # Uses 1-based
                dwelling.cooling_system.get_cooling_system_power_demand(minute) if dwelling.cooling_system else 0.0  # Uses 1-based
            ]

            self._minute_writer.writerow(row)

        # Flush to disk periodically
        self._minute_file.flush()

    def write_daily_summary(self, dwelling_idx: int, dwelling):
        """
        Write daily summary for one dwelling - all 17 VBA-matching columns.

        VBA Reference: DailyTotals (mdlThermalElectricalModel.bas lines 1057-1121)

        Parameters
        ----------
        dwelling_idx : int
            Dwelling index (0-based)
        dwelling : Dwelling
            Dwelling object with simulation results
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
        self_consumption_kwh = dwelling.pv_system.get_daily_sum_p_self() / 60.0 / 1000.0 if dwelling.pv_system else 0.0

        # VBA line 1088: dblNetElectricityDemand = aPVSystem(intRunNumber).GetDailySumP_net / 60 / 1000
        net_electricity_kwh = dwelling.pv_system.get_daily_sum_p_net() / 60.0 / 1000.0 if dwelling.pv_system else total_electricity_kwh

        # VBA line 1092: dblHotWaterDemand = aHotWater(intRunNumber).GetDailySumHotWaterDemand
        hot_water_litres = dwelling.hot_water.get_daily_hot_water_volume()

        # VBA line 1094: dblAverageIndoorTemperature = aBuilding(intRunNumber).GetMeanTheta_i
        mean_temp = np.mean(dwelling.building.theta_i)

        # VBA line 1096: dblThermalEnergySpace = aPrimaryHeatingSystem(intRunNumber).GetDailySumThermalEnergySpace / 60 / 1000
        thermal_energy_space_kwh = dwelling.heating_system.get_daily_sum_thermal_energy_space() / 60.0 / 1000.0

        # VBA line 1097: dblThermalEnergyWater = aPrimaryHeatingSystem(intRunNumber).GetDailySumThermalEnergyWater / 60 / 1000
        thermal_energy_water_kwh = dwelling.heating_system.get_daily_sum_thermal_energy_water() / 60.0 / 1000.0

        # VBA line 1099: dblGasDemand = aPrimaryHeatingSystem(intRunNumber).GetDailySumFuelFlow / 60
        gas_m3 = dwelling.heating_system.get_daily_fuel_consumption() / 60.0

        # VBA line 1118: aHeatingControls(intRunNumber).GetSpaceThermostatSetpoint
        thermostat_setpoint = dwelling.heating_controls.get_space_thermostat_setpoint() if hasattr(dwelling, 'heating_controls') else 20.0

        # VBA line 1119: aSolarThermal(intRunNumber).GetDailySumPhi_s / 60 / 1000
        solar_thermal_kwh = dwelling.solar_thermal.get_daily_sum_phi_s() / 60.0 / 1000.0 if dwelling.solar_thermal else 0.0

        # Write all 17 columns matching VBA DailyTotals (lines 1103-1119)
        row = [
            dwelling_idx + 1,              # Column 1: Dwelling index (1-based)
            "2015-01-01",                  # Column 2: Date (placeholder - could be passed as parameter)
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
