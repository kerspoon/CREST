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
        """Initialize daily summary file with headers."""
        filename = self.output_dir / "results_daily_summary.csv"
        self._summary_file = open(filename, 'w', newline='')
        self._summary_writer = csv.writer(self._summary_file)

        # Write header
        header = [
            'Dwelling',
            'Total_Electricity_kWh',
            'Total_Gas_m3',
            'Total_Hot_Water_L',
            'Peak_Electricity_W',
            'Peak_Heating_W',
            'Mean_Internal_Temp_C',
            'Mean_Occupancy_At_Home',
            'Mean_Occupancy_Active'
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
        Write daily summary for one dwelling.

        Parameters
        ----------
        dwelling_idx : int
            Dwelling index (0-based)
        dwelling : Dwelling
            Dwelling object with simulation results
        """
        if not self.config.save_daily_summary or self._summary_writer is None:
            return

        # Calculate daily totals
        # VBA line 1086: dblTotalElectricityDemand = dblLightingDemand + dblApplianceDemand
        # NOTE: This does NOT include heating/cooling/pump electricity - only lighting + appliances
        lighting_kwh = dwelling.lighting.get_daily_energy() / 60.0 / 1000.0
        appliances_kwh = dwelling.appliances.get_daily_energy() / 60.0 / 1000.0
        total_electricity = (lighting_kwh + appliances_kwh) * 1000.0  # Convert back to Wh for consistency

        total_gas = dwelling.heating_system.get_daily_fuel_consumption() / 60.0  # m³ (sum of rates in m³/h, divide by 60)
        total_hot_water = dwelling.hot_water.get_daily_hot_water_volume()  # litres

        # Calculate peaks
        peak_electricity = max(dwelling.get_total_electricity_demand(t)
                              for t in range(1, 1441))
        peak_heating = max(dwelling.heating_system.phi_h_output[t-1]
                          for t in range(1, 1441))

        # Calculate means
        mean_temp = np.mean(dwelling.building.theta_i)

        # Occupancy (expand from 10-min to 1-min for consistency)
        occupancy_1min = self._expand_10min_to_1min(dwelling.occupancy.active_occupancy)
        at_home_1min = self._expand_10min_to_1min(
            self._calculate_at_home(dwelling.occupancy.combined_states)
        )
        mean_at_home = np.mean(at_home_1min)
        mean_active = np.mean(occupancy_1min)

        row = [
            dwelling_idx + 1,  # 1-based dwelling index
            total_electricity / 1000.0,  # kWh
            total_gas,
            total_hot_water,
            peak_electricity,
            peak_heating,
            mean_temp,
            mean_at_home,
            mean_active
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
