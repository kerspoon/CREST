"""
Visualization module for CREST simulation results.

Generates summary plots for simulation outputs including demand profiles,
occupancy patterns, temperatures, and PV generation.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


class ResultsPlotter:
    """
    Generates visualizations for CREST simulation results.

    Creates plots in {output_dir}/plots/ directory.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize plotter.

        Parameters
        ----------
        output_dir : Path
            Base output directory (plots saved to output_dir/plots/)
        """
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)

        # Time axis for plots (minutes 0-1439 as datetime)
        base_date = datetime(2024, 1, 1)
        self.time_axis = [base_date + timedelta(minutes=m) for m in range(1440)]

        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.figsize = (12, 6)
        self.dpi = 150

    def plot_all(self, dwellings: list, global_climate) -> list:
        """
        Generate all plots for simulation results.

        Parameters
        ----------
        dwellings : list
            List of Dwelling objects after simulation
        global_climate : GlobalClimate
            Global climate object after simulation

        Returns
        -------
        list
            List of paths to generated plot files
        """
        plot_files = []

        # Climate plot (shared across dwellings)
        plot_files.append(self.plot_climate(global_climate))

        # Per-dwelling plots
        for dwelling in dwellings:
            idx = dwelling.config.dwelling_index
            plot_files.append(self.plot_demand_profile(dwelling, idx))
            plot_files.append(self.plot_occupancy(dwelling, idx))
            plot_files.append(self.plot_temperatures(dwelling, global_climate, idx))

            if dwelling.pv_system:
                plot_files.append(self.plot_pv(dwelling, idx))

        return [p for p in plot_files if p is not None]

    def plot_demand_profile(self, dwelling, dwelling_idx: int) -> Path:
        """
        Plot stacked demand profile showing all electricity components.

        Parameters
        ----------
        dwelling : Dwelling
            Dwelling object after simulation
        dwelling_idx : int
            Dwelling index for filename

        Returns
        -------
        Path
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Extract demand arrays (W)
        lighting = np.array([dwelling.lighting.get_total_demand(t) for t in range(1, 1441)])
        appliances = np.array([dwelling.appliances.get_total_demand(t) for t in range(1, 1441)])
        heating_elec = np.array([dwelling.heating_system.get_heating_system_power_demand(t) for t in range(1, 1441)])

        cooling_elec = np.zeros(1440)
        if dwelling.cooling_system:
            cooling_elec = np.array([dwelling.cooling_system.get_cooling_system_power_demand(t) for t in range(1, 1441)])

        # Stacked area plot
        ax.fill_between(self.time_axis, 0, lighting,
                        label='Lighting', color='#FFD700', alpha=0.8)
        ax.fill_between(self.time_axis, lighting, lighting + appliances,
                        label='Appliances', color='#4169E1', alpha=0.8)
        ax.fill_between(self.time_axis, lighting + appliances,
                        lighting + appliances + heating_elec,
                        label='Heating (elec)', color='#DC143C', alpha=0.8)
        if cooling_elec.sum() > 0:
            ax.fill_between(self.time_axis, lighting + appliances + heating_elec,
                            lighting + appliances + heating_elec + cooling_elec,
                            label='Cooling', color='#00CED1', alpha=0.8)

        # Formatting
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Power Demand (W)')
        ax.set_title(f'Dwelling {dwelling_idx} - Electricity Demand Profile')
        ax.legend(loc='upper left')
        ax.set_xlim(self.time_axis[0], self.time_axis[-1])
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

        # Add daily total annotation
        total_kwh = (lighting.sum() + appliances.sum() + heating_elec.sum() + cooling_elec.sum()) / 60 / 1000
        ax.annotate(f'Daily total: {total_kwh:.2f} kWh',
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filepath = self.plot_dir / f'dwelling_{dwelling_idx}_demand.png'
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

        return filepath

    def plot_occupancy(self, dwelling, dwelling_idx: int) -> Path:
        """
        Plot occupancy pattern over the day.

        Parameters
        ----------
        dwelling : Dwelling
            Dwelling object after simulation
        dwelling_idx : int
            Dwelling index for filename

        Returns
        -------
        Path
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get active occupancy array (1-minute resolution)
        active_occ = dwelling.occupancy.get_active_occupancy_1min()

        # Step plot for occupancy
        ax.fill_between(self.time_axis, 0, active_occ,
                        step='mid', color='#2E8B57', alpha=0.7, label='Active occupants')
        ax.step(self.time_axis, active_occ, where='mid', color='#1E5631', linewidth=1.5)

        # Formatting
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Number of Active Occupants')
        ax.set_title(f'Dwelling {dwelling_idx} - Occupancy Pattern ({dwelling.config.num_residents} residents)')
        ax.set_xlim(self.time_axis[0], self.time_axis[-1])
        ax.set_ylim(bottom=0, top=max(dwelling.config.num_residents + 0.5, active_occ.max() + 0.5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add mean annotation
        mean_occ = dwelling.occupancy.get_mean_active_occupancy()
        prop_occupied = dwelling.occupancy.get_proportion_actively_occupied()
        ax.annotate(f'Mean active: {mean_occ:.2f}\nProportion occupied: {prop_occupied:.1%}',
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filepath = self.plot_dir / f'dwelling_{dwelling_idx}_occupancy.png'
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

        return filepath

    def plot_temperatures(self, dwelling, global_climate, dwelling_idx: int) -> Path:
        """
        Plot indoor and outdoor temperatures with heating setpoint.

        Parameters
        ----------
        dwelling : Dwelling
            Dwelling object after simulation
        global_climate : GlobalClimate
            Global climate object
        dwelling_idx : int
            Dwelling index for filename

        Returns
        -------
        Path
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get temperature arrays (direct array access, 0-indexed)
        indoor_temp = dwelling.building.theta_i
        outdoor_temp = global_climate.theta_o
        emitter_temp = dwelling.building.theta_em
        setpoint = dwelling.heating_controls.get_space_thermostat_setpoint()

        # Plot temperatures
        ax.plot(self.time_axis, outdoor_temp, color='#4169E1', linewidth=1.5,
                label='Outdoor', alpha=0.8)
        ax.plot(self.time_axis, indoor_temp, color='#DC143C', linewidth=2,
                label='Indoor')
        ax.plot(self.time_axis, emitter_temp, color='#FF8C00', linewidth=1,
                label='Emitter', alpha=0.7)

        # Setpoint as dashed line
        if setpoint > -50:  # Valid setpoint
            ax.axhline(y=setpoint, color='#DC143C', linestyle='--', linewidth=1.5,
                       label=f'Setpoint ({setpoint:.0f}°C)', alpha=0.7)

        # Formatting
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Dwelling {dwelling_idx} - Temperature Profiles')
        ax.legend(loc='upper right')
        ax.set_xlim(self.time_axis[0], self.time_axis[-1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

        # Add mean indoor temp annotation
        mean_indoor = dwelling.building.get_mean_theta_i()
        ax.annotate(f'Mean indoor: {mean_indoor:.1f}°C',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filepath = self.plot_dir / f'dwelling_{dwelling_idx}_temperature.png'
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

        return filepath

    def plot_pv(self, dwelling, dwelling_idx: int) -> Path:
        """
        Plot PV generation vs demand with self-consumption.

        Parameters
        ----------
        dwelling : Dwelling
            Dwelling object after simulation (must have PV system)
        dwelling_idx : int
            Dwelling index for filename

        Returns
        -------
        Path
            Path to saved plot file
        """
        if not dwelling.pv_system:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        # Get arrays (W)
        pv_output = np.array([dwelling.pv_system.get_pv_output(t) for t in range(1, 1441)])
        total_demand = np.array([
            dwelling.appliances.get_total_demand(t) + dwelling.lighting.get_total_demand(t)
            for t in range(1, 1441)
        ])
        self_consumption = np.array([dwelling.pv_system.get_self_consumption(t) for t in range(1, 1441)])

        # Plot demand as grey area
        ax.fill_between(self.time_axis, 0, total_demand,
                        color='#808080', alpha=0.3, label='Demand')

        # Plot PV generation
        ax.fill_between(self.time_axis, 0, pv_output,
                        color='#32CD32', alpha=0.5, label='PV generation')

        # Highlight self-consumption
        ax.fill_between(self.time_axis, 0, self_consumption,
                        color='#006400', alpha=0.7, label='Self-consumption')

        # Formatting
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Power (W)')
        ax.set_title(f'Dwelling {dwelling_idx} - PV Generation and Self-Consumption')
        ax.legend(loc='upper left')
        ax.set_xlim(self.time_axis[0], self.time_axis[-1])
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

        # Add summary annotation
        pv_kwh = pv_output.sum() / 60 / 1000
        self_kwh = self_consumption.sum() / 60 / 1000
        self_pct = (self_kwh / pv_kwh * 100) if pv_kwh > 0 else 0
        ax.annotate(f'PV: {pv_kwh:.2f} kWh\nSelf-consumed: {self_kwh:.2f} kWh ({self_pct:.0f}%)',
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filepath = self.plot_dir / f'dwelling_{dwelling_idx}_pv.png'
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

        return filepath

    def plot_climate(self, global_climate) -> Path:
        """
        Plot outdoor temperature and solar irradiance.

        Parameters
        ----------
        global_climate : GlobalClimate
            Global climate object after simulation

        Returns
        -------
        Path
            Path to saved plot file
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)

        # Get climate arrays (direct array access, 0-indexed)
        temperature = global_climate.theta_o
        irradiance = global_climate.g_o  # W/m²

        # Temperature on left axis
        color_temp = '#DC143C'
        ax1.set_xlabel('Time of Day')
        ax1.set_ylabel('Temperature (°C)', color=color_temp)
        ax1.plot(self.time_axis, temperature, color=color_temp, linewidth=2, label='Temperature')
        ax1.tick_params(axis='y', labelcolor=color_temp)
        ax1.set_xlim(self.time_axis[0], self.time_axis[-1])

        # Irradiance on right axis
        ax2 = ax1.twinx()
        color_irr = '#FFD700'
        ax2.set_ylabel('Solar Irradiance (W/m²)', color=color_irr)
        ax2.fill_between(self.time_axis, 0, irradiance, color=color_irr, alpha=0.5)
        ax2.plot(self.time_axis, irradiance, color='#FFA500', linewidth=1)
        ax2.tick_params(axis='y', labelcolor=color_irr)
        ax2.set_ylim(bottom=0)

        # Formatting
        ax1.set_title('Climate Conditions')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))

        # Add summary annotation
        mean_temp = temperature.mean()
        max_irr = irradiance.max()
        daily_irr = irradiance.sum() / 60 / 1000  # kWh/m²
        ax1.annotate(f'Mean temp: {mean_temp:.1f}°C\nPeak irradiance: {max_irr:.0f} W/m²\nDaily irradiation: {daily_irr:.2f} kWh/m²',
                     xy=(0.02, 0.98), xycoords='axes fraction',
                     ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filepath = self.plot_dir / 'climate.png'
        fig.savefig(filepath, dpi=self.dpi)
        plt.close(fig)

        return filepath
