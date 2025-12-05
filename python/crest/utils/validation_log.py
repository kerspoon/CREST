"""
Validation logging for RNG debugging.

Logs selection results, bulb configurations, appliance ownership,
and switch-on decisions for comparison with VBA output.
"""

from pathlib import Path
from typing import Optional, TextIO
import numpy as np


class ValidationLogger:
    """
    Logger for validation data to compare Python vs VBA outputs.

    Output format is tab-separated for easy diff comparison.
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        verbose: bool = False
    ):
        """
        Initialize the validation logger.

        Parameters
        ----------
        log_file : Path, optional
            Path to log file. If None, logging is disabled.
        verbose : bool
            If True, log switch-on decisions (much larger output)
        """
        self.log_file = log_file
        self.verbose = verbose
        self._file_handle: Optional[TextIO] = None

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(log_file, 'w')

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def _log(self, line: str) -> None:
        """Write a line to the log file."""
        if self._file_handle is not None:
            self._file_handle.write(line + '\n')
            self._file_handle.flush()

    def log_selection(
        self,
        context: str,
        rand_val: float,
        cumulative: np.ndarray,
        selected_idx: int,
        value: int
    ) -> None:
        """
        Log a selection from distribution.

        Parameters
        ----------
        context : str
            What is being selected (e.g., "residents", "building")
        rand_val : float
            Random value used for selection
        cumulative : np.ndarray
            Cumulative probability array
        selected_idx : int
            Selected 0-based index
        value : int
            Actual value (e.g., number of residents, 1-based building index)
        """
        if self._file_handle is None:
            return

        # Format cumulative as compact list
        cumul_str = '[' + ','.join(f'{c:.6f}' for c in cumulative) + ']'

        self._log(
            f"SELECTION\t{context}\trand={rand_val:.15f}\t"
            f"cumul={cumul_str}\tidx={selected_idx}\tvalue={value}"
        )

    def log_dwelling_params(
        self,
        dwelling_idx: int,
        residents: int,
        building: int,
        heating: int,
        pv: int,
        solar: int,
        cooling: int
    ) -> None:
        """
        Log final dwelling parameters after all selections.
        """
        if self._file_handle is None:
            return

        self._log(
            f"DWELLING\t{dwelling_idx}\tresidents={residents}\t"
            f"building={building}\theating={heating}\tpv={pv}\t"
            f"solar={solar}\tcooling={cooling}"
        )

    def log_bulb_config(
        self,
        dwelling_idx: int,
        config_idx: int,
        num_bulbs: int,
        powers: list,
        irradiance_threshold: int,
        calibration_scalar: float
    ) -> None:
        """
        Log bulb configuration for a dwelling.

        Parameters
        ----------
        dwelling_idx : int
            Dwelling identifier
        config_idx : int
            Bulb configuration index (1-100)
        num_bulbs : int
            Number of bulbs
        powers : list
            Power rating for each bulb
        irradiance_threshold : int
            Irradiance threshold for low-light condition
        calibration_scalar : float
            Calibration scalar for relative use
        """
        if self._file_handle is None:
            return

        powers_str = '[' + ','.join(f'{int(p)}' for p in powers) + ']'

        self._log(
            f"BULB_CONFIG\t{dwelling_idx}\tconfig={config_idx}\t"
            f"num_bulbs={num_bulbs}\tpowers={powers_str}\t"
            f"irrad_thresh={irradiance_threshold}\tcalib_scalar={calibration_scalar:.15f}"
        )

    def log_bulb_use(
        self,
        dwelling_idx: int,
        bulb_idx: int,
        calibrated_use: float
    ) -> None:
        """
        Log per-bulb calibrated relative use.
        """
        if self._file_handle is None:
            return

        self._log(
            f"BULB_USE\t{dwelling_idx}\tbulb={bulb_idx}\t"
            f"calibrated_use={calibrated_use:.15f}"
        )

    def log_switch_decision(
        self,
        dwelling_idx: int,
        bulb_idx: int,
        minute: int,
        irradiance: int,
        irrad_threshold: int,
        active_occ: int,
        effective_occ: float,
        calibrated_use: float,
        rand_5pct: float,
        rand_switch: float,
        low_irrad: bool,
        switched_on: bool,
        duration: int = 0
    ) -> None:
        """
        Log a switch-on decision (verbose mode only).

        Only logs when verbose=True due to large output size.
        """
        if self._file_handle is None or not self.verbose:
            return

        result = "ON" if switched_on else "OFF"
        dur_str = f"\tduration={duration}" if switched_on else ""

        self._log(
            f"SWITCH\tD{dwelling_idx}\tbulb={bulb_idx}\tmin={minute}\t"
            f"irrad={irradiance}\tthresh={irrad_threshold}\t"
            f"occ={active_occ}\teff_occ={effective_occ:.6f}\t"
            f"use={calibrated_use:.9f}\trand_5pct={rand_5pct:.15f}\t"
            f"rand_sw={rand_switch:.15f}\tlow_irrad={low_irrad}\t"
            f"result={result}{dur_str}"
        )

    def log_switch_summary(
        self,
        dwelling_idx: int,
        total_switch_ons: int,
        total_minutes_on: int
    ) -> None:
        """
        Log summary of switch-on events per dwelling.
        """
        if self._file_handle is None:
            return

        self._log(
            f"SWITCH_SUMMARY\t{dwelling_idx}\t"
            f"switch_ons={total_switch_ons}\tminutes_on={total_minutes_on}"
        )

    def log_appliances(
        self,
        dwelling_idx: int,
        names: list,
        owned: list,
        rand_values: list
    ) -> None:
        """
        Log appliance ownership for a dwelling.

        Parameters
        ----------
        dwelling_idx : int
            Dwelling identifier
        names : list[str]
            Appliance names (31 items)
        owned : list[bool]
            Ownership flags (31 items)
        rand_values : list[float]
            Random values used for each ownership decision
        """
        if self._file_handle is None:
            return

        # Log each appliance on its own line for easier diff
        for i, (name, is_owned, rand_val) in enumerate(zip(names, owned, rand_values)):
            flag = 1 if is_owned else 0
            self._log(
                f"APPLIANCE\t{dwelling_idx}\tidx={i}\t{name}={flag}\t"
                f"rand={rand_val:.15f}"
            )

        # Also log summary
        total_owned = sum(1 for o in owned if o)
        self._log(
            f"APPLIANCES_SUMMARY\t{dwelling_idx}\t"
            f"owned={total_owned}/31"
        )

    def log_active_occupancy(
        self,
        dwelling_idx: int,
        residents: int,
        is_24hr: bool,
        values: np.ndarray,
        combined_states: np.ndarray = None
    ) -> None:
        """
        Log active occupancy array for a dwelling.

        Parameters
        ----------
        dwelling_idx : int
            Dwelling identifier
        residents : int
            Number of residents in dwelling
        is_24hr : bool
            Whether dwelling has 24-hour occupancy
        values : np.ndarray
            Active occupancy values for each 10-min timestep (144 values)
        combined_states : np.ndarray, optional
            Combined state strings for each timestep (e.g., "10", "11")
        """
        if self._file_handle is None:
            return

        # Format values as compact list
        values_str = '[' + ','.join(str(int(v)) for v in values) + ']'

        self._log(
            f"ACTIVE_OCC\t{dwelling_idx}\tresidents={residents}\t"
            f"is_24hr={is_24hr}\tvalues={values_str}"
        )

        # Also log combined states if provided
        if combined_states is not None:
            states_str = '[' + ','.join(str(s) for s in combined_states) + ']'
            self._log(
                f"COMBINED_STATES\t{dwelling_idx}\tstates={states_str}"
            )
