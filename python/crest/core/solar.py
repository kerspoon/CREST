"""
CREST Demand Model - Solar Geometry Calculations

Shared solar geometry calculations for PV and solar thermal systems.
Extracts and consolidates duplicate code from clsPVSystem.cls and clsSolarThermal.cls.

VBA Source:
- clsPVSystem.cls lines 122-337 (solar geometry in CalculatePVOutput)
- clsSolarThermal.cls lines 307-496 (solar geometry in GetIncidentRadiation)

Note: This module uses the corrected logic from clsPVSystem. The clsSolarThermal
      VBA code has a bug at line 428: Tan(Declination)/Tan(Declination) should
      be Tan(Declination)/Tan(Latitude).
"""

import math
import numpy as np
from typing import Tuple

from crest.simulation.config import (
    PI,
    GROUND_REFLECTANCE,
    DAY_SUMMER_TIME_STARTS,
    DAY_SUMMER_TIME_END
)


class SolarGeometry:
    """
    Solar geometry calculations for photovoltaic and solar thermal systems.

    Calculates solar position (altitude, azimuth) and incident radiation
    (direct, diffuse, reflected) on a tilted surface.

    All calculations use 1-minute resolution (1440 timesteps per day).
    Implements daylight saving time adjustments for UK (days 87-304).
    """

    def __init__(self,
                 day_of_year: int,
                 latitude: float,
                 longitude: float,
                 meridian: float,
                 enable_daylight_saving: bool = True):
        """
        Initialize solar geometry calculator for a specific day and location.

        Args:
            day_of_year: Day of year (1-365)
            latitude: Latitude in degrees (positive North)
            longitude: Longitude in degrees (positive East)
            meridian: Standard meridian for time zone in degrees
            enable_daylight_saving: Enable UK daylight saving time (days 87-304)

        VBA Source: clsPVSystem.cls lines 194-212, 221-231
        """
        self.day_of_year = day_of_year
        self.latitude = latitude
        self.longitude = longitude
        self.meridian = meridian
        self.enable_daylight_saving = enable_daylight_saving

        # Calculate B (VBA line 222)
        self.B = 360 * (day_of_year - 81) / 364

        # Calculate equation of time (VBA line 225)
        self.equation_of_time = (
            (9.87 * math.sin(2 * self.B * PI / 180)) -
            (7.53 * math.cos(self.B * PI / 180)) -
            (1.5 * math.sin(self.B * PI / 180))
        )

        # Calculate time correction factor (VBA line 228)
        self.time_correction_factor = (
            (4 * (longitude - meridian)) + self.equation_of_time
        )

        # Calculate sky diffuse factor (VBA line 231)
        self.sky_diffuse_factor = 0.095 + (
            0.04 * math.sin(2 * PI * (day_of_year - 100) / 365)
        )

        # Calculate optical depth (VBA line 257)
        self.optical_depth = 0.174 + (
            0.035 * math.sin(2 * PI * (day_of_year - 100) / 365)
        )

    def _apply_daylight_saving(self, timestep: int) -> Tuple[int, int]:
        """
        Convert timestep to hour and minute with daylight saving adjustment.

        Args:
            timestep: Timestep (1-1440) - VBA 1-based indexing

        Returns:
            Tuple of (local_standard_time_hour, local_standard_time_minute)

        VBA Source: clsPVSystem.cls lines 237-251
        """
        # Convert timestep to hour and minute (0-based internally)
        timestep_0based = timestep - 1
        hour = timestep_0based // 60
        minute = (timestep_0based % 60) + 1  # VBA uses 1-60 for minutes

        # Apply daylight saving time adjustment (VBA lines 240-245)
        if (self.enable_daylight_saving and
            self.day_of_year >= DAY_SUMMER_TIME_STARTS and
            self.day_of_year < DAY_SUMMER_TIME_END):
            local_standard_time_hour = hour - 1
        else:
            local_standard_time_hour = hour

        local_standard_time_minute = minute

        return local_standard_time_hour, local_standard_time_minute

    def calculate_solar_position(self, timestep: int) -> dict:
        """
        Calculate solar position (altitude, azimuth, hour angle, declination).

        Args:
            timestep: Timestep (1-1440) - VBA 1-based indexing

        Returns:
            Dictionary with:
                - altitude: Solar altitude angle (degrees)
                - azimuth: Solar azimuth angle (degrees, adjusted)
                - hour_angle: Hour angle (degrees)
                - declination: Solar declination (degrees)
                - hours_before_solar_noon: Hours before solar noon

        VBA Source: clsPVSystem.cls lines 253-286
        """
        # Get local standard time with daylight saving adjustment
        hour, minute = self._apply_daylight_saving(timestep)

        # Calculate hours before solar noon (VBA line 254)
        hours_before_solar_noon = 12 - (
            hour + (minute / 60) + (self.time_correction_factor / 60)
        )

        # Calculate hour angle (VBA line 260)
        hour_angle = 15 * hours_before_solar_noon

        # Calculate declination (VBA line 263)
        declination = 23.45 * math.sin(2 * PI * (284 + self.day_of_year) / 365.25)

        # Calculate solar altitude angle (VBA line 266)
        altitude_rad = math.asin(
            (math.cos(self.latitude * PI / 180) *
             math.cos(declination * PI / 180) *
             math.cos(hour_angle * PI / 180)) +
            (math.sin(self.latitude * PI / 180) *
             math.sin(declination * PI / 180))
        )
        solar_altitude_angle = altitude_rad * 180 / PI

        # Calculate azimuth of sun (VBA line 270)
        # Handle edge case when cos(altitude) = 0 (sun at zenith)
        if abs(math.cos(solar_altitude_angle * PI / 180)) < 1e-10:
            azimuth_of_sun = 0.0
        else:
            azimuth_of_sun = (180 / PI) * math.asin(
                math.cos(declination * PI / 180) *
                math.sin(hour_angle * PI / 180) /
                math.cos(solar_altitude_angle * PI / 180)
            )

        # Check whether the acute angle is the correct azimuth (VBA lines 273-285)
        # This is the CORRECTED logic from clsPVSystem (not the buggy clsSolarThermal version)
        if math.cos(hour_angle * PI / 180) >= (
            math.tan(declination * PI / 180) / math.tan(self.latitude * PI / 180)
        ):
            # Azimuth is acute. Output of Asin should be an acute angle anyway.
            adjusted_azimuth_of_sun = azimuth_of_sun
        else:
            # Azimuth is obtuse.
            if azimuth_of_sun > 0:
                adjusted_azimuth_of_sun = 180 - azimuth_of_sun
            elif azimuth_of_sun < 0:
                adjusted_azimuth_of_sun = -180 - azimuth_of_sun
            else:
                adjusted_azimuth_of_sun = azimuth_of_sun

        return {
            'altitude': solar_altitude_angle,
            'azimuth': adjusted_azimuth_of_sun,
            'hour_angle': hour_angle,
            'declination': declination,
            'hours_before_solar_noon': hours_before_solar_noon
        }

    def calculate_incident_radiation(self,
                                     timestep: int,
                                     slope: float,
                                     azimuth: float,
                                     G_o_clearsky: float,
                                     clearness_index: float) -> dict:
        """
        Calculate incident radiation on a tilted surface.

        Calculates direct beam, diffuse, and reflected radiation components
        on a surface with specified tilt and azimuth.

        Args:
            timestep: Timestep (1-1440) - VBA 1-based indexing
            slope: Panel tilt angle from horizontal (degrees, 0-90)
            azimuth: Panel azimuth angle (degrees, 0=South, +East, -West)
            G_o_clearsky: Clear sky beam radiation on horizontal surface (W/m²)
            clearness_index: Clearness index from Markov chain (0-1)

        Returns:
            Dictionary with:
                - G_incident: Total incident radiation on panel (W/m²)
                - G_direct: Direct beam radiation (W/m²)
                - G_diffuse: Diffuse radiation (W/m²)
                - G_reflected: Ground-reflected radiation (W/m²)
                - incident_angle: Solar incident angle on panel (degrees)

        VBA Source: clsPVSystem.cls lines 288-325
        """
        # Get solar position
        solar_pos = self.calculate_solar_position(timestep)
        solar_altitude = solar_pos['altitude']
        solar_azimuth = solar_pos['azimuth']

        # Calculate solar incident angle on panel (VBA lines 288-293)
        # Dot product of sun direction and panel normal
        dot_product = (
            (math.cos(solar_altitude * PI / 180) *
             math.cos(solar_azimuth * PI / 180 - azimuth * PI / 180) *
             math.sin(slope * PI / 180)) +
            (math.sin(solar_altitude * PI / 180) *
             math.cos(slope * PI / 180))
        )

        # Handle numerical errors that might put dot_product outside [-1, 1]
        dot_product = max(-1.0, min(1.0, dot_product))

        solar_incident_angle = math.acos(dot_product) * 180 / PI

        # Calculate direct beam radiation on panel (VBA lines 300-304)
        if abs(solar_incident_angle) > 90:
            # Sun is behind the panel
            G_direct = 0.0
        else:
            G_direct = G_o_clearsky * math.cos(solar_incident_angle * PI / 180)

        # Calculate diffuse radiation on panel (VBA line 308)
        # Sky diffuse radiation assuming isotropic sky
        G_diffuse = (
            self.sky_diffuse_factor * G_o_clearsky *
            ((1 + math.cos(slope * PI / 180)) / 2)
        )

        # Calculate reflected radiation on panel (VBA line 312)
        # Ground-reflected radiation
        G_reflected = (
            GROUND_REFLECTANCE * G_o_clearsky *
            (math.sin(solar_altitude * PI / 180) + self.sky_diffuse_factor) *
            ((1 - math.cos(slope * PI / 180)) / 2)
        )

        # Total clearsky incident radiation (VBA line 316)
        G_i_clearsky = G_direct + G_diffuse + G_reflected

        # Apply clearness index to get actual incident radiation (VBA line 324)
        G_incident = G_i_clearsky * clearness_index

        return {
            'G_incident': G_incident,
            'G_direct': G_direct,
            'G_diffuse': G_diffuse,
            'G_reflected': G_reflected,
            'incident_angle': solar_incident_angle
        }

    def calculate_all_day_radiation(self,
                                    slope: float,
                                    azimuth: float,
                                    G_o_clearsky_array: np.ndarray,
                                    clearness_index_array: np.ndarray) -> dict:
        """
        Calculate incident radiation for all 1440 timesteps of the day.

        Convenience method for batch processing (used by PVSystem).

        Args:
            slope: Panel tilt angle (degrees)
            azimuth: Panel azimuth angle (degrees)
            G_o_clearsky_array: Array of clear sky radiation (1440 elements, W/m²)
            clearness_index_array: Array of clearness indices (1440 elements)

        Returns:
            Dictionary with arrays (1440 elements each):
                - G_incident: Total incident radiation (W/m²)
                - G_direct: Direct beam radiation (W/m²)
                - G_diffuse: Diffuse radiation (W/m²)
                - G_reflected: Ground-reflected radiation (W/m²)

        Note: Uses 0-based indexing for arrays, but passes 1-based timesteps
              to calculate_incident_radiation to match VBA behavior.
        """
        G_incident = np.zeros(1440)
        G_direct = np.zeros(1440)
        G_diffuse = np.zeros(1440)
        G_reflected = np.zeros(1440)

        for timestep_0based in range(1440):
            timestep_1based = timestep_0based + 1

            result = self.calculate_incident_radiation(
                timestep=timestep_1based,
                slope=slope,
                azimuth=azimuth,
                G_o_clearsky=G_o_clearsky_array[timestep_0based],
                clearness_index=clearness_index_array[timestep_0based]
            )

            G_incident[timestep_0based] = result['G_incident']
            G_direct[timestep_0based] = result['G_direct']
            G_diffuse[timestep_0based] = result['G_diffuse']
            G_reflected[timestep_0based] = result['G_reflected']

        return {
            'G_incident': G_incident,
            'G_direct': G_direct,
            'G_diffuse': G_diffuse,
            'G_reflected': G_reflected
        }
