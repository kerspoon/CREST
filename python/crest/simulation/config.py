"""
CREST Demand Model - Configuration and Constants

Physical constants, simulation parameters, and configuration values used throughout the model.
"""

import math
from enum import Enum

# ===============================================================================================
# LOCATION AND REGION ENUMS
# ===============================================================================================

class Country(Enum):
    """Country selection for appliance ownership and water temperature."""
    UK = "UK"
    INDIA = "India"

class City(Enum):
    """City/region selection for climate data."""
    ENGLAND = "England"
    N_DELHI = "N Delhi"
    MUMBAI = "Mumbai"
    BENGALURU = "Bengaluru"
    CHENNAI = "Chennai"
    KOLKATA = "Kolkata"
    ITANAGAR = "Itanagar"

class UrbanRural(Enum):
    """Urban or rural location for appliance ownership."""
    URBAN = "Urban"
    RURAL = "Rural"

# ===============================================================================================
# COUNTRY-SPECIFIC CONSTANTS
# ===============================================================================================

# Cold water inlet temperature by country (°C)
# VBA: clsHotWater.cls lines 156-162
COLD_WATER_TEMPERATURE_BY_COUNTRY = {
    Country.UK: 10.0,
    Country.INDIA: 20.0
}

# ===============================================================================================
# PHYSICAL CONSTANTS
# ===============================================================================================

# Mathematical constant pi
PI = math.pi

# Specific heat capacity of water (J/kg/K)
SPECIFIC_HEAT_CAPACITY_WATER = 4200.0

# Ground reflectance for solar calculations
GROUND_REFLECTANCE = 0.2

# ===============================================================================================
# DAYLIGHT SAVING TIME
# ===============================================================================================

# Day of year when British Summer Time starts (typically late March)
DAY_SUMMER_TIME_STARTS = 87

# Day of year when British Summer Time ends (typically late October)
DAY_SUMMER_TIME_END = 304

# ===============================================================================================
# OCCUPANCY MODEL CONSTANTS
# ===============================================================================================

# Thermal gain per active occupant (W)
OCCUPANT_THERMAL_GAIN_ACTIVE = 147

# Thermal gain per dormant/sleeping occupant (W)
OCCUPANT_THERMAL_GAIN_DORMANT = 84

# Number of 10-minute time steps in a day
TIMESTEPS_PER_DAY_10MIN = 144

# ===============================================================================================
# SIMULATION PARAMETERS
# ===============================================================================================

# Number of 1-minute time steps in a day
TIMESTEPS_PER_DAY_1MIN = 1440

# Simulation timestep in seconds (for thermal calculations)
THERMAL_TIMESTEP_SECONDS = 60

# Number of minutes per hour
MINUTES_PER_HOUR = 60

# ===============================================================================================
# BUILDING THERMAL MODEL PARAMETERS
# ===============================================================================================

# Default cold water temperature (°C) - UK standard
COLD_WATER_TEMPERATURE = 10.0

# ===============================================================================================
# HEATING CONTROL DEADBANDS
# ===============================================================================================

# Space heating thermostat deadband (±°C)
THERMOSTAT_DEADBAND_SPACE = 2.0

# Hot water cylinder thermostat deadband (±°C)
THERMOSTAT_DEADBAND_WATER = 5.0

# Emitter/radiator thermostat deadband (±°C)
THERMOSTAT_DEADBAND_EMITTER = 5.0

# Timer random time shift (±minutes) for diversity
TIMER_RANDOM_SHIFT_MINUTES = 15

# ===============================================================================================
# HEATING SYSTEM PARAMETERS
# ===============================================================================================

# Default boiler thermal efficiency
BOILER_THERMAL_EFFICIENCY = 0.75

# ===============================================================================================
# APPLIANCE AND LIGHTING PARAMETERS
# ===============================================================================================

# Maximum number of bulbs per dwelling
MAX_BULBS_PER_DWELLING = 60

# Maximum number of appliance types
MAX_APPLIANCE_TYPES = 31

# Number of water fixture types
WATER_FIXTURE_TYPES = 4

# ===============================================================================================
# VALIDATION AND LIMITS
# ===============================================================================================

# VBA Integer limit (for compatibility checking)
VBA_INTEGER_MAX = 32767
