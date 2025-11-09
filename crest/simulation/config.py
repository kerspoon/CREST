"""
CREST Demand Model - Configuration and Constants

Physical constants, simulation parameters, and configuration values used throughout the model.
"""

import math

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
