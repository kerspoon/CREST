#!/bin/bash
# CREST Simulation Run Script
# Generated from: excel/lcg_fixed.xlsm
# Generated at: 2025-12-06 10:40:35

# ============================================================
# SIMULATION SETTINGS (extracted from Excel Main Sheet)
# ============================================================

# Date settings
DAY=1
MONTH=1
WEEKDAY="wd"  # 'wd' = weekday, 'we' = weekend

# Location settings
LATITUDE=53.4794892
LONGITUDE=-2.2451148
MERIDIAN=0.0  # Local standard time meridian
CITY="England"
COUNTRY="UK"
YEAR=2019
URBAN_RURAL="Urban"

# Simulation settings
NUM_DWELLINGS=2
SEED=""  # Empty for random seed

# Checkbox settings (from Excel form controls)
ASSIGN_DWELLING_PARAMS=true  # Stochastically assign dwelling parameters
SAVE_DETAILED=true  # Include high-resolution dynamic output
SAVE_DAILY_TOTALS=true  # Include daily demand totals
OVERWRITE_DATA=true  # Overwrite existing data
PV_ENABLED=true  # PV included as an option
DAYLIGHT_SAVING=true  # Country uses daylight saving time

# Python-specific settings
USE_PORTABLE_RNG=false  # Use portable LCG for RNG validation

# ============================================================
# PATHS (set these before running)
# ============================================================

# Config file containing dwelling configurations
DWELLINGS_FILE="${DWELLINGS_FILE:-excel/lcg_fixed/Dwellings.csv}"

# Output directory for results
OUTPUT_DIR="${OUTPUT_DIR:-output/run}"

# ============================================================
# RUN THE SIMULATION
# ============================================================

# Build command line arguments
CMD_ARGS=()
CMD_ARGS+=(--day "$DAY")
CMD_ARGS+=(--month "$MONTH")
CMD_ARGS+=(--latitude "$LATITUDE")
CMD_ARGS+=(--longitude "$LONGITUDE")
CMD_ARGS+=(--meridian "$MERIDIAN")
CMD_ARGS+=(--country "$COUNTRY")
CMD_ARGS+=(--city "$CITY")
CMD_ARGS+=(--year "$YEAR")
CMD_ARGS+=(--urban-rural "$URBAN_RURAL")
CMD_ARGS+=(--config-file "$DWELLINGS_FILE")
CMD_ARGS+=(--output-dir "$OUTPUT_DIR")

# Add seed if specified
if [ -n "$SEED" ]; then
    CMD_ARGS+=(--seed "$SEED")
fi

# Add optional flags based on checkbox settings
if [ "$SAVE_DETAILED" = "true" ]; then
    CMD_ARGS+=(--save-detailed)
fi

if [ "$USE_PORTABLE_RNG" = "true" ]; then
    CMD_ARGS+=(--portable-rng)
fi

# Run the simulation
echo "Running CREST simulation with settings from: ${excel_path}"
echo "Output directory: $OUTPUT_DIR"
echo ""

venv/bin/python python/main.py "${CMD_ARGS[@]}"
