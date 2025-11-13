#!/bin/bash
# Template script for running CREST Python simulation
#
# This script demonstrates how to run the Python CREST model with custom settings.
# Copy and modify this script to create custom simulation configurations.
#
# Usage:
#   bash scripts/run_python.sh
#
# To create a custom run:
#   cp scripts/run_python.sh my_simulation.sh
#   # Edit settings below
#   bash my_simulation.sh

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================

# Number of dwellings to simulate (or use --config-file to load from CSV)
NUM_DWELLINGS=1

# Date settings
DAY=1
MONTH=1
YEAR=2006

# Location settings
COUNTRY="UK"
CITY="England"
URBAN_RURAL="Urban"  # Options: Urban, Rural

# Random seed (optional - leave empty for random, or set for reproducibility)
SEED=""

# Output settings
SAVE_DETAILED="--save-detailed"  # Include minute-level output (remove to disable)
PORTABLE_RNG=""  # Use portable RNG for validation (add "--portable-rng" to enable)

# Output directory (leave empty to auto-generate as output/run_YYYYMMDD_NN)
OUTPUT_DIR=""

# Dwelling configuration file (leave empty to generate stochastically)
CONFIG_FILE=""

# ============================================================================
# RUN SIMULATION
# ============================================================================

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
source venv/bin/activate

# Build command
CMD="python python/main.py"

# Add dwelling count or config file
if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config-file \"$CONFIG_FILE\""
else
    CMD="$CMD --num-dwellings $NUM_DWELLINGS"
fi

# Add date settings
CMD="$CMD --day $DAY --month $MONTH --year $YEAR"

# Add location settings
CMD="$CMD --country \"$COUNTRY\" --city \"$CITY\" --urban-rural \"$URBAN_RURAL\""

# Add seed if specified
if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

# Add output settings
if [ -n "$SAVE_DETAILED" ]; then
    CMD="$CMD $SAVE_DETAILED"
fi

if [ -n "$PORTABLE_RNG" ]; then
    CMD="$CMD $PORTABLE_RNG"
fi

# Add output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

# Print and run command
echo "Running simulation:"
echo "$CMD"
echo
eval "$CMD"
