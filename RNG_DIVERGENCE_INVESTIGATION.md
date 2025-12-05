# RNG Divergence Investigation

**Date**: 2025-12-05
**Status**: RNG synchronized, fixing output discrepancies

## Overview

Python and VBA now produce identical RNG call sequences (156,068 calls matching, giving the same random numbers to many decimal places - within 1e-10 tolerance - and being called from equivalent places in the same order the same number of times). The results for both are now deterministic and should give identical output on `lcg_fixed`.

But there are errors, mostly with how we read in the data files.

## Common Bug Patterns 

These specific examples have been fixed but there is likely more like this.

### 1. Index Off-by-One Errors (1-based vs 0-based)

Dwelling config indices use `value_offset=1` in selection, making them 1-based. But CSV lookups were using `.iloc[]` directly without subtracting 1.

Examples fixed:
- `controls.py`: `heating_systems.iloc[config.heating_system_index - 1]`
- `building.py`: `buildings_data.iloc[config.building_index - 1]`
- `loader.py`: `get_heating_type()` had incorrect row indexing

### 2. CSV Row Indexing Errors

Row indices in pandas (0-based) vs file line numbers (1-based) cause confusion.

Examples fixed:
- `climate.py`: Monthly temperature loaded with `range(2, 14)` instead of `range(1, 13)`
- `loader.py`: Hot water thermostat settings used `iloc[24:36]` instead of `iloc[23:35]`

### 3. Unit Conversion Errors

- `writer.py`: FuelRate was multiplied by 60 (m_fuel is already in m³/h, not m³/min)

### 4. Missing Component Connections

- `dwelling.py`: Appliances object wasn't connected to heating/cooling/solar_thermal systems, so `calculate_total_demand()` missed their electricity usage

## Validation Process

### Step 1: Run validation scripts

```bash
# Run Python simulation with validation logging
venv/bin/python3 scripts/rng_validation_run.py --validation-log

# Compare dwelling parameters
venv/bin/python3 scripts/rng_compare_params.py output/rng_validation/python_2houses_YYYYMMDD_XX/ output/rng_validation/excel_2houses_20251205_01/

# Compare RNG call sequences
venv/bin/python3 scripts/rng_log_compare.py output/rng_validation/python_2houses_YYYYMMDD_XX/ output/rng_validation/excel_2houses_20251205_01/
```

### Step 2: Compare minute-level output

Compare `results_minute_level.csv` (Python) with `Results - disaggregated.csv` (Excel).

**Note**: Row numbers differ - Python data starts row 5, Excel starts row 7 (Excel has 2 blank lines at start).

```python
# Load both files
python_df = pd.read_csv('output/.../results_minute_level.csv', skiprows=[0,2,3])
excel_df = pd.read_csv('excel/lcg_fixed/Results - disaggregated.csv', skiprows=[0,1,2,4,5])
```

### Step 3: Identify discrepancies

For each column, compare Python vs Excel values:
- Start with minute 1, dwelling 1
- Check which columns have different data
- Investigate the root cause
- Fix and re-run validation
- Repeat through all minutes

## Current Status

### Matching ✓
- RNG sequences (156,068 calls)
- Dwelling parameters (residents, building, heating, pv, solar, cooling indices)
- Occupancy and activity states
- Lighting demand
- Hot water demand (litres)
- Outdoor temperature
- FuelRate (after fix)

### Issues Fixed (2025-12-05)

#### Issue #1: Heat Gains Ratio Column Index - FIXED

**Symptom:** Casual thermal gains Python=132.2 W, Excel=129.6 W (diff=2.6 W)

**Root cause:** Python was loading heat_gains_ratio from wrong CSV column.
- Python used `iloc[31]` ("Appliance mean power factor")
- VBA uses column AG = `iloc[32]` ("Heat gains ratio for casual thermal gains")

**Fix:** `appliances.py` line 189 - changed from `row.iloc[31]` to `row.iloc[32]`

**Verification:** After fix, first 60 minutes of dwelling 1 match exactly (129.60 W)

#### Issue #2: Electricity Used by Heating System - FIXED

**Symptom:** Python=10.0 W, Excel=0.0 W at minute 0

**Root cause:** Python output included pump power (`p_h`), but VBA only writes `aHeatingElectricity`.
- VBA: `WriteHeatingSystem` writes only `aHeatingElectricity` to column AN
- Python: `get_heating_system_power_demand()` returns `p_h + heating_electricity`

For gas heating systems (index 1-3), `aHeatingElectricity` is always 0. The 10W was pump power.

**Fix:**
- Added `get_heating_electricity(timestep)` method to `heating.py`
- Updated `writer.py` to use `get_heating_electricity()` instead of `get_heating_system_power_demand()`

**Verification:** After fix, heating electricity shows 0.0 W (matches Excel)

### Column Index Reference

These are the correct column indices (0-based) for AppliancesAndWaterFixtures.csv:

| Column | Index | Description |
|--------|-------|-------------|
| E | 4 | Short name |
| F | 5 | Proportion of dwellings with appliance |
| G | 6 | Activity use profile |
| P | 15 | Mean cycle power (W) |
| R | 17 | Mean cycle length (min) |
| S | 18 | Restart delay (min) |
| T | 19 | Standby power (W) |
| AD | 29 | Probability of switch on |
| AF | 31 | Appliance mean power factor (NOT heat gains!) |
| AG | 32 | Heat gains ratio (for casual thermal gains) |

### Current Status

**Matching ✓**
- RNG sequences (156,068 calls)
- Dwelling parameters
- Occupancy and activity states
- Lighting demand
- Hot water demand
- Outdoor temperature
- FuelRate
- Casual thermal gains (first 60 minutes verified)
- Electricity used by heating system

**Remaining discrepancies in later minutes:**
Large differences (~100-160 W) appear in dwelling 2 during active periods. This is likely due to:
1. Different appliance switching behavior (needs further investigation)
2. Different lighting switch-on counts

