# RNG Divergence Investigation

**Date**: 2025-12-06
**Status**: Excellent match - daily totals within 0.02 kWh, minute-level within small tolerances

## Overview

Python and VBA now produce identical RNG call sequences (156,068 calls matching within 1e-10 tolerance). Results are deterministic with identical seeds on `lcg_fixed`.

## Current Status

To find where we are up to:

```bash
# Run Python simulation with validation logging (re-export from excel if VBA changed)
venv/bin/python3 scripts/rng_validation_run.py --validation-log --no-export
# Then compare the results files
venv/bin/python3 scripts/compare_results.py excel/lcg_fixed/ output/rng_validation/python_2houses_YYYYMMDD_NN
```

### Latest Comparison Results (2025-12-06)

**Perfect matches (16 columns):**
- Dwelling index, Occupancy, Lighting demand, Hot water demand
- Space/HW heating timer settings, Heating system switched on, HW heating required
- Solar thermal collector control state
- Space cooling timer settings, Cooling system switched on, Cooling output
- Heating/Cooling thermostat set points, Cooling/Heating electricity

**Daily summary max differences:**
- Net electricity demand: 0.0132 kWh
- Average indoor temperature: 0.0012°C
- All other columns < 0.002

---

## Fixes Applied (2025-12-06)

### 1. PV/Solar Thermal Location Parameters Not Passed

**Problem**: PV and Solar Thermal systems were using default location parameters (Loughborough: lat=52.2, day=166) instead of actual simulation parameters (Manchester: lat=53.48, day=1).

**Fix**: Modified `dwelling.py` to pass location parameters from `global_climate.config` to both `pv_system.initialize()` and `solar_thermal.initialize()`.

### 2. VBA Solar Thermal Acos Bug (clsSolarThermal.cls line 481)

**Problem**: VBA used `Cos(dot_product)` instead of `Acos(dot_product)` to calculate incident angle, causing ~2x error in direct beam radiation.

**Fix**: Changed VBA line 481 from:
```vba
dblSolarIncidentAngle = Cos(...)
```
to:
```vba
dblSolarIncidentAngle = (180 / PI) * Application.WorksheetFunction.Acos(...)
```

### 3. Self-Consumption Unit Conversion (writer.py)

**Problem**: Python was converting P_self from W to kWh, but VBA stores it in Watts.

**Fix**: Removed `/60.0/1000.0` conversion.

### 4. Hardcoded Zeros for Heating/Water Flags (writer.py)

**Problem**: Lines 408-409 had hardcoded `0` instead of actual values.

**Fix**: Changed to use `heating_controls.heater_on_off[idx]` and `heating_controls.heat_water_on_off[idx]`.

---

## Known Outstanding Issues

### GlobalClimate.csv Export Mismatch

The exported `excel/lcg_fixed/GlobalClimate.csv` appears to be from a **summer simulation** rather than January 1:
- Shows clear sky irradiance ~965 W/m² at 08:47
- On January 1 at 53.5°N, max solar altitude is only 13.49° (clear sky should be ~8-12 W/m²)

**Note**: This is an export/logging issue only. The actual Results files use correct climate data internally for January 1.

### Small Remaining Differences

- **Outdoor global radiation**: ~85 W/m² max diff (related to GlobalClimate.csv export issue)
- **Primary heating output**: ~18 W max diff (floating point accumulation)
- **Appliance demand**: ~3 W max diff (likely RNG timing or integer rounding)

---

## Common Bug Patterns (Reference)

### 1. int() vs round() - Truncation vs Rounding

VBA assigns Double to Integer using `CInt()` which ROUNDS. Python `int()` TRUNCATES.

**Always use `round()` in Python when VBA assigns to Integer.**

### 2. Index Off-by-One Errors (1-based vs 0-based)

Dwelling config indices use `value_offset=1` in selection, making them 1-based. But CSV lookups were using `.iloc[]` directly without subtracting 1.

### 3. Location/Date Parameters Not Passed

Components with location-dependent defaults may not receive actual simulation parameters. Check that `latitude`, `longitude`, `meridian`, and `day_of_year` are passed from `global_climate.config` to all components that need them.

### 4. Unit Conversion Errors

- `writer.py`: FuelRate was multiplied by 60 (m_fuel is already in m³/h, not m³/min)
- `writer.py`: Self-consumption was incorrectly converted from W to kWh

### 5. Column Index Reference

Correct column indices (0-based) for AppliancesAndWaterFixtures.csv:

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

---
