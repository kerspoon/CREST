# RNG Divergence Investigation

**Date**: 2025-12-08
**Status**: COMPLETE - All 40 columns match within floating-point precision (max diff ~4e-8)

## Overview

Python and VBA now produce identical RNG call sequences (1.6M+ calls for 20 houses matching within 1e-10 tolerance). Results are deterministic with identical seeds on `lcg_fixed`.

## Current Status

To find where we are up to:

```bash
# Run Python simulation with validation logging (re-export from excel if VBA changed)
venv/bin/python3 scripts/rng_validation_run.py --validation-log --no-export
# Then compare the results files
venv/bin/python3 scripts/compare_results.py excel/lcg_fixed/ output/rng_validation/python_20houses_YYYYMMDD_NN
```

### Latest Comparison Results (2025-12-08, 20 Houses)

**All 40 columns match within floating-point precision:**
- Dwelling index, Occupancy, Activity, Lighting demand, Hot water demand
- Space/HW heating timer settings, Heating system switched on, HW heating required
- Solar thermal collector control state, Solar thermal collector temperature
- Space cooling timer settings, Cooling system switched on, Cooling output
- Heating/Cooling thermostat set points, Cooling/Heating electricity
- Appliance demand, Casual thermal gains
- Primary heating output, Indoor temperature
- PV output, Self-consumption, Net electricity demand

**Daily summary max differences:**
- All columns: < 1e-7 (floating-point precision only)

**Minute-level max differences:**
- All columns: < 1e-7 (floating-point precision only)

---

## Fixes Applied (2025-12-06 to 2025-12-08)

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

### 5. Climate Data 1-Based vs 0-Based Index Bug (writer.py)

**Problem**: `get_temperature(idx)` and `get_irradiance(idx)` were called with 0-based `idx`, but these functions expect 1-based `minute`. This caused a 1-timestep shift in outdoor temperature and radiation values.

**Fix**: Changed lines 398-399 from:
```python
dwelling.local_climate.get_temperature(idx),
dwelling.local_climate.get_irradiance(idx),
```
to:
```python
dwelling.local_climate.get_temperature(minute),
dwelling.local_climate.get_irradiance(minute),
```

### 6. Hardcoded Zeros for Cooling Timer/Thermostat (writer.py)

**Problem**: Lines 422-423 had hardcoded `0` for "Space cooling timer" and "Cooling system switched on" columns.

**Fix**: Changed to use actual values from `heating_controls`:
```python
# BEFORE (buggy):
0,                                                   # 33. Space cooling timer
0,                                                   # 34. Cooling system switched on

# AFTER (fixed):
int(heating_controls.space_cooling_timer[idx]),      # 33. Space cooling timer
int(heating_controls.space_cooling_thermostat[idx]), # 34. Cooling system switched on
```

### 7. Activity Column Using min() Instead of Raw Second Digit (writer.py)

**Problem**: Python used `active_occupancy` (which is `min(at_home, active)`) for the "Activity" output column, but VBA outputs the raw second digit of the combined state.

**Fix**: Added `_calculate_activity()` method to extract second digit, and changed output column to use it:
```python
# BEFORE (buggy):
occupancy_1min = self._expand_10min_to_1min(dwelling.occupancy.active_occupancy)

# AFTER (fixed):
activity_1min = self._expand_10min_to_1min(
    self._calculate_activity(dwelling.occupancy.combined_states)
)
```

Note: `active_occupancy` (min of both digits) is still used internally for thermal gains calculations.

### 8. Appliance Rated Power Truncation (appliances.py)

**Problem**: Python used `int(float(...))` to load rated power from CSV, which TRUNCATES. VBA uses implicit `CInt()` which ROUNDS.

**Fix**: Changed `appliances.py:181-186` from `int()` to `round()`:
```python
# BEFORE (buggy):
rated_power = int(float(row.iloc[15])) if len(row) > 15 else 100

# AFTER (fixed):
rated_power = round(float(row.iloc[15])) if len(row) > 15 else 100
```

Same fix applied to `cycle_length`, `restart_delay`, and `standby_power`.

### 9. Solar Thermal Object Creation (dwelling.py)

**Problem**: Python only created `SolarThermal` objects for dwellings with `solar_thermal_index > 0`. VBA creates SolarThermal for ALL dwellings regardless, initializing `theta_collector` to outdoor temperature.

**Fix**: Modified `dwelling.py:195-211` to always create SolarThermal:
```python
# BEFORE (buggy):
if config.solar_thermal_index > 0:
    self.solar_thermal = SolarThermal(...)
    self.solar_thermal.initialize(...)
else:
    self.solar_thermal = None

# AFTER (fixed):
# VBA creates SolarThermal for ALL dwellings
self.solar_thermal = SolarThermal(data_loader, self.rng)
self.solar_thermal.initialize(...)
if config.solar_thermal_index > 0:
    self.building.set_solar_thermal(self.solar_thermal)
    self.appliances.set_solar_thermal(self.solar_thermal)
```

---

## All Issues Resolved

### Appliance Demand - FIXED (2025-12-08)
- **Root cause**: Python used `int(float(...))` (TRUNCATES) when loading rated power from CSV, but VBA uses implicit `CInt()` (ROUNDS) when assigning to Integer variables
- **Fix**: Changed `appliances.py:181-186` to use `round()` instead of `int()`
- **Affected appliances** (had decimal values in CSV):
  - PC: 140.7 → was 140 (truncated), now 141 (rounded)
  - VCR_DVD: 33.55 → was 33, now 34
  - RECEIVER: 26.82 → was 26, now 27
  - DISH_WASHER: 1130.61 → was 1130, now 1131
  - WASHING_MACHINE: 405.54 → was 405, now 406
- **Result**: Appliance demand now has **PERFECT MATCH (max diff = 0)**

### Heating Output - FIXED (2025-12-08)
- **Status**: With appliance demand fixed, all downstream values now match
- **Cascade effect confirmed**: Appliance demand → casual gains → indoor temp → heating demand all now match within floating-point precision

### Solar Thermal Collector Temperature - FIXED (2025-12-08)
- **Root cause**: Python only created `SolarThermal` objects for dwellings with `solar_thermal_index > 0`, but VBA creates SolarThermal for ALL dwellings regardless
- **Symptom**: 3.84°C difference at minute 0 for 7 dwellings that had no solar thermal system (`solar_thermal_index=0`)
- **Fix**: Changed `dwelling.py:195-211` to always create `SolarThermal` object for all dwellings, matching VBA behavior
- **Result**: Solar thermal collector temperature now has **PERFECT MATCH**

---

## Common Bug Patterns (Reference)

### 1. int() vs round() - Truncation vs Rounding

VBA assigns Double to Integer using `CInt()` which ROUNDS. Python `int()` TRUNCATES.

**Always use `round()` in Python when VBA assigns to Integer.**

### 2. Index Off-by-One Errors (1-based vs 0-based)

Dwelling config indices use `value_offset=1` in selection, making them 1-based. But CSV lookups were using `.iloc[]` directly without subtracting 1.

**Climate getters expect 1-based minute, not 0-based index!**

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
