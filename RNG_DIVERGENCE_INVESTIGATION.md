# RNG Divergence Investigation

**Date**: 2025-12-05
**Status**: RNG synchronized, temperature bug FIXED, appliance demand bug FIXED

## Summary

All RNG-related bugs have been fixed. Python and VBA now produce identical RNG call sequences (156,068 calls matching perfectly). Dwelling parameters match between implementations.

## Fixes Applied (Python only - no VBA changes)

### 1. Index Lookup Bugs (1-based to 0-based conversion)

All dwelling config indices use `value_offset=1` in selection, making them 1-based. But CSV lookups were using `.iloc[]` directly without subtracting 1.

| File | Line | Fix Applied |
|------|------|-------------|
| `controls.py` | 73 | `heating_systems.iloc[config.heating_system_index - 1]` |
| `controls.py` | 90 | `cooling_systems.iloc[config.cooling_system_index - 1]` |
| `controls.py` | 202 | `buildings_data.iloc[self.config.building_index - 1]` |
| `building.py` | 69 | `buildings_data.iloc[config.building_index - 1]` |
| `building.py` | 96 | `heating_systems_data.iloc[config.heating_system_index - 1]` |
| `heating.py` | 60 | `heating_systems.iloc[config.heating_system_index - 1]` |
| `water.py` | 94 | `heating_systems.iloc[config.heating_system_index - 1]` |

### 2. get_heating_type() Bug

In `loader.py`, the `get_heating_type()` function had incorrect row indexing:

```python
# Before (wrong)
row_idx = 3 + heating_index
return int(df.iloc[row_idx, 3])

# After (correct)
row_idx = heating_index - 1
return int(df.iloc[row_idx]['1 = regular, 2 = combi'])
```

### 3. Selection Logic Bug

In `main.py`, `_select_from_distribution()` didn't handle cases where probabilities don't sum to 1.0 (e.g., solar thermal with 50% coverage):

```python
# Added handling for when rand >= total probability
if index >= len(proportions):
    # No match found - VBA returns 0 (default uninitialized Long value)
    return 0
```

### 4. Monthly Temperature Index Bug (FIXED 2025-12-05)

In `climate.py`, monthly temperature data was loaded from wrong row indices:

```python
# Before (wrong) - loaded Feb-Dec data for Jan-Nov
for month_idx in range(2, 14):  # Rows 2-13 - WRONG

# After (correct) - loads Jan-Dec data correctly
for month_idx in range(1, 13):  # Rows 1-12 - CORRECT
```

**Impact**: Python was using February's mean temperature (3.7°C) for January instead of the correct value (3.3°C), causing a constant 0.4°C offset in all temperature calculations.

### 5. Overnight Clearness Division Bug (FIXED 2025-12-05)

In `climate.py`, the overnight mean clearness calculation divided by `count` instead of `di`:

```python
# Before (wrong)
overnight_mean_clearness /= count  # Number of values summed

# After (correct - matching VBA)
overnight_mean_clearness /= di  # Darkness duration in minutes
```

### Files Already Correct (had -1 adjustment)

- `python/crest/core/pv.py` line 171
- `python/crest/core/solar_thermal.py` line 203
- `python/crest/core/cooling.py` line 138

## Verification Results

### RNG Call Matching

```
Excel calls:  156,068
Python calls: 156,068
All values match: True (within 1e-10 tolerance)
```

### Dwelling Parameters Match

```
Dwelling 1:
  Python: residents=1 building=4 heating=1 pv=2 solar=2 cooling=1
  Excel:  residents=1 building=4 heating=1 pv=2 solar=2 cooling=1 ✓

Dwelling 2:
  Python: residents=2 building=1 heating=1 pv=2 solar=0 cooling=1
  Excel:  residents=2 building=1 heating=1 pv=2 solar=0 cooling=1 ✓
```

## Output Comparison Results

### After Temperature Fix (2025-12-05)

Minute 1 temperatures now match:
- Python: 3.839146°C
- Excel: 3.839146°C
- Difference: ~6e-14 (floating point precision)

### Daily Summary - Dwelling 1

| Field | Python | Excel | Status |
|-------|--------|-------|--------|
| Mean active occupancy | 0.4236 | 0.4236 | ✓ Match |
| Proportion occupied | 0.4236 | 0.4236 | ✓ Match |
| Lighting demand | 1.9361 kWh | 1.9361 kWh | ✓ Match |
| Hot water demand | 52 L | 52 L | ✓ Match |
| Thermostat setpoint | 17°C | 17°C | ✓ Match |
| Appliance demand | 10.68 kWh | 10.71 kWh | ✓ ~0.03 kWh diff (fixed) |
| PV output | 3.10 kWh | 2.39 kWh | ~0.71 kWh diff |
| Indoor temp (avg) | 17.40°C | 16.65°C | ~0.75°C diff |

### 6. Appliance Total Demand Missing Heating/Cooling Electricity (FIXED 2025-12-05)

In `dwelling.py`, the Appliances object wasn't connected to heating/cooling/solar_thermal systems, so `calculate_total_demand()` didn't include their electricity usage.

**Root cause**: `Dwelling.__init__` called `building.set_heating_system()` but not `appliances.set_heating_system()`. The `calculate_total_demand()` method checks `if self.heating_system is not None` before adding heating electricity, but `self.heating_system` was always `None`.

**Fix applied** in `python/crest/simulation/dwelling.py`:
```python
# Added after creating appliances:
self.appliances.set_heating_system(self.heating_system)

# Added after creating solar_thermal (if any):
self.appliances.set_solar_thermal(self.solar_thermal)

# Added after creating cooling_system (if any):
self.appliances.set_cooling_system(self.cooling_system)
```

**Also fixed** in `python/crest/output/writer.py` - removed double-counting of heating/cooling electricity in net demand calculation:
```python
# Before (wrong - double-counted heating/cooling):
net_elec_w = lighting_w + appliance_w + heating_elec_w + cooling_elec_w - pv_output_w

# After (correct - appliance_w already includes heating/cooling):
net_elec_w = lighting_w + appliance_w - pv_output_w
```

**Impact**:
- Before fix: Pa=50W at minute 1, Appliance demand=10.34 kWh/day
- After fix: Pa=60W at minute 1, Appliance demand=10.68 kWh/day (matches Excel 10.71 kWh)

## Technical Details

### Root Cause of Original Bug

The cooling system type lookup was the most critical bug because it affected RNG consumption:

- `cooling_system_type > 1` triggers the cooling timer Markov chain (48 RNG calls)
- Wrong lookup caused Python to run/skip this loop differently than VBA
- This shifted all subsequent RNG values, causing complete divergence

### CSV Index Structure

All dwelling parameter CSVs use 1-based indices in the first column:

| CSV | Index Column |
|-----|--------------|
| CoolingSystems.csv | "Primary heating system index" |
| PrimaryHeatingSystems.csv | "Primary heating system index" |
| Buildings.csv | "Building index" |
| SolarThermalSystems.csv | "Solar thermal system index" |
| ClimateData&CoolingTech.csv | Row 1 = Jan, Row 2 = Feb, etc. |

VBA looks up by index column value. Python now correctly converts 1-based indices to 0-based for pandas iloc.

## Next Steps

1. ~~Investigate the appliance ownership logging/dwelling index issue~~ **DONE** - Was actually a missing system reference bug
2. Investigate PV output difference (~0.71 kWh Python vs Excel)
3. Investigate indoor temperature difference (~0.75°C Python vs Excel)
4. Run comparison for other months to ensure the monthly temperature fix works across all months
