# RNG Divergence Investigation

**Date**: 2025-12-05
**Status**: RNG synchronization complete, output comparison pending

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

## Next Steps: Output Comparison

The simulation outputs cannot be compared yet because the Excel files are from different runs:

| File | Timestamp | Description |
|------|-----------|-------------|
| `random_debug.txt` | 15:48 | RNG log used for validation |
| `Results - disaggregated.csv` | 17:14 | Simulation results |

**Action required**: Re-run Excel once to generate both RNG log and results from the same run, then compare outputs.

### Expected Matching Fields (based on partial analysis)

These fields matched in daily summary:
- Mean active occupancy
- Proportion of day actively occupied
- Lighting demand
- Hot water demand (litres)
- Space thermostat set point

These fields differed (needs re-validation with synchronized runs):
- Appliance demand
- PV output
- Average indoor temperature
- Thermal energy values
- Solar thermal gains

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

VBA looks up by index column value. Python now correctly converts 1-based indices to 0-based for pandas iloc.
