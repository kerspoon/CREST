# RNG Divergence Investigation

## Status: IN PROGRESS

Last updated: 2024-12-04

## Problem Summary

RNG sequences between Python and Excel (VBA) diverge at call #83118. At this point:
- **Excel**: Finished lighting simulation, moved to appliances module
- **Python**: Still in lighting simulation

## Fixes Applied

### 1. appliances.py (lines 193-197)
**Issue**: Python was skipping RNG calls when appliance ownership was pre-configured in Dwellings.csv.

**Finding**: VBA ALWAYS calls `g_PortableRNG.Random()` for each of 31 appliances, regardless of configuration. The ownership columns in Dwellings sheet are OUTPUT only (written by `WriteAppliances`), never INPUT.

**Fix**: Changed to always call `rng.random()` for appliance ownership:
```python
# VBA ALWAYS generates appliance ownership randomly - it never reads from Dwellings sheet.
# The ownership columns in Dwellings are OUTPUT only (written by WriteAppliances).
self.has_appliance.append(self.rng.random() < ownership)
```

### 2. lighting.py (line 130)
**Issue**: Python stored irradiance threshold as a float, VBA uses integer (`CInt()`).

**Finding**: With threshold ~54.46, comparison `54 < 54.46` gives different results than `54 < 54`.

**Fix**: Round threshold to integer:
```python
self.irradiance_threshold = round(self._get_monte_carlo_normal_dist_guess(
    irradiance_mean,
    irradiance_sd
))
```

## Root Cause Identified

**Python has far fewer lighting switch-on events than Excel** (38 vs 144 total).

When a light switches on:
1. Duration is determined (2 RNG calls)
2. Light stays on for N minutes (no RNG calls during this time)
3. Fewer switch-ons = more minute iterations = more RNG calls consumed in lighting

This explains why Python's lighting finishes later (more calls) even though it has fewer switch-on events.

## The Mystery (Unsolved)

Manual tracing shows the switch-on condition at call 6854 **should succeed**:

```
rand_val = 0.01212
calibrated_relative_use = 0.01489  (from bulb 2, generated at call 4834)
effective_occ = 1.0  (for 1 active occupant)
threshold = 1.0 × 0.01489 = 0.01489
Condition: 0.01212 < 0.01489 = True
```

But in actual Python execution, the switch-on does NOT occur. The next RNG call goes to the 5% check (next minute) instead of duration determination.

## Likely Culprits (Not Yet Verified)

1. **Variable scoping**: `calibrated_relative_use` might have wrong value in actual execution
2. **Occupancy state**: `effective_occ` might be 0 at that specific minute
3. **Low irradiance condition**: `low_irradiance` might be False
4. **Loop structure**: Some subtle difference in how bulbs/minutes are iterated

## Key Data Points

### RNG Call Counts
- Excel total: 161,925 calls
- Python total: 178,364 calls
- First divergence: call #83,118

### Lighting Statistics
- Excel switch-on events: 144
- Python switch-on events: 38
- Excel minute iterations (5% checks): 71,582
- Python minute iterations (5% checks): 74,460

### Calibration Values (Both Match)
- Calibration scalar: 0.00815368639667705
- Effective occupancy for 1 person: 1.0
- Irradiance threshold (dwelling 1): 54

## Files Modified
- `python/crest/core/appliances.py` - Always call RNG for ownership
- `python/crest/core/lighting.py` - Round irradiance threshold to integer

## Validation Commands

```bash
# Run Python with RNG logging
venv/bin/python scripts/rng_validation_run.py

# Compare with Excel run
venv/bin/python scripts/rng_log_compare.py \
    output/rng_validation/python_5houses_YYYYMMDD_NN \
    output/rng_validation/excel_2houses_YYYYMMDD_NN
```

## Next Steps

1. Add detailed logging to lighting switch-on condition to capture actual values
2. Compare `calibrated_relative_use` values at runtime vs expected
3. Check if `low_irradiance` or `effective_occ` differ from expected at call 6854
4. Verify bulb loop iteration matches VBA exactly
