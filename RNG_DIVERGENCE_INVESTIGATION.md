# RNG Divergence Investigation

**Date**: 2025-12-06
**Status**: Near-complete match, minor floating point precision differences remain

## Overview

Python and VBA now produce identical RNG call sequences (156,068 calls matching within 1e-10 tolerance). Results are deterministic with identical seeds on `lcg_fixed`.

## Current Status

| Column | Python | Excel | Diff | Status |
|--------|--------|-------|------|--------|
| Lighting demand | 284,070 | 284,070 | 0 | ✓ MATCH |
| Appliance demand | 1,430,817 | 1,431,885 | -1,068 | ~0.07% |
| Outdoor temperature | 9,874.9 | 9,874.9 | 0 | ✓ MATCH |

**Remaining issue:** ~1,000 W total appliance diff across 2880 rows (~0.04% per row). Caused by floating point precision differences between Excel's `NormInv` and scipy's `norm.ppf` for the normal distribution inverse CDF.

---

## Issues Fixed

### Issue #3: int() vs round() in Appliances (2025-12-06)

**Symptom:** 10,578 W appliance demand difference

**Root cause:** Python used `int()` (truncation) where VBA uses `CInt()` (rounding).

**Fix:** Changed `int()` to `round()` in three places in `appliances.py`:
1. Line 539: `_get_monte_carlo_normal_dist_guess()` return value
2. Line 270: restart delay calculation
3. Line 438: TV cycle length calculation

**Result:** 90% improvement (10,578 W → 1,068 W diff)

### Issue #2: Electricity Used by Heating System (2025-12-05)

**Symptom:** Python=10.0 W, Excel=0.0 W at minute 0

**Root cause:** Python output included pump power (`p_h`), but VBA only writes `aHeatingElectricity`.

**Fix:** Added `get_heating_electricity()` method to `heating.py`, updated `writer.py` to use it.

### Issue #1: Heat Gains Ratio Column Index (2025-12-05)

**Symptom:** Casual thermal gains Python=132.2 W, Excel=129.6 W

**Root cause:** Python loaded heat_gains_ratio from wrong CSV column (`iloc[31]` instead of `iloc[32]`).

**Fix:** `appliances.py` line 189 - changed to `row.iloc[32]`

---

## Common Bug Patterns (Reference)

### 1. int() vs round() - Truncation vs Rounding

VBA assigns Double to Integer using `CInt()` which ROUNDS. Python `int()` TRUNCATES.

**Always use `round()` in Python when VBA assigns to Integer.**

### 2. Index Off-by-One Errors (1-based vs 0-based)

Dwelling config indices use `value_offset=1` in selection, making them 1-based. But CSV lookups were using `.iloc[]` directly without subtracting 1.

Examples fixed:
- `controls.py`: `heating_systems.iloc[config.heating_system_index - 1]`
- `building.py`: `buildings_data.iloc[config.building_index - 1]`
- `loader.py`: `get_heating_type()` had incorrect row indexing

### 3. CSV Row Indexing Errors

Row indices in pandas (0-based) vs file line numbers (1-based) cause confusion.

### 4. Unit Conversion Errors

- `writer.py`: FuelRate was multiplied by 60 (m_fuel is already in m³/h, not m³/min)

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

## Validation Process

### Step 1: Run validation scripts

```bash
# Run Python simulation with validation logging
venv/bin/python3 scripts/rng_validation_run.py --validation-log

# Compare RNG call sequences
venv/bin/python3 scripts/rng_log_compare.py output/rng_validation/python_2houses_YYYYMMDD_XX/ output/rng_validation/excel_2houses_20251205_01/
```

### Step 2: Compare minute-level output

Compare `results_minute_level.csv` (Python) with `Results - disaggregated.csv` (Excel).

```python
# Load both files
python_df = pd.read_csv('output/.../results_minute_level.csv', skiprows=[0,2,3])
excel_df = pd.read_csv('excel/lcg_fixed/Results - disaggregated.csv', skiprows=[0,1,2,4,5])

# Compare key columns
for col in ['Lighting demand', 'Appliance demand']:
    diff = (python_df[col] - excel_df[col]).sum()
    print(f"{col}: diff = {diff}")
```

### Step 3: Identify discrepancies

For each column, compare Python vs Excel values:
- Filter for rows where diff > 0
- Identify minute/dwelling
- Trace root cause in code
- Fix and re-run validation
