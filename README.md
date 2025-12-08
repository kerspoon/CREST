# CREST Demand Model - Python Port

A high-resolution (1-minute) stochastic integrated thermal-electrical domestic energy demand simulator, ported from the original Excel VBA implementation.

**Project Goal:** Achieve 100% feature parity with the Excel VBA model - identical outputs for identical inputs (same configurations + same random seeds → same results).

This has been converted by an LLM (Claude Code) with minor handholding but we do know that the differences between the two are minor (as you can confirm with the two tests below) so the functionality is either the same and the differences come only from rounding errors (though both use 64-bit IEEE 754 double-precision floats) or there are bugs that don't make much difference to the output (the recent bugs we found are in how we round numbers up/down). 

The two tests are:

1. modify excel and python to use the same deterministic "random" number generator with the same seed. Thus every output should be exactly the same. This has been run for 20 houses and all 40 output columns now match within floating-point precision (~1e-8). We have also confirmed that the random number generator is called exactly the same number of times from the equivalent functions. 

2. run both excel and python lots of times (1000 for python and 100 for excel). Get the interquartile range for the python and check that 50% of all the excel values all within this range. This also works fairly well.

At this point I'm pretty happy that it's ok to use. Especially as going through this process found 3 bugs in the original excel file. 



## Quick Start

```bash
# Install
pip install -r python/requirements.txt 
python python/main.py --help

# Run with default settings (auto-creates output/run_YYYYMMDD_NN/)
python python/main.py --save-detailed

# Run once using the same settings as an excel file (see `output/run_YYYYMMDD_01/` with results)
python scripts/excel_run_and_compare.py excel/original.xlsm

# Run 20 iteration of excel and 1000 of python then compare them statistically
./run_excel_example.bat (in windows powershell)
python scripts/monte_carlo_run.py
python scripts/monte_carlo_compare.py output/monte_carlo/python_1000runs_20251114_06 output/monte_carlo/excel_20runs_20251113_02

# Run using a stable portable random number generator in both excel and python and compare where they diverge as they should give identical outputs
## copy excel/lcg_fixed.xlsm to windows and in file properties disable protection, then open in excel and run then copy random_debug.txt to output\rng_validation\excel_20251204_01
python scripts/rng_validation_run.py
python scripts/rng_log_compare.py

# Compare Python vs Excel minute-level results (outputs to output/comparison/comparison_YYYYMMDD_NN/)
python scripts/compare_results.py excel/lcg_fixed/ output/rng_validation/python_2houses_YYYYMMDD_NN
```

Results include:
- `results_minute_level.csv` - 1440 rows of minute-by-minute data
- `results_daily_summary.csv` - Daily totals per dwelling
- `global_climate.csv` - Climate conditions
- `rerun_simulation.sh` - Script to reproduce this exact run
- `metadata.json` - Traceability information

---


## Current Status

**Validation:** Probably identical for practical purposes, but not proven mathematically identical.

### Monte Carlo Validation (1000 Python runs vs 100 Excel runs × 5 dwellings)

| Metric | Result | Expected | Status |
|--------|--------|----------|--------|
| Daily totals in IQR | 52.7% | 50% (95% CI: 45.6-54.4%) | ✓ PASS |
| Disaggregated in IQR | 77.9% | ~50% | ⚠ High |
| Range violations (daily) | 9/7500 (0.12%) | ~0.2% | ✓ PASS |
| Range violations (disagg) | 1-2% | ~0.2% | ⚠ High (solar) |

**Verdict:** Daily totals match well. Disaggregated at 77.9% suggests Python may have slightly wider distributions than Excel for some variables. Solar range violations at 10× expected rate indicate minor differences in edge-case handling.

See [`VALIDATION_RANGE_REPORT.md`](./VALIDATION_RANGE_REPORT.md) for detailed analysis of edge cases.

### RNG Validation (deterministic comparison)

See [`RNG_DIVERGENCE_INVESTIGATION.md`](./RNG_DIVERGENCE_INVESTIGATION.md) for detailed findings.

| Metric | Value |
|--------|-------|
| Perfect match columns | 40 of 40 |
| RNG calls verified | 1.6M+ identical |
| Max daily electricity diff | ~0 kWh |
| Max temperature diff | ~0°C |

**All 40 columns match within floating-point precision:**
- Occupancy, Activity, Lighting, Hot water demand
- Appliance demand, Casual thermal gains
- All timer settings and on/off states
- All thermostat setpoints, Heating/Cooling electricity
- Primary heating output, Indoor temperature
- Solar thermal collector temperature, PV output

**Issues resolved (2025-12-08):**
- ~~Appliance demand: max 7W~~ → **FIXED** - Root cause was `int()` vs `round()` when loading rated power from CSV
- ~~Heating output: max 27W at transitions~~ → **FIXED** - Cascade from appliance fix
- ~~Solar thermal collector temperature: max 3.8°C~~ → **FIXED** - Python now creates SolarThermal for all dwellings (matching VBA)

### VBA Bug Fixes

Three bugs were discovered in the original Excel VBA code during validation. See [`EXCEL_VBA_FIXES.md`](./EXCEL_VBA_FIXES.md) for fix instructions.

| Bug | Location | Impact |
|-----|----------|--------|
| Day of year undefined | `clsSolarThermal.cls` ~378 | Solar position uses day 0 |
| Tan(x)/Tan(x) = 1 | `clsSolarThermal.cls` ~427 | Sunrise check always passes |
| Cos() not Acos() | `clsSolarThermal.cls` ~465 | ~2x error in beam radiation |

**Fixed files:** `lcg_fixed.xlsm` has all fixes. Create `monte_carlo_fixed.xlsm` or `original_fixed.xlsm` using the instructions.

---


## Directory Structure

```
crest/
├── README.md                    # This file
├── CLAUDE.md                    # Development instructions
├── EXCEL_VBA_FIXES.md           # VBA bug fix instructions
│
├── excel/                       # Excel/VBA reference implementation
│   ├── original.xlsm                    # Base v2.3.3 model (has Acos bug)
│   ├── original/                        # Exports from original.xlsm
│   │   ├── *.cls, *.bas                 # VBA code
│   │   ├── Dwellings.csv                # Dwelling configurations
│   │   ├── Main_Sheet.csv               # Run parameters
│   │   └── *.csv                        # Data sheets (ActivityStats, etc.)
│   ├── original_100houses.xlsm          # Reference 100-house run
│   ├── monte_carlo_base.xlsm            # 5 varied test houses (has Acos bug)
│   ├── monte_carlo_fixed.xlsm           # Bug-fixed version (create using EXCEL_VBA_FIXES.md)
│   ├── lcg_fixed.xlsm                   # LCG + bug fixes (deterministic comparison)
│   └── lcg_fixed/                       # Exports from LCG model
│
├── python/                      # Python implementation
│   ├── crest/                   # Main package
│   │   ├── core/                # Simulation modules
│   │   ├── simulation/          # Dwelling orchestration
│   │   ├── data/                # Data loading
│   │   ├── output/              # Results writing
│   │   └── utils/               # Utilities (RNG, etc.)
│   ├── data/                    # CSV data files
│   ├── main.py                  # Entry point
│   ├── requirements.txt         # Python dependencies
│   └── mypy.ini                 # Type checking config
│
├── scripts/                     # Validation and utility scripts
│   ├── excel_run_and_compare.py # Run Python from Excel settings + compare
│   ├── extract_settings.py      # Extract simulation params from Excel Main Sheet
│   ├── export_excel.py          # Export VBA + CSV from .xlsm
│   ├── monte_carlo_run.py       # Run N Monte Carlo iterations
│   ├── monte_carlo_compare.py   # IQR validation (Objective #2)
│   ├── rng_validation_run.py    # Run with LCG logging
│   ├── rng_log_compare.py       # RNG sequence comparison (Objective #1)
│   ├── compare_results.py       # Compare Python vs Excel minute-level results
│   ├── check_types.sh           # Run mypy type checking
│   ├── run_excel_example.bat    # Run Excel N times (Windows batch)
│   ├── run_excel_macro.ps1      # PowerShell automation for Excel runs
│   └── utils.py                 # Helper functions
│
└── output/                      # Simulation results
    ├── run_YYYYMMDD_NN/         # General simulation runs (auto-numbered)
    │   ├── results_minute_level.csv      # Minute-by-minute results
    │   ├── results_daily_summary.csv     # Daily totals
    │   ├── dwellings_config.csv          # Copy of dwelling config
    │   ├── rerun_simulation.sh           # Script to reproduce run
    │   ├── comparison_report.txt         # Excel vs Python (if comparing)
    │   └── metadata.json                 # Traceability info
    ├── monte_carlo/             # Monte Carlo validation runs
    │   ├── python_YYYYMMDD_NN/  # Python baseline (1000 runs)
    │   ├── excel_YYYYMMDD_NN/   # Excel comparison (20 runs)
    │   └── validation_pYYYYMMDD_NN_eYYYYMMDD_NN/  # IQR analysis
    ├── rng_validation/          # RNG call sequence validation
    │   ├── python_YYYYMMDD_NN/  # Python run with LCG logging
    │   ├── excel_YYYYMMDD_NN/   # Excel run with LCG logging
    │   └── validation_pYYYYMMDD_NN_eYYYYMMDD_NN/  # Comparison report
    └── experiments/             # Ad-hoc test runs
```

### Output Directory Naming Convention

- **Runs**: `{type}_YYYYMMDD_NN` (auto-increments daily: _01, _02, ...)
- **Validations**: `validation_pYYYYMMDD_NN_eYYYYMMDD_NN` (links Python + Excel runs)

Each validation directory contains `metadata.json` with paths to source runs.

---

## Validation Workflow

The project has **two primary objectives** for validating that the Python port matches the Excel VBA implementation:

### Objective #1: RNG Call Sequence Matching

**Goal:** Verify that Python makes identical RNG calls in the same order as Excel.

**Process:**

1. **Generate Python run with LCG logging:**
   ```bash
   python scripts/rng_validation_run.py
   ```
   Creates: `output/rng_validation/python_5houses_YYYYMMDD_NN/`
   - `results_minute_level.csv` - Simulation results
   - `rng_calls.log` - Every RNG call logged (16MB+)

2. **Manually run Excel** (`excel/lcg_fixed.xlsm`):
   - Load `excel/lcg_fixed/Dwellings.csv` (or your test config)
   - Use seed `42`
   - Save output to: `output/rng_validation/excel_5houses_YYYYMMDD_NN/`
   - Include `rng_calls.log` from VBA

3. **Compare RNG logs:**
   ```bash
   python scripts/rng_log_compare.py \\
     output/rng_validation/python_5houses_YYYYMMDD_01 \\
     output/rng_validation/excel_5houses_YYYYMMDD_01
   ```
   Creates: `output/rng_validation/validation_pYYYYMMDD_01_eYYYYMMDD_01/`
   - `comparison_report.txt` - Match/mismatch summary
   - `call_sequence_diff.csv` - Detailed differences (if any)

**Expected Result:** ✓ 100% match in call count, order, and location

---

### Objective #2: Statistical Distribution Validation (IQR Test)

**Goal:** Verify that Python output distributions match Excel using the Interquartile Range (IQR) test.


**Process:**

1. **Generate Python baseline (1000 runs):**
   ```bash
   python scripts/monte_carlo_run.py 1000
   ```
   Creates: `output/monte_carlo/python_1000runs_YYYYMMDD_NN/`
   - `seed_001/` through `seed_1000/` - Individual run results
   - `daily_summary.csv` - Combined daily totals
   - `minute_level.parquet` - Combined minute data (compressed)

2. **Run Excel 20 times:**
   - First, create `monte_carlo_fixed.xlsm` using instructions in [`EXCEL_VBA_FIXES.md`](./EXCEL_VBA_FIXES.md)
   - Use `run_excel_example.bat` with `excel/monte_carlo_fixed.xlsm` (needs to be done on Windows)
   - Save each run to: `output/monte_carlo/excel_20runs_YYYYMMDD_NN/run_01/` through `run_20/`
   - Each run should contain:
     - `results_minute_level.csv`
     - `results_daily_summary.csv`

3. **Run IQR validation:**
   ```bash
   python scripts/monte_carlo_compare.py \\
     output/monte_carlo/python_1000runs_YYYYMMDD_01 \\
     output/monte_carlo/excel_20runs_YYYYMMDD_01
   ```
   Creates: `output/monte_carlo/validation_pYYYYMMDD_01_eYYYYMMDD_01/`
   - `iqr_analysis.csv` - Detailed test results (72K+ rows)
   - `summary_statistics.csv` - Overall statistics by variable
   - `validation_report.txt` - Pass/fail summary

**Expected Result:** >50% of Excel samples fall within Python IQR

For each (dwelling, minute, variable) combination:
- Python: Compute Q1, median, Q3 from 1000 runs
- Excel: Check if each of 20 values falls in [Q1, Q3]
- Expected: ~50% in IQR (by definition of quartiles)

---

## What the Model Simulates

- **Occupancy:** 4-state Markov chain (home/away × active/dormant) with 1-minute resolution
- **Electrical demand:** 31 appliance types with activity-based switching, up to 60 light bulbs
- **Thermal demand:** 5-node RC thermal network, gas/electric boilers, hot water (4 fixtures)
- **Renewables:** PV systems, solar thermal collectors
- **Cooling:** Fans, air coolers, AC units (India-specific)
- **Climate:** Stochastic weather (temperature, solar irradiance) with seasonal variability

---

## Excel-Based Workflow

### Run Using Excel Settings

The easiest way to run Python simulations is to use settings directly from an Excel file:

```bash
# Run using settings from any Excel file
python scripts/excel_run_and_compare.py excel/original.xlsm
python scripts/excel_run_and_compare.py excel/lcg_fixed.xlsm

```

**What it does:**
1. Exports VBA and CSV to `excel/{basename}/`
2. Reads run parameters from Main Sheet (day, month, country, seed, etc.)
3. Runs Python with those settings
4. Creates `output/run_YYYYMMDD_NN/` with results
5. Generates `rerun_simulation.sh` for reproducibility
6. Compares with Excel output (if Excel results exist)
---

## Advanced Usage

### Run with Custom Configuration

```bash
python python/main.py \\
  --config-file my_dwellings.csv \\
  --day 15 \\
  --month 7 \\
  --country UK \\
  --city England \\
  --urban-rural Urban \\
  --seed 12345 \\
  --output-dir output/experiments/summer_test \\
  --save-detailed
```

### Use Portable RNG (for exact Excel matching)

```bash
python python/main.py \\
  --config-file excel/original/Dwellings.csv \\
  --portable-rng \\
  --seed 42 \\
  --output-dir output/experiments/lcg_test
```

### Type Checking

```bash
# Run mypy type checker
scripts/check_types.sh

# Check specific file
venv/bin/mypy python/crest/core/occupancy.py
```

---

## Development

### Project Instructions

See [`CLAUDE.md`](./CLAUDE.md) for detailed development guidelines, including:
- Feature parity requirements
- Auditing process
- Line-by-line VBA comparison methodology


### Exporting from Excel

```bash
# Export VBA code and CSV sheets from any Excel file
python scripts/export_excel.py excel/original.xlsm
```

**Note:** The `excel_run_and_compare.py` workflow automatically handles exports,
so manual exporting is rarely needed.

---

## Original Model Reference

- **Authors:** Eoghan McKenna, Murray Thomson (Loughborough University)
- **Publication:** McKenna, E., & Thomson, M. (2016). High-resolution stochastic integrated thermal–electrical domestic demand model. *Applied Energy*, 165, 445-461.
- **Original:** Excel VBA implementation (CREST Demand Model v2.3.3)

---

## License

This Python port maintains compatibility with the original Excel VBA model's licensing.

---

## Data Files

**37 CSV files** in `python/data/` (extracted from Excel sheets):
- **12 occupancy TPMs** (6 resident counts × weekday/weekend)
- **25 config/spec files:**
  - Appliances (ownership, specs, activity profiles)
  - Buildings (thermal properties, proportions)
  - Heating systems (boiler types, efficiencies)
  - Cooling systems (India-specific)
  - PV systems (panel specs, inverter characteristics)
  - Solar thermal (collector specs)
  - Climate data (temperature profiles by city/month/hour)
  - Activity statistics (72 activity profiles)

