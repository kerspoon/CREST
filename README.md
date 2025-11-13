# CREST Demand Model - Python Port

A high-resolution (1-minute) stochastic integrated thermal-electrical domestic energy demand simulator, ported from the original Excel VBA implementation.

**Project Goal:** Achieve 100% feature parity with the Excel VBA model - identical outputs for identical inputs (same configurations + same random seeds → same results).

---

## TODO & status

We have just doing a major reorganisation of the folder structure. This has likely broken the three major tests:

1. `excel_run_and_compare` (take an excel file, and run the equivalent in python) this has been checked, it runs under the new structure but the analysis/comparison is very basic. We have scripts that do a better comparison but these are not being used. 
	1a. improve the analysis to look at the variance in every bit of _summary_ data (min level is not going to be a fair comparison). but the total power should be similar (within X%)

2. `monte_carlo` (run 1000 iterations of `monte_carlo_base` then compare statistically to multiple runs of the same in excel) this is being worked on. 
	2a. we have both `scripts/analyse_montecarlo.py` and `scripts/monte_carlo_compare.py` (the former being older, probably more comprehensive but also not working). Combine the best of these. We want to test each excel output against the interquartile range of the python runs. By each output I mean 1) from the sheet "Results - daily totals" compare every row (dwelling) from column C "Mean active occupancy" to Q "Solar thermal collector heat gains" = 15 columns x 5 houses × 20 Excel runs) from sheet "Results - disaggregated" take columns D "Occupancy" to AN "Electricity used by heating system" (37 columns x 5 houses x 1440 mins_per_day × 20 Excel runs = about 5M data points). Currently we only look at a small subset of this. Output two tables of the summary results (one for daily total and one for disaggregated). in the `disaggregated` show how many timestamps were in the IQR, columns as the 5 houses, and rows as the 37 variables with each cell being a % of timestamps in the IQR (out of the 1440 x 20 samples). Also give a final summary for the given data (which might not be 1000 python and 20 excel runs) what sort of variance would we expect to see - how many should be falling withing the IQR with this many samples? how unlikely is it to have +/-10%, +/-1% or +/-0.1%?
 
3. `rnd` (aim to get full parity by checking the random number generator is called by the same function in the same order in both codebases, the main output of this is to show where the calls first diverged and to save the overall running order of the calls), it was running before the restructure.



We have a mess of files in ./scripts some are used some are not. I only want to keep those used by the above 3 tests (e.g. extracting files from a xlsm is part of running the above tests) but it would be good to check the purpose of each to see if any others are useful.

Once we have 1) got the tests running and 2) tidied the scripts we need to 3) try to get the tests passing (i.e. exactly the same output in the case of rnd, and statistically similar for monte_carlo). Before the reorganisation there were significant differences between the outputs of the two code bases. The reorganisation was to make it easier to check we are comparing like with like. 

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r python/requirements.txt

# Verify installation
python python/main.py --help
```

### Run Your First Simulation

**Option 1: Use an Excel file's settings (easiest)**

```bash
# Run using settings from Excel file
python scripts/excel_run_and_compare.py excel/original.xlsm
```

This automatically:
- Exports VBA and CSV from Excel to `excel/original/`
- Extracts run parameters from Main Sheet
- Runs Python with those settings
- Creates `output/run_YYYYMMDD_01/` with results
- Compares with Excel output (if available)

**Option 2: Run Python directly**

```bash
# Run with default settings (auto-creates output/run_YYYYMMDD_NN/)
python python/main.py 

# Or specify output directory
python python/main.py \\
  --config-file excel/original/Dwellings.csv \\
  --day 1 --month 1 \\
  --output-dir output/my_test \\
  --save-detailed \\
  --seed 42
```

Results include:
- `results_minute_level.csv` - 1440 rows of minute-by-minute data
- `results_daily_summary.csv` - Daily totals per dwelling
- `global_climate.csv` - Climate conditions
- `rerun_simulation.sh` - Script to reproduce this exact run
- `metadata.json` - Traceability information

---

## Directory Structure

```
crest/
├── README.md                    # This file
├── CLAUDE.md                    # Development instructions
├── API_REFERENCE.md             # Technical API documentation
├
├── excel/                       # Excel/VBA reference implementation
│   ├── original.xlsm                    # Base v2.3.3 model
│   ├── original/                        # Exports from original.xlsm
│   │   ├── *.cls, *.bas                 # VBA code
│   │   ├── Dwellings.csv                # Dwelling configurations
│   │   ├── Main_Sheet.csv               # Run parameters
│   │   └── *.csv                        # Data sheets (ActivityStats, etc.)
│   ├── original_100houses.xlsm          # Reference 100-house run
│   ├── original_100houses/              # Exports from 100-house model
│   ├── monte_carlo_base.xlsm            # 5 varies test houses to use for monte carlo
│   ├── monte_carlo_base/                # export from the monte carlo base
│   ├── lcg_fixed.xlsm                   # LCG version with bug fixes
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
│   ├── run_100_houses.py        # Convenience: run 100-house validation
│   ├── extract_settings.py      # Extract settings from Excel Main Sheet
│   ├── export_excel.py          # Export VBA + CSV from .xlsm
│   ├── monte_carlo_run.py       # Run N Monte Carlo iterations
│   ├── monte_carlo_compare.py   # IQR validation (Objective #2)
│   ├── rng_validation_run.py    # Run with LCG logging
│   ├── rng_log_compare.py       # RNG sequence comparison (Objective #1)
│   ├── check_types.sh           # Run mypy type checking
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
   - Use seed `12345`
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

2. **Manually run Excel 20 times:**
   - Use `excel/monte_carlo_base.xlsm`
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

