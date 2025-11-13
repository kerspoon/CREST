# CREST Demand Model - Python Port

A high-resolution (1-minute) stochastic integrated thermal-electrical domestic energy demand simulator, ported from the original Excel VBA implementation.

**Project Goal:** Achieve 100% feature parity with the Excel VBA model - identical outputs for identical inputs (same configurations + same random seeds → same results).

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

**Option 2: Use template script**

```bash
# Use the template script (edit settings at top of file)
bash scripts/run_python.sh

# Or create a custom configuration
cp scripts/run_python.sh my_simulation.sh
# Edit settings in my_simulation.sh
bash my_simulation.sh
```

**Option 3: Run Python directly**

```bash
# Run with default settings (auto-creates output/run_YYYYMMDD_NN/)
python python/main.py --config-file excel/original/Dwellings.csv --save-detailed

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

**Test Scope:** 72,000+ data points
- 5 houses × 20 Excel runs × 5 variables × 1440 minutes = 72,000 tests

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
   - Use `excel/original.xlsm` or `excel/lcg_fixed.xlsm`
   - Load same dwelling config (e.g., from `excel/original/Dwellings.csv`)
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

**Variables tested:**
- Electricity (W)
- Gas consumption (m³/min)
- Hot water demand (L/min)
- Indoor temperature (°C)
- Lighting (W)

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

# Run 100-house validation
python scripts/run_100_houses.py
```

**What it does:**
1. Exports VBA and CSV to `excel/{basename}/`
2. Reads run parameters from Main Sheet (day, month, country, seed, etc.)
3. Runs Python with those settings
4. Creates `output/run_YYYYMMDD_NN/` with results
5. Generates `rerun_simulation.sh` for reproducibility
6. Compares with Excel output (if Excel results exist)

**Skip comparison:**
```bash
python scripts/excel_run_and_compare.py excel/original.xlsm --no-compare
```

### Add New Excel File

To add a new Excel file to the workflow:

```bash
# 1. Copy your .xlsm file to excel/
cp my_model.xlsm excel/

# 2. Run it (exports happen automatically)
python scripts/excel_run_and_compare.py excel/my_model.xlsm
```

This creates `excel/my_model/` with all VBA code and CSV data.

### Re-run a Simulation

Every run creates a `rerun_simulation.sh` script:

```bash
# Re-run the exact same simulation
cd output/run_20250113_01/
bash rerun_simulation.sh
```

The script contains the exact command-line arguments used.

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

### API Documentation

See [`API_REFERENCE.md`](./API_REFERENCE.md) for:
- Detailed module documentation
- Function signatures
- VBA cross-references

### Exporting from Excel

```bash
# Export VBA code and CSV sheets from any Excel file
python scripts/export_excel.py excel/original.xlsm

# By default exports to excel/{basename}/
# You can specify output directory:
python scripts/export_excel.py excel/original.xlsm --output excel/my_export/

# Export only VBA or only CSV
python scripts/export_excel.py excel/original.xlsm --vba-only
python scripts/export_excel.py excel/original.xlsm --csv-only
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

All data files have been validated against the original Excel sheets.
