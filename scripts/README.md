# Scripts Directory

Validation and utility scripts for the CREST Demand Model Python port.

## Validation Workflow Scripts

### Objective #1: RNG Call Sequence Matching

**`rng_validation_run.py`** - Run single iteration with LCG logging
- Uses portable LCG for exact Excel matching
- Logs every RNG call to `rng_calls.log`
- Creates: `output/rng_validation/python_5houses_YYYYMMDD_NN/`

**`rng_log_compare.py`** - Compare Excel vs Python RNG logs
- Compares call-by-call sequence
- Reports first divergence location
- Creates: `output/rng_validation/validation_pYYYYMMDD_NN_eYYYYMMDD_NN/`

### Objective #2: Statistical Distribution Validation

**`monte_carlo_run.py`** - Run N Monte Carlo iterations
- Default: 1000 iterations
- Auto-incrementing run numbers
- Creates: `output/monte_carlo/python_NNNNruns_YYYYMMDD_NN/`

**`monte_carlo_compare.py`** - IQR validation
- Tests 72,000+ data points
- Compares Excel samples against Python IQR
- Creates: `output/monte_carlo/validation_pYYYYMMDD_NN_eYYYYMMDD_NN/`

## Utility Scripts

**`check_types.sh`** - Run mypy type checker
**`vba_export.py`** - Export VBA code from .xlsm
**`csv_export.py`** - Export Excel sheets as CSV
**`utils.py`** - Shared helper functions

## Other Scripts

**`compare.py`** - Statistical comparison tools
**`validate_algorithm.py`** - Algorithm validation
**`create_test_config.py`** - Create test configurations
**`diagnose_*.py`** - Debugging utilities

See [detailed documentation](../README.md#validation-workflow) for full usage instructions.
