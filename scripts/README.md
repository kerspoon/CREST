# Validation Scripts

Utility scripts for validating Python implementation against Excel VBA baseline.

## Scripts

### `parse_excel_configs.py`
Extracts dwelling configurations from Excel file to JSON.
- Reads `CREST_Demand_Model_v2.3.3__100_houses.xlsm`
- Outputs `configs/excel_100houses.json`
- Allows running Python with identical dwelling setups as Excel

### `read_excel_100houses.py`
Reads baseline results from Excel simulation.
- Extracts minute-level and daily results
- Formats for comparison with Python output

### `compare_results.py`
Compares Python simulation results against Excel baseline.
- Checks electricity, gas, hot water consumption
- Reports differences and statistical summaries
- Validates model accuracy

### `clean_excel_data.py`
Data cleaning utility for Excel outputs.

### `compare_rng_logs.py`
Compares Excel VBA and Python random number generator call sequences.
- Validates that both implementations call RNG in identical order
- Compares random values with configurable tolerance
- Reports exact location where sequences diverge
- Usage: `python compare_rng_logs.py excel_rnd_calls.txt debug_random_calls.log`
- Options:
  - `--tolerance/-t`: Set numerical tolerance (default: 1e-10)
  - `--verbose/-v`: Show all comparisons, not just mismatches
  - `--max-diff/-m`: Maximum differences to display (default: 50)

**Log Formats:**
- **Excel**: Alternating lines with location and value
  ```
  1: clsGlobalClimate:123 - transition steps
  2: r 0.252345174783841
  ```
- **Python**: Single line per call (after header)
  ```
  Call #   1: 0.25234517478384078  @ climate.py:simulate_clearness_index:112
  ```

## Workflow

1. Run Excel simulation → generates baseline results
2. Extract configs: `python parse_excel_configs.py`
3. Run Python simulation: `python crest_simulate.py --num-dwellings 100 --seed 42 --output-dir output`
4. Compare results: `python compare_results.py`

## Future

Once all audits are complete, these scripts will validate that Python produces statistically equivalent results to Excel VBA.
