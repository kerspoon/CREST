# CREST Demand Model - Python Port

A high-resolution (1-minute) stochastic integrated thermal-electrical domestic energy demand simulator.

## Overview - PYCREST (Centre for Renewable Energy Systems Technology) Demand Model

A python port of a high-resolution (1-minute) stochastic integrated thermal-electrical domestic energy demand simulator originally developed in Excel VBA by researchers at Loughborough University (McKenna & Thomson,
  2016).

What it models:

  - Occupancy: 4-state Markov chain (at home/away × active/dormant)
  - Electrical demand: 31 appliance types, up to 60 light bulbs
  - Thermal demand: Building physics (5-node RC thermal network), gas boilers, hot water (4 fixtures)
  - Renewables: PV systems, solar thermal collectors
  - Cooling: Fans, air coolers, AC units
  - Climate: Stochastic weather (temperature, solar irradiance) with seasonal variability

Purpose:

  - Low-voltage network analysis (simulating aggregations of dwellings)
  - Urban energy systems modeling
  - Bottom-up activity-based approach captures appropriate demand diversity
  - Stochastic methods produce realistic statistical properties

---

### Conversion Goal

Convert the VBA Excel model to Python like-for-like - exact functionality match so outputs are identical (same inputs, same random seeds → same results, accounting for floating point differences).


### Data Files

37 CSV files extracted from Excel sheets in original/excel_data/:
  - 12 occupancy TPMs (6 resident counts × weekday/weekend)
  - 25 config/spec files (appliances, buildings, heating systems, PV, etc.)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python crest_simulate.py --help
```

# Type Checking

This project uses [mypy](https://mypy-lang.org/) for static type checking to catch interface bugs before runtime. 

```bash
  # Run type checker
  ./check_types.sh

  # or directly 
  venv/bin/mypy crest/core/building.py
```

later improvements TODO:

1. disallow_incomplete_defs = True
2. disallow_untyped_calls = True
3. disallow_untyped_defs = True
4. warn_return_any = True
5. .git/hooks/pre-commit


## Citation

If you use this model in research, please cite the original CREST model:

> Richardson, I., Thomson, M., Infield, D., & Clifford, C. (2010).
> Domestic electricity use: A high-resolution energy demand model.
> Energy and Buildings, 42(10), 1878-1887.

## License

This Python port maintains the original GNU GPL v3 license from the CREST model.

## Original Model

Original Excel/VBA model developed by:
- John Barton (J.P.Barton@lboro.ac.uk)
- Murray Thomson (M.Thomson@lboro.ac.uk)
- Centre for Renewable Energy Systems Technology (CREST)
- Loughborough University

## Python Port Version

Version: 1.0.0
Port completed: 2025
