# CREST Demand Model - Python Port

A high-resolution (1-minute) stochastic integrated thermal-electrical domestic energy demand simulator.

## Overview

The CREST Demand Model is a sophisticated simulation tool for modeling residential energy demand, originally developed by Loughborough University in Excel/VBA. This is a like-for-like Python port that maintains statistical equivalence with the original model.

### Key Features

- **High resolution**: 1-minute timestep thermal and electrical simulation
- **Stochastic modeling**: Markov chain-based occupancy and activity patterns
- **Integrated thermal-electrical**: Coupled building thermal model with appliances, lighting, and heating
- **Comprehensive systems**: Includes heating, hot water, appliances, lighting, PV, solar thermal, and cooling
- **UK-calibrated**: Based on UK Time Use Survey 2000 and building stock data

### Model Components

**Core Stochastic Models:**
- Occupancy (4-state Markov chain)
- Global climate (irradiance, temperature)
- Local climate (dwelling-specific)

**Thermal Systems:**
- Building (5-node thermal capacitance network)
- Hot water (cylinder thermal model + 4 fixtures)
- Heating system (gas boilers, electric heating)
- Heating controls (hysteresis thermostats + timers)

**Electrical Demand:**
- Appliances (31 appliance types)
- Lighting (irradiance-based, 60 bulbs)

**Renewable Systems:**
- PV systems
- Solar thermal collectors
- Cooling systems

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python crest_simulate.py --help
```

### Requirements

- Python 3.9+
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- SciPy >= 1.10.0

## Usage

### Basic Simulation

Simulate a single dwelling with default parameters:

```bash
python crest_simulate.py
```

### Multiple Dwellings

Simulate 10 dwellings for statistical analysis:

```bash
python crest_simulate.py --num-dwellings 10 --residents 3
```

### Weekend Simulation

Simulate a weekend day:

```bash
python crest_simulate.py --weekend --day 15 --month 6
```

### Reproducible Results

Use a seed for reproducible results:

```bash
python crest_simulate.py --seed 42 --num-dwellings 5
```

### Command-Line Options

```
--num-dwellings N    Number of dwellings to simulate (default: 1)
--residents N        Number of residents per dwelling (1-6, default: 2)
--weekend            Simulate weekend day (default: weekday)
--day N              Day of month (default: 15)
--month N            Month of year (default: 6)
--seed N             Random seed for reproducibility
--data-dir PATH      Path to data directory (default: ./data)
```

## Project Structure

```
crest/
├── crest/                          # Main package
│   ├── core/                       # Model classes
│   │   ├── occupancy.py           # 4-state occupancy model
│   │   ├── climate.py             # Global/local climate
│   │   ├── building.py            # Thermal building model
│   │   ├── water.py               # Hot water demand
│   │   ├── heating.py             # Heating system
│   │   ├── controls.py            # Thermostats & timers
│   │   ├── appliances.py          # Appliances model
│   │   ├── lighting.py            # Lighting model
│   │   └── renewables.py          # PV, solar thermal, cooling
│   ├── data/
│   │   └── loader.py              # CSV data loader
│   ├── simulation/
│   │   ├── config.py              # Constants & parameters
│   │   └── dwelling.py            # Dwelling orchestrator
│   └── utils/
│       ├── random.py              # RNG utilities
│       └── markov.py              # Markov chain helpers
├── data/                           # CSV data files (37 files)
├── crest_simulate.py               # CLI entry point
├── requirements.txt
└── README.md
```

## Model Architecture

The simulation follows this execution order:

1. **Global Climate** generates shared irradiance and temperature
2. **Occupancy** creates activity patterns (10-minute resolution)
3. **Hot Water** generates stochastic fixture events
4. **Appliances** generates electrical demand based on activities
5. **Lighting** generates demand based on irradiance and occupancy
6. **Renewable Systems** calculate PV and solar thermal output
7. **Building Thermal Loop** (minute-by-minute):
   - Heating controls calculate thermostat/timer states
   - Heating system allocates heat to space/water
   - Cooling system provides cooling if needed
   - Building thermal model updates temperatures (Euler's method)

## Data Files

The model requires 37 CSV data files in the `data/` directory:

**Occupancy & Activity:**
- `tpm1_wd.csv` through `tpm6_we.csv` (12 occupancy TPMs)
- `Starting_states.csv`
- `24hr_occupancy.csv`
- `ActivityStats.csv`

**Buildings & Systems:**
- `Buildings.csv`
- `PrimaryHeatingSystems.csv`
- `CoolingSystems.csv`
- `HeatingControls.csv`
- `HeatingControlsTPM.csv`

**Climate:**
- `GlobalClimate.csv`
- `Irradiance.csv`
- `ClearnessIndexTPM.csv`
- `ClimateDataandCoolingTech.csv`

**Appliances & Fixtures:**
- `AppliancesAndWaterFixtures.csv`
- `WaterUsage.csv`
- `light_config.csv`
- `bulbs.csv`

**Renewables:**
- `PV_systems.csv`
- `SolarThermalSystems.csv`

**Configuration:**
- `Dwellings.csv`

## Technical Notes

### Thermal Model

The building thermal model uses a 5-node RC network:
- External building node
- Internal air node
- Heating emitters
- Cooling emitters
- Hot water cylinder

Solved using explicit Euler method with 60-second timesteps.

### Stochastic Processes

Random number generation uses NumPy's modern `Generator` interface for proper seeding and reproducibility. All Markov chains use transition probability matrices from empirical UK data.

### Validation

The Python port produces statistically equivalent results to the original Excel/VBA model:
- Same stochastic processes and distributions
- Same thermal model equations (Euler solver)
- Same activity profiles and switching logic
- Differences due to RNG implementation are expected

## Performance

- Single dwelling simulation: ~2-5 seconds
- 100 dwellings: ~3-5 minutes
- Memory usage: ~100-500 MB depending on number of dwellings

Performance optimizations (JIT compilation with Numba, vectorization) can be added if needed.

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
