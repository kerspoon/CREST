# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CREST Demand Model: High-resolution (1-minute) stochastic integrated thermal-electrical domestic energy demand simulator. Python port of the original Excel/VBA model from Loughborough University, maintaining statistical equivalence while providing better modularity and extensibility.

## Running Simulations

```bash
# Basic single dwelling simulation
python crest_simulate.py

# Multiple dwellings with reproducible seed
python crest_simulate.py --num-dwellings 10 --seed 42

# Weekend simulation for specific date
python crest_simulate.py --weekend --day 15 --month 6 --residents 3
```

Full CLI options: `--num-dwellings`, `--residents` (1-6), `--weekend`, `--day`, `--month`, `--seed`, `--data-dir`

## Architecture Overview

### Simulation Execution Flow

The model follows a strict execution order where components depend on outputs from previous stages:

1. **GlobalClimate** (`core/climate.py`) - Runs first, generates shared weather data for all dwellings
   - Clearness index (Markov chain, 1-min resolution)
   - Solar irradiance (solar geometry calculations)
   - Outdoor temperature (ARMA model + diurnal profile)

2. **Occupancy** (`core/occupancy.py`) - 4-state Markov chain (at home/away, active/asleep)
   - 10-minute resolution, uses TPM CSVs based on number of residents
   - Generates thermal gains (147W active, 84W dormant)

3. **Pre-simulation Phase** - These run before the thermal loop:
   - **HotWater** (`core/water.py`) - Stochastic fixture events (basin, sink, shower, bath)
   - **Appliances** (`core/appliances.py`) - 31 appliance types with activity-based switching
   - **Lighting** (`core/lighting.py`) - Irradiance-based switching for up to 60 bulbs
   - **Renewables** (`core/renewables.py`) - PV and solar thermal output calculations

4. **Thermal Loop** (minute-by-minute, 1-1440) - The core building physics simulation:
   - **HeatingControls** (`core/controls.py`) - Hysteresis thermostats + timer states
   - **HeatingSystem** (`core/heating.py`) - Allocates heat to space/water
   - **CoolingSystem** - Provides cooling if needed
   - **Building** (`core/building.py`) - Updates temperatures using Euler method

### Building Thermal Model (5-Node RC Network)

`Building.calculate_temperature_change()` solves coupled differential equations using explicit Euler method (60-second timesteps):

- **theta_b**: External building node temperature
- **theta_i**: Internal air temperature (what occupants feel)
- **theta_em**: Heating emitter temperature (radiators)
- **theta_cool**: Cooling emitter temperature
- **theta_cyl**: Hot water cylinder temperature

Heat flows calculated from:
- Passive solar gains (irradiance × aperture area)
- Casual gains (occupants + appliances + lighting)
- Heating system output (space + hot water)
- Ventilation losses
- Cylinder standing losses

### Dwelling Orchestration

`Dwelling` (`simulation/dwelling.py`) is the main coordinator that:
1. Creates all subsystem instances with proper configuration
2. Wires dependencies (e.g., Building needs Occupancy, HeatingSystem needs HeatingControls)
3. Runs simulation in correct order
4. Provides aggregated results (total electricity demand = appliances + lighting + heating pump - PV)

Critical dependency pattern: Components store references to other components (set via `set_*` methods) rather than passing data through function calls. This allows the minute-by-minute thermal loop to access current states efficiently.

### Data Loading Architecture

`CRESTDataLoader` (`data/loader.py`) handles 37 CSV files:
- **Transition Probability Matrices (TPMs)**: 12 occupancy files (tpm1-6, wd/we), ClearnessIndexTPM, HeatingControlsTPM
- **Configuration**: Buildings, Heating/Cooling systems, Appliances, Lighting
- **Empirical Distributions**: WaterUsage (151 bins), ActivityStats, Starting_states

CSVs use 1-based indexing with headers. TPMs have state labels in row 0, data starts row 1, probabilities in columns 2+.

### Stochastic Processes

All randomness uses `RandomGenerator` (`utils/random.py`) wrapping NumPy's `Generator`:
- **Seeding**: Call `random.set_seed(N)` before simulation for reproducibility
- **Markov chains**: Use `markov.select_next_state()` helper for inverse transform sampling
- **Hysteresis thermostats**: Deadband logic prevents rapid cycling (±2°C space, ±5°C water/emitter)

## Key Implementation Details

### Indexing Conventions

**Mixed indexing throughout codebase:**
- **0-based**: Internal arrays (occupancy.active_occupancy[0:144], building.theta_i[0:1440])
- **1-based**: External API (get_temperature(timestep) expects timestep 1-1440)
- **CSV TPMs**: Row 0 = state labels, Row 1+ = data, Columns 0-1 = labels, Columns 2+ = probabilities

When porting VBA code, note VBA uses 1-based arrays. Convert by subtracting 1 from indices.

### Time Resolution

- **Occupancy**: 10-minute (144 steps), expanded to 1-minute via `np.repeat(arr, 10)`
- **Thermal calculations**: 1-minute (1440 steps), 60-second Euler timestep
- **Timer schedules**: 30-minute (48 steps) from Markov chain, expanded to 1-minute with random time shift (±15 min)

### Constants and Configuration

Physical constants in `simulation/config.py`:
- Water specific heat: 4200 J/kg/K
- Occupant gains: 147W active, 84W dormant
- Thermostat deadbands: 2°C space, 5°C water/emitter
- Cold water temp: 10°C (UK standard)

Modifying physics requires editing config.py. Dwelling-specific parameters (building type, heating system) come from CSV files.

### Circular Dependencies Pattern

The model has intentional circular references to enable coupled physics:
- Building ↔ HeatingSystem ↔ HeatingControls
- Building references HotWater for H_demand (variable thermal transfer coefficient)
- HeatingControls references Building for temperature feedback

These are handled by:
1. Creating all objects first
2. Calling `set_*()` methods to wire dependencies
3. Running simulation loop where each component queries others as needed

Do NOT try to eliminate these - they reflect physical coupling in the system.

## Extending the Model

### Adding a New Appliance Type

1. Add row to `data/AppliancesAndWaterFixtures.csv` with spec (power, cycle length, ownership, use profile)
2. `Appliances._load_appliance_specs()` auto-loads from CSV
3. No code changes needed unless adding new switching logic

### Adding a New Building Type

1. Add row to `data/Buildings.csv` with thermal parameters (H_ob, H_bi, C_b, C_i, etc.)
2. Reference by index in DwellingConfig.building_index
3. Thermal model automatically uses new parameters

### Modifying Thermal Model

To change thermal equations (e.g., add new node, different solver):
1. Update `Building.calculate_temperature_change()` differential equations
2. Add new temperature array if needed
3. Update `Building.initialize_temperatures()` for new state
4. Test against Excel validation data if maintaining equivalence is needed

### Adding Output Variables

Current `Dwelling.get_total_electricity_demand()` returns single value. To export time series:
1. Collect arrays from components (e.g., `building.theta_i`, `appliances.total_demand`)
2. Create results DataFrame
3. Add CSV export to `crest_simulate.py`

## Data Requirements

All 37 CSV files in `data/` directory are required. Missing files cause immediate failure on load.

Critical TPMs:
- `tpm{1-6}_{wd,we}.csv`: Occupancy transitions (different for 1-6 residents, weekday/weekend)
- `ClearnessIndexTPM.csv`: 101×101 matrix for cloud cover simulation
- `HeatingControlsTPM.csv`: Timer on/off transitions

If adding new regions/climates:
- Modify `GlobalClimate._generate_daily_temperatures()` to use different monthly temps
- Update `ClimateDataandCoolingTech.csv` with regional data
- May need new occupancy TPMs if behavior differs significantly from UK

## Performance Characteristics

- Single dwelling: ~2-5 seconds (dominated by Python overhead in loops)
- 100 dwellings: ~3-5 minutes (scales linearly)
- Memory: ~100-500 MB (1440 timesteps × components × dwellings)

Bottlenecks:
1. Thermal loop (1440 iterations × differential equations)
2. Stochastic switching checks (appliances, lighting, hot water)

Optimization opportunities:
- Numba JIT on `Building.calculate_temperature_change()`
- Vectorize appliance switching across multiple dwellings
- Pre-compute Markov chain transitions for batch runs

Do NOT optimize prematurely - model is already fast enough for typical use cases (≤100 dwellings).
