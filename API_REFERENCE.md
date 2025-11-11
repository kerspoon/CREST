# CREST API Reference

## crest_simulate.py
**CLI simulation script.** Orchestrates GlobalClimate, Dwelling, and ResultsWriter. Loads activity stats, assigns stochastic/manual dwelling parameters, runs simulations, outputs daily summaries (17 metrics matching VBA).

### CLI Arguments
- `--num-dwellings INT` (default: 1) - Number of dwellings
- `--residents INT` (optional) - Fixed residents for single dwelling, else stochastic
- `--weekend` - Weekend day (default: weekday)
- `--day INT` (default: 15) - Day of month
- `--month INT` (default: 6) - Month (1-12)
- `--seed INT` (optional) - Random seed for reproducibility
- `--data-dir PATH` (optional) - Data directory (default: ./data)
- `--output-dir PATH` (optional) - Save results to directory
- `--save-detailed` - Save minute-level CSV (large files)
- `--country {UK,India}` (default: UK) - Appliance ownership, water temp
- `--city {England,N Delhi,Mumbai,Bengaluru,Chennai,Kolkata,Itanagar}` (default: England) - Temperature profiles
- `--urban-rural {Urban,Rural}` (default: Urban) - Appliance ownership

### Functions
- `load_activity_statistics(data_loader) -> dict` - Load 72 activity profiles from CSV rows 30-101
- `assign_dwelling_parameters(data_loader, dwelling_index, rng) -> DwellingConfig` - Stochastic inverse transform sampling for building/heating/PV/solar thermal/cooling
- `main()` - Entry point

---

## crest/simulation/config.py
**Physical constants and configuration enums.** Defines Country, City, UrbanRural enums; thermal parameters (specific heat, thermostat deadbands, occupant gains); simulation resolution constants.

### Enums
- `Country(Enum)`: UK, INDIA
- `City(Enum)`: ENGLAND, N_DELHI, MUMBAI, BENGALURU, CHENNAI, KOLKATA, ITANAGAR
- `UrbanRural(Enum)`: URBAN, RURAL

### Constants
- `TIMESTEPS_PER_DAY_1MIN = 1440` - Thermal loop resolution
- `TIMESTEPS_PER_DAY_10MIN = 144` - Occupancy resolution
- `THERMAL_TIMESTEP_SECONDS = 60` - Euler integration timestep
- `OCCUPANT_THERMAL_GAIN_ACTIVE = 147` W
- `OCCUPANT_THERMAL_GAIN_DORMANT = 84` W
- `THERMOSTAT_DEADBAND_SPACE = 2.0` °C
- `THERMOSTAT_DEADBAND_WATER = 5.0` °C
- `THERMOSTAT_DEADBAND_EMITTER = 5.0` °C
- `COLD_WATER_TEMPERATURE_BY_COUNTRY = {UK: 10.0, INDIA: 20.0}` °C
- `SPECIFIC_HEAT_CAPACITY_WATER = 4200.0` J/kg/K
- `MAX_BULBS_PER_DWELLING = 60`
- `MAX_APPLIANCE_TYPES = 31`

---

## crest/simulation/dwelling.py
**Main dwelling orchestrator.** Creates and wires all subsystems (occupancy, building, heating, controls, appliances, lighting, PV, solar thermal, cooling). Runs simulation in correct execution order: occupancy → pre-simulation (water, appliances, lighting, PV output) → thermal loop (controls, heating, cooling, building physics) → post-simulation (totals, net demand).

### Classes
- `DwellingConfig(dataclass)`: dwelling_index, num_residents, building_index, heating_system_index, country, urban_rural, cooling_system_index, pv_system_index, solar_thermal_index, is_weekend
- `Dwelling`: Main coordinator class

### Dwelling Methods
- `__init__(config, global_climate, data_loader, activity_statistics, rng)`
- `run_simulation()` - Execute full day simulation
- `get_total_electricity_demand(timestep: int) -> float` - Total demand (W) at 1-based timestep
- `get_daily_totals() -> dict` - 15 metrics: occupancy, lighting/appliances kWh, PV output, net demand, self-consumption, hot water L, indoor temp, thermal energy, gas m³, setpoints
- `write_dwelling_index(current_date, dwelling_index_row_offset) -> List[tuple]` - Generate (dwelling_index, date, time_str) for 1440 timesteps

---

## crest/core/occupancy.py
**4-state Markov chain occupancy model (10-min resolution).** States: at_home × active (0-residents each). Uses UK Time Use Survey TPMs. Generates active_occupancy[144], thermal_gains[144] (147W active, 84W dormant).

### Classes
- `OccupancyConfig(dataclass)`: num_residents (1-6), is_weekend, dwelling_index
- `Occupancy`: Markov chain model

### Occupancy Methods
- `__init__(config, data_loader, rng)`
- `run_simulation()` - Generate day of occupancy states
- `get_mean_active_occupancy() -> float` - Mean active occupants
- `get_probability_actively_occupied() -> float` - Proportion of time active
- `get_thermal_gains(timestep: int) -> float` - Thermal gains (W) at 10-min timestep (1-144)

**Public Arrays**: `active_occupancy[144]`, `combined_states[144]` (strings like "11"), `thermal_gains[144]`

---

## crest/core/climate.py
**Global and local climate models (1-min resolution).** GlobalClimate generates clearness index (Markov chain, 101 bins), solar irradiance (solar geometry), outdoor temperature (ARMA + diurnal). LocalClimate wraps per-dwelling access.

### Classes
- `ClimateConfig(dataclass)`: day_of_month, month_of_year, city, latitude, longitude, meridian, use_daylight_saving
- `GlobalClimate`: Shared climate for all dwellings
- `LocalClimate`: Per-dwelling wrapper

### GlobalClimate Methods
- `__init__(config, data_loader, rng)`
- `simulate_clearness_index()` - Markov chain cloud cover
- `calculate_clear_sky_irradiance()` - Solar geometry G_o_clearsky[1440]
- `calculate_global_irradiance()` - Apply clearness index G_o[1440]
- `_generate_daily_temperatures()` - ARMA + diurnal profile theta_o[1440]
- `run_all()` - Execute all climate sub-models

**Public Arrays**: `clearness_index[1440]`, `g_o_clearsky[1440]`, `g_o[1440]`, `theta_o[1440]`

### LocalClimate Methods
- `__init__(global_climate, dwelling_index)`
- `get_irradiance(timestep: int) -> float` - W/m² (0-based index)
- `get_temperature(timestep: int) -> float` - °C (0-based index)

---

## crest/core/building.py
**5-node RC thermal network (1-min resolution).** Nodes: theta_b (external building), theta_i (internal air), theta_em (heating emitters), theta_cool (cooling emitters), theta_cyl (hot water cylinder). Solves coupled ODEs with explicit Euler (60s timestep). Calculates passive solar gains, casual gains, ventilation losses, cylinder losses.

### Classes
- `BuildingConfig(dataclass)`: building_index, heating_system_index, dwelling_index, run_number
- `Building`: Low-order thermal model

### Building Methods
- `__init__(config, data_loader)`
- `set_local_climate(local_climate)` - Wire climate dependency
- `set_occupancy(occupancy)` - Wire occupancy for casual gains
- `set_heating_system(heating_system)` - Wire heating
- `set_hot_water(hot_water)` - Wire for H_demand variable transfer coefficient
- `set_solar_thermal(solar_thermal)` - Wire solar thermal input
- `set_cooling_system(cooling_system)` - Wire cooling
- `set_appliances(appliances)` - Wire appliance heat gains
- `set_lighting(lighting)` - Wire lighting heat gains
- `set_heating_controls(heating_controls)` - Wire thermostat feedback
- `initialize_temperatures(outdoor_temp, random_gen)` - Set initial node temps
- `calculate_temperature_change(timestep: int)` - Euler step for all 5 nodes
- `get_mean_theta_i() -> float` - Mean internal temp °C
- `get_required_heating() -> float` - Heating demand (W) for current timestep
- `get_required_cooling() -> float` - Cooling demand (W) for current timestep

**Public Arrays**: `theta_b[1440]`, `theta_i[1440]`, `theta_em[1440]`, `theta_cool[1440]`, `theta_cyl[1440]`, `phi_s[1440]` (passive solar), `phi_c[1440]` (casual gains)

**Thermal Parameters**: `h_ob`, `h_bi`, `h_v`, `h_em`, `h_cool` (W/K), `c_b`, `c_i`, `c_em`, `c_cool` (J/K), `a_s` (m²), `h_loss` (W/K)

---

## crest/core/heating.py
**Primary heating system (boilers, electric).** Allocates heat to space/water based on HeatingControls signals. Calculates fuel flow (gas m³/min, electric kW), pump/standby electricity. Types: 1=regular, 2=combi, 4=no heating, 5=electric water heater.

### Classes
- `HeatingSystemConfig(dataclass)`: heating_system_index, dwelling_index, run_number
- `HeatingSystem`: Heat distribution model

### HeatingSystem Methods
- `__init__(config, data_loader)`
- `set_heating_controls(heating_controls)` - Wire control signals
- `set_building(building)` - Wire building temperatures
- `calculate_heat_output(timestep: int)` - Allocate phi_h_max to space/water
- `get_heating_system_power_demand(timestep: int) -> float` - Pump/standby electricity (W)
- `get_daily_thermal_energy_space() -> float` - W·min
- `get_daily_thermal_energy_water() -> float` - W·min
- `get_daily_fuel_consumption() -> float` - m³ for gas

**Public Arrays**: `phi_h_output[1440]` (total W), `phi_h_space[1440]` (W), `phi_h_water[1440]` (W), `m_fuel[1440]` (m³/min), `heating_electricity[1440]` (W)

**Parameters**: `phi_h_max` (W), `p_standby` (W), `p_pump` (W), `eta_h` (efficiency)

---

## crest/core/controls.py
**Hysteresis thermostats + timer schedules.** Controls space heating, hot water, cooling with deadbands (±2°C space, ±5°C water/emitter). Timer uses 30-min Markov chain expanded to 1-min with ±15min random shift. Combi boilers (type 2) have simplified hot water logic.

### Classes
- `HeatingControlsConfig(dataclass)`: dwelling_index, building_index, heating_system_index, cooling_system_index, is_weekend, run_number
- `HeatingControls`: Thermostat and timer logic

### HeatingControls Methods
- `__init__(config, data_loader, rng)`
- `set_building(building)` - Wire temperature feedback
- `set_hot_water(hot_water)` - Wire hot water demand signal
- `initialize_thermostat_states(theta_i, theta_cyl, theta_em, theta_cool)` - Initial states
- `calculate_control_states(timestep: int)` - Update all thermostats and timer
- `get_space_thermostat_setpoint() -> float` - °C setpoint

**Public Arrays**: `space_heating_thermostat[1440]` (bool), `space_heating_timer[1440]` (bool), `hot_water_thermostat[1440]` (bool), `hot_water_timer[1440]` (bool), `cooling_thermostat[1440]` (bool), `cooling_timer[1440]` (bool), `emitter_thermostat[1440]` (bool), `cooler_emitter_thermostat[1440]` (bool)

---

## crest/core/water.py
**Hot water demand model (4 fixtures).** Stochastic switching for basin, sink, shower, bath using activity profiles. Draws volumes from empirical distributions (151 bins). Calculates flow rate (L/min), energy demand, cylinder thermal dynamics.

### Classes
- `WaterFixtureSpec(dataclass)`: name, prob_switch_on, mean_flow, use_profile, restart_delay, volume_column
- `HotWaterConfig(dataclass)`: dwelling_index, heating_system_index, num_residents, country, run_number
- `HotWater`: Fixture event model

### HotWater Methods
- `__init__(config, data_loader, activity_statistics, is_weekend, rng)`
- `set_occupancy(occupancy)` - Wire occupancy dependency
- `run_simulation()` - Generate day of fixture events
- `get_daily_hot_water_volume() -> float` - Total litres
- `get_H_demand(timestep: int) -> float` - Variable transfer coefficient H_bi to cylinder (W/K)

**Public Arrays**: `hot_water_demand[1440]` (L/min), `hot_water_energy[1440]` (W)

**Fixtures**: Basin (7L, 0.5 L/min, Act_WashDress), Sink (10L, 5 L/min, Act_Cooking), Shower (42L, 7 L/min, Act_Shower), Bath (50L, 12 L/min, Act_Bath)

---

## crest/core/appliances.py
**31 appliance types with activity-based switching.** Stochastic on/off events using time-use profiles. Monte Carlo power variation (±30%). Ownership determined by country + urban/rural. Calculates electrical demand and thermal gains (fraction of power).

### Classes
- `ApplianceSpec(dataclass)`: name, rated_power, standby_power, cycle_length, restart_delay, ownership_prob, use_profile, prob_switch_on, heat_gains_ratio
- `AppliancesConfig(dataclass)`: dwelling_index, country, urban_rural, run_number
- `Appliances`: Multi-appliance model

### Appliances Methods
- `__init__(config, data_loader, activity_statistics, is_weekend, rng)`
- `set_occupancy(occupancy)` - Wire occupancy dependency
- `run_simulation()` - Generate day of appliance events
- `calculate_total_demand()` - Sum across appliances + heating/cooling electricity
- `get_total_demand(timestep: int) -> float` - Total demand (W) at 1-based timestep
- `get_daily_energy() -> float` - W·min

**Public Arrays**: `appliance_demands[1440, 31]` (W per appliance), `total_demand[1440]` (W), `thermal_gains[1440]` (W)

**Appliances**: Lighting_A (TV, cooking, wash-dress, iron), Lighting_B (active occupancy), Cold (fridge, freezer, fridge-freezer), Wet (washing machine, dishwasher, dryer), Consumer Electronics (TV, Set-top box, PC, Laptop, Phone), Cooking (Kettle, Microwave, Hob, Oven), Others (Vacuum, Answer phone)

---

## crest/core/lighting.py
**Irradiance-based lighting (up to 60 bulbs).** Switching probability inversely proportional to G_o (lights on when dark). Requires active occupancy. Bulbs assigned stochastic power ratings (20-100W) and relative use (0.25-1.5). 100% thermal conversion.

### Classes
- `LightingConfig(dataclass)`: dwelling_index, country, run_number
- `Lighting`: Multi-bulb model

### Lighting Methods
- `__init__(config, data_loader, rng)`
- `set_occupancy(occupancy)` - Wire occupancy dependency
- `set_local_climate(local_climate)` - Wire irradiance for switching
- `run_simulation()` - Generate day of lighting events
- `get_total_demand(timestep: int) -> float` - Total demand (W) at 1-based timestep
- `get_daily_energy() -> float` - W·min

**Public Arrays**: `bulb_demands[1440, 60]` (W per bulb), `total_demand[1440]` (W), `thermal_gains[1440]` (W)

**Parameters**: `num_bulbs` (random 10-60), bulb ratings (random 20-100W), `irradiance_threshold` (60 W/m²), `prob_switch_on` (0.2), `prob_switch_off` (varies with irradiance)

---

## crest/core/pv.py
**Photovoltaic system model.** Calculates tilted irradiance (direct + diffuse + reflected) using SolarGeometry. Power: P_pv = G_i × A_array × eta_pv. Computes net demand (appliances + lighting - PV) and self-consumption (min of generation and demand).

### Classes
- `PVSystem`: Solar PV model

### PVSystem Methods
- `__init__(data_loader, random_gen)`
- `initialize(dwelling_index, run_number, climate, appliances, lighting, pv_system_index, latitude, longitude, meridian, day_of_year, month, day)` - Set config and references
- `calculate_pv_output()` - Pre-simulation: compute P_pv[1440] from irradiance
- `calculate_net_demand()` - Post-simulation: P_net = demand - P_pv
- `calculate_self_consumption()` - Post-simulation: P_self = min(P_pv, demand)
- `get_pv_output(timestep: int) -> float` - W at 1-based timestep
- `get_daily_sum_pv_output() -> float` - W·min
- `get_daily_sum_net_demand() -> float` - W·min
- `get_daily_sum_self_consumption() -> float` - W·min

**Public Arrays**: `G_i[1440]` (W/m²), `P_pv[1440]` (W), `P_net[1440]` (W), `P_self[1440]` (W)

**Parameters**: `A_array` (m²), `eta_pv` (efficiency), `slope` (tilt °), `azimuth` (° from South)

---

## crest/core/solar_thermal.py
**Solar thermal collector model.** Calculates collector temperature evolution (Euler integration, 60s timestep). Hysteresis pump control (on when collector > cylinder + 8°C, off when < cylinder + 4°C). Heat transfer: phi_s = m_pump × c_p × (theta_collector - theta_cyl) when pump on.

### Classes
- `SolarThermal`: Solar thermal model

### SolarThermal Methods
- `__init__(data_loader, random_gen)`
- `initialize(dwelling_index, run_number, climate, building, solar_thermal_index, latitude, longitude, meridian, day_of_year, month, day)` - Set config and references
- `calculate_solar_thermal_output(timestep: int)` - Thermal loop: update collector temp, pump state, heat transfer
- `get_P_pumpsolar(timestep: int) -> float` - Pump electricity (W) at 1-based timestep
- `get_daily_sum_phi_s() -> float` - W·min

**Public Arrays**: `theta_collector[1440]` (°C), `phi_s[1440]` (W to cylinder), `P_pump_solar_array[1440]` (W), `solar_thermal_on_off[1440]` (bool)

**Parameters**: `N` (number of collectors), `A_coll_aperture` (m²), `eta_zero`, `k1`, `k2` (efficiency curve), `m_pump` (kg/s), `P_pump_solar` (W), `C_collector` (J/K), `slope` (°), `azimuth` (°)

---

## crest/core/cooling.py
**Space cooling system (fans, air coolers, AC).** Simple on/off based on thermostat + timer + emitter signals from HeatingControls. Capacity limiting: phi_h_cooling = min(0, max(phi_h_cool, required_cooling)). Types: 1=none, 2=fans (20W), 3=air cooler (1000W cooling, 100W elec, COP=10), 4=AC (5000W cooling, 1250W elec, COP=4).

### Classes
- `CoolingSystem`: Space cooling model

### CoolingSystem Methods
- `__init__(data_loader, random_gen)`
- `initialize(dwelling_index, run_number, controls, building, cooling_system_index)` - Set config and references
- `calculate_cooling_output(timestep: int)` - Thermal loop: compute cooling delivered
- `get_cooling_system_power_demand(timestep: int) -> float` - Electricity (W) at 1-based timestep

**Public Arrays**: `phi_h_cooling[1440]` (W, negative), `phi_cooling[1440]` (W electricity)

**Parameters**: `phi_h_cool` (max cooling W, negative), `P_standby_cool` (W), `P_pump_cool` (W), `eta_h_cool` (COP)

---

## crest/core/solar.py
**Solar geometry calculations for PV and solar thermal.** Computes solar altitude, azimuth, incidence angle, direct/diffuse/reflected radiation on tilted surfaces. Handles UK daylight saving (days 87-304). Corrects VBA bug in clsSolarThermal (Tan(Declination)/Tan(Declination) → Tan(Declination)/Tan(Latitude)).

### Classes
- `SolarGeometry`: Solar position and radiation calculator

### SolarGeometry Methods
- `__init__(day_of_year, latitude, longitude, meridian, enable_daylight_saving)` - Pre-compute invariants (B, equation_of_time, time_correction_factor, sky_diffuse_factor, optical_depth)
- `calculate_incident_radiation(minute_of_day, g_o, clearness_index, slope, azimuth) -> Tuple[float, float, float, float]` - Returns (G_i_total, G_i_direct, G_i_diffuse, G_i_reflected) W/m²
- `get_solar_altitude(minute_of_day) -> float` - Solar altitude angle (°)
- `get_solar_azimuth(minute_of_day) -> float` - Solar azimuth angle (°)

**Constants**: Ground reflectance = 0.2

---

## crest/data/loader.py
**CSV data loader with caching.** Loads all 37 CSV files: 12 occupancy TPMs, 25 configuration/spec files, clearness TPM, heating controls TPM. Handles VBA-style 1-based indexing, header skipping. Provides proportion arrays for stochastic parameter assignment.

### Classes
- `CRESTDataLoader`: Central data loader

### CRESTDataLoader Methods
- `__init__(data_dir: Optional[Path])` - Default: ./data
- `load_occupancy_tpm(num_residents: int, is_weekend: bool) -> DataFrame` - TPM for 1-6 residents, wd/we
- `load_starting_states() -> DataFrame` - Initial occupancy probabilities
- `load_24hr_occupancy() -> DataFrame` - 24hr occupancy correction
- `load_activity_stats() -> DataFrame` - 72 activity profiles
- `load_appliances_and_fixtures() -> DataFrame` - 31 appliances + 4 water fixtures
- `load_buildings() -> DataFrame` - Building thermal parameters
- `load_primary_heating_systems() -> DataFrame` - Heating system specs
- `load_cooling_systems() -> DataFrame` - Cooling system specs
- `load_heating_controls() -> DataFrame` - Setpoint distributions
- `load_heating_controls_tpm() -> DataFrame` - Timer Markov chain
- `load_global_climate() -> DataFrame` - Historical climate
- `load_irradiance() -> DataFrame` - Solar irradiance lookup
- `load_clearness_index_tpm() -> DataFrame` - 101×101 Markov chain
- `load_climate_data_and_cooling_tech() -> DataFrame` - Regional climate
- `load_lighting_config() -> DataFrame` - Lighting parameters
- `load_bulbs() -> DataFrame` - Example bulb configs
- `load_water_usage() -> DataFrame` - Volume distributions (151 bins)
- `load_pv_systems() -> DataFrame` - PV specs
- `load_solar_thermal_systems() -> DataFrame` - Solar thermal specs
- `load_dwellings() -> DataFrame` - Dwelling assignments
- `load_resident_proportions() -> ndarray` - 5 probabilities (1-5 residents)
- `load_building_proportions() -> ndarray` - Building type probabilities
- `load_heating_proportions() -> ndarray` - Heating system probabilities
- `load_pv_proportions() -> ndarray` - PV system probabilities
- `load_solar_thermal_proportions() -> ndarray` - Solar thermal probabilities
- `load_cooling_proportions() -> ndarray` - Cooling system probabilities
- `get_heating_type(heating_index: int) -> int` - Returns type code (1=regular, 2=combi, 4=no heating, 5=electric)
- `clear_cache()` - Free memory
- `preload_all()` - Load all 37 files

**Singleton**: `get_default_loader(data_dir) -> CRESTDataLoader`

---

## crest/output/writer.py
**CSV results writer.** Writes minute-level data (23 columns × 1440 rows per dwelling), daily summaries (8 metrics), and global climate (4 columns × 1440 rows). Format matches Excel model "Results - disaggregated" sheet for validation.

### Classes
- `OutputConfig(dataclass)`: output_dir, save_minute_data, save_daily_summary, save_global_climate
- `ResultsWriter`: CSV writer with buffering

### ResultsWriter Methods
- `__init__(config: OutputConfig)` - Creates output_dir, opens CSV files
- `write_minute_data(dwelling_idx: int, dwelling: Dwelling)` - Write 1440 rows to results_minute_level.csv
- `write_daily_summary(dwelling_idx: int, dwelling: Dwelling)` - Write 1 row to results_daily_summary.csv
- `write_global_climate(global_climate: GlobalClimate)` - Write 1440 rows to global_climate.csv
- `close()` - Close all files
- Context manager: `with ResultsWriter(config) as writer:`

**Minute CSV Columns**: Dwelling, Minute, At_Home, Active, Lighting_W, Appliances_W, Total_Electricity_W, Outdoor_Temp_C, Irradiance_Wm2, Internal_Temp_C, External_Building_Temp_C, Hot_Water_Demand_L_per_min, Cylinder_Temp_C, Emitter_Temp_C, Cooling_Emitter_Temp_C, Total_Heat_Output_W, Space_Heating_W, Water_Heating_W, Gas_Consumption_m3_per_min, Passive_Solar_Gains_W, Casual_Gains_W, Heating_Electricity_W, PV_Output_W, Cooling_Electricity_W

**Daily Summary Columns**: Dwelling, Total_Electricity_kWh, Total_Gas_m3, Total_Hot_Water_L, Peak_Electricity_W, Peak_Heating_W, Mean_Internal_Temp_C, Mean_Occupancy_At_Home, Mean_Occupancy_Active

**Climate CSV Columns**: Minute, Clearness_Index, Clear_Sky_Irradiance_Wm2, Global_Irradiance_Wm2, Outdoor_Temperature_C

---

## crest/utils/markov.py
**Markov chain utilities.** Inverse transform sampling, probability normalization, 24hr occupancy modifications, TPM row indexing helpers.

### Functions
- `select_next_state(transition_probabilities: ndarray, rng_value: float) -> int` - Inverse transform method, returns 0-based state index
- `normalize_probabilities(probabilities: ndarray, zero_threshold: float) -> ndarray` - Normalize to sum=1, handle dead-end states
- `modify_24hr_occupancy_probabilities(probabilities: ndarray, num_residents: int) -> ndarray` - Zero out unoccupied state transitions
- `calculate_tpm_row_index(timestep: int, current_state_str: str, num_residents: int, possible_states: int, vba_compatible: bool) -> int` - VBA row formula for TPM lookup
- `extract_tpm_row(tpm: ndarray, row_index: int, col_offset: int, vba_compatible: bool) -> ndarray` - Extract probabilities from TPM row
- `get_state_labels_from_tpm(tpm: ndarray, col_offset: int, vba_compatible: bool) -> ndarray` - Extract state labels from row 0

---

## crest/utils/random.py
**NumPy random number generation wrapper.** Provides global seeded RNG for reproducibility. Replaces VBA Rnd() function.

### Classes
- `RandomGenerator`: Wrapper for np.random.Generator

### RandomGenerator Methods
- `__init__(seed: Optional[int])` - Create RNG with seed
- `random() -> float` - Uniform [0, 1), equivalent to VBA Rnd()
- `randint(low: int, high: int) -> int` - [low, high)
- `choice(a, p) -> Any` - Sample from array with probabilities
- `normal(loc, scale, size) -> float | ndarray` - Gaussian
- `exponential(scale, size) -> float | ndarray` - Exponential
- `uniform(low, high, size) -> float | ndarray` - Uniform

### Global Functions
- `set_seed(seed: Optional[int])` - Initialize global RNG
- `get_rng() -> RandomGenerator` - Get global RNG instance
- `random()`, `randint()`, `choice()`, `normal()`, `exponential()`, `uniform()` - Convenience wrappers using global RNG
