# Multi-Dwelling Orchestration Audit - Progress Tracker

**File**: `original/mdlThermalElectricalModel.bas` (1399 lines) → `crest_simulate.py`
**Status**: IN PROGRESS (2/9 phases complete)
**Started**: 2025-11-11
**Last Updated**: 2025-11-11

---

## Overview

This audit covers the remaining VBA orchestration code for multi-dwelling simulations. The per-dwelling logic (lines 282-488) was completed in the dwelling.py audit. This focuses on:
- Stochastic parameter assignment
- Database selection (UK vs India, year interpolation)
- Multi-dwelling aggregation
- Daily summary statistics
- Activity statistics loading

---

## Progress Summary

### ✅ COMPLETED (2/9 phases) - 22%

#### Phase 1: Activity Statistics Loading ✅
**File**: `crest_simulate.py` lines 27-89
**Status**: COMPLETE
**VBA Reference**: LoadActivityStatistics (mdlThermalElectricalModel.bas lines 761-801)

**What was done:**
- Replaced stub implementation that generated dummy data
- Now properly loads all 72 activity profiles from ActivityStats.csv
- Parses weekend flag, active occupant count, profile ID, and 144 modifiers
- Keys formatted as: `"{weekend}_{occupants}_{profile_id}"`
- Handles variable CSV row positions robustly

**Testing**: ✅ Verified 72 profiles loaded correctly with real data

---

#### Phase 3: CRESTDataLoader Extensions ✅
**File**: `crest/data/loader.py` lines 385-476
**Status**: COMPLETE
**VBA Reference**: Multiple proportion ranges across worksheets

**What was done:**
Added 7 new methods to CRESTDataLoader:

1. `load_resident_proportions()` - Returns array[5] for 1-5 residents
   - VBA: rPrNumberResidents range in ActivityStats
   - Reads pandas rows 8-12, column 2

2. `load_building_proportions()` - Returns array[N] for building types
   - VBA: rBuildingProportion range in Buildings worksheet
   - Reads column B, skips 4 header rows

3. `load_heating_proportions()` - Returns array[N] for heating systems
   - VBA: rPrimaryHeatingSystemProportion range
   - Reads column B, skips 4 header rows

4. `load_pv_proportions()` - Returns array[N] for PV systems
   - VBA: rPVProportion range
   - Index 0 = no PV, 1+ = PV types

5. `load_solar_thermal_proportions()` - Returns array[N] for solar thermal
   - VBA: rSolarThermalProportion range
   - Index 0 = no solar thermal, 1+ = system types

6. `load_cooling_proportions()` - Returns array[N] for cooling systems
   - VBA: rCoolingSystemProportion range
   - Index 0 = no cooling, 1+ = system types

7. `get_heating_type(heating_index)` - Returns heating system type code
   - Used to check if combi boiler (type 2)
   - Reads column E from PrimaryHeatingSystems

**Testing**: ✅ Verified resident proportions sum to 1.0, building proportions loaded

---

### ⏳ PENDING (7/9 phases) - 78%

#### Phase 2: Stochastic Parameter Assignment ⏳
**File**: `crest_simulate.py` (new function)
**Status**: NOT STARTED
**VBA Reference**: AssignDwellingParameters (lines 1130-1288)
**Estimated**: ~100 lines of code

**Plan:**
```python
def assign_dwelling_parameters(
    data_loader: CRESTDataLoader,
    dwelling_index: int,
    rng: np.random.Generator
) -> DwellingConfig:
    """
    Stochastically assign all parameters for one dwelling.

    VBA Reference: AssignDwellingParameters (mdlThermalElectricalModel.bas lines 1130-1288)
    Uses inverse transform sampling on cumulative probability distributions.
    """
    # Load all proportion arrays
    resident_props = data_loader.load_resident_proportions()
    building_props = data_loader.load_building_proportions()
    heating_props = data_loader.load_heating_proportions()
    pv_props = data_loader.load_pv_proportions()
    solar_thermal_props = data_loader.load_solar_thermal_proportions()
    cooling_props = data_loader.load_cooling_proportions()

    # Inverse transform sampling for each parameter
    num_residents = _select_from_distribution(resident_props, rng) + 1  # 1-5
    building_index = _select_from_distribution(building_props, rng) + 1  # 1-based
    heating_index = _select_from_distribution(heating_props, rng) + 1
    pv_index = _select_from_distribution(pv_props, rng)

    # SPECIAL LOGIC: Combi boiler (type 2) precludes solar thermal
    heating_type = data_loader.get_heating_type(heating_index)
    if heating_type == 2:  # Combi boiler
        solar_thermal_index = 0
    else:
        solar_thermal_index = _select_from_distribution(solar_thermal_props, rng)

    cooling_index = _select_from_distribution(cooling_props, rng)

    return DwellingConfig(
        dwelling_index=dwelling_index,
        num_residents=num_residents,
        building_index=building_index,
        heating_system_index=heating_index,
        pv_system_index=pv_index,
        solar_thermal_index=solar_thermal_index,
        cooling_system_index=cooling_index,
        is_weekend=False,  # Set by simulation config
        country=Country.UK,
        urban_rural=UrbanRural.URBAN
    )

def _select_from_distribution(proportions: np.ndarray, rng: np.random.Generator) -> int:
    """
    Inverse transform sampling from discrete probability distribution.

    VBA Reference: Inverse transform logic in AssignDwellingParameters
    """
    cumulative = np.cumsum(proportions)
    rand = rng.random()
    return int(np.searchsorted(cumulative, rand))
```

**Key Implementation Notes:**
- VBA lines 1194-1205: Resident selection with cumulative probabilities
- VBA lines 1207-1218: Building index selection
- VBA lines 1220-1231: Heating system selection
- VBA lines 1233-1244: PV system selection
- VBA lines 1246-1262: Solar thermal selection with combi boiler check
- VBA lines 1264-1275: Cooling system selection

**Integration:**
- Remove JSON config file loading (lines 189-220 in crest_simulate.py)
- Replace with stochastic generation for all dwellings
- Call `assign_dwelling_parameters()` in main loop

---

#### Phase 4: Database Selection Functions ⏳
**File**: `crest_simulate.py` (4 new functions)
**Status**: NOT STARTED
**VBA References**:
- SetApplianceDatabase (lines 548-636)
- SetBuildingProportions (lines 637-675)
- SetHeatingSystemProportions (lines 677-715)
- SetCoolingSystemProportions (lines 716-754)
**Estimated**: ~200 lines of code

**Plan:**

```python
def set_appliance_database(
    data_loader: CRESTDataLoader,
    country: Country,
    urban_rural: UrbanRural,
    year: int
) -> dict:
    """
    Select appliance ownership proportions based on country, year, urban/rural.

    VBA Reference: SetApplianceDatabase (lines 548-636)
    Handles UK (single column) vs India (interpolation between 5-year intervals)
    """
    # Load 4 databases: proportions, energies, operating powers, standby powers
    # For UK: Use column 1 directly
    # For India: Interpolate between years, select urban/rural column
    pass

def set_building_proportions(
    data_loader: CRESTDataLoader,
    country: Country,
    urban_rural: UrbanRural,
    year: int
) -> np.ndarray:
    """VBA Reference: SetBuildingProportions (lines 637-675)"""
    pass

def set_heating_proportions(
    data_loader: CRESTDataLoader,
    country: Country,
    urban_rural: UrbanRural,
    year: int
) -> np.ndarray:
    """VBA Reference: SetHeatingSystemProportions (lines 677-715)"""
    pass

def set_cooling_proportions(
    data_loader: CRESTDataLoader,
    country: Country,
    urban_rural: UrbanRural,
    year: int
) -> np.ndarray:
    """VBA Reference: SetCoolingSystemProportions (lines 716-754)"""
    pass
```

**Key Implementation Notes:**
- Year parameter needs to be added to CLI (currently missing)
- UK data: Use column 1 directly (no interpolation)
- India data: Interpolate between 2006, 2011, 2016, 2021, 2026, 2031
- Interpolation formula: `weight1 * col1 + weight2 * col2`
- Urban/rural affects column selection for India

**Integration:**
- Call these functions before main dwelling loop
- Update CRESTDataLoader to read proportion databases
- May need to extend CSV files with additional columns

---

#### Phase 5: Multi-Dwelling Aggregation ⏳
**File**: `crest_simulate.py` (new function)
**Status**: NOT STARTED
**VBA Reference**: AggregateResults (lines 810-1048)
**Estimated**: ~150 lines of code

**Plan:**

```python
def aggregate_results(dwellings: list[Dwelling], num_dwellings: int) -> dict:
    """
    Calculate aggregated time series results across all dwellings.

    VBA Reference: AggregateResults (mdlThermalElectricalModel.bas lines 810-1048)
    Returns dict with 23 output variables as 1440-element arrays.
    """
    # Initialize accumulators for 23 variables
    aggregated = {
        'occupancy': np.zeros(1440),
        'active_occupancy': np.zeros(1440),
        'electricity_demand': np.zeros(1440),
        'lighting_demand': np.zeros(1440),
        'appliance_demand': np.zeros(1440),
        'pv_output': np.zeros(1440),
        'heating_power': np.zeros(1440),
        'cooling_power': np.zeros(1440),
        'theta_i': np.zeros(1440),
        'theta_o': np.zeros(1440),
        # ... 13 more variables
    }

    # Calculate total population
    total_population = sum(d.config.num_residents for d in dwellings)

    # Aggregate minute by minute (VBA lines 943-1020)
    for minute in range(1, 1441):
        for dwelling in dwellings:
            # Accumulate each variable
            aggregated['occupancy'][minute-1] += dwelling.occupancy.get_total_occupancy(minute)
            aggregated['electricity_demand'][minute-1] += dwelling.get_total_electricity_demand(minute)
            # ... etc for all 23 variables

        # Normalize by population or dwelling count (VBA lines 998-1019)
        aggregated['occupancy'][minute-1] /= total_population
        aggregated['electricity_demand'][minute-1] /= 1000  # W to kW
        # ... etc

    return aggregated
```

**23 Output Variables** (VBA lines 891-933):
1. Total occupancy
2. Active occupancy
3. Total electricity demand
4. Lighting demand
5. Appliance demand
6. PV output
7. Net demand (after PV)
8. Self-consumption
9. Hot water demand
10. Heating power (space)
11. Heating power (water)
12. Fuel flow
13. Indoor temperature
14. Outdoor temperature
15. Building external temperature
16. Emitter temperature
17. Cylinder temperature
18. Cooler temperature
19. Solar thermal output
20. Clearness index
21. Solar irradiance
22. Clearsky irradiance
23. (Additional variable TBD)

---

#### Phase 6: Component Getter Methods ⏳
**Files**: 7 component files
**Status**: NOT STARTED
**Estimated**: ~140 lines total (14 methods × ~10 lines each)

**Required Methods:**

**`crest/core/occupancy.py`:**
```python
def get_mean_active_occupancy(self) -> float:
    """VBA: GetMeanActiveOccupancy"""
    return np.mean(self.active_occupancy)

def get_proportion_actively_occupied(self) -> float:
    """VBA: GetPrActivelyOccupied"""
    return np.sum(self.active_occupancy > 0) / len(self.active_occupancy)
```

**`crest/core/lighting.py`:**
```python
def get_daily_sum_lighting(self) -> float:
    """VBA: GetDailySumLighting"""
    return np.sum(self.total_demand)
```

**`crest/core/appliances.py`:**
```python
def get_daily_sum_appliance_demand(self) -> float:
    """VBA: GetDailySumApplianceDemand"""
    return np.sum(self.total_demand)
```

**`crest/core/pv.py`:**
```python
def get_daily_sum_pv_output(self) -> float:
    """VBA: GetDailySumPvOutput"""
    return np.sum(self.pv_output)

def get_daily_sum_p_net(self) -> float:
    """VBA: GetDailySumP_net"""
    return np.sum(self.p_net)

def get_daily_sum_p_self(self) -> float:
    """VBA: GetDailySumP_self"""
    return np.sum(self.p_self)
```

**`crest/core/building.py`:**
```python
def get_mean_theta_i(self) -> float:
    """VBA: GetMeanTheta_i"""
    return np.mean(self.theta_i)
```

**`crest/core/heating.py`:**
```python
def get_daily_sum_thermal_energy_space(self) -> float:
    """VBA: GetDailySumThermalEnergySpace"""
    return np.sum(self.phi_h_space)

def get_daily_sum_thermal_energy_water(self) -> float:
    """VBA: GetDailySumThermalEnergyWater"""
    return np.sum(self.phi_h_water)
```

**`crest/core/controls.py`:**
```python
def get_space_thermostat_setpoint(self) -> float:
    """VBA: Space thermostat setpoint"""
    return self.theta_setpoint_space
```

**`crest/core/solar_thermal.py`:**
```python
def get_daily_sum_phi_s(self) -> float:
    """VBA: GetDailySumPhi_s"""
    return np.sum(self.phi_s)
```

**`crest/core/water.py`:**
```python
def get_daily_hot_water_volume(self) -> float:
    """VBA: GetDailyHotWaterVolume"""
    return np.sum(self.hot_water_volume)
```

---

#### Phase 7: Extend Daily Totals ⏳
**File**: `crest_simulate.py` lines 249-264
**Status**: PARTIALLY IMPLEMENTED (3/17 metrics)
**VBA Reference**: DailyTotals (lines 1057-1121)
**Estimated**: ~50 lines of code

**Current Implementation:**
```python
# Lines 251-253 - Only 3 metrics calculated
daily_electricity = sum(dwelling.get_total_electricity_demand(t)
                       for t in range(1, 1441))
daily_fuel = dwelling.heating_system.get_daily_fuel_consumption()
daily_water = dwelling.hot_water.get_daily_hot_water_volume()
```

**Missing 14 Metrics:**
1. Mean active occupancy → `occupancy.get_mean_active_occupancy()`
2. Proportion actively occupied → `occupancy.get_proportion_actively_occupied()`
3. Daily lighting sum → `lighting.get_daily_sum_lighting()`
4. Daily appliance sum → `appliances.get_daily_sum_appliance_demand()`
5. Daily PV output → `pv_system.get_daily_sum_pv_output()`
6. Daily net demand → `pv_system.get_daily_sum_p_net()`
7. Daily self-consumption → `pv_system.get_daily_sum_p_self()`
8. Mean indoor temperature → `building.get_mean_theta_i()`
9. Space thermal energy → `heating_system.get_daily_sum_thermal_energy_space()`
10. Water thermal energy → `heating_system.get_daily_sum_thermal_energy_water()`
11. Space setpoint → `heating_controls.get_space_thermostat_setpoint()`
12. Water setpoint → `heating_controls.get_water_thermostat_setpoint()`
13. Solar thermal output → `solar_thermal.get_daily_sum_phi_s()`
14. Emitter nominal temp → `building.theta_em_nominal`

**Updated Implementation:**
```python
daily_totals = {
    'dwelling_index': dwelling_idx,
    'mean_active_occupancy': dwelling.occupancy.get_mean_active_occupancy(),
    'proportion_actively_occupied': dwelling.occupancy.get_proportion_actively_occupied(),
    'daily_lighting': dwelling.lighting.get_daily_sum_lighting(),
    'daily_appliances': dwelling.appliances.get_daily_sum_appliance_demand(),
    'daily_electricity': daily_electricity,
    'daily_pv_output': dwelling.pv_system.get_daily_sum_pv_output() if dwelling.pv_system else 0,
    'daily_p_net': dwelling.pv_system.get_daily_sum_p_net() if dwelling.pv_system else 0,
    'daily_p_self': dwelling.pv_system.get_daily_sum_p_self() if dwelling.pv_system else 0,
    'daily_hot_water': daily_water,
    'mean_theta_i': dwelling.building.get_mean_theta_i(),
    'daily_space_heat': dwelling.heating_system.get_daily_sum_thermal_energy_space(),
    'daily_water_heat': dwelling.heating_system.get_daily_sum_thermal_energy_water(),
    'daily_fuel': daily_fuel,
    'space_setpoint': dwelling.heating_controls.get_space_thermostat_setpoint(),
    'water_setpoint': dwelling.heating_controls.get_water_thermostat_setpoint(),
    'daily_solar_thermal': dwelling.solar_thermal.get_daily_sum_phi_s() if dwelling.solar_thermal else 0,
    'emitter_nominal_temp': dwelling.building.theta_em_nominal
}
```

---

#### Phase 8: Comprehensive AUDIT_LOG Documentation ⏳
**File**: `AUDIT_LOG.md`
**Status**: NOT STARTED
**Estimated**: ~300 lines

**Structure:**
```markdown
### 13. Multi-Dwelling Orchestration (mdlThermalElectricalModel.bas → crest_simulate.py)

**Status**: ✅ COMPLETE

**VBA File**: original/mdlThermalElectricalModel.bas (1399 lines)
**Python File**: crest_simulate.py (~600 lines after completion)

#### VBA Functions Analysis

1. **RunThermalElectricalDemandModel** (lines 132-498)
   - [Detailed line-by-line comparison]

2. **SetApplianceDatabase** (lines 548-636)
   - [Implementation notes]

[... continue for all 13 VBA functions ...]

#### Fixes Applied

##### Fix 13.1: Activity Statistics Loading
[Details]

##### Fix 13.2: Stochastic Parameter Assignment
[Details]

[... continue for all fixes ...]

#### Testing Results
[Test results and verification]
```

---

#### Phase 9: Testing ⏳
**Status**: NOT STARTED
**Estimated**: Manual testing + verification

**Test Plan:**

1. **Unit Tests:**
   - Test `_select_from_distribution()` with known seed
   - Test `assign_dwelling_parameters()` produces valid configs
   - Test all getter methods return correct types/values
   - Test `aggregate_results()` with 2-3 dwellings

2. **Integration Test:**
   - Run 100 dwellings with fixed seed
   - Verify parameter distributions match proportions
   - Verify combi boiler → no solar thermal rule
   - Compare daily totals to VBA (within ±5% for stochastic)

3. **Regression Test:**
   - Run existing test cases
   - Ensure no breaking changes to existing functionality

---

## Implementation Order

**Recommended sequence:**

1. ✅ Phase 1: Activity statistics (DONE)
2. ✅ Phase 3: CRESTDataLoader extensions (DONE)
3. ⏳ Phase 2: Stochastic parameter assignment
4. ⏳ Phase 4: Database selection functions
5. ⏳ Phase 6: Component getter methods (enables Phase 7)
6. ⏳ Phase 7: Extend daily totals
7. ⏳ Phase 5: Multi-dwelling aggregation
8. ⏳ Phase 8: Documentation
9. ⏳ Phase 9: Testing

---

## Key Design Decisions

### 1. Remove JSON Config Support ✅
**Decision**: Replace JSON config loading with stochastic generation
**Rationale**: VBA always generates stochastically; JSON was a Python-specific feature
**Impact**: Breaking change - users must switch to stochastic mode
**Status**: Approved by user

### 2. Year Parameter Addition
**Decision**: Add `--year` CLI parameter (2006-2031)
**Rationale**: Required for India data interpolation
**Impact**: New optional parameter, defaults to 2015
**Status**: Pending implementation

### 3. Country/Urban-Rural Configuration
**Decision**: Add `--country` and `--urban-rural` CLI parameters
**Rationale**: Required for database selection functions
**Impact**: New optional parameters, default to UK/Urban
**Status**: Pending implementation

---

## Testing Strategy

### Stochastic Parameter Distribution Test
```python
# Run 1000 dwellings, check distributions
configs = [assign_dwelling_parameters(loader, i, rng) for i in range(1000)]
resident_counts = [c.num_residents for c in configs]
actual_dist = np.histogram(resident_counts, bins=[1,2,3,4,5,6])[0] / 1000
expected_dist = loader.load_resident_proportions()
assert np.allclose(actual_dist, expected_dist, atol=0.05)  # Within 5%
```

### Combi Boiler Rule Test
```python
# Verify combi boiler precludes solar thermal
for config in configs:
    heating_type = loader.get_heating_type(config.heating_system_index)
    if heating_type == 2:  # Combi boiler
        assert config.solar_thermal_index == 0
```

### Aggregation Test
```python
# Verify aggregation sums correctly
agg = aggregate_results(dwellings, num_dwellings)
total_elec_agg = sum(agg['electricity_demand'])
total_elec_individual = sum(
    sum(d.get_total_electricity_demand(t) for t in range(1, 1441))
    for d in dwellings
)
assert np.isclose(total_elec_agg, total_elec_individual / 1000, rtol=0.01)
```

---

## Files Modified

### Completed ✅
1. `crest_simulate.py` - Activity statistics loading fixed
2. `crest/data/loader.py` - 7 new proportion loading methods added

### Pending ⏳
3. `crest_simulate.py` - Stochastic parameter assignment
4. `crest_simulate.py` - Database selection functions
5. `crest_simulate.py` - Multi-dwelling aggregation
6. `crest_simulate.py` - Extended daily totals
7. `crest/core/occupancy.py` - Add 2 getter methods
8. `crest/core/lighting.py` - Add 1 getter method
9. `crest/core/appliances.py` - Add 1 getter method
10. `crest/core/pv.py` - Add 3 getter methods
11. `crest/core/building.py` - Add 1 getter method
12. `crest/core/heating.py` - Add 2 getter methods
13. `crest/core/controls.py` - Add 2 getter methods
14. `crest/core/solar_thermal.py` - Add 1 getter method
15. `crest/core/water.py` - Add 1 getter method
16. `AUDIT_LOG.md` - Comprehensive documentation

**Total**: 16 files (2 done, 14 pending)

---

## Estimated Remaining Work

| Phase | Lines of Code | Complexity | Est. Time |
|-------|--------------|------------|-----------|
| Phase 2 | ~100 | Medium | 30 min |
| Phase 4 | ~200 | High | 45 min |
| Phase 5 | ~150 | Medium | 30 min |
| Phase 6 | ~140 | Low | 20 min |
| Phase 7 | ~50 | Low | 10 min |
| Phase 8 | ~300 (docs) | Medium | 30 min |
| Phase 9 | N/A (testing) | High | 30 min |
| **Total** | **~940 lines** | | **~3 hours** |

---

## Next Session Plan

**Priority Order:**
1. Start with Phase 2 (stochastic parameter assignment) - Core functionality
2. Continue with Phase 6 (getter methods) - Enables Phase 7
3. Complete Phase 7 (daily totals) - Quick win
4. Tackle Phase 4 (database selection) - More complex
5. Implement Phase 5 (aggregation) - Depends on getters
6. Document in Phase 8 (AUDIT_LOG)
7. Test everything in Phase 9

**Breaking Points:**
- After Phase 2: Core stochastic generation working
- After Phase 2+6+7: Daily totals complete
- After Phase 2+4: Database selection complete
- After all phases: Full VBA parity

---

## Notes

- Token usage at commit: 99k / 200k (50%)
- All proportion loading methods tested and working
- Activity statistics loading tested with 72 profiles
- Resident proportions sum to 1.0 as expected
- No TODOs left in completed code
- Ready for Phase 2 implementation

**Last Updated**: 2025-11-11 08:45 UTC
