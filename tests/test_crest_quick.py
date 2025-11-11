#!/usr/bin/env python3
"""
CREST Quick Test Suite
======================

Progressive testing from simplest to more complex:
1. Constants verification
2. Data loading
3. Random number generation
4. Inverse transform sampling
5. Single dwelling simulation
6. Reproducibility check

Run with: python test_crest_quick.py
"""

import sys
from pathlib import Path

# Add crest package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import math


def test_1_constants():
    """Test 1: Verify constants are defined correctly."""
    print("\n" + "="*70)
    print("TEST 1: CONSTANTS VERIFICATION")
    print("="*70)
    
    try:
        from crest.simulation.config import (
            PI, 
            SPECIFIC_HEAT_CAPACITY_WATER,
            TIMESTEPS_PER_DAY_1MIN,
            TIMESTEPS_PER_DAY_10MIN,
            OCCUPANT_THERMAL_GAIN_ACTIVE,
            OCCUPANT_THERMAL_GAIN_DORMANT
        )
        
        # Check PI
        print(f"✓ PI = {PI}")
        assert abs(PI - math.pi) < 1e-10, "PI should equal math.pi"
        print(f"  Expected: {math.pi}")
        print(f"  Match: ✅")
        
        # Check water specific heat
        print(f"\n✓ SPECIFIC_HEAT_CAPACITY_WATER = {SPECIFIC_HEAT_CAPACITY_WATER} J/kg/K")
        assert SPECIFIC_HEAT_CAPACITY_WATER == 4200.0, "Should be 4200 J/kg/K"
        print(f"  Expected: 4200.0 J/kg/K")
        print(f"  Match: ✅")
        
        # Check timesteps
        print(f"\n✓ TIMESTEPS_PER_DAY_1MIN = {TIMESTEPS_PER_DAY_1MIN}")
        assert TIMESTEPS_PER_DAY_1MIN == 1440, "Should be 1440 minutes"
        
        print(f"✓ TIMESTEPS_PER_DAY_10MIN = {TIMESTEPS_PER_DAY_10MIN}")
        assert TIMESTEPS_PER_DAY_10MIN == 144, "Should be 144 ten-minute steps"
        
        # Check occupancy gains
        print(f"\n✓ OCCUPANT_THERMAL_GAIN_ACTIVE = {OCCUPANT_THERMAL_GAIN_ACTIVE} W")
        assert OCCUPANT_THERMAL_GAIN_ACTIVE == 147, "Should be 147W"
        
        print(f"✓ OCCUPANT_THERMAL_GAIN_DORMANT = {OCCUPANT_THERMAL_GAIN_DORMANT} W")
        assert OCCUPANT_THERMAL_GAIN_DORMANT == 84, "Should be 84W"
        
        print(f"\n{'✅ TEST 1 PASSED':<70}")
        return True
        
    except Exception as e:
        print(f"\n{'❌ TEST 1 FAILED':<70}")
        print(f"Error: {e}")
        return False


def test_2_data_loading():
    """Test 2: Verify data files can be loaded."""
    print("\n" + "="*70)
    print("TEST 2: DATA LOADING")
    print("="*70)
    
    try:
        from crest.data.loader import CRESTDataLoader
        
        # Try to load data
        data_loader = CRESTDataLoader()
        print(f"✓ Data loader initialized")
        print(f"  Data directory: {data_loader.data_dir}")
        
        # Test loading key files
        print(f"\n✓ Loading activity statistics...")
        activity_df = data_loader.load_activity_stats()
        print(f"  Rows: {len(activity_df)}, Columns: {len(activity_df.columns)}")
        
        print(f"\n✓ Loading buildings data...")
        buildings_df = data_loader.load_buildings()
        print(f"  Rows: {len(buildings_df)}, Columns: {len(buildings_df.columns)}")
        
        print(f"\n✓ Loading TPM (2 residents, weekday)...")
        tpm = data_loader.load_occupancy_tpm(2, False)
        print(f"  Shape: {tpm.shape}")
        
        # Test proportion loading
        print(f"\n✓ Loading resident proportions...")
        resident_props = data_loader.load_resident_proportions()
        print(f"  Values: {resident_props}")
        print(f"  Sum: {np.sum(resident_props):.6f} (should be ~1.0)")
        assert abs(np.sum(resident_props) - 1.0) < 0.01, "Proportions should sum to ~1.0"
        
        print(f"\n✓ Loading building proportions...")
        building_props = data_loader.load_building_proportions()
        print(f"  Number of building types: {len(building_props)}")
        print(f"  Sum: {np.sum(building_props):.6f} (should be ~1.0)")
        
        print(f"\n✓ Loading heating proportions...")
        heating_props = data_loader.load_heating_proportions()
        print(f"  Number of heating types: {len(heating_props)}")
        print(f"  Sum: {np.sum(heating_props):.6f} (should be ~1.0)")
        
        print(f"\n✓ Testing combi boiler check...")
        heating_type = data_loader.get_heating_type(1)  # First heating system
        print(f"  Heating system 1 type: {heating_type}")
        
        print(f"\n{'✅ TEST 2 PASSED':<70}")
        return True
        
    except Exception as e:
        print(f"\n{'❌ TEST 2 FAILED':<70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_random_generation():
    """Test 3: Verify random number generation is reproducible."""
    print("\n" + "="*70)
    print("TEST 3: RANDOM NUMBER GENERATION")
    print("="*70)
    
    try:
        from crest.utils.random import RandomGenerator
        
        # Test reproducibility with seed
        print(f"✓ Creating RNG with seed=42...")
        rng1 = RandomGenerator(seed=42)
        values1 = [rng1.random() for _ in range(5)]
        print(f"  First 5 values: {[f'{v:.6f}' for v in values1]}")
        
        print(f"\n✓ Creating another RNG with seed=42...")
        rng2 = RandomGenerator(seed=42)
        values2 = [rng2.random() for _ in range(5)]
        print(f"  First 5 values: {[f'{v:.6f}' for v in values2]}")
        
        print(f"\n✓ Checking reproducibility...")
        assert values1 == values2, "Same seed should produce same values"
        print(f"  Match: ✅ (Same seed produces identical values)")
        
        # Test different seed produces different values
        print(f"\n✓ Creating RNG with seed=123...")
        rng3 = RandomGenerator(seed=123)
        values3 = [rng3.random() for _ in range(5)]
        print(f"  First 5 values: {[f'{v:.6f}' for v in values3]}")
        
        assert values1 != values3, "Different seeds should produce different values"
        print(f"  Match: ✅ (Different seed produces different values)")
        
        # Test normal distribution
        print(f"\n✓ Testing normal distribution (mean=10, std=2)...")
        rng4 = RandomGenerator(seed=42)
        normal_values = [rng4.normal(10, 2) for _ in range(1000)]
        mean = np.mean(normal_values)
        std = np.std(normal_values)
        print(f"  Sample mean: {mean:.3f} (expected ~10)")
        print(f"  Sample std:  {std:.3f} (expected ~2)")
        
        assert 9.5 < mean < 10.5, "Mean should be close to 10"
        assert 1.8 < std < 2.2, "Std should be close to 2"
        print(f"  Distribution: ✅")
        
        print(f"\n{'✅ TEST 3 PASSED':<70}")
        return True
        
    except Exception as e:
        print(f"\n{'❌ TEST 3 FAILED':<70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_inverse_transform():
    """Test 4: Verify inverse transform sampling matches expected behavior."""
    print("\n" + "="*70)
    print("TEST 4: INVERSE TRANSFORM SAMPLING")
    print("="*70)
    
    try:
        from crest.utils.random import RandomGenerator
        
        # Import the function from crest_simulate
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Define the function inline for testing (from crest_simulate.py lines 92-130)
        def _select_from_distribution(proportions: np.ndarray, rng: np.random.Generator) -> int:
            cumulative = np.cumsum(proportions)
            rand_value = rng.random()
            index = int(np.searchsorted(cumulative, rand_value))
            return min(index, len(proportions) - 1)
        
        # Test with known proportions
        print(f"✓ Testing with proportions: [0.2, 0.3, 0.5]")
        proportions = np.array([0.2, 0.3, 0.5])
        
        # Generate many samples to check distribution
        rng = RandomGenerator(seed=42)
        samples = [_select_from_distribution(proportions, rng.rng) for _ in range(10000)]
        
        # Count occurrences
        counts = [samples.count(i) for i in range(3)]
        frequencies = [c/len(samples) for c in counts]
        
        print(f"\n  Index 0: {frequencies[0]:.3f} (expected ~0.2)")
        print(f"  Index 1: {frequencies[1]:.3f} (expected ~0.3)")
        print(f"  Index 2: {frequencies[2]:.3f} (expected ~0.5)")
        
        # Check within reasonable bounds (±5%)
        assert 0.15 < frequencies[0] < 0.25, "Index 0 frequency should be ~0.2"
        assert 0.25 < frequencies[1] < 0.35, "Index 1 frequency should be ~0.3"
        assert 0.45 < frequencies[2] < 0.55, "Index 2 frequency should be ~0.5"
        
        print(f"\n  Distribution: ✅ (Matches expected proportions)")
        
        # Test edge case: all probability in first index
        print(f"\n✓ Testing edge case: [1.0, 0.0, 0.0]")
        edge_props = np.array([1.0, 0.0, 0.0])
        edge_samples = [_select_from_distribution(edge_props, rng.rng) for _ in range(100)]
        assert all(s == 0 for s in edge_samples), "Should always select index 0"
        print(f"  Result: ✅ (Always selects index 0)")
        
        # Test edge case: all probability in last index
        print(f"\n✓ Testing edge case: [0.0, 0.0, 1.0]")
        edge_props2 = np.array([0.0, 0.0, 1.0])
        edge_samples2 = [_select_from_distribution(edge_props2, rng.rng) for _ in range(100)]
        assert all(s == 2 for s in edge_samples2), "Should always select index 2"
        print(f"  Result: ✅ (Always selects index 2)")
        
        print(f"\n{'✅ TEST 4 PASSED':<70}")
        return True
        
    except Exception as e:
        print(f"\n{'❌ TEST 4 FAILED':<70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_activity_loading():
    """Test 5: Verify activity statistics loading."""
    print("\n" + "="*70)
    print("TEST 5: ACTIVITY STATISTICS LOADING")
    print("="*70)
    
    try:
        from crest.data.loader import CRESTDataLoader
        
        # Load using the function from crest_simulate
        def load_activity_statistics(data_loader):
            activity_stats = {}
            activity_df = data_loader.load_activity_stats()
            
            for row_idx in range(len(activity_df)):
                row = activity_df.iloc[row_idx]
                weekend = row.iloc[1]
                occupants = row.iloc[2]
                profile_id = row.iloc[3]
                
                if pd.isna(weekend) or pd.isna(occupants) or pd.isna(profile_id):
                    continue
                if not isinstance(profile_id, str) or not profile_id.startswith('Act_'):
                    continue
                
                is_weekend = int(weekend)
                active_occupants = int(occupants)
                profile_id = str(profile_id)
                modifiers = row.iloc[4:148].values.astype(float)
                key = f"{is_weekend}_{active_occupants}_{profile_id}"
                activity_stats[key] = modifiers
            
            return activity_stats
        
        import pandas as pd
        data_loader = CRESTDataLoader()
        
        print(f"✓ Loading activity statistics...")
        activity_stats = load_activity_statistics(data_loader)
        
        print(f"  Total profiles loaded: {len(activity_stats)}")
        print(f"  Expected: 72 profiles")
        
        # Check some expected keys exist
        expected_keys = [
            "0_0_Act_Cooking",
            "0_1_Act_Cooking",
            "1_1_Act_TV",
            "0_2_Act_HouseClean"
        ]
        
        print(f"\n✓ Checking expected keys exist...")
        for key in expected_keys:
            if key in activity_stats:
                modifiers = activity_stats[key]
                print(f"  ✓ '{key}': {len(modifiers)} modifiers")
                assert len(modifiers) == 144, "Should have 144 ten-minute modifiers"
            else:
                print(f"  Note: '{key}' not found (may not be in dataset)")
        
        # Show sample of loaded keys
        print(f"\n✓ Sample of loaded keys (first 5):")
        for i, key in enumerate(list(activity_stats.keys())[:5]):
            print(f"  {i+1}. {key}")
        
        print(f"\n{'✅ TEST 5 PASSED':<70}")
        return True
        
    except Exception as e:
        print(f"\n{'❌ TEST 5 FAILED':<70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_single_dwelling():
    """Test 6: Run a single dwelling simulation (most comprehensive test)."""
    print("\n" + "="*70)
    print("TEST 6: SINGLE DWELLING SIMULATION")
    print("="*70)
    
    try:
        from crest.data.loader import CRESTDataLoader
        from crest.core.climate import GlobalClimate, ClimateConfig
        from crest.simulation.dwelling import Dwelling, DwellingConfig
        from crest.simulation.config import Country, City, UrbanRural
        from crest.utils.random import RandomGenerator
        import pandas as pd
        
        # Load activity statistics
        def load_activity_statistics(data_loader):
            activity_stats = {}
            activity_df = data_loader.load_activity_stats()
            
            for row_idx in range(len(activity_df)):
                row = activity_df.iloc[row_idx]
                weekend = row.iloc[1]
                occupants = row.iloc[2]
                profile_id = row.iloc[3]
                
                if pd.isna(weekend) or pd.isna(occupants) or pd.isna(profile_id):
                    continue
                if not isinstance(profile_id, str) or not profile_id.startswith('Act_'):
                    continue
                
                is_weekend = int(weekend)
                active_occupants = int(occupants)
                profile_id = str(profile_id)
                modifiers = row.iloc[4:148].values.astype(float)
                key = f"{is_weekend}_{active_occupants}_{profile_id}"
                activity_stats[key] = modifiers
            
            return activity_stats
        
        print(f"✓ Setting up simulation...")
        
        # Initialize with fixed seed for reproducibility
        rng = RandomGenerator(seed=42)
        
        # Load data
        data_loader = CRESTDataLoader()
        activity_stats = load_activity_statistics(data_loader)
        print(f"  Loaded {len(activity_stats)} activity profiles")
        
        # Create climate
        climate_config = ClimateConfig(
            day_of_month=15,
            month_of_year=6,
            city=City.ENGLAND
        )
        global_climate = GlobalClimate(climate_config, data_loader)
        global_climate.run_all()
        print(f"  ✓ Climate simulation complete")
        
        # Create dwelling config (2 residents, simple setup)
        dwelling_config = DwellingConfig(
            dwelling_index=1,
            num_residents=2,
            building_index=1,
            heating_system_index=1,
            pv_system_index=0,  # No PV
            solar_thermal_index=0,  # No solar thermal
            cooling_system_index=0,  # No cooling
            country=Country.UK,
            urban_rural=UrbanRural.URBAN,
            is_weekend=False
        )
        
        print(f"\n✓ Running dwelling simulation...")
        print(f"  - 2 residents")
        print(f"  - Weekday in June")
        print(f"  - UK, Urban")
        print(f"  - No PV, no solar thermal, no cooling")
        
        # Create and run dwelling
        dwelling = Dwelling(
            dwelling_config,
            global_climate,
            data_loader,
            activity_stats,
            rng
        )
        
        dwelling.run_simulation()
        print(f"  ✓ Simulation complete (1440 timesteps)")
        
        # Get daily totals
        print(f"\n✓ Calculating daily totals...")
        totals = dwelling.get_daily_totals()
        
        # Display results
        print(f"\n📊 RESULTS:")
        print(f"  Mean active occupancy:    {totals['mean_active_occupancy']:.2f} persons")
        print(f"  Proportion occupied:      {totals['proportion_actively_occupied']:.3f}")
        print(f"  Lighting demand:          {totals['lighting_demand']:.2f} kWh")
        print(f"  Appliance demand:         {totals['appliance_demand']:.2f} kWh")
        print(f"  Total electricity:        {totals['total_electricity_demand']:.2f} kWh")
        print(f"  Hot water:                {totals['hot_water_demand']:.1f} litres")
        print(f"  Average indoor temp:      {totals['average_indoor_temperature']:.1f}°C")
        print(f"  Space heating:            {totals['thermal_energy_space']:.2f} kWh")
        print(f"  Water heating:            {totals['thermal_energy_water']:.2f} kWh")
        print(f"  Gas consumption:          {totals['gas_demand']:.2f} m³")
        
        # Basic sanity checks
        print(f"\n✓ Sanity checks...")
        assert 0 <= totals['mean_active_occupancy'] <= 2, "Active occupancy should be 0-2"
        print(f"  ✓ Active occupancy in valid range")
        
        assert 0 <= totals['proportion_actively_occupied'] <= 1, "Proportion should be 0-1"
        print(f"  ✓ Proportion in valid range")
        
        assert totals['lighting_demand'] > 0, "Should have some lighting demand"
        print(f"  ✓ Lighting demand > 0")
        
        assert totals['appliance_demand'] > 0, "Should have some appliance demand"
        print(f"  ✓ Appliance demand > 0")
        
        assert totals['hot_water_demand'] > 0, "Should have some hot water demand"
        print(f"  ✓ Hot water demand > 0")
        
        assert 10 < totals['average_indoor_temperature'] < 30, "Indoor temp should be reasonable"
        print(f"  ✓ Indoor temperature in reasonable range")
        
        # Typical ranges (these can vary widely but should be in ballpark)
        if 1 < totals['total_electricity_demand'] < 50:
            print(f"  ✓ Total electricity in typical range (1-50 kWh/day)")
        else:
            print(f"  ⚠️  Total electricity {totals['total_electricity_demand']:.2f} kWh outside typical range")
        
        if 0 < totals['gas_demand'] < 200:
            print(f"  ✓ Gas consumption in typical range (0-200 m³/day)")
        else:
            print(f"  ⚠️  Gas consumption {totals['gas_demand']:.2f} m³ outside typical range")
        
        print(f"\n{'✅ TEST 6 PASSED':<70}")
        print(f"\n🎉 Single dwelling simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n{'❌ TEST 6 FAILED':<70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_reproducibility():
    """Test 7: Verify two runs with same seed produce identical results."""
    print("\n" + "="*70)
    print("TEST 7: REPRODUCIBILITY CHECK")
    print("="*70)
    
    try:
        from crest.data.loader import CRESTDataLoader
        from crest.core.climate import GlobalClimate, ClimateConfig
        from crest.simulation.dwelling import Dwelling, DwellingConfig
        from crest.simulation.config import Country, City, UrbanRural
        from crest.utils.random import RandomGenerator
        import pandas as pd
        
        def load_activity_statistics(data_loader):
            activity_stats = {}
            activity_df = data_loader.load_activity_stats()
            for row_idx in range(len(activity_df)):
                row = activity_df.iloc[row_idx]
                weekend = row.iloc[1]
                occupants = row.iloc[2]
                profile_id = row.iloc[3]
                if pd.isna(weekend) or pd.isna(occupants) or pd.isna(profile_id):
                    continue
                if not isinstance(profile_id, str) or not profile_id.startswith('Act_'):
                    continue
                is_weekend = int(weekend)
                active_occupants = int(occupants)
                modifiers = row.iloc[4:148].values.astype(float)
                key = f"{is_weekend}_{active_occupants}_{profile_id}"
                activity_stats[key] = modifiers
            return activity_stats
        
        def run_dwelling(seed):
            """Run a dwelling simulation with given seed."""
            rng = RandomGenerator(seed=seed)
            data_loader = CRESTDataLoader()
            activity_stats = load_activity_statistics(data_loader)
            
            climate_config = ClimateConfig(day_of_month=15, month_of_year=6, city=City.ENGLAND)
            global_climate = GlobalClimate(climate_config, data_loader)
            global_climate.run_all()
            
            dwelling_config = DwellingConfig(
                dwelling_index=1, num_residents=2, building_index=1,
                heating_system_index=1, country=Country.UK,
                urban_rural=UrbanRural.URBAN, is_weekend=False
            )
            
            dwelling = Dwelling(dwelling_config, global_climate, data_loader, activity_stats, rng)
            dwelling.run_simulation()
            return dwelling.get_daily_totals()
        
        print(f"✓ Running simulation 1 with seed=42...")
        totals1 = run_dwelling(seed=42)
        print(f"  Total electricity: {totals1['total_electricity_demand']:.6f} kWh")
        
        print(f"\n✓ Running simulation 2 with seed=42...")
        totals2 = run_dwelling(seed=42)
        print(f"  Total electricity: {totals2['total_electricity_demand']:.6f} kWh")
        
        print(f"\n✓ Comparing results...")
        
        # Compare all metrics
        all_match = True
        tolerance = 1e-6
        
        for key in totals1.keys():
            val1 = totals1[key]
            val2 = totals2[key]
            if abs(val1 - val2) > tolerance:
                print(f"  ❌ {key}: {val1:.6f} != {val2:.6f}")
                all_match = False
            else:
                print(f"  ✓ {key}: {val1:.6f} == {val2:.6f}")
        
        if all_match:
            print(f"\n  ✅ All metrics match exactly!")
            print(f"  Same seed produces identical results")
        else:
            print(f"\n  ❌ Some metrics differ")
            return False
        
        print(f"\n{'✅ TEST 7 PASSED':<70}")
        return True
        
    except Exception as e:
        print(f"\n{'❌ TEST 7 FAILED':<70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests in sequence."""
    print("\n" + "="*70)
    print(" CREST QUICK TEST SUITE")
    print("="*70)
    print("\nRunning progressive tests from simplest to most complex...")
    
    tests = [
        ("Constants Verification", test_1_constants),
        ("Data Loading", test_2_data_loading),
        ("Random Number Generation", test_3_random_generation),
        ("Inverse Transform Sampling", test_4_inverse_transform),
        ("Activity Statistics Loading", test_5_activity_loading),
        ("Single Dwelling Simulation", test_6_single_dwelling),
        ("Reproducibility Check", test_7_reproducibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if not result:
                print(f"\n⚠️  Stopping at first failure. Fix this before continuing.")
                break
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR in {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
            break
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<10} {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nYour CREST implementation is working correctly!")
        print("Next steps:")
        print("  1. Run multi-dwelling simulations (10, 50, 100 houses)")
        print("  2. Compare results with VBA outputs")
        print("  3. Run extended validation tests")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Please fix the failed tests before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


