#!/usr/bin/env python3
"""
CREST Super Quick Test - One Dwelling, Visual Output
=====================================================

The absolute simplest test: Run one dwelling and show the results.
Takes about 30 seconds.

Usage: python test_one_dwelling.py
"""

import sys
from pathlib import Path

# Add crest package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🏠 CREST One-Dwelling Quick Test")
print("=" * 60)

try:
    # Imports
    print("\n📦 Importing modules...")
    from crest.data.loader import CRESTDataLoader
    from crest.core.climate import GlobalClimate, ClimateConfig
    from crest.simulation.dwelling import Dwelling, DwellingConfig
    from crest.simulation.config import Country, City, UrbanRural
    from crest.utils.random import RandomGenerator
    import pandas as pd
    print("   ✅ All imports successful")
    
    # Load activity stats
    print("\n📊 Loading activity statistics...")
    def load_activity_statistics(data_loader):
        activity_stats = {}
        activity_df = data_loader.load_activity_stats()
        for row_idx in range(len(activity_df)):
            row = activity_df.iloc[row_idx]
            weekend, occupants, profile_id = row.iloc[1], row.iloc[2], row.iloc[3]
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
    
    data_loader = CRESTDataLoader()
    activity_stats = load_activity_statistics(data_loader)
    print(f"   ✅ Loaded {len(activity_stats)} activity profiles")
    
    # Create climate
    print("\n🌤️  Simulating climate (June 15, England)...")
    climate_config = ClimateConfig(day_of_month=15, month_of_year=6, city=City.ENGLAND)
    global_climate = GlobalClimate(climate_config, data_loader)
    global_climate.run_all()
    print("   ✅ Climate simulation complete")
    
    # Create dwelling
    print("\n🏡 Creating dwelling...")
    print("   - 2 residents")
    print("   - Weekday")
    print("   - UK, Urban")
    print("   - Seed: 42 (for reproducibility)")
    
    rng = RandomGenerator(seed=42)
    dwelling_config = DwellingConfig(
        dwelling_index=1,
        num_residents=2,
        building_index=1,
        heating_system_index=1,
        pv_system_index=0,
        solar_thermal_index=0,
        cooling_system_index=0,
        country=Country.UK,
        urban_rural=UrbanRural.URBAN,
        is_weekend=False
    )
    
    print("\n⚙️  Running simulation (1440 minutes)...")
    dwelling = Dwelling(dwelling_config, global_climate, data_loader, activity_stats, rng)
    dwelling.run_simulation()
    print("   ✅ Simulation complete")
    
    # Get results
    print("\n📈 Calculating daily totals...")
    totals = dwelling.get_daily_totals()
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 DAILY ENERGY CONSUMPTION RESULTS")
    print("=" * 60)
    
    print(f"\n👥 OCCUPANCY:")
    print(f"   Mean active occupancy:     {totals['mean_active_occupancy']:.2f} persons")
    print(f"   Time actively occupied:    {totals['proportion_actively_occupied']*100:.1f}%")
    
    print(f"\n⚡ ELECTRICITY:")
    print(f"   Lighting:                  {totals['lighting_demand']:.2f} kWh")
    print(f"   Appliances:                {totals['appliance_demand']:.2f} kWh")
    print(f"   TOTAL ELECTRICITY:         {totals['total_electricity_demand']:.2f} kWh/day")
    
    if totals['pv_output'] > 0:
        print(f"\n☀️  SOLAR PV:")
        print(f"   PV generation:             {totals['pv_output']:.2f} kWh")
        print(f"   Self-consumption:          {totals['self_consumption']:.2f} kWh")
        print(f"   Net demand:                {totals['net_electricity_demand']:.2f} kWh")
    
    print(f"\n🔥 HEATING:")
    print(f"   Space heating:             {totals['thermal_energy_space']:.2f} kWh")
    print(f"   Water heating:             {totals['thermal_energy_water']:.2f} kWh")
    print(f"   Gas consumption:           {totals['gas_demand']:.2f} m³/day")
    print(f"   Indoor temperature:        {totals['average_indoor_temperature']:.1f}°C")
    print(f"   Thermostat setpoint:       {totals['space_thermostat_setpoint']:.1f}°C")
    
    print(f"\n💧 HOT WATER:")
    print(f"   Volume used:               {totals['hot_water_demand']:.1f} litres/day")
    
    if totals['solar_thermal_output'] > 0:
        print(f"\n☀️  SOLAR THERMAL:")
        print(f"   Solar thermal output:      {totals['solar_thermal_output']:.2f} kWh")
    
    # Sanity checks
    print("\n" + "=" * 60)
    print("🔍 SANITY CHECKS")
    print("=" * 60)
    
    checks = [
        ("Active occupancy in range", 0 <= totals['mean_active_occupancy'] <= 2),
        ("Lighting demand positive", totals['lighting_demand'] > 0),
        ("Appliance demand positive", totals['appliance_demand'] > 0),
        ("Hot water demand positive", totals['hot_water_demand'] > 0),
        ("Indoor temp reasonable", 10 < totals['average_indoor_temperature'] < 30),
        ("Total electricity reasonable", 1 < totals['total_electricity_demand'] < 50),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ SUCCESS! Your CREST implementation is working!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run the full test suite: python test_crest_quick.py")
        print("  2. Test with different configurations (more residents, PV, etc.)")
        print("  3. Run multiple dwellings and compare with VBA")
    else:
        print("⚠️  Some sanity checks failed - review the results")
        print("=" * 60)
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nMake sure:")
    print("  1. You're in the correct directory")
    print("  2. The 'crest' package is in the parent directory")
    print("  3. All required CSV files are in the 'data' directory")
    sys.exit(1)
    
except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("\nMake sure:")
    print("  1. The 'data' directory exists")
    print("  2. All CSV files are present")
    print("  3. You're running from the correct directory")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

