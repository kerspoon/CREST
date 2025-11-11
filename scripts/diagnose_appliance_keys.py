#!/usr/bin/env python3
"""
Diagnostic script to check if appliance activity keys are being found.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crest.data.loader import CRESTDataLoader
from crest_simulate import load_activity_statistics
import pandas as pd


def check_appliance_keys():
    """Check which appliance profile keys would be looked up."""
    print("="*80)
    print("APPLIANCE ACTIVITY KEY DIAGNOSTIC")
    print("="*80)

    loader = CRESTDataLoader()
    activity_stats = load_activity_statistics(loader)

    # Load appliance specs
    appliances_df = loader.load_appliances_and_fixtures()

    print(f"\nTotal activity profiles available: {len(activity_stats)}")
    print(f"Available keys: {sorted(list(activity_stats.keys())[:10])}...")

    # Check which appliance profiles are used
    print(f"\nAppliance use profiles:")
    profile_counts = appliances_df['use_profile'].value_counts()
    print(profile_counts)

    # For each profile type, check if keys exist for different occupancy levels
    print(f"\n{'Profile':<20} {'Weekday Keys':<30} {'Weekend Keys':<30}")
    print("-"*80)

    unique_profiles = appliances_df['use_profile'].unique()
    for profile in sorted(unique_profiles):
        if profile in ['LEVEL', 'ACTIVE_OCC', 'CUSTOM']:
            print(f"{profile:<20} (Does not use activity stats)")
            continue

        weekday_keys = []
        weekend_keys = []

        # Check for occupancy levels 0-6
        for occ in range(7):
            wd_key = f"0_{occ}_{profile}"
            we_key = f"1_{occ}_{profile}"

            if wd_key in activity_stats:
                weekday_keys.append(f"{occ}✅")
            else:
                weekday_keys.append(f"{occ}❌")

            if we_key in activity_stats:
                weekend_keys.append(f"{occ}✅")
            else:
                weekend_keys.append(f"{occ}❌")

        wd_str = ", ".join(weekday_keys)
        we_str = ", ".join(weekend_keys)
        print(f"{profile:<20} {wd_str:<30} {we_str:<30}")

    print("\n" + "="*80)
    print("Legend: Number = occupancy level, ✅ = key found, ❌ = key missing")
    print("="*80)


if __name__ == "__main__":
    check_appliance_keys()
