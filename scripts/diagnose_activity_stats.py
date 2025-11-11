#!/usr/bin/env python3
"""
Diagnostic script to check activity statistics loading and usage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crest.data.loader import CRESTDataLoader
from crest_simulate import load_activity_statistics
import pandas as pd


def check_activity_stats_loading():
    """Check if activity statistics are loaded correctly."""
    print("="*80)
    print("ACTIVITY STATISTICS LOADING TEST")
    print("="*80)

    loader = CRESTDataLoader()
    activity_stats = load_activity_statistics(loader)

    print(f"\nTotal activity profiles loaded: {len(activity_stats)}")
    print(f"\nExpected: 72 profiles (36 weekday + 36 weekend)")

    # Check for some expected keys
    test_keys = [
        "0_1_Act_TV",  # Weekday, 1 occupant, TV
        "1_1_Act_TV",  # Weekend, 1 occupant, TV
        "0_2_Act_Cooking",  # Weekday, 2 occupants, Cooking
        "1_2_Act_Cooking",  # Weekend, 2 occupants, Cooking
    ]

    print("\nChecking for expected keys:")
    for key in test_keys:
        if key in activity_stats:
            profile = activity_stats[key]
            print(f"  ✅ {key}: Found, length={len(profile)}, sum={profile.sum():.4f}, mean={profile.mean():.6f}")
        else:
            print(f"  ❌ {key}: NOT FOUND")

    # List all keys
    print(f"\nAll loaded keys (showing first 20):")
    for i, key in enumerate(sorted(activity_stats.keys())[:20]):
        profile = activity_stats[key]
        print(f"  {key}: length={len(profile)}, sum={profile.sum():.4f}, mean={profile.mean():.6f}")

    # Check profile lengths
    print(f"\nProfile length validation:")
    for key, profile in activity_stats.items():
        if len(profile) != 144:
            print(f"  ❌ {key}: Wrong length {len(profile)}, expected 144")

    all_correct = all(len(profile) == 144 for profile in activity_stats.values())
    if all_correct:
        print(f"  ✅ All profiles have correct length (144 ten-minute periods)")

    print("="*80)


def check_raw_csv_data():
    """Check the raw ActivityStats.csv file."""
    print("\nRAW CSV DATA CHECK")
    print("="*80)

    loader = CRESTDataLoader()
    df = loader.load_activity_stats()

    print(f"CSV shape: {df.shape}")
    print(f"Column count: {len(df.columns)}")
    print(f"Row count: {len(df)}")

    # Check for activity profile rows
    print(f"\nChecking for 'Act_' prefixed profile IDs (column 3, 0-indexed):")

    activity_rows = []
    for idx, row in df.iterrows():
        profile_id = row.iloc[3]
        if pd.notna(profile_id) and isinstance(profile_id, str) and profile_id.startswith('Act_'):
            weekend = row.iloc[1]
            occupants = row.iloc[2]
            activity_rows.append((idx, int(weekend), int(occupants), profile_id))

    print(f"Found {len(activity_rows)} activity profile rows")

    # Show first 10
    print(f"\nFirst 10 activity profiles:")
    for idx, weekend, occupants, profile_id in activity_rows[:10]:
        row = df.iloc[idx]
        modifiers = row.iloc[4:148].values.astype(float)
        weekend_str = "weekend" if weekend == 1 else "weekday"
        print(f"  Row {idx}: {weekend_str}, {occupants} occ, {profile_id}, sum={modifiers.sum():.4f}")

    print("="*80)


if __name__ == "__main__":
    check_activity_stats_loading()
    check_raw_csv_data()

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
