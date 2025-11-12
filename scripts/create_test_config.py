#!/usr/bin/env python3
"""Create 5-dwelling config for algorithm validation test."""

import pandas as pd
import sys

# The 5 test dwellings we selected
TEST_DWELLINGS = [27, 8, 30, 37, 7]

def main():
    input_csv = sys.argv[1] if len(sys.argv) > 1 else 'CREST_Demand_Model_v2_3_3__100_houses_dwellings.csv'
    output_csv = 'test_5_identical_dwellings.csv'
    
    # Read original (skip decorative rows)
    df = pd.read_csv(input_csv, header=1, skiprows=[2, 3])
    
    # Extract the 5 test dwellings
    test_df = df[df.iloc[:, 0].isin(TEST_DWELLINGS)].copy()
    
    # Reset dwelling index to 1-5
    test_df.iloc[:, 0] = range(1, 6)
    
    print(f"Created 5-dwelling config:")
    print(f"  Dwelling 1 (original {TEST_DWELLINGS[0]}): {test_df.iloc[0, 1]} residents")
    print(f"  Dwelling 2 (original {TEST_DWELLINGS[1]}): {test_df.iloc[1, 1]} residents")
    print(f"  Dwelling 3 (original {TEST_DWELLINGS[2]}): {test_df.iloc[2, 1]} residents")
    print(f"  Dwelling 4 (original {TEST_DWELLINGS[3]}): {test_df.iloc[3, 1]} residents")
    print(f"  Dwelling 5 (original {TEST_DWELLINGS[4]}): {test_df.iloc[4, 1]} residents")
    
    # Save with header rows
    with open(output_csv, 'w') as f:
        f.write('Dwelling parameters,,,,,,,,Appliance and water fixtures owned,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n')
        test_df.to_csv(f, index=False)
    
    print(f"\nSaved: {output_csv}")
    print("\nNext steps:")
    print("1. Copy this file to Excel for VBA runs")
    print("2. Run VBA 20 times (manually change seed each time)")
    print("3. Run Python 1000 times: python scripts/run_monte_carlo.py")

if __name__ == '__main__':
    main()
