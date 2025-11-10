#!/usr/bin/env python3
"""
Clean and reformat extracted Excel data to match Python output format.
"""

import pandas as pd
import sys

# Read the raw Excel CSV
df = pd.read_csv('results/excel_100houses/results_daily_summary.csv', skiprows=2)

# The data should start from row 3 (0-indexed row 2 after skipping 2 rows)
# Let's inspect what we have
print("Columns:", list(df.columns[:10]))
print("First row:", df.iloc[0, :10].to_dict())
print("Shape:", df.shape)

# Try to identify the correct columns
# Based on the Excel model, we expect:
# Dwelling index, Date, Mean active occupancy, ..., Total dwelling electricity demand, Hot water demand, etc.

# Save cleaned version
df_clean = pd.DataFrame({
    'Dwelling': df.iloc[:, 0],  # First column is dwelling index
    'Total_Electricity_kWh': pd.to_numeric(df.iloc[:, 7], errors='coerce') / 1000.0,  # Convert Wh to kWh
    'Total_Gas_m3': pd.to_numeric(df.iloc[:, 14], errors='coerce'),
    'Total_Hot_Water_L': pd.to_numeric(df.iloc[:, 10], errors='coerce'),
    'Mean_Internal_Temp_C': pd.to_numeric(df.iloc[:, 11], errors='coerce')
})

# Remove any rows with NaN dwelling numbers
df_clean = df_clean.dropna(subset=['Dwelling'])

# Convert Dwelling to int
df_clean['Dwelling'] = df_clean['Dwelling'].astype(int)

# Keep only first 100 dwellings
df_clean = df_clean[df_clean['Dwelling'] <= 100]

print("\nCleaned data:")
print(df_clean.head())
print(f"Rows: {len(df_clean)}")

# Save
df_clean.to_csv('results/excel_100houses/results_daily_summary_clean.csv', index=False)
print("\n✓ Saved to results/excel_100houses/results_daily_summary_clean.csv")
