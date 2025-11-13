#!/usr/bin/env python3
"""
Extract run settings from Excel CREST model Main Sheet.

This script reads simulation parameters directly from the Excel Main Sheet
and outputs them in various formats for use with Python simulations.

How it works:
1. Opens Excel file using openpyxl (reads calculated values only)
2. Extracts settings from specific cells in Main Sheet:
   - F6: Day of month
   - H6: Month of year
   - F9: City/location
   - F10: Country
   - H10: Year (for India simulations)
   - K10: Urban/Rural setting
   - F12: Number of dwellings
3. Sets defaults for checkboxes (cannot read ActiveX controls)
4. Outputs in requested format (text, json, or shell script)

Limitations:
- Cannot read Excel checkboxes (ActiveX/Form controls)
- Checkbox settings must be specified via command-line flags:
  --save-detailed, --portable-rng, etc.

Usage:
    python scripts/extract_settings.py excel/original.xlsm
    python scripts/extract_settings.py excel/original.xlsm --format json
    python scripts/extract_settings.py excel/original.xlsm --format shell
    python scripts/extract_settings.py excel/original.xlsm --output settings.json
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

try:
    from openpyxl import load_workbook
except ImportError:
    print("ERROR: openpyxl not installed. Install with: pip install openpyxl")
    sys.exit(1)


def extract_settings(excel_path: Path) -> Dict[str, Any]:
    """
    Extract run settings from Main Sheet.

    Settings locations (based on original Excel layout):
    - Column H contains most parameters (starting around row 6)
    - K10: Additional parameter
    - M8: Additional parameter
    - Checkboxes need special handling

    Args:
        excel_path: Path to .xlsm file

    Returns:
        Dictionary of settings
    """
    wb = load_workbook(excel_path, data_only=True)

    # Try to find the Main Sheet (handle variations in naming)
    main_sheet = None
    for sheet_name in wb.sheetnames:
        if 'main' in sheet_name.lower():
            main_sheet = wb[sheet_name]
            break

    if main_sheet is None:
        raise ValueError(f"Could not find 'Main Sheet' in {excel_path}. Available sheets: {wb.sheetnames}")

    print(f"Reading settings from sheet: '{main_sheet.title}'")

    settings = {}

    # Extract settings based on actual Main Sheet layout
    # Row 6: Date - Day in F6, Month in H6
    # Row 7: Weekday/Weekend in F7
    # Row 8: Latitude in F8, Longitude in H8, LST Meridian in L8
    # Row 9: City in F9
    # Row 10: Country in F10, Year in H10, Urban/Rural in K10
    # Row 12: Number of dwellings in F12

    # Day of month (F6)
    day_val = main_sheet['F6'].value
    settings['day'] = int(day_val) if day_val is not None else 1

    # Month of year (H6)
    month_val = main_sheet['H6'].value
    settings['month'] = int(month_val) if month_val is not None else 1

    # City/location (F9)
    city_val = main_sheet['F9'].value
    settings['city'] = str(city_val) if city_val is not None else 'England'

    # Country (F10)
    country_val = main_sheet['F10'].value
    settings['country'] = str(country_val) if country_val is not None else 'UK'

    # Year (H10) - for India simulations
    year_val = main_sheet['H10'].value
    if year_val is not None:
        try:
            settings['year'] = int(float(year_val))
        except (ValueError, TypeError):
            settings['year'] = 2006
    else:
        settings['year'] = 2006

    # Urban/Rural (K10)
    urban_rural_val = main_sheet['K10'].value
    settings['urban_rural'] = str(urban_rural_val) if urban_rural_val is not None else 'Urban'

    # Number of dwellings (F12)
    num_dwellings_val = main_sheet['F12'].value
    settings['num_dwellings'] = int(num_dwellings_val) if num_dwellings_val is not None else 1

    # Seed - not typically in Main Sheet, default to None
    settings['seed'] = None

    # Checkboxes: Excel uses ActiveX/Form controls which cannot be read by openpyxl
    # These are UI-only controls in Excel that don't link to cell values
    # For Python simulation, use command-line flags instead:
    #   --save-detailed: Write minute-level results (corresponds to Q9 "Include high-resolution output")
    #   --portable-rng: Use portable LCG for RNG validation (Excel uses VBA Rnd())
    # Defaults:
    settings['save_detailed'] = True  # Write detailed output
    settings['use_portable_rng'] = False  # Use standard Python random

    # Print what we found for debugging
    print("\nExtracted settings:")
    for key, value in sorted(settings.items()):
        print(f"  {key:20} = {value}")

    return settings


def format_as_shell_script(settings: Dict[str, Any], excel_path: Path) -> str:
    """
    Format settings as a shell script for re-running the simulation.

    Args:
        settings: Settings dictionary
        excel_path: Original Excel file path

    Returns:
        Shell script content
    """
    script_lines = [
        "#!/bin/bash",
        "# CREST Simulation Run Script",
        f"# Generated from: {excel_path}",
        f"# Date: {Path('.').absolute()}",
        "",
        "# Run the Python CREST simulation with these settings",
        "",
        "python python/main.py \\",
    ]

    # Build command line arguments
    args = []

    if 'num_dwellings' in settings and settings['num_dwellings']:
        args.append(f"  --num-dwellings {settings['num_dwellings']} \\")

    if 'day' in settings:
        args.append(f"  --day {settings['day']} \\")

    if 'month' in settings:
        args.append(f"  --month {settings['month']} \\")

    if 'year' in settings:
        args.append(f"  --year {settings['year']} \\")

    if 'country' in settings:
        args.append(f"  --country {settings['country']} \\")

    if 'city' in settings:
        args.append(f"  --city '{settings['city']}' \\")

    if 'urban_rural' in settings:
        args.append(f"  --urban-rural {settings['urban_rural']} \\")

    if 'seed' in settings and settings['seed'] is not None:
        args.append(f"  --seed {settings['seed']} \\")

    if settings.get('save_detailed', False):
        args.append(f"  --save-detailed \\")

    if settings.get('use_portable_rng', False):
        args.append(f"  --portable-rng \\")

    # Add config file (will be set by calling script)
    args.append(f"  --config-file \"$DWELLINGS_FILE\" \\")

    # Add output directory (will be set by calling script)
    args.append(f"  --output-dir \"$OUTPUT_DIR\"")

    script_lines.extend(args)
    script_lines.append("")

    return '\n'.join(script_lines)


def format_as_json(settings: Dict[str, Any]) -> str:
    """Format settings as JSON."""
    return json.dumps(settings, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Extract run settings from Excel CREST model')
    parser.add_argument('excel_file', type=Path, help='Path to .xlsm Excel file')
    parser.add_argument('--format', choices=['json', 'shell', 'text'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--output', type=Path, help='Output file (default: stdout)')

    args = parser.parse_args()

    if not args.excel_file.exists():
        print(f"ERROR: File not found: {args.excel_file}")
        sys.exit(1)

    # Extract settings
    try:
        settings = extract_settings(args.excel_file)
    except Exception as e:
        print(f"ERROR: Failed to extract settings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Format output
    if args.format == 'json':
        output = format_as_json(settings)
    elif args.format == 'shell':
        output = format_as_shell_script(settings, args.excel_file)
    else:  # text
        output = "\n".join([f"{k}={v}" for k, v in sorted(settings.items())])

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nSettings written to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(output)
        print("=" * 60)


if __name__ == '__main__':
    main()
