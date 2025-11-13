#!/usr/bin/env python3
"""
Convenience script to run 100-house simulation using Excel baseline.

This is a wrapper around excel_run_and_compare.py specifically for the
100-house validation test.

Usage:
    python scripts/run_100_houses.py
    python scripts/run_100_houses.py --no-compare
"""

import sys
import subprocess
from pathlib import Path


def main():
    # Path to 100-house Excel file
    excel_file = Path("excel/original_100houses.xlsm")

    if not excel_file.exists():
        print(f"ERROR: 100-house Excel file not found: {excel_file}")
        print("\nExpected location: excel/original_100houses.xlsm")
        print("Make sure you've reorganized the excel/ directory")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable,
        'scripts/excel_run_and_compare.py',
        str(excel_file)
    ]

    # Pass through any command-line arguments
    cmd.extend(sys.argv[1:])

    print("="*60)
    print("CREST 100-HOUSE VALIDATION RUN")
    print("="*60)
    print()

    # Run the main script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
