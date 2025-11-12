#!/usr/bin/env python3
"""
RNG Log Comparison Tool

Compares Excel VBA and Python random number generator call logs to verify
they generate the same sequence of random numbers in the same order.

The Excel log format alternates between location and value lines:
    1: clsGlobalClimate:123 - transition steps
    2: r 0.252345174783841

The Python log format has a header followed by single lines per call:
    Call #   1: 0.25234517478384078  @ climate.py:simulate_clearness_index:112

Usage:
    python compare_rng_logs.py <excel_log> <python_log> [--tolerance TOLERANCE] [--verbose]

Arguments:
    excel_log     Path to Excel RNG call log file
    python_log    Path to Python RNG call log file
    --tolerance   Numerical tolerance for float comparison (default: 1e-10)
    --verbose     Show detailed output for all comparisons
    --max-diff    Maximum number of differences to show (default: 50)
"""

import sys
import re
import argparse
from typing import List, Tuple, Optional


class RNGCall:
    """Represents a single RNG call with its location and value."""

    def __init__(self, call_num: int, value: float, location: str, source: str):
        self.call_num = call_num
        self.value = value
        self.location = location
        self.source = source  # 'excel' or 'python'

    def __repr__(self):
        return f"RNGCall(#{self.call_num}, {self.value:.15f}, {self.location})"


def parse_excel_log(filepath: str) -> List[RNGCall]:
    """
    Parse Excel VBA RNG log file.

    Expected format (alternating lines):
        1: clsGlobalClimate:123 - transition steps
        2: r 0.252345174783841
    """
    calls = []

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Process pairs of lines (location, then value)
    i = 0
    call_num = 0
    while i < len(lines) - 1:
        # Parse location line
        loc_match = re.match(r'(\d+):\s*(.+)', lines[i])
        if not loc_match:
            print(f"Warning: Could not parse Excel location line {i+1}: {lines[i]}")
            i += 1
            continue

        location = loc_match.group(2)

        # Parse value line
        val_match = re.match(r'(\d+):\s*r\s+([\d.E+-]+)', lines[i + 1])
        if not val_match:
            # Skip special markers like "rN3" silently
            if re.match(r'\d+:\s*rN\d+', lines[i + 1]):
                i += 2
                continue
            print(f"Warning: Could not parse Excel value line {i+2}: {lines[i+1]}")
            i += 2
            continue

        try:
            value = float(val_match.group(2))
            call_num += 1
            calls.append(RNGCall(call_num, value, location, 'excel'))
        except ValueError as e:
            print(f"Warning: Could not convert value on line {i+2}: {val_match.group(2)}")

        i += 2

    return calls


def parse_python_log(filepath: str) -> List[RNGCall]:
    """
    Parse Python RNG log file.

    Expected format (after header):
        Call #   1: 0.25234517478384078  @ climate.py:simulate_clearness_index:112 → random.py:random:175
        Call #1000: 0.82545396359637380  @ climate.py:simulate_clearness_index:112 → random.py:random:175
    """
    calls = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip header lines and empty lines
            if not line or not line.startswith('Call #'):
                continue

            # Parse: Call #<spaces>N: VALUE  @ location
            # Note: spaces after # are optional (depends on number of digits)
            match = re.match(r'Call\s+#\s*(\d+):\s+([\d.E+-]+)\s+@\s+(.+)', line)
            if match:
                call_num = int(match.group(1))
                value = float(match.group(2))
                location = match.group(3)
                calls.append(RNGCall(call_num, value, location, 'python'))
            else:
                print(f"Warning: Could not parse Python line: {line}")

    return calls


def compare_values(val1: float, val2: float, tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Compare two floating-point values with tolerance.

    Returns:
        (match: bool, difference: float)
    """
    diff = abs(val1 - val2)
    return (diff <= tolerance, diff)


def compare_logs(excel_calls: List[RNGCall],
                python_calls: List[RNGCall],
                tolerance: float = 1e-10,
                verbose: bool = False,
                max_diff: int = 50) -> bool:
    """
    Compare two lists of RNG calls and report differences.

    Returns:
        True if logs match, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"RNG Log Comparison Report")
    print(f"{'='*80}")
    print(f"Excel calls:  {len(excel_calls)}")
    print(f"Python calls: {len(python_calls)}")
    print(f"Tolerance:    {tolerance}")
    print(f"{'='*80}\n")

    # Check if counts match
    if len(excel_calls) != len(python_calls):
        print(f"⚠️  WARNING: Different number of RNG calls!")
        print(f"   Excel:  {len(excel_calls)} calls")
        print(f"   Python: {len(python_calls)} calls")
        print(f"   Will compare up to {min(len(excel_calls), len(python_calls))} calls\n")

    all_match = True
    differences = []
    max_calls = min(len(excel_calls), len(python_calls))

    # Compare each call
    for i in range(max_calls):
        excel = excel_calls[i]
        python = python_calls[i]

        # Check call numbers match
        if excel.call_num != python.call_num:
            diff_msg = f"Call #{i+1}: Call number mismatch! Excel: {excel.call_num}, Python: {python.call_num}"
            differences.append(diff_msg)
            all_match = False
            continue

        # Compare values
        match, diff = compare_values(excel.value, python.value, tolerance)

        if not match:
            diff_msg = (f"Call #{excel.call_num}: VALUE MISMATCH\n"
                       f"   Excel:  {excel.value:.17f}\n"
                       f"   Python: {python.value:.17f}\n"
                       f"   Diff:   {diff:.17e}\n"
                       f"   Excel location:  {excel.location}\n"
                       f"   Python location: {python.location}")
            differences.append(diff_msg)
            all_match = False
        elif verbose:
            print(f"✓ Call #{excel.call_num}: {excel.value:.15f} (match)")

    # Report results
    if all_match:
        print(f"✅ SUCCESS: All {max_calls} RNG calls match perfectly!\n")
        print("The Excel VBA and Python implementations are generating")
        print("identical random number sequences in the same order.\n")
        return True
    else:
        print(f"❌ FAILURE: Found {len(differences)} difference(s)\n")
        print(f"Showing first {min(len(differences), max_diff)} differences:\n")

        for i, diff_msg in enumerate(differences[:max_diff]):
            print(diff_msg)
            print()

        if len(differences) > max_diff:
            print(f"... and {len(differences) - max_diff} more differences (use --max-diff to show more)\n")

        # Report first divergence
        if differences:
            first_diff = differences[0]
            match = re.search(r'Call #(\d+)', first_diff)
            if match:
                first_call = int(match.group(1))
                print(f"📍 First divergence at call #{first_call}")
                if first_call > 1:
                    print(f"   (Calls 1-{first_call-1} matched successfully)")
                print()

        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compare Excel and Python RNG call logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('excel_log', help='Path to Excel RNG call log file')
    parser.add_argument('python_log', help='Path to Python RNG call log file')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-10,
                       help='Numerical tolerance for float comparison (default: 1e-10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output for all comparisons')
    parser.add_argument('--max-diff', '-m', type=int, default=50,
                       help='Maximum number of differences to show (default: 50)')

    args = parser.parse_args()

    try:
        print("Parsing Excel log...")
        excel_calls = parse_excel_log(args.excel_log)

        print("Parsing Python log...")
        python_calls = parse_python_log(args.python_log)

        success = compare_logs(excel_calls, python_calls,
                              args.tolerance, args.verbose, args.max_diff)

        sys.exit(0 if success else 1)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == '__main__':
    main()
