#!/usr/bin/env python3
"""
RNG Log Comparison Tool for CREST Validation

Implements Objective #1: RNG Call Sequence Matching
Compares Excel VBA and Python random number generator call logs to verify
they generate the same sequence of random numbers in the same order.

Usage:
    python scripts/rng_log_compare.py <python_dir> <excel_dir> [--tolerance TOLERANCE]

Arguments:
    python_dir    Directory containing Python run with rng_calls.log
    excel_dir     Directory containing Excel run with rng_calls.log
    --tolerance   Numerical tolerance for float comparison (default: 1e-10)
    --verbose     Show detailed output for all comparisons
    --max-diff    Maximum number of differences to show (default: 50)

Example:
    python scripts/rng_log_compare.py \\
        output/rng_validation/python_5houses_20250113_01 \\
        output/rng_validation/excel_5houses_20250113_01
"""

import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple

# Import helper utilities
from utils import create_validation_dir, save_metadata


class RNGCall:
    """Represents a single RNG call with its location and value."""

    def __init__(self, call_num: int, value: float, location: str, source: str):
        self.call_num = call_num
        self.value = value
        self.location = location
        self.source = source  # 'excel' or 'python'

    def __repr__(self):
        return f"RNGCall(#{self.call_num}, {self.value:.15f}, {self.location})"


def parse_excel_log(filepath: Path) -> List[RNGCall]:
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
            i += 2
            continue

        try:
            value = float(val_match.group(2))
            call_num += 1
            calls.append(RNGCall(call_num, value, location, 'excel'))
        except ValueError:
            pass

        i += 2

    return calls


def parse_python_log(filepath: Path) -> List[RNGCall]:
    """
    Parse Python RNG log file.

    Expected format (after header):
        Call #   1: 0.25234517478384078  @ climate.py:simulate_clearness_index:112
    """
    calls = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip header lines and empty lines
            if not line or not line.startswith('Call #'):
                continue

            # Parse: Call #<spaces>N: VALUE  @ location
            match = re.match(r'Call\s+#\s*(\d+):\s+([\d.E+-]+)\s+@\s+(.+)', line)
            if match:
                call_num = int(match.group(1))
                value = float(match.group(2))
                location = match.group(3)
                calls.append(RNGCall(call_num, value, location, 'python'))

    return calls


def compare_values(val1: float, val2: float, tolerance: float = 1e-10) -> Tuple[bool, float]:
    """Compare two floating-point values with tolerance."""
    diff = abs(val1 - val2)
    return (diff <= tolerance, diff)


def extract_module_name(location: str, source: str) -> str:
    """Extract the module/class name from a location string."""
    if source == 'excel':
        match = re.match(r'cls(\w+)', location)
        if match:
            name = match.group(1).lower()
            # Map VBA class names to Python module names
            if 'climate' in name:
                return 'climate'
            elif 'occupancy' in name:
                return 'occupancy'
            elif 'lighting' in name:
                return 'lighting'
            elif 'appliance' in name:
                return 'appliances'
            elif 'water' in name or 'hotwater' in name:
                return 'water'
            elif 'heating' in name:
                return 'heating'
            elif 'cooling' in name:
                return 'cooling'
            elif 'solar' in name:
                return 'solar'
            elif 'pv' in name:
                return 'pv'
            return name
        return location.lower()
    else:  # python
        match = re.match(r'(\w+)\.py:', location)
        if match:
            return match.group(1)
        return location


def compare_locations(excel_loc: str, python_loc: str) -> Tuple[bool, str]:
    """Compare if two locations are from the same module/class."""
    excel_module = extract_module_name(excel_loc, 'excel')
    python_module = extract_module_name(python_loc, 'python')

    if excel_module == python_module:
        return (True, "")
    else:
        return (False, f"Module mismatch: Excel={excel_module}, Python={python_module}")


def compare_logs(excel_calls: List[RNGCall],
                python_calls: List[RNGCall],
                tolerance: float = 1e-10,
                verbose: bool = False,
                max_diff: int = 50) -> Tuple[bool, List[str]]:
    """
    Compare two lists of RNG calls and report differences.

    Returns:
        Tuple of (success: bool, report_lines: List[str])
    """
    report = []
    report.append("=" * 80)
    report.append("RNG LOG COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"Excel calls:  {len(excel_calls)}")
    report.append(f"Python calls: {len(python_calls)}")
    report.append(f"Tolerance:    {tolerance}")
    report.append("=" * 80)
    report.append("")

    # Check if counts match
    counts_match = len(excel_calls) == len(python_calls)
    if not counts_match:
        report.append("✗ FAILURE: Different number of RNG calls!")
        report.append(f"   Excel:  {len(excel_calls)} calls")
        report.append(f"   Python: {len(python_calls)} calls")
        report.append(f"   Difference: {abs(len(excel_calls) - len(python_calls))} calls")
        report.append(f"   Will compare up to {min(len(excel_calls), len(python_calls))} calls")
        report.append("")

    all_match = counts_match
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

        # Compare locations (module/class)
        loc_match, loc_msg = compare_locations(excel.location, python.location)

        # Compare values
        val_match, diff = compare_values(excel.value, python.value, tolerance)

        if not val_match or not loc_match:
            diff_msg = f"Call #{excel.call_num}: MISMATCH\n"
            if not loc_match:
                diff_msg += f"   {loc_msg}\n"
            if not val_match:
                diff_msg += (f"   Value difference: {diff:.17e}\n")
            diff_msg += (f"   Excel:  {excel.value:.17f} from {excel.location}\n"
                        f"   Python: {python.value:.17f} from {python.location}")
            differences.append(diff_msg)
            all_match = False
        elif verbose:
            report.append(f"✓ Call #{excel.call_num}: {excel.value:.15f} (match)")

    # Report results
    if all_match and counts_match:
        report.append(f"✓ SUCCESS: All {max_calls} RNG calls match perfectly!")
        report.append("")
        report.append("The Excel VBA and Python implementations are generating")
        report.append("identical random number sequences in the same order.")
        report.append("")
    else:
        # Determine what failed
        failure_reasons = []
        if not counts_match:
            failure_reasons.append(f"Call count mismatch ({abs(len(excel_calls) - len(python_calls))} calls difference)")
        if differences:
            failure_reasons.append(f"{len(differences)} value/location mismatch(es)")

        report.append(f"✗ FAILURE: {', '.join(failure_reasons)}")
        report.append("")

        if differences:
            report.append(f"Showing first {min(len(differences), max_diff)} differences:")
            report.append("")
            for i, diff_msg in enumerate(differences[:max_diff]):
                report.append(diff_msg)
                report.append("")

            if len(differences) > max_diff:
                report.append(f"... and {len(differences) - max_diff} more differences")
                report.append("")

            # Report first divergence
            first_diff = differences[0]
            match = re.search(r'Call #(\d+)', first_diff)
            if match:
                first_call = int(match.group(1))
                report.append(f"First divergence at call #{first_call}")
                if first_call > 1:
                    report.append(f"(Calls 1-{first_call-1} matched successfully)")
                report.append("")

    return (all_match and counts_match, report)


def main():
    parser = argparse.ArgumentParser(
        description='Compare Excel and Python RNG call logs for CREST validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('python_dir', help='Directory containing Python run with rng_calls.log')
    parser.add_argument('excel_dir', help='Directory containing Excel run with rng_calls.log')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-10,
                       help='Numerical tolerance for float comparison (default: 1e-10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output for all comparisons')
    parser.add_argument('--max-diff', '-m', type=int, default=50,
                       help='Maximum number of differences to show (default: 50)')

    args = parser.parse_args()

    python_dir = Path(args.python_dir)
    excel_dir = Path(args.excel_dir)

    # Validate directories exist
    if not python_dir.exists():
        print(f"ERROR: Python directory not found: {python_dir}")
        sys.exit(1)

    if not excel_dir.exists():
        print(f"ERROR: Excel directory not found: {excel_dir}")
        sys.exit(1)

    print("=" * 80)
    print("CREST RNG LOG COMPARISON")
    print("=" * 80)
    print(f"Python run: {python_dir}")
    print(f"Excel run:  {excel_dir}")
    print()

    # Find RNG log files
    python_log = python_dir / 'rng_calls.log'
    excel_log = excel_dir / 'rng_calls.log'

    if not python_log.exists():
        print(f"ERROR: Python RNG log not found: {python_log}")
        sys.exit(1)

    if not excel_log.exists():
        print(f"ERROR: Excel RNG log not found: {excel_log}")
        sys.exit(1)

    try:
        print("Parsing Excel log...")
        excel_calls = parse_excel_log(excel_log)
        print(f"  Found {len(excel_calls)} RNG calls")

        print("Parsing Python log...")
        python_calls = parse_python_log(python_log)
        print(f"  Found {len(python_calls)} RNG calls")
        print()

        success, report_lines = compare_logs(
            excel_calls, python_calls,
            args.tolerance, args.verbose, args.max_diff
        )

        # Print report to console
        for line in report_lines:
            print(line)

        # Create validation directory
        validation_dir = create_validation_dir(str(python_dir), str(excel_dir), "rng_validation")
        print()
        print(f"Validation directory: {validation_dir}")

        # Save report
        report_file = validation_dir / 'comparison_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"Report saved to: {report_file}")

        # Save call sequence diff (if there are differences)
        if not success:
            diff_file = validation_dir / 'call_sequence_diff.csv'
            import pandas as pd

            diff_data = []
            max_calls = min(len(excel_calls), len(python_calls))
            for i in range(max_calls):
                excel = excel_calls[i]
                python = python_calls[i]

                val_match, diff = compare_values(excel.value, python.value, args.tolerance)
                loc_match, _ = compare_locations(excel.location, python.location)

                diff_data.append({
                    'call_num': excel.call_num,
                    'excel_value': excel.value,
                    'python_value': python.value,
                    'value_diff': diff,
                    'values_match': val_match,
                    'excel_location': excel.location,
                    'python_location': python.location,
                    'locations_match': loc_match
                })

            pd.DataFrame(diff_data).to_csv(diff_file, index=False)
            print(f"Detailed diff saved to: {diff_file}")

        # Save metadata
        save_metadata(
            validation_dir,
            str(python_dir),
            str(excel_dir),
            excel_calls=len(excel_calls),
            python_calls=len(python_calls),
            calls_match=success,
            tolerance=args.tolerance
        )
        print(f"Metadata saved to: {validation_dir / 'metadata.json'}")

        print()
        print("=" * 80)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
