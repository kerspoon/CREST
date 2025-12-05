#!/usr/bin/env python3
"""
Validation Parameters Comparison Script

Compares dwelling_params.log files from Python and VBA to identify divergences
in dwelling parameters, bulb configurations, appliance ownership, and switch-on decisions.

Usage:
    python scripts/validation_params_compare.py <python_dir> <excel_dir>

Example:
    python scripts/validation_params_compare.py \
        output/rng_validation/python_01 \
        output/rng_validation/excel_01
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogEntry:
    """Parsed log entry."""
    line_number: int
    entry_type: str
    raw_line: str
    fields: dict


def parse_python_log(log_path: Path) -> list[LogEntry]:
    """Parse Python dwelling_params.log file."""
    entries = []

    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            entry_type = parts[0] if parts else ""

            fields = {}
            for part in parts[1:]:
                if '=' in part:
                    key, val = part.split('=', 1)
                    fields[key] = val
                else:
                    # Positional fields (like dwelling index)
                    if 'pos_0' not in fields:
                        fields['pos_0'] = part
                    elif 'pos_1' not in fields:
                        fields['pos_1'] = part

            entries.append(LogEntry(
                line_number=line_num,
                entry_type=entry_type,
                raw_line=line,
                fields=fields
            ))

    return entries


def parse_vba_log(log_path: Path) -> list[LogEntry]:
    """
    Parse VBA dwelling_params.txt file.

    VBA format should match Python for easy comparison.
    """
    # VBA uses same format as Python
    return parse_python_log(log_path)


def compare_entries(py_entry: LogEntry, vba_entry: LogEntry, tolerance: float = 1e-10) -> Tuple[bool, str]:
    """
    Compare two log entries.

    Returns (match: bool, description: str)
    """
    if py_entry.entry_type != vba_entry.entry_type:
        return False, f"Type mismatch: Python={py_entry.entry_type}, VBA={vba_entry.entry_type}"

    # Compare fields
    all_keys = set(py_entry.fields.keys()) | set(vba_entry.fields.keys())
    mismatches = []

    for key in sorted(all_keys):
        py_val = py_entry.fields.get(key, "MISSING")
        vba_val = vba_entry.fields.get(key, "MISSING")

        if py_val == "MISSING" or vba_val == "MISSING":
            mismatches.append(f"{key}: Python={py_val}, VBA={vba_val}")
            continue

        # Try numeric comparison
        try:
            py_float = float(py_val)
            vba_float = float(vba_val)
            if abs(py_float - vba_float) > tolerance:
                mismatches.append(f"{key}: Python={py_val}, VBA={vba_val}")
        except ValueError:
            # String comparison
            if py_val != vba_val:
                mismatches.append(f"{key}: Python={py_val}, VBA={vba_val}")

    if mismatches:
        return False, "; ".join(mismatches)

    return True, "Match"


def main():
    parser = argparse.ArgumentParser(description="Compare Python and VBA dwelling params logs")
    parser.add_argument("python_dir", type=Path, help="Python output directory")
    parser.add_argument("excel_dir", type=Path, help="Excel/VBA output directory")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Numeric comparison tolerance")
    parser.add_argument("--max-diffs", type=int, default=20, help="Maximum differences to show")
    args = parser.parse_args()

    # Find log files
    py_log = args.python_dir / "dwelling_params.log"
    vba_log = args.excel_dir / "dwelling_params.txt"

    # Also check alternative names
    if not vba_log.exists():
        vba_log = args.excel_dir / "dwelling_params.log"

    if not py_log.exists():
        print(f"ERROR: Python log not found: {py_log}")
        sys.exit(1)

    if not vba_log.exists():
        print(f"ERROR: VBA log not found: {vba_log}")
        print("  Looked for dwelling_params.txt and dwelling_params.log")
        sys.exit(1)

    print(f"Comparing:")
    print(f"  Python: {py_log}")
    print(f"  VBA:    {vba_log}")
    print()

    # Parse logs
    py_entries = parse_python_log(py_log)
    vba_entries = parse_vba_log(vba_log)

    print(f"Python entries: {len(py_entries)}")
    print(f"VBA entries:    {len(vba_entries)}")
    print()

    # Compare entry by entry
    differences = []
    max_entries = max(len(py_entries), len(vba_entries))

    for i in range(max_entries):
        if i >= len(py_entries):
            differences.append((i + 1, "Python log ended", None, vba_entries[i]))
        elif i >= len(vba_entries):
            differences.append((i + 1, "VBA log ended", py_entries[i], None))
        else:
            match, desc = compare_entries(py_entries[i], vba_entries[i], args.tolerance)
            if not match:
                differences.append((i + 1, desc, py_entries[i], vba_entries[i]))

    # Report results
    if not differences:
        print("=" * 60)
        print("RESULT: PERFECT MATCH!")
        print(f"All {len(py_entries)} entries match.")
        print("=" * 60)
        return

    print("=" * 60)
    print(f"RESULT: {len(differences)} DIFFERENCES FOUND")
    print("=" * 60)
    print()

    # Show first N differences
    for idx, (entry_num, desc, py_entry, vba_entry) in enumerate(differences[:args.max_diffs]):
        print(f"Difference {idx + 1} at entry #{entry_num}:")
        print(f"  Description: {desc}")
        if py_entry:
            print(f"  Python: {py_entry.raw_line[:100]}...")
        if vba_entry:
            print(f"  VBA:    {vba_entry.raw_line[:100]}...")
        print()

    if len(differences) > args.max_diffs:
        print(f"... and {len(differences) - args.max_diffs} more differences")

    # Create output directory and save full diff
    output_dir = args.python_dir.parent / f"validation_{args.python_dir.name}_{args.excel_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    diff_file = output_dir / "params_diff.csv"
    with open(diff_file, 'w') as f:
        f.write("entry_num,description,python_line,vba_line\n")
        for entry_num, desc, py_entry, vba_entry in differences:
            py_line = py_entry.raw_line.replace('"', '""') if py_entry else ""
            vba_line = vba_entry.raw_line.replace('"', '""') if vba_entry else ""
            f.write(f'{entry_num},"{desc}","{py_line}","{vba_line}"\n')

    print(f"Full diff saved to: {diff_file}")


if __name__ == "__main__":
    main()
