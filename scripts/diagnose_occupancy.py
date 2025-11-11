#!/usr/bin/env python3
"""
Diagnostic script to identify occupancy bug by comparing TPM extraction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from crest.data.loader import CRESTDataLoader
from crest.utils import markov
from crest.utils.random import RandomGenerator
import numpy as np

def test_tpm_extraction():
    """Test TPM row extraction for a specific case."""
    loader = CRESTDataLoader()

    # Test case: 2 residents, weekday
    num_residents = 2
    tpm = loader.load_occupancy_tpm(num_residents=num_residents, is_weekend=False)

    print("="*80)
    print("TPM EXTRACTION TEST")
    print("="*80)
    print(f"\nTest case: {num_residents} residents, weekday")
    print(f"TPM shape: {tpm.shape}")
    print(f"Number of possible states: {(num_residents + 1) ** 2} = {9}")

    # Test timestep 1, state "10"
    timestep = 1  # 1-based (VBA compatible)
    current_state = "10"
    possible_states = 9

    print(f"\nTest: Timestep {timestep}, State '{current_state}'")

    # Calculate row index using VBA formula
    row_idx = markov.calculate_tpm_row_index(
        timestep,
        current_state,
        num_residents,
        possible_states,
        vba_compatible=True
    )

    print(f"Calculated row index (1-based): {row_idx}")

    # VBA formula breakdown:
    # intRow = 2 + (intTimeStep - 1) * intPossibleStates + (intResidents + 1) * left + right
    # intRow = 2 + (1 - 1) * 9 + (2 + 1) * 1 + 0
    # intRow = 2 + 0 + 3 + 0 = 5
    left = 1
    right = 0
    vba_row = 2 + (timestep - 1) * possible_states + (num_residents + 1) * left + right
    print(f"VBA formula check: 2 + ({timestep}-1)*{possible_states} + ({num_residents}+1)*{left} + {right} = {vba_row}")

    # Extract probabilities
    probs = tpm.iloc[row_idx - 1, 2:].astype(float).values
    print(f"\nExtracted probabilities ({len(probs)} values):")
    print(f"  Sum: {np.sum(probs):.6f}")
    print(f"  Non-zero: {np.count_nonzero(probs)}")
    print(f"  Values: {probs}")

    # Get state labels
    state_labels = tpm.iloc[0, 2:].values
    print(f"\nState labels: {list(state_labels)}")

    print("\n" + "="*80)
    return probs, state_labels


def test_state_selection():
    """Test if state selection matches expected behavior."""
    print("\nSTATE SELECTION TEST")
    print("="*80)

    # Simple test case
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    rng_values = [0.05, 0.15, 0.35, 0.75, 0.95]
    expected_states = [0, 1, 2, 3, 3]

    print("Test probabilities: [0.1, 0.2, 0.3, 0.4]")
    print("Cumulative: [0.1, 0.3, 0.6, 1.0]")
    print()

    for rng_val, expected in zip(rng_values, expected_states):
        selected = markov.select_next_state(probs, rng_val)
        status = "✅" if selected == expected else "❌"
        print(f"{status} RNG={rng_val:.2f} → Selected={selected}, Expected={expected}")

    print("="*80)


def test_24hr_modification():
    """Test 24-hour occupancy modification."""
    print("\n24-HOUR OCCUPANCY MODIFICATION TEST")
    print("="*80)

    num_residents = 2

    # Original probabilities for 9 states (2 residents)
    # States: 00, 01, 02, 10, 11, 12, 20, 21, 22
    # First 3 states (00, 01, 02) are "unoccupied" (0 at home)
    original = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05])

    print(f"Original probabilities (9 states):")
    print(f"  Unoccupied (00,01,02): {original[:3]}, sum={np.sum(original[:3]):.2f}")
    print(f"  Occupied (10-22):      {original[3:]}, sum={np.sum(original[3:]):.2f}")
    print(f"  Total sum: {np.sum(original):.2f}")

    modified = markov.modify_24hr_occupancy_probabilities(original, num_residents)

    print(f"\nModified probabilities:")
    print(f"  Unoccupied (00,01,02): {modified[:3]}, sum={np.sum(modified[:3]):.2f}")
    print(f"  Occupied (10-22):      {modified[3:]}, sum={np.sum(modified[3:]):.2f}")
    print(f"  Total sum: {np.sum(modified):.2f}")

    print("\nExpected behavior:")
    print("  - Unoccupied states should be zero")
    print("  - Occupied states should be scaled up")
    print("  - Total should sum to 1.0")

    # Calculate expected
    unoccupied_sum = np.sum(original[:3])
    occupied_sum = 1.0 - unoccupied_sum
    expected = original.copy()
    expected[:3] = 0.0
    expected[3:] = original[3:] / occupied_sum

    print(f"\nExpected: {expected}")
    print(f"Got:      {modified}")

    if np.allclose(modified, expected):
        print("✅ Modification matches expected behavior")
    else:
        print("❌ Modification differs from expected!")
        print(f"   Diff: {modified - expected}")

    print("="*80)


def compare_starting_states():
    """Compare starting state probabilities."""
    print("\nSTARTING STATE TEST")
    print("="*80)

    loader = CRESTDataLoader()
    starting_df = loader.load_starting_states()

    # Check for 2 residents, weekday
    num_residents = 2
    col_idx = num_residents
    row_offset = 6  # Weekday offset (0-based)

    print(f"Starting states for {num_residents} residents, weekday:")
    print(f"  Column index: {col_idx}")
    print(f"  Row offset: {row_offset}")

    probs = []
    states = []
    for i in range(49):
        row_idx = row_offset + i
        if row_idx < len(starting_df):
            prob = starting_df.iloc[row_idx, col_idx]
            state = starting_df.iloc[row_idx, 0]
            if pd.notna(prob) and pd.notna(state):
                probs.append(float(prob))
                states.append(str(state))

    print(f"\nFound {len(probs)} valid starting states")
    print(f"Probability sum: {np.sum(probs):.6f}")

    # Show top 5 most likely states
    sorted_indices = np.argsort(probs)[::-1][:5]
    print("\nTop 5 most likely starting states:")
    for idx in sorted_indices:
        print(f"  {states[idx]}: {probs[idx]:.4f}")

    print("="*80)


if __name__ == "__main__":
    import pandas as pd

    # Run all tests
    test_tpm_extraction()
    test_state_selection()
    test_24hr_modification()
    compare_starting_states()

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
