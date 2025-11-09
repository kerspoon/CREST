"""
Markov Chain Utilities

Helper functions for Markov chain state transitions used in the occupancy and climate models.
"""

import numpy as np
from typing import Union, Tuple, Optional


def select_next_state(transition_probabilities: np.ndarray, rng_value: float) -> int:
    """
    Select the next state based on transition probabilities.

    Uses the inverse transform method: generate a random number and find which
    state it corresponds to in the cumulative probability distribution.

    Parameters
    ----------
    transition_probabilities : np.ndarray
        Array of transition probabilities (should sum to 1.0)
    rng_value : float
        Random number between 0 and 1

    Returns
    -------
    int
        Index of the selected next state (0-based)
    """
    # Calculate cumulative probabilities
    cumulative_prob = np.cumsum(transition_probabilities)

    # Find the first state where cumulative probability exceeds the random value
    next_state_idx = np.searchsorted(cumulative_prob, rng_value)

    # Ensure we don't exceed array bounds (in case of floating point rounding)
    next_state_idx = min(next_state_idx, len(transition_probabilities) - 1)

    return next_state_idx


def normalize_probabilities(probabilities: np.ndarray, zero_threshold: float = 1e-10) -> np.ndarray:
    """
    Normalize probabilities to sum to 1.0.

    If all probabilities are zero (or below threshold), sets the first probability to 1.0
    as per the VBA model's handling of "dead-end" states.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of probabilities
    zero_threshold : float, optional
        Threshold below which sum is considered zero

    Returns
    -------
    np.ndarray
        Normalized probabilities that sum to 1.0
    """
    prob_sum = np.sum(probabilities)

    if prob_sum < zero_threshold:
        # Dead-end state: force transition to first state
        normalized = np.zeros_like(probabilities)
        normalized[0] = 1.0
        return normalized

    # Normal case: normalize
    return probabilities / prob_sum


def modify_24hr_occupancy_probabilities(
    probabilities: np.ndarray,
    num_residents: int
) -> np.ndarray:
    """
    Modify transition probabilities to prevent transitions to unoccupied states.

    Used for dwellings that require 24-hour occupancy (e.g., someone always home).
    The first (num_residents + 1) states represent unoccupied states (people away/asleep).

    Parameters
    ----------
    probabilities : np.ndarray
        Original transition probabilities
    num_residents : int
        Number of residents in the dwelling

    Returns
    -------
    np.ndarray
        Modified probabilities with zero probability for unoccupied states
    """
    modified = probabilities.copy()

    # Sum probability of transitioning to unoccupied states
    num_unoccupied_states = num_residents + 1
    unoccupied_prob_sum = np.sum(modified[:num_unoccupied_states])

    # Set unoccupied state probabilities to zero
    modified[:num_unoccupied_states] = 0.0

    # Remaining probability for occupied states
    occupied_prob_sum = 1.0 - unoccupied_prob_sum

    if occupied_prob_sum <= 0:
        # No probability for occupied states: force transition to first occupied+active state
        # State (num_residents + 1) + 1 in 0-based indexing = num_residents + 1
        modified[num_residents + 1] = 1.0
    else:
        # Proportionally adjust occupied state probabilities
        modified[num_unoccupied_states:] /= occupied_prob_sum

    return modified


def calculate_tpm_row_index(
    timestep: int,
    current_state_str: str,
    num_residents: int,
    possible_states: int,
    vba_compatible: bool = True
) -> int:
    """
    Calculate the row index in the transition probability matrix.

    Matches the VBA formula:
    intRow = 2 + (intTimeStep - 1) * intPossibleStates
             + (intResidents + 1) * IIf(Left(strCombinedState, 1) = "0", 0, CInt(Left(strCombinedState, 1)))
             + CInt(Right(strCombinedState, 1))

    Parameters
    ----------
    timestep : int
        Current timestep (1-based if vba_compatible=True, else 0-based)
    current_state_str : str
        Current state as string (e.g., "10", "11", "00")
    num_residents : int
        Number of residents in dwelling
    possible_states : int
        Total number of possible states
    vba_compatible : bool, optional
        If True, expects 1-based timestep and returns 1-based row index (default: True)

    Returns
    -------
    int
        Row index in the TPM (1-based if vba_compatible, else 0-based)
    """
    # Extract left and right characters from state string
    left_char = current_state_str[0]
    right_char = current_state_str[-1]

    # Parse to integers
    left_val = 0 if left_char == "0" else int(left_char)
    right_val = int(right_char)

    if vba_compatible:
        # VBA formula (1-based indexing, row 1 is headers, row 2 is state labels)
        row_index = 2 + (timestep - 1) * possible_states + (num_residents + 1) * left_val + right_val
    else:
        # Python 0-based indexing (assumes TPM data starts at row 0)
        row_index = (timestep) * possible_states + (num_residents + 1) * left_val + right_val

    return row_index


def extract_tpm_row(
    tpm: np.ndarray,
    row_index: int,
    col_offset: int = 2,
    vba_compatible: bool = True
) -> np.ndarray:
    """
    Extract a row of transition probabilities from the TPM array.

    Parameters
    ----------
    tpm : np.ndarray
        Full transition probability matrix
    row_index : int
        Row index (1-based if vba_compatible, else 0-based)
    col_offset : int, optional
        Column offset for probability data (VBA TPMs have 2 header columns)
    vba_compatible : bool, optional
        If True, uses 1-based indexing (default: True)

    Returns
    -------
    np.ndarray
        Array of transition probabilities for the specified row
    """
    if vba_compatible:
        # Convert to 0-based indexing
        row_idx = row_index - 1
    else:
        row_idx = row_index

    # Extract the row, skipping header columns
    return tpm[row_idx, col_offset:].copy()


def get_state_labels_from_tpm(
    tpm: np.ndarray,
    col_offset: int = 2,
    vba_compatible: bool = True
) -> np.ndarray:
    """
    Extract state labels from the first row of the TPM.

    In VBA, state labels are stored in row 1 (0 in Python), columns starting at col_offset.

    Parameters
    ----------
    tpm : np.ndarray
        Full transition probability matrix
    col_offset : int, optional
        Column offset for state labels
    vba_compatible : bool, optional
        If True, reads from row 0 (VBA row 1)

    Returns
    -------
    np.ndarray
        Array of state labels as strings
    """
    if vba_compatible:
        state_row = 0  # VBA row 1 = Python row 0
    else:
        state_row = 0

    return tpm[state_row, col_offset:].copy()
