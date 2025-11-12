"""
Random Number Generation Utilities

Provides consistent random number generation to replace VBA's Rnd() function.
Uses numpy's modern random number generation with proper seeding for reproducibility.

Also provides PortableLCG - a Linear Congruential Generator that produces
identical sequences in both Python and VBA for cross-platform validation.
"""

import numpy as np
from typing import Optional, Union


class PortableLCG:
    """
    Portable Linear Congruential Generator (LCG) for cross-platform validation.

    This RNG produces identical sequences in both Python and VBA when given
    the same seed. Uses parameters from Numerical Recipes:
    - Multiplier (a) = 1664525
    - Increment (c) = 1013904223
    - Modulus (m) = 2^32

    Formula: next = (a * current + c) mod m
    Output: next / m to get float in [0, 1)

    This implementation is designed to be bit-identical to the VBA version.
    """

    # LCG parameters (Numerical Recipes)
    MULTIPLIER = 1664525
    INCREMENT = 1013904223
    MODULUS = 2**32  # 4294967296

    def __init__(self, seed: int = 1, debug: bool = False):
        """
        Initialize the LCG with a seed.

        Parameters
        ----------
        seed : int
            Seed value (will be taken modulo 2^32)
        debug : bool
            If True, log every random call with caller info
        """
        # Ensure seed is in valid range [0, MODULUS)
        self.state = seed % self.MODULUS
        self.initial_seed = self.state
        self.debug = debug
        self.call_count = 0

    def random(self) -> float:
        """
        Generate the next random float in [0.0, 1.0).

        Returns
        -------
        float
            Random number between 0.0 (inclusive) and 1.0 (exclusive)
        """
        # Update state: next = (a * current + c) mod m
        self.state = (self.MULTIPLIER * self.state + self.INCREMENT) % self.MODULUS

        # Convert to float in [0, 1)
        result = self.state / self.MODULUS

        # Debug logging
        if self.debug:
            import traceback
            import inspect
            self.call_count += 1

            # Get caller info
            frame = inspect.currentframe().f_back
            caller_filename = frame.f_code.co_filename.split('/')[-1]
            caller_function = frame.f_code.co_name
            caller_line = frame.f_lineno

            # Get more context - who called the caller
            if frame.f_back:
                parent_frame = frame.f_back
                parent_filename = parent_frame.f_code.co_filename.split('/')[-1]
                parent_function = parent_frame.f_code.co_name
                parent_line = parent_frame.f_lineno
                location = f"{parent_filename}:{parent_function}:{parent_line} → {caller_filename}:{caller_function}:{caller_line}"
            else:
                location = f"{caller_filename}:{caller_function}:{caller_line}"

            print(f"Call #{self.call_count:4d}: {result:.17f}  @ {location}")

        return result

    def randint(self, low: int, high: int) -> int:
        """
        Generate a random integer in [low, high).

        Parameters
        ----------
        low : int
            Lowest integer (inclusive)
        high : int
            Highest integer (exclusive)

        Returns
        -------
        int
            Random integer in [low, high)
        """
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")

        # Scale random() to the desired range
        range_size = high - low
        return low + int(self.random() * range_size)

    def reset(self):
        """Reset the generator to its initial seed."""
        self.state = self.initial_seed


class RandomGenerator:
    """
    Wrapper for random number generation.

    Provides a clean interface for generating random numbers with proper seeding
    to ensure reproducibility of stochastic simulations.

    Supports two modes:
    - use_portable_lcg=False: Uses numpy's modern RNG (default)
    - use_portable_lcg=True: Uses PortableLCG for cross-platform validation with VBA
    """

    def __init__(self, seed: Optional[int] = None, use_portable_lcg: bool = False, debug: bool = False):
        """
        Initialize the random number generator.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. If None, results will be non-deterministic.
            Note: PortableLCG requires a seed, so if None and use_portable_lcg=True, seed=1 is used.
        use_portable_lcg : bool, optional
            If True, use PortableLCG for cross-platform validation. Default: False (use numpy).
        debug : bool, optional
            If True, log every random call with caller info. Default: False.
        """
        self.seed = seed
        self.use_portable_lcg = use_portable_lcg
        self.debug = debug

        if use_portable_lcg:
            # Use PortableLCG for cross-platform validation
            if seed is None:
                seed = 1  # PortableLCG requires a seed
            self._lcg = PortableLCG(seed, debug=debug)
            self.rng = None
        else:
            # Use numpy's default RNG
            self.rng = np.random.default_rng(seed)
            self._lcg = None

    def random(self) -> float:
        """
        Generate a random float in [0.0, 1.0).

        Equivalent to VBA's Rnd() function.

        Returns
        -------
        float
            Random number between 0.0 (inclusive) and 1.0 (exclusive)
        """
        if self.use_portable_lcg:
            return self._lcg.random()
        else:
            return self.rng.random()

    def randint(self, low: int, high: int) -> int:
        """
        Generate a random integer in [low, high).

        Parameters
        ----------
        low : int
            Lowest integer to be drawn (inclusive)
        high : int
            One above the highest integer to be drawn (exclusive)

        Returns
        -------
        int
            Random integer in [low, high)
        """
        if self.use_portable_lcg:
            return self._lcg.randint(low, high)
        else:
            return self.rng.integers(low, high)

    def choice(self, a: Union[np.ndarray, list], p: Optional[np.ndarray] = None):
        """
        Generate a random sample from a given array.

        Parameters
        ----------
        a : array_like
            Array to sample from
        p : array_like, optional
            Probabilities associated with each entry in a. If None, uniform distribution.

        Returns
        -------
        single item or ndarray
            Random sample from a
        """
        if self.use_portable_lcg:
            # Implement choice using portable LCG
            if p is None:
                # Uniform choice
                idx = self._lcg.randint(0, len(a))
                return a[idx]
            else:
                # Weighted choice using cumulative probabilities
                r = self._lcg.random()
                cumsum = 0.0
                for i, prob in enumerate(p):
                    cumsum += prob
                    if r < cumsum:
                        return a[i]
                return a[-1]  # In case of rounding errors
        else:
            return self.rng.choice(a, p=p)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Draw samples from a normal (Gaussian) distribution.

        Parameters
        ----------
        loc : float
            Mean of the distribution
        scale : float
            Standard deviation of the distribution
        size : int or tuple of ints, optional
            Output shape. If None, return a single value.

        Returns
        -------
        float or ndarray
            Random samples from the normal distribution

        Notes
        -----
        When use_portable_lcg=True, uses inverse CDF method (scipy.stats.norm.ppf)
        to match Excel's NormInv function. This ensures exactly 1 random call per value,
        maintaining synchronization with VBA code.
        """
        if self.use_portable_lcg:
            # Use inverse CDF method to match Excel's NormInv
            # This uses exactly 1 random call per value (critical for sync with VBA)
            from scipy import stats

            EPSILON = 1e-10  # Match VBA clamping

            if size is None:
                # Special case: If SD is 0 or very small, return the mean (no variation possible)
                # Note: We do NOT skip when mean=0, because VBA's NormInv always calls Rnd()
                if scale <= 1e-7:
                    return loc

                u = self._lcg.random()
                # Clamp to avoid edge cases (match Excel's NormInv requirements)
                if u <= EPSILON:
                    u = EPSILON
                if u >= (1.0 - EPSILON):
                    u = 1.0 - EPSILON
                # Inverse normal CDF (equivalent to Excel's NormInv)
                z = stats.norm.ppf(u)
                return loc + scale * z
            else:
                # Generate array of samples
                result = []
                for _ in range(size):
                    # Special case: If SD is 0 or very small, return the mean (no variation possible)
                    # Note: We do NOT skip when mean=0, because VBA's NormInv always calls Rnd()
                    if scale <= 1e-7:
                        result.append(loc)
                        continue

                    u = self._lcg.random()
                    # Clamp to avoid edge cases
                    if u <= EPSILON:
                        u = EPSILON
                    if u >= (1.0 - EPSILON):
                        u = 1.0 - EPSILON
                    z = stats.norm.ppf(u)
                    result.append(loc + scale * z)
                return np.array(result)
        else:
            return self.rng.normal(loc, scale, size)

    def exponential(self, scale: float = 1.0, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Draw samples from an exponential distribution.

        Parameters
        ----------
        scale : float
            Scale parameter (1/lambda)
        size : int or tuple of ints, optional
            Output shape. If None, return a single value.

        Returns
        -------
        float or ndarray
            Random samples from the exponential distribution
        """
        if self.use_portable_lcg:
            # Inverse transform: -scale * log(1 - U)
            import math
            if size is None:
                u = self._lcg.random()
                # Avoid log(0)
                if u > 0.9999999999:
                    u = 0.9999999999
                return -scale * math.log(1.0 - u)
            else:
                result = []
                for _ in range(size):
                    u = self._lcg.random()
                    if u > 0.9999999999:
                        u = 0.9999999999
                    result.append(-scale * math.log(1.0 - u))
                return np.array(result)
        else:
            return self.rng.exponential(scale, size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Draw samples from a uniform distribution.

        Parameters
        ----------
        low : float
            Lower boundary of the output interval
        high : float
            Upper boundary of the output interval
        size : int or tuple of ints, optional
            Output shape. If None, return a single value.

        Returns
        -------
        float or ndarray
            Random samples from the uniform distribution
        """
        if self.use_portable_lcg:
            if size is None:
                return low + (high - low) * self._lcg.random()
            else:
                result = []
                for _ in range(size):
                    result.append(low + (high - low) * self._lcg.random())
                return np.array(result)
        else:
            return self.rng.uniform(low, high, size)


# Global random generator instance
# Can be re-initialized with a seed for reproducible results
_global_rng: Optional[RandomGenerator] = None
_use_portable_lcg: bool = False


def set_seed(seed: Optional[int] = None, use_portable_lcg: bool = False, debug: bool = False):
    """
    Set the global random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value. If None, random behavior will be non-deterministic.
    use_portable_lcg : bool, optional
        If True, use PortableLCG for cross-platform validation. Default: False.
    debug : bool, optional
        If True, log every random call with caller info. Default: False.
    """
    global _global_rng, _use_portable_lcg
    _use_portable_lcg = use_portable_lcg
    _global_rng = RandomGenerator(seed, use_portable_lcg=use_portable_lcg, debug=debug)


def get_rng() -> RandomGenerator:
    """
    Get the global random number generator.

    Returns
    -------
    RandomGenerator
        The global random generator instance
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = RandomGenerator()
    return _global_rng


# Convenience functions that use the global RNG
def random() -> float:
    """Generate a random float in [0.0, 1.0) using the global RNG."""
    return get_rng().random()


def randint(low: int, high: int) -> int:
    """Generate a random integer in [low, high) using the global RNG."""
    return get_rng().randint(low, high)


def choice(a: Union[np.ndarray, list], p: Optional[np.ndarray] = None):
    """Generate a random sample from array using the global RNG."""
    return get_rng().choice(a, p=p)


def normal(loc: float = 0.0, scale: float = 1.0, size: Optional[int] = None) -> Union[float, np.ndarray]:
    """Draw samples from normal distribution using the global RNG."""
    return get_rng().normal(loc, scale, size)


def exponential(scale: float = 1.0, size: Optional[int] = None) -> Union[float, np.ndarray]:
    """Draw samples from exponential distribution using the global RNG."""
    return get_rng().exponential(scale, size)


def uniform(low: float = 0.0, high: float = 1.0, size: Optional[int] = None) -> Union[float, np.ndarray]:
    """Draw samples from uniform distribution using the global RNG."""
    return get_rng().uniform(low, high, size)
