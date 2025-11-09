"""
Random Number Generation Utilities

Provides consistent random number generation to replace VBA's Rnd() function.
Uses numpy's modern random number generation with proper seeding for reproducibility.
"""

import numpy as np
from typing import Optional, Union


class RandomGenerator:
    """
    Wrapper for numpy random number generation.

    Provides a clean interface for generating random numbers with proper seeding
    to ensure reproducibility of stochastic simulations.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random number generator.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. If None, results will be non-deterministic.
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def random(self) -> float:
        """
        Generate a random float in [0.0, 1.0).

        Equivalent to VBA's Rnd() function.

        Returns
        -------
        float
            Random number between 0.0 (inclusive) and 1.0 (exclusive)
        """
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
        """
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
        return self.rng.uniform(low, high, size)


# Global random generator instance
# Can be re-initialized with a seed for reproducible results
_global_rng: Optional[RandomGenerator] = None


def set_seed(seed: Optional[int] = None):
    """
    Set the global random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value. If None, random behavior will be non-deterministic.
    """
    global _global_rng
    _global_rng = RandomGenerator(seed)


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
