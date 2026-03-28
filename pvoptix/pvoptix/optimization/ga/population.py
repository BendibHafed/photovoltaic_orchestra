"""
Population initialization for GA.
"""

import numpy as np


def initialize_population(
    pop_size: int,
    genome_length: int = 5,
    random: bool = True,
    init_params: dict = None,
    noise: float = 0.0,
) -> np.ndarray:
    """
    Initialize a population of individuals for the GA.
    Each individual is represented as a vector of normalized genes in [0,1].

    Parameters
    ----------
    pop_size : int
        Number of individuals in the population
    genome_length : int
        Number of genes per individual (5 for single, 7 for double)
    random : bool
        If True, initialize randomly. If False, initialize with predefined params.
    init_params : dict or None
        Dictionary of initial parameters if random=False
    noise : float
        Optional noise level (0.0 = no noise). Adds diversity around init_params.

    Returns
    -------
    population : ndarray
        Array of shape (pop_size, genome_length) with values in [0,1]
    """
    # For now, just return random population
    # Will be fully implemented when genome mapping is added
    if not random:
        if init_params is None:
            raise ValueError("init_params must be provided when random=False")
        # Placeholder: create population around init_params
        # Will use encode_individual when available
        base = np.array([0.5] * genome_length)
        population = np.tile(base, (pop_size, 1))
        if noise > 0.0:
            population += np.random.normal(0, noise, population.shape)
            population = np.clip(population, 0.0, 1.0)
    else:
        # Random uniform initialization in [0,1]
        population = np.random.rand(pop_size, genome_length)

    return population