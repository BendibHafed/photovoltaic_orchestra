"""
Population initialization for GA.
"""

import numpy as np


def initialize_population(pop_size, genome_length=5, random=True, init_params=None, noise=0.0):
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
    import numpy as np
    from pvoptix.pvoptix.optimization.ga.genome_mapping import encode_individual
    from pvoptix.pvoptix.optimization.ga.genome_mapping_double import encode_individual_double

    if not random:
        if init_params is None:
            raise ValueError("init_params must be provided when random=False")
        
        # Encode based on parameter count
        if genome_length == 7:
            base_individual = encode_individual_double(init_params)
        else:
            base_individual = encode_individual(init_params)

        if noise > 0.0:
            population = np.tile(base_individual, (pop_size, 1))
            population += np.random.normal(0, noise, population.shape)
            population = np.clip(population, 0.0, 1.0)
        else:
            population = np.tile(base_individual, (pop_size, 1))
    else:
        # Random uniform initialization in [0,1]
        population = np.random.rand(pop_size, genome_length)

    return population