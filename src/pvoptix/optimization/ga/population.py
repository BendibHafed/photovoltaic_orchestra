"""
Population initialization for GA.
"""

import numpy as np
from pvoptix.optimization.ga.genome_mapping_double import encode_individual_double


def initialize_population(
    pop_size: int,
    genome_length: int = 7,
    random: bool = True,
    init_params: dict = None,
    noise: float = 0.0,
) -> np.ndarray:
    """
    Initialize a population of individuals for the GA.

    Args:
        pop_size: Number of individuals
        genome_length: Number of genes (7 for double-diode)
        random: If True, initialize randomly. If False, use init_params.
        init_params: Initial parameter set (if random=False)
        noise: Noise level for diversity around init_params

    Returns:
        Population array of shape (pop_size, genome_length) in [0,1]
    """
    if not random:
        if init_params is None:
            raise ValueError("init_params must be provided when random=False")

        # Encode initial parameters to normalized genome
        base_individual = encode_individual_double(init_params)

        if noise > 0.0:
            # Add Gaussian noise for diversity
            population = np.tile(base_individual, (pop_size, 1))
            population += np.random.normal(0, noise, population.shape)
            population = np.clip(population, 0.0, 1.0)
        else:
            # Exact replication
            population = np.tile(base_individual, (pop_size, 1))
    else:
        # Random uniform initialization in [0,1]
        population = np.random.rand(pop_size, genome_length)

    return population