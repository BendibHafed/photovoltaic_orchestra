"""
Mutation operators for GA.
"""

import numpy as np


def mutate(
    population: np.ndarray,
    mutation_rate: float = 0.1,
    sigma: float = 0.05,
) -> np.ndarray:
    """
    Apply Gaussian mutation to a normalized population.

    Args:
        population: Population of normalized individuals [0,1]
        mutation_rate: Probability of mutation per gene
        sigma: Standard deviation of Gaussian noise

    Returns:
        Mutated population (clipped to [0,1])
    """
    mutated = np.array(population, copy=True)
    pop_size, genome_length = mutated.shape

    for i in range(pop_size):
        for j in range(genome_length):
            if np.random.rand() < mutation_rate:
                mutated[i, j] += np.random.normal(0.0, sigma)

    return np.clip(mutated, 0.0, 1.0)