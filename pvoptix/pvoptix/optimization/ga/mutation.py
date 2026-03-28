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

    Notes:
        - Assumes genes are normalized in [0, 1].
        - Mutation is applied per gene with probability `mutation_rate`.

    Args:
        population: Population of normalized individuals
        mutation_rate: Probability of mutation per gene
        sigma: Standard deviation of Gaussian noise

    Returns:
        Mutated population
    """
    mutated = np.array(population, copy=True)

    pop_size, genome_length = mutated.shape
    for i in range(pop_size):
        for j in range(genome_length):
            if np.random.rand() < mutation_rate:
                mutated[i, j] += np.random.normal(0.0, sigma)

    return np.clip(mutated, 0.0, 1.0)