"""
Tournament selection for GA.
"""

import numpy as np


def tournament_selection(
    population: np.ndarray,
    fitness_values: np.ndarray,
    tournament_size: int = 3,
    diversity_prob: float = 0.1,
) -> np.ndarray:
    """
    Tournament selection for GA.

    Args:
        population: Current population of shape (pop_size, genome_length)
        fitness_values: Fitness values (RMSE) for each individual
        tournament_size: Number of individuals per tournament
        diversity_prob: Probability of selecting a non-best candidate

    Returns:
        Selected parent population
    """
    pop_size, genome_length = population.shape
    parents = np.zeros_like(population)

    for i in range(pop_size):
        # Randomly choose tournament participants
        candidates_idx = np.random.choice(pop_size, tournament_size, replace=False)
        candidate_fitness = fitness_values[candidates_idx]

        # Sort candidates by fitness (ascending RMSE)
        sorted_idx = candidates_idx[np.argsort(candidate_fitness)]

        # Normally pick the best, but sometimes pick another for diversity
        if np.random.rand() < diversity_prob and len(sorted_idx) > 1:
            winner_idx = np.random.choice(sorted_idx[1:])
        else:
            winner_idx = sorted_idx[0]

        parents[i] = population[winner_idx]

    return parents