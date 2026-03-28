"""
Crossover operators for GA.
"""

import numpy as np


def crossover(
    parents: np.ndarray,
    crossover_rate: float = 0.8,
    method: str = "one_point",
    uniform_rate: float = 0.5,
) -> np.ndarray:
    """
    Apply crossover to the parent population to generate offspring.

    Parameters
    ----------
    parents : ndarray
        Parent population of shape (pop_size, genome_length)
    crossover_rate : float
        Probability of crossover (default = 0.8)
    method : str
        Crossover method: "one_point", "two_point", or "uniform"
    uniform_rate : float
        Probability of taking gene from parent1 in uniform crossover

    Returns
    -------
    offspring : ndarray
        New population after crossover
    """
    pop_size, genome_length = parents.shape
    offspring = np.copy(parents)

    for i in range(0, pop_size - 1, 2):  # Process in pairs
        parent1, parent2 = parents[i], parents[i + 1]

        if np.random.rand() < crossover_rate:
            if method == "one_point":
                point = np.random.randint(1, genome_length)
                offspring[i, :point] = parent1[:point]
                offspring[i, point:] = parent2[point:]
                offspring[i + 1, :point] = parent2[:point]
                offspring[i + 1, point:] = parent1[point:]

            elif method == "two_point":
                p1, p2 = sorted(
                    np.random.choice(range(1, genome_length), 2, replace=False)
                )
                offspring[i, p1:p2] = parent2[p1:p2]
                offspring[i + 1, p1:p2] = parent1[p1:p2]

            elif method == "uniform":
                mask = np.random.rand(genome_length) < uniform_rate
                offspring[i] = np.where(mask, parent1, parent2)
                offspring[i + 1] = np.where(mask, parent2, parent1)

    return offspring