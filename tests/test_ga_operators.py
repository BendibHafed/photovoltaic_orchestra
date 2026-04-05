# tests/test_ga_operators.py
"""Test GA operators: crossover, mutation, selection, population."""

import numpy as np
import pytest
from pvoptix.optimization.ga.crossover import crossover
from pvoptix.optimization.ga.mutation import mutate
from pvoptix.optimization.ga.selection import tournament_selection
from pvoptix.optimization.ga.population import initialize_population
from pvoptix.optimization.ga.genome_mapping_double import BOUNDS_DOUBLE


class TestGAOperators:
    """Test GA operators."""

    def test_crossover_output_shape(self):
        """Crossover should return the same shape as the input."""
        pop_size = 10
        genome_length = 7
        parents = np.random.rand(pop_size, genome_length)

        offspring = crossover(parents, crossover_rate=0.8)

        assert offspring.shape == parents.shape

    def test_crossover_values_in_range(self):
        """Offspring values should remain within [0, 1]."""
        parents = np.random.rand(10, 7)
        offspring = crossover(parents, crossover_rate=0.8)

        assert np.all(offspring >= 0)
        assert np.all(offspring <= 1)

    def test_crossover_no_crossover(self):
        """With crossover_rate=0, offspring should equal parents."""
        parents = np.random.rand(10, 7)
        offspring = crossover(parents, crossover_rate=0.0)

        assert np.allclose(offspring, parents)

    def test_mutation_output_shape(self):
        """Mutation should return the same shape as the input."""
        population = np.random.rand(10, 7)
        mutated = mutate(population, mutation_rate=0.1)

        assert mutated.shape == population.shape

    def test_mutation_values_in_range(self):
        """Mutated values should remain within [0, 1]."""
        population = np.random.rand(10, 7)
        mutated = mutate(population, mutation_rate=0.5)

        assert np.all(mutated >= 0)
        assert np.all(mutated <= 1)

    def test_mutation_no_mutation(self):
        """With mutation_rate=0, the population should remain unchanged."""
        population = np.random.rand(10, 7)
        original = population.copy()
        mutated = mutate(population, mutation_rate=0.0)

        assert np.allclose(mutated, original)

    def test_tournament_selection_output_shape(self):
        """Tournament selection should return the correct shape."""
        pop_size = 10
        genome_length = 7
        population = np.random.rand(pop_size, genome_length)
        fitness = np.random.rand(pop_size)

        parents = tournament_selection(population, fitness, tournament_size=3)

        assert parents.shape == population.shape

    def test_tournament_selection_selects_best(self):
        """Tournament selection should favor better individuals."""
        pop_size = 10
        genome_length = 7
        population = np.random.rand(pop_size, genome_length)
        fitness = np.random.rand(pop_size)
        fitness[0] = 0.0  # Best individual
        fitness[1:] = 1.0  # Others worse

        n_trials = 100
        found_count = 0
        
        for _ in range(n_trials):
            parents = tournament_selection(population, fitness, tournament_size=3, diversity_prob=0.0)
            found = np.any([np.allclose(parents[i], population[0]) for i in range(pop_size)])
            if found:
                found_count += 1
    
        assert found_count > n_trials * 0.5, f"Best selected only {found_count}/{n_trials} times"

    def test_initialize_population_random(self):
        """Random population initialization should produce valid values."""
        pop_size = 10
        genome_length = 7
        population = initialize_population(pop_size, genome_length, random=True)

        assert population.shape == (pop_size, genome_length)
        assert np.all(population >= 0)
        assert np.all(population <= 1)

    def test_initialize_population_with_init_params(self, true_params):
        """Initialization with given parameters should produce identical individuals when noise=0."""
        pop_size = 10
        population = initialize_population(
            pop_size,
            genome_length=7,
            random=False,
            init_params=true_params,
            noise=0.0
        )

        assert np.allclose(population[0], population[1])

    def test_initialize_population_with_noise(self, true_params):
        """Initialization with noise should add diversity among individuals."""
        pop_size = 10
        population_no_noise = initialize_population(
            pop_size, genome_length=7, random=False, init_params=true_params, noise=0.0
        )
        population_with_noise = initialize_population(
            pop_size, genome_length=7, random=False, init_params=true_params, noise=0.05
        )

        assert not np.allclose(population_no_noise, population_with_noise)


if __name__ == "__main__":
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }

    test = TestGAOperators()
    test.test_crossover_output_shape()
    test.test_crossover_values_in_range()
    test.test_crossover_no_crossover()
    test.test_mutation_output_shape()
    test.test_mutation_values_in_range()
    test.test_mutation_no_mutation()
    test.test_tournament_selection_output_shape()
    test.test_tournament_selection_selects_best()
    test.test_initialize_population_random()
    test.test_initialize_population_with_init_params(true_params)
    test.test_initialize_population_with_noise(true_params)
    print("All GA operators tests passed.")
