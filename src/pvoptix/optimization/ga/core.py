"""
Genetic Algorithm (GA) core module.
Fully generic: works with any objective_function(individual) -> float.
"""

import numpy as np


class CancelledError(Exception):
    """Raised when GA execution is cancelled cooperatively."""


def run_ga(
    objective_function,
    pop_size=40,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism=True,
    random_init=True,
    init_params=None,
    noise=0.0,
    seed=None,
    tournament_size=3,
    diversity_prob=0.1,
    verbose=True,
    live_plot=False,
    figsize=(10, 4),
    on_progress=None,
    should_cancel=None,
    genome_length=7,
    auto_close_plot=True,
    plot_display_seconds=10,
):
    """
    Generic Genetic Algorithm engine.

    Args:
        objective_function: Function individual -> fitness (lower is better)
        pop_size: Population size
        generations: Number of generations
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability per gene
        elitism: Keep best individual
        random_init: Random initialization vs using init_params
        init_params: Initial parameter set (if random_init=False)
        noise: Noise for diversity around init_params
        seed: Random seed
        tournament_size: Tournament selection size
        diversity_prob: Probability to select non-best (diversity)
        verbose: Print progress
        live_plot: Show live convergence plot
        figsize: Figure size for live plotting
        on_progress: Progress callback
        should_cancel: Cancellation callback
        genome_length: Number of genes (7 for double-diode)
        auto_close_plot: Auto-close plot after GA finishes
        plot_display_seconds: Seconds to display plot before closing

    Returns:
        Tuple of (best_params, best_fitness, history)
    """
    # Local imports (to avoid circular imports)
    import matplotlib

    from pvoptix.optimization.ga.crossover import crossover
    from pvoptix.optimization.ga.mutation import mutate
    from pvoptix.optimization.ga.population import initialize_population
    from pvoptix.optimization.ga.selection import tournament_selection
    from pvoptix.optimization.ga.genome_mapping_double import decode_individual_double

    # Note: Do not change the global matplotlib backend here.
    # The user should set their preferred backend before calling this function.
    # This prevents interfering with interactive plots in the main program.
    # if not live_plot:
    #     matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if seed is not None:
        np.random.seed(int(seed))

    # Initialize population
    population = initialize_population(
        pop_size,
        genome_length=genome_length,
        random=random_init,
        init_params=init_params,
        noise=noise,
    )

    history = []
    best_individual = None
    best_fitness = float("inf")

    # Setup live plot
    fig = None
    ax = None
    line_best = None
    line_mean = None

    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=figsize)
        (line_best,) = ax.plot([], [], "b-o", label="Best fitness")
        (line_mean,) = ax.plot([], [], "r--", label="Mean fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("GA Convergence")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

    # Main GA loop
    for gen in range(generations):
        if should_cancel is not None and should_cancel():
            raise CancelledError()

        # Evaluate population
        fitness_values = np.array([objective_function(ind) for ind in population])

        # Track best
        gen_best_idx = int(np.argmin(fitness_values))
        gen_best_fitness = fitness_values[gen_best_idx]
        gen_best_individual = population[gen_best_idx]

        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual

        # Decode best individual for history
        decoded = decode_individual_double(gen_best_individual)
        history.append((gen_best_fitness, decoded))

        # Progress reporting
        if on_progress is not None:
            best_decoded = decode_individual_double(best_individual) if best_individual is not None else {}
            on_progress(gen + 1, generations, float(best_fitness), best_decoded)

        if verbose:
            mean_fit = float(np.mean(fitness_values))
            print(f"Gen {gen:03d} | Best = {gen_best_fitness:.4e} | Mean = {mean_fit:.4e}")

        # Update live plot
        if live_plot and fig is not None:
            rmse_best = [entry[0] for entry in history]
            rmse_mean = rmse_best[:-1] + [np.mean(fitness_values)]
            line_best.set_xdata(range(len(rmse_best)))
            line_best.set_ydata(rmse_best)
            line_mean.set_xdata(range(len(rmse_mean)))
            line_mean.set_ydata(rmse_mean)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

        # Selection, crossover, mutation
        parents = tournament_selection(
            population, fitness_values, tournament_size, diversity_prob
        )
        offspring = crossover(parents, crossover_rate, method="uniform")
        offspring = mutate(offspring, mutation_rate)

        # Elitism
        if elitism:
            worst_idx = np.argmax(fitness_values)
            offspring[worst_idx] = gen_best_individual

        population = offspring

    # Decode best individual
    if best_individual is not None:
        best_params = decode_individual_double(best_individual)
    else:
        best_params = {}

    # Handle plot closing
    if live_plot and fig is not None:
        plt.ioff()
        if auto_close_plot:
            plt.show(block=False)
            print(f"\nPlot closing in {plot_display_seconds} seconds...")
            plt.pause(plot_display_seconds)
            plt.close(fig)
        else:
            plt.show()

    return best_params, best_fitness, history