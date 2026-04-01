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
    genome_length=None,
    auto_close_plot=True,
    plot_display_seconds=3,
):
    """
    Core GA loop.
    
    Parameters
    ----------
    objective_function : callable
        Function that takes an individual (normalized genome) and returns a float fitness
    pop_size : int
        Number of individuals
    generations : int
        Number of generations to evolve
    crossover_rate : float
        Probability of crossover
    mutation_rate : float
        Probability of mutation
    elitism : bool
        Whether to copy best individual to next generation
    random_init : bool
        Initialize population randomly or around init_params
    init_params : dict or None
        Predefined parameter set for initialization if random_init=False
    noise : float
        Noise applied when initializing around init_params
    tournament_size : int
        Tournament selection size
    diversity_prob : float
        Probability to select non-best in tournament
    verbose : bool
        Print progress
    live_plot : bool
        Show live matplotlib convergence
    figsize : tuple
        Figure size for live plotting
    on_progress : callable, optional
        Callback for progress updates
    should_cancel : callable, optional
        Cancellation callback
    genome_length : int, optional
        Explicit genome length. If None, inferred from init_params or default 5.
    auto_close_plot : bool
        If True, automatically close the plot after GA finishes
    plot_display_seconds : float
        Seconds to display the plot before auto-closing (if auto_close_plot=True)

    Returns
    -------
    best_params : dict
        Decoded best individual
    best_fitness : float
        Fitness of best individual
    history : list
        History of (fitness, decoded_params) per generation
    """
    # --- Import GA utilities locally ---
    import matplotlib

    from pvoptix.pvoptix.optimization.ga.crossover import crossover
    from pvoptix.pvoptix.optimization.ga.mutation import mutate
    from pvoptix.pvoptix.optimization.ga.population import initialize_population
    from pvoptix.pvoptix.optimization.ga.selection import tournament_selection
    from pvoptix.pvoptix.optimization.ga.genome_mapping import decode_individual
    from pvoptix.pvoptix.optimization.ga.genome_mapping_double import decode_individual_double

    # Set matplotlib backend based on live_plot setting
    if not live_plot:
        matplotlib.use("Agg")
    
    import matplotlib.pyplot as plt

    # Determine genome length
    if genome_length is None:
        if init_params is not None:
            genome_length = len(init_params)
        else:
            genome_length = 5  # Default for single-diode
    
    if seed is not None:
        np.random.seed(int(seed))

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

    # --- Setup live plotting ---
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

        # Decode the best individual for history (based on genome length)
        if genome_length == 7:
            decoded = decode_individual_double(gen_best_individual)
        else:
            decoded = decode_individual(gen_best_individual)
        
        history.append((gen_best_fitness, decoded))

        if on_progress is not None:
            # Report progress using 1-based generation index
            if genome_length == 7:
                progress_params = decode_individual_double(best_individual) if best_individual is not None else {}
            else:
                progress_params = decode_individual(best_individual) if best_individual is not None else {}
            on_progress(gen + 1, generations, float(best_fitness), progress_params)
            
        if verbose:
            mean_fit = float(np.mean(fitness_values))
            print(
                f"Gen {gen:03d} | Best fitness = {gen_best_fitness:.4e} | "
                f"Mean fitness = {mean_fit:.4e}"
            )

        # --- Live plot update ---
        if live_plot and fig is not None and ax is not None:
            rmse_best = [entry[0] for entry in history]
            rmse_mean = rmse_best[:-1] + [np.mean(fitness_values)]
            line_best.set_xdata(range(len(rmse_best)))
            line_best.set_ydata(rmse_best)
            line_mean.set_xdata(range(len(rmse_mean)))
            line_mean.set_ydata(rmse_mean)
            
            # Update axis limits
            ax.relim()
            ax.autoscale_view()
            
            # Redraw
            plt.draw()
            plt.pause(0.01)

        # --- Selection, crossover, mutation ---
        parents = tournament_selection(
            population,
            fitness_values,
            tournament_size=tournament_size,
            diversity_prob=diversity_prob,
        )
        offspring = crossover(parents, crossover_rate)
        offspring = mutate(offspring, mutation_rate)

        # --- Elitism ---
        if elitism:
            worst_idx = np.argmax(fitness_values)
            offspring[worst_idx] = gen_best_individual

        population = offspring

    # --- Return decoded best individual ---
    if best_individual is not None:
        if genome_length == 7:
            best_params = decode_individual_double(best_individual)
        else:
            best_params = decode_individual(best_individual)
    else:
        best_params = {}
        
    # --- Handle plot display and auto-closing ---
    if live_plot and fig is not None:
        plt.ioff()
        
        if auto_close_plot:
            # Non-blocking display with auto-close
            plt.show(block=False)
            print(f"\n📊 GA convergence plot displayed. Closing in {plot_display_seconds} seconds...")
            plt.pause(plot_display_seconds)
            plt.close(fig)
        else:
            # Block until user closes the window
            plt.show()
            
    return best_params, best_fitness, history