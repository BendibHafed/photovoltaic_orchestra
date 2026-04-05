"""
PV-specific objective function for double-diode model (7 parameters).
"""

import numpy as np

from pvoptix.optimization.ga.fitness_double import global_rmse_double
from pvoptix.optimization.ga.genome_mapping_double import decode_individual_double
from pvoptix.models.parameters import default_coeffs


def pv_rmse_objective_double(
    individual: np.ndarray,
    datasets: list[dict],
    Ns: int,
    coefficients=None,
) -> float:
    """
    Evaluate the RMSE of a double-diode PV model for a given genome.

    Args:
        individual: Normalized genome vector [0,1] of length 7
        datasets: Experimental I-V datasets
        Ns: Number of series-connected cells
        coefficients: Model coefficients

    Returns:
        Global RMSE across all datasets
    """
    if coefficients is None:
        coefficients = default_coeffs

    # Directly pass the individual to global_rmse_double
    return global_rmse_double(individual, datasets, Ns, coefficients=coefficients)