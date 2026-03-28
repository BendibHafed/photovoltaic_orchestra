"""
PV-specific objective function for single-diode model (5 parameters).
"""

from pvoptix.pvoptix.optimization.ga.fitness import global_rmse
from pvoptix.pvoptix.optimization.ga.genome_mapping import decode_individual
from pvoptix.pvoptix.models.parameters import default_coeffs
import numpy as np


def pv_rmse_objective(
    individual: np.ndarray,
    datasets: list[dict],
    Ns: int,
    coefficients=None,
) -> float:
    """
    Evaluate the RMSE of a single diode PV model for a given genome.

    Args:
        individual: Normalized genome vector [0,1] of length 5
        datasets: Experimental I-V datasets
        Ns: Number of series-connected cells
        coefficients: Model coefficients

    Returns:
        Global RMSE across all datasets
    """
    if coefficients is None:
        coefficients = default_coeffs

    stc_params = decode_individual(individual)
    return global_rmse(stc_params, datasets, Ns, coefficients=coefficients)