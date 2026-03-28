"""
Fitness function for double-diode model (7 parameters).

Implements Equation (13) from NCMAI'26 paper.
"""

import numpy as np

from pvoptix.pvoptix.models.parameters import (
    iph_model,
    i01_model,
    i02_model,
    rs_model,
    rsh_model_double,
    default_coeffs,
)
from pvoptix.pvoptix.optimization.ga.genome_mapping_double import decode_individual_double
from pvoptix.pvoptix.solvers.double import solve_current_double


def global_rmse_double(
    individual: np.ndarray,
    datasets: list[dict],
    Ns: int,
    coefficients=None,
) -> float:
    """
    Compute global RMSE for double-diode model across multiple datasets.

    Implements Equation (13) from NCMAI'26 paper.

    Args:
        individual: Normalized genome vector [0,1]⁷
        datasets: List of I-V datasets (each with V, I, T, G)
        Ns: Number of series-connected cells
        coefficients: Model coefficients

    Returns:
        Global RMSE across all datasets (lower is better)
    """
    if coefficients is None:
        coefficients = default_coeffs

    # Decode STC parameters from genome (individual is the normalized genome)
    stc_params = decode_individual_double(individual)

    Rs_stc = float(stc_params["Rs"])
    Rsh_stc = float(stc_params["Rsh"])
    I01_stc = float(stc_params["I01"])
    I02_stc = float(stc_params["I02"])
    Iph_stc = float(stc_params["Iph"])
    n1 = float(stc_params["n1"])
    n2 = float(stc_params["n2"])

    total_sq_error = 0.0
    total_points = 0

    for data in datasets:
        G = data["G"]
        T = data["T"]
        V_exp = data["V"]
        I_exp = data["I"]

        # Translate STC parameters to operating conditions
        Iph = iph_model(G, T, Iph_stc, alpha_I=coefficients.alpha_I)
        I01 = i01_model(T, I01_stc, n1)
        I02 = i02_model(T, I02_stc, n2)
        Rs = rs_model(
            G, T, Rs_stc,
            alpha_Rs=coefficients.alpha_Rs,
            beta_Rs=coefficients.beta_Rs,
            Rs_min=coefficients.Rs_min,
            Rs_max=coefficients.Rs_max,
        )
        Rsh = rsh_model_double(G, Rsh_stc)

        params_op = {
            "Rs": Rs, "Rsh": Rsh, "I01": I01, "I02": I02,
            "Iph": Iph, "n1": n1, "n2": n2,
        }

        # Calculate errors for this curve
        for V_meas, I_meas in zip(V_exp, I_exp, strict=True):
            try:
                I_cal = solve_current_double(V_meas, T, params_op, Ns)
                if not np.isfinite(I_cal):
                    raise ValueError("Invalid current")

                error = I_cal - I_meas
                total_sq_error += error * error
                total_points += 1

            except Exception:
                # Penalize solver failures
                total_sq_error += 1.0
                total_points += 1

    if total_points == 0:
        return float(1e6)

    return float(np.sqrt(total_sq_error / total_points))