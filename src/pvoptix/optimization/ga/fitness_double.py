"""
Fitness function for double-diode model (7 parameters).

Implements Equation (14) from NCMAI'26 paper:
RMSE_global = sqrt( 1/M * sum_{j=1}^{M} (1/N_j * sum_{i=1}^{N_j} (I_mes - I_cal)²) )
"""

import numpy as np

from pvoptix.models.parameters import (
    iph_model,
    i01_model,
    i02_model,
    rs_model,
    rsh_model_double,
    default_coeffs,
)
from pvoptix.optimization.ga.genome_mapping_double import decode_individual_double
from pvoptix.solvers.double import solve_current_double


def global_rmse_double(
    individual: np.ndarray,
    datasets: list[dict],
    Ns: int,
    coefficients=None,
) -> float:
    """
    Compute global RMSE for double-diode model across multiple datasets.

    Implements Equation (14) from NCMAI'26 paper:
    RMSE_global = sqrt( 1/M * sum_{j=1}^{M} (1/N_j * sum_{i=1}^{N_j} (I_mes - I_cal)²) )

    The RMSE is computed in absolute Amperes (no normalization by I_ph),
    which matches the paper's Equation (11) and provides physically meaningful units.
    """
    if coefficients is None:
        coefficients = default_coeffs

    stc_params = decode_individual_double(individual)

    Rs_stc = float(stc_params["Rs"])
    Rsh_stc = float(stc_params["Rsh"])
    I01_stc = float(stc_params["I01"])
    I02_stc = float(stc_params["I02"])
    Iph_stc = float(stc_params["Iph"])
    n1 = float(stc_params["n1"])
    n2 = float(stc_params["n2"])

    sum_condition_mse = 0.0
    num_conditions = len(datasets)

    for data in datasets:
        G = data["G"]
        T = data["T"]
        V_exp = data["V"]
        I_exp = data["I"]

        # Translate STC to operating conditions
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

        # Compute mean squared error for this condition (absolute error in Amperes)
        sq_errors = []
        for V_meas, I_meas in zip(V_exp, I_exp, strict=True):
            try:
                I_cal = solve_current_double(V_meas, T, params_op, Ns)
                if not np.isfinite(I_cal):
                    raise ValueError("Invalid current")
                # Absolute error squared (Amperes²) - NO division by I_ph
                sq_errors.append((I_cal - I_meas) ** 2)
            except Exception:
                # Penalize solver failures (1A error)
                sq_errors.append(1.0)

        # MSE for this condition (Amperes²)
        condition_mse = np.mean(sq_errors)
        sum_condition_mse += condition_mse

    # Equation (14): outer square root gives RMSE in Amperes
    global_rmse = np.sqrt(sum_condition_mse / num_conditions)

    return float(global_rmse)