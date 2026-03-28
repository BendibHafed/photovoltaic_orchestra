"""
Fitness function for single-diode model (5 parameters).
"""

import numpy as np

from pvoptix.pvoptix.models.parameters import (
    iph_model,
    i0_model,
    rs_model,
    rsh_model_double,
    default_coeffs,
)
from pvoptix.pvoptix.solvers.single import solve_current_single


def global_rmse(
    stc_params: dict[str, float],
    datasets: list[dict],
    Ns: int,
    coefficients=None,
) -> float:
    """
    Compute normalized global RMSE over all I–V curves for single-diode model.

    Args:
        stc_params: STC parameters (Rs, Rsh, I0, Iph, n)
        datasets: List of I-V datasets
        Ns: Number of cells in series
        coefficients: Model coefficients

    Returns:
        Global RMSE value (lower is better)
    """
    if coefficients is None:
        coefficients = default_coeffs

    Rs_stc = float(stc_params["Rs"])
    Rsh_stc = float(stc_params["Rsh"])
    I0_stc = float(stc_params["I0"])
    Iph_stc = float(stc_params["Iph"])
    n = float(stc_params["n"])

    total_sq_error = 0.0
    total_points = 0

    for data in datasets:
        G = data["G"]
        T = data["T"]
        V_exp = data["V"]
        I_exp = data["I"]

        # Translate STC parameters to operating conditions
        Iph = iph_model(G, T, Iph_stc, alpha_I=coefficients.alpha_I)
        I0 = i0_model(T, I0_stc, n)
        Rs = rs_model(
            G, T, Rs_stc,
            alpha_Rs=coefficients.alpha_Rs,
            beta_Rs=coefficients.beta_Rs,
            Rs_min=coefficients.Rs_min,
            Rs_max=coefficients.Rs_max,
        )
        Rsh = rsh_model_double(G, Rsh_stc)

        params_op = {"Rs": Rs, "Rsh": Rsh, "I0": I0, "Iph": Iph, "n": n}

        # Calculate errors for this curve
        for V_meas, I_meas in zip(V_exp, I_exp, strict=True):
            try:
                I_cal = solve_current_single(V_meas, T, params_op, Ns)
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