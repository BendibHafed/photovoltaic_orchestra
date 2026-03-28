"""
Numerical solver for single-diode PV model equation.

Implements robust hybrid solver combining Brent's method and Newton-Raphson.
"""

import numpy as np
from scipy.optimize import brentq, fsolve

from pvoptix.pvoptix.models.single_diode import single_diode_equation


def solve_current_single(
    V: float,
    T: float,
    params: dict[str, float],
    Ns: int,
) -> float:
    """
    Solve the single-diode equation at a single voltage point.

    Args:
        V: Voltage [V]
        T: Temperature [K]
        params: PV module parameters with keys: Rs, Rsh, I0, Iph, n
        Ns: Number of cells in series

    Returns:
        Solved current [A] clamped to [0, Iph]
    """
    # Extract parameters
    Rs = float(params["Rs"])
    Rsh = float(params["Rsh"])
    I0 = float(params["I0"])
    Iph = float(params["Iph"])
    n = float(params["n"])

    # Physical constants
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = (k / q) * T

    # Estimate Voc for bracketing
    try:
        Voc_est = n * Vt * Ns * np.log1p(Iph / max(I0, 1e-12))
    except Exception:
        Voc_est = 21.6  # Fallback constant

    def equation(I_val: float) -> float:
        return single_diode_equation(
            I_val, V, T, Rs, Rsh, I0, Iph, n, Ns
        )

    # Initial guess
    if V < 0.5:
        I_guess = Iph
    else:
        I_guess = max(0.0, Iph * (1 - V / max(Voc_est, 1e-3)))

    # Try brentq first (bracket-based, robust)
    try:
        I_final = brentq(equation, -0.1 * Iph, 1.2 * Iph, maxiter=500)
    except Exception:
        # Fallback to fsolve (Newton-Raphson)
        try:
            I_final, info, ier, msg = fsolve(
                equation, I_guess, maxfev=500, full_output=True
            )
            if ier != 1:  # ier=1 means convergence
                I_final = 0.0
            else:
                I_final = I_final[0]
        except Exception:
            I_final = 0.0  # Last resort fallback

    # Clamp to physical range
    I_final = float(np.clip(I_final, 0.0, Iph))

    return I_final


def iv_model_single(
    V: np.ndarray,
    params: dict[str, float],
    T: float,
    Ns: int,
) -> np.ndarray:
    """
    Solve the I-V curve for a PV module using single-diode model.

    Args:
        V: Voltage array [V]
        params: PV module parameters (Rs, Rsh, I0, Iph, n)
        T: Temperature [K]
        Ns: Number of cells in series

    Returns:
        Current array [A]
    """
    V = np.atleast_1d(V)
    I_array = np.array([solve_current_single(v, T, params, Ns) for v in V])
    return I_array