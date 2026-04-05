"""
Numerical solver for double-diode PV model equation.

Implements robust hybrid solver combining Brent's method and Newton-Raphson.
"""

import numpy as np
from scipy.optimize import brentq, fsolve

from pvoptix.models.double_diode import double_diode_equation


def solve_current_double(
    V: float,
    T: float,
    params: dict[str, float],
    Ns: int,
) -> float:
    """
    Solve the double-diode equation at a single voltage point.

    Args:
        V: Voltage [V]
        T: Temperature [K]
        params: PV module parameters with keys: Rs, Rsh, I01, I02, Iph, n1, n2
        Ns: Number of cells in series

    Returns:
        Solved current [A] clamped to [0, Iph]
    """
    # Extract parameters
    Rs = float(params["Rs"])
    Rsh = float(params["Rsh"])
    I01 = float(params["I01"])
    I02 = float(params["I02"])
    Iph = float(params["Iph"])
    n1 = float(params["n1"])
    n2 = float(params["n2"])

    # Physical constants
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = (k / q) * T

    # Estimate Voc for bracketing
    try:
        a1 = n1 * Ns * Vt
        Voc_est = a1 * np.log(Iph / max(I01, 1e-12))
        Voc_est = min(Voc_est, 50.0)
    except Exception:
        Voc_est = 21.6  # Fallback for S75 module

    def equation(I_val: float) -> float:
        return double_diode_equation(
            I_val, V, T, Rs, Rsh, I01, I02, Iph, n1, n2, Ns
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
            I_final = 0.0

    # Clamp to physical range
    I_final = float(np.clip(I_final, 0.0, Iph))

    return I_final


def iv_model_double(
    V: np.ndarray,
    params: dict[str, float],
    T: float,
    Ns: int,
) -> np.ndarray:
    """
    Solve the I-V curve for a PV module using double-diode model.

    Args:
        V: Voltage array [V]
        params: PV module parameters (Rs, Rsh, I01, I02, Iph, n1, n2)
        T: Temperature [K]
        Ns: Number of cells in series

    Returns:
        Current array [A]
    """
    V = np.atleast_1d(V)
    I_array = np.array([solve_current_double(v, T, params, Ns) for v in V])
    return I_array