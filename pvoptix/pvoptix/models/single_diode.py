"""
Single-diode PV model equation (5 parameters).
"""

import numpy as np


def single_diode_equation(
    I: float,
    V: float,
    T: float,
    Rs: float,
    Rsh: float,
    I0: float,
    Iph: float,
    n: float,
    Ns: int,
) -> float:
    """
    Implicit single-diode PV model equation (module level).

    Returns f(I) = 0 for the solver.

    Args:
        I: Current [A] (unknown, being solved)
        V: Voltage [V]
        T: Temperature [K]
        Rs: Series resistance [Ω]
        Rsh: Shunt resistance [Ω]
        I0: Diode saturation current [A]
        Iph: Photogenerated current [A]
        n: Diode ideality factor
        Ns: Number of cells in series

    Returns:
        f(I) value (should be zero at solution)
    """
    # Physical constants
    k = 1.380649e-23  # Boltzmann constant [J/K]
    q = 1.602176634e-19  # Elementary charge [C]

    # Thermal voltage
    Vt = (k * T) / q  # [V]

    # Effective thermal voltage for Ns cells
    Vt_eq = n * Ns * Vt

    # Diode current with clipping to avoid overflow
    exponent = (V + I * Rs) / Vt_eq
    diode_term = I0 * (np.exp(np.clip(exponent, -100, 100)) - 1.0)

    # Shunt current
    shunt_term = (V + I * Rs) / Rsh

    # Implicit equation: Iph - Id - Ishunt - I = 0
    return Iph - diode_term - shunt_term - I