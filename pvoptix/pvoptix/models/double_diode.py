"""
Double-diode PV model equation (7 parameters).

Implements equation (1) from NCMAI'26 paper.
"""

import numpy as np


def double_diode_equation(
    I: float,
    V: float,
    T: float,
    Rs: float,
    Rsh: float,
    I01: float,
    I02: float,
    Iph: float,
    n1: float,
    n2: float,
    Ns: int,
) -> float:
    """
    Implicit double-diode PV model equation (module level).

    Returns f(I) = 0 for the numerical solver.

    Args:
        I: Current [A] (unknown, being solved)
        V: Voltage [V]
        T: Temperature [K]
        Rs: Series resistance [Ω]
        Rsh: Shunt resistance [Ω]
        I01: Diode 1 saturation current [A]
        I02: Diode 2 saturation current [A]
        Iph: Photogenerated current [A]
        n1: Diode 1 ideality factor
        n2: Diode 2 ideality factor
        Ns: Number of cells in series

    Returns:
        f(I) value (should be zero at solution)
    """
    # Physical constants
    k = 1.380649e-23  # Boltzmann constant [J/K]
    q = 1.602176634e-19  # Elementary charge [C]

    # Thermal voltage
    Vt = (k * T) / q  # [V]

    # Exponential parameters (equations 2 and 3 from paper)
    a1 = n1 * Ns * Vt
    a2 = n2 * Ns * Vt

    # Diode currents with clipping to avoid overflow
    exp_arg1 = (V + I * Rs) / a1
    exp_arg2 = (V + I * Rs) / a2

    diode1 = I01 * (np.exp(np.clip(exp_arg1, -100, 100)) - 1.0)
    diode2 = I02 * (np.exp(np.clip(exp_arg2, -100, 100)) - 1.0)

    # Shunt current
    shunt = (V + I * Rs) / Rsh

    # Implicit equation: Iph - I01 - I02 - Ishunt - I = 0
    return Iph - diode1 - diode2 - shunt - I