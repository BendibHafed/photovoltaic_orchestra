"""
Parameter models for PV module translation equations.

Implements equations (4-11) from NCMAI'26 paper:
- Equation (4): Iph(G,T)
- Equations (5-6): I01(T), I02(T)
- Equations (8-9): a1(T), a2(T)
- Equation (10): Rsh(G)
- Equation (11): Rs(T) (constant)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Physical constants (SI units)
q = 1.602176634e-19  # Electron charge [C]
k = 1.380649e-23     # Boltzmann constant [J/K]

# STC reference conditions
G_STC = 1000.0       # Irradiance at STC [W/m^2]
T_STC = 298.15       # Temperature at STC [K] (25°C)

# Silicon bandgap energy (converted from eV to J)
Eg = 1.12 * q        # [J]


@dataclass
class PVModelCoefficients:
    """
    Coefficients for PV parameter models.

    Allows injection of different sets of coefficients from external sources.
    """
    alpha_I: float = 0.0005      # Temperature coefficient for Iph [1/K]
    alpha_Rs: float = 0.0         # Temperature coefficient for Rs [1/K]
    beta_Rs: float = 0.0          # Irradiance coefficient for Rs [1/(W/m²)]
    beta_Rsh: float = -0.002      # Temperature coefficient for Rsh [1/K]
    Rs_min: float = 0.0           # Minimum series resistance [Ω]
    Rs_max: Optional[float] = None  # Maximum series resistance [Ω]


default_coeffs = PVModelCoefficients()


def iph_model(G: float, T: float, Iph_stc: float, alpha_I: float = None) -> float:
    """
    Photogenerated current model.

    Equation (4) from NCMAI'26 paper:
    Iph(G,T) = Iph_ref * (G/G_ref) * [1 + α_I * (T - T_ref)]

    Args:
        G: Irradiance [W/m²]
        T: Temperature [K]
        Iph_stc: Photocurrent at STC [A]
        alpha_I: Temperature coefficient [1/K]

    Returns:
        Photocurrent at operating conditions [A]
    """
    if alpha_I is None:
        alpha_I = default_coeffs.alpha_I
    return Iph_stc * (G / G_STC) * (1.0 + alpha_I * (T - T_STC))


def i01_model(T: float, I01_stc: float, n1: float) -> float:
    """
    Temperature-dependent diode 1 saturation current.

    Equation (5) from NCMAI'26 paper.

    Args:
        T: Temperature [K]
        I01_stc: Diode 1 saturation current at STC [A]
        n1: Diode 1 ideality factor

    Returns:
        Diode 1 saturation current at temperature T [A]
    """
    exponent = (Eg / (n1 * k)) * (1.0 / T_STC - 1.0 / T)
    exponent = np.clip(exponent, -100, 100)
    return I01_stc * (T / T_STC) ** 3 * np.exp(exponent)


def i02_model(T: float, I02_stc: float, n2: float) -> float:
    """
    Temperature-dependent diode 2 saturation current.

    Equation (6) from NCMAI'26 paper.

    Args:
        T: Temperature [K]
        I02_stc: Diode 2 saturation current at STC [A]
        n2: Diode 2 ideality factor

    Returns:
        Diode 2 saturation current at temperature T [A]
    """
    exponent = (Eg / (n2 * k)) * (1.0 / T_STC - 1.0 / T)
    exponent = np.clip(exponent, -100, 100)
    return I02_stc * (T / T_STC) ** 3 * np.exp(exponent)


def rs_model(
    G: float,
    T: float,
    Rs_stc: float,
    alpha_Rs: float = None,
    beta_Rs: float = None,
    Rs_min: float = None,
    Rs_max: float = None,
) -> float:
    """
    Series resistance model.

    Equation (11) from paper (constant in this work).

    Args:
        G: Irradiance [W/m²]
        T: Temperature [K]
        Rs_stc: Series resistance at STC [Ω]
        alpha_Rs: Temperature coefficient [1/K]
        beta_Rs: Irradiance coefficient [1/(W/m²)]
        Rs_min: Minimum allowed value [Ω]
        Rs_max: Maximum allowed value [Ω]

    Returns:
        Series resistance at operating conditions [Ω]
    """
    if alpha_Rs is None:
        alpha_Rs = default_coeffs.alpha_Rs
    if beta_Rs is None:
        beta_Rs = default_coeffs.beta_Rs
    if Rs_min is None:
        Rs_min = default_coeffs.Rs_min
    if Rs_max is None:
        Rs_max = default_coeffs.Rs_max

    # Temperature effect
    temp_effect = 1.0 + alpha_Rs * (T - T_STC)

    # Irradiance effect
    irrad_effect = 1.0 + beta_Rs * (G - G_STC) / G_STC

    # Combined effect
    Rs = Rs_stc * temp_effect * irrad_effect

    # Apply bounds
    Rs = max(Rs, Rs_min)
    if Rs_max is not None:
        Rs = min(Rs, Rs_max)

    return Rs


def rsh_model_double(G: float, Rsh_stc: float) -> float:
    """
    Shunt resistance model for double-diode.

    Equation (10) from NCMAI'26 paper:
    Rsh(G) = Rsh_ref * (G_ref / G)

    Args:
        G: Irradiance [W/m²]
        Rsh_stc: Shunt resistance at STC [Ω]

    Returns:
        Shunt resistance at operating conditions [Ω]
    """
    return Rsh_stc * (G_STC / max(G, 1e-6))