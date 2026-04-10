"""
Genome mapping for double-diode model (7 parameters).

Converts between normalized genome [0,1]⁷ and physical parameters.
"""

import numpy as np

# Parameter bounds for double-diode model (STC ranges)
# Based on S75 module specifications and typical values
BOUNDS_DOUBLE = {
    "Rs": (0.001, 1.0),      # Series resistance [Ω]
    "Rsh": (10.0, 6000.0),   # Shunt resistance [Ω]
    "I01": (1e-12, 1e-7),    # Diode 1 saturation current [A] (log scale)
    "I02": (1e-12, 1e-7),    # Diode 2 saturation current [A] (log scale)
    "Iph": (0.1, 10.0),      # Photogenerated current [A]
    "n1": (1.0, 2.0),        # Diode 1 ideality factor
    "n2": (1.0, 2.0),        # Diode 2 ideality factor
}


def decode_individual_double(individual: np.ndarray) -> dict[str, float]:
    """
    Decode a normalized individual [0,1]⁷ into physical parameters.

    Args:
        individual: Normalized genome vector of length 7

    Returns:
        Physical parameters dict {Rs, Rsh, I01, I02, Iph, n1, n2}
    """
    params = {}
    keys = list(BOUNDS_DOUBLE.keys())

    for i, key in enumerate(keys):
        low, high = BOUNDS_DOUBLE[key]

        if key in ["I01", "I02"]:
            # Logarithmic mapping for saturation currents (wide range)
            log_low, log_high = np.log10(low), np.log10(high)
            val = log_low + individual[i] * (log_high - log_low)
            params[key] = float(10**val)
        else:
            # Linear mapping for other parameters
            params[key] = float(low + individual[i] * (high - low))

    # Enforce physical constraint: n2 >= n1 (typical for double-diode)
    if params["n2"] < params["n1"]:
        params["n2"] = params["n1"] + 0.05

    return params


def encode_individual_double(params: dict[str, float]) -> np.ndarray:
    """
    Encode physical parameters back into normalized [0,1]⁷ genome.

    Args:
        params: Physical parameters dict

    Returns:
        Normalized genome vector of length 7
    """
    individual = np.zeros(len(BOUNDS_DOUBLE))
    keys = list(BOUNDS_DOUBLE.keys())

    for i, key in enumerate(keys):
        low, high = BOUNDS_DOUBLE[key]
        val = float(np.clip(params[key], low, high))

        if key in ["I01", "I02"]:
            # Logarithmic encoding
            log_low, log_high = np.log10(low), np.log10(high)
            individual[i] = (np.log10(val) - log_low) / (log_high - log_low)
        else:
            # Linear encoding
            individual[i] = (val - low) / (high - low)

    return individual