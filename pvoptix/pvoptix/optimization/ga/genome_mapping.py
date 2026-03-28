"""
Genome mapping for single-diode model (5 parameters).
"""

import numpy as np

# Parameter bounds for single-diode model (STC ranges)
BOUNDS = {
    "Rs": (0.001, 1.0),   # Series resistance [ohm]
    "Rsh": (10.0, 5000.0), # Shunt resistance [ohm]
    "I0": (1e-10, 1e-7),   # Diode saturation current [A] (log scale)
    "Iph": (0.1, 10.0),    # Photogenerated current [A]
    "n": (1.0, 2.0),       # Diode ideality factor
}


def decode_individual(individual: np.ndarray) -> dict[str, float]:
    """
    Decode a normalized individual [0,1] into physical STC parameters.

    Args:
        individual: Normalized genome vector of length 5

    Returns:
        Physical parameters {Rs, Rsh, I0, Iph, n}
    """
    params = {}
    keys = list(BOUNDS.keys())

    for i, key in enumerate(keys):
        low, high = BOUNDS[key]

        if key == "I0":
            # Logarithmic mapping for I0
            log_low, log_high = np.log10(low), np.log10(high)
            val = log_low + individual[i] * (log_high - log_low)
            params[key] = float(10**val)
        else:
            # Linear mapping for other parameters
            params[key] = float(low + individual[i] * (high - low))

    return params


def encode_individual(params: dict[str, float]) -> np.ndarray:
    """
    Encode physical parameters back into normalized [0,1] genome.

    Args:
        params: Physical parameters {Rs, Rsh, I0, Iph, n}

    Returns:
        Normalized genome vector of length 5
    """
    individual = np.zeros(len(BOUNDS))
    keys = list(BOUNDS.keys())

    for i, key in enumerate(keys):
        low, high = BOUNDS[key]
        val = float(np.clip(params[key], low, high))

        if key == "I0":
            # Logarithmic encoding for I0
            log_low, log_high = np.log10(low), np.log10(high)
            individual[i] = (np.log10(val) - log_low) / (log_high - log_low)
        else:
            # Linear encoding for other parameters
            individual[i] = (val - low) / (high - low)

    return individual