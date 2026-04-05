"""
I-V data loader for .dat files.
"""

import numpy as np


def load_iv_dat(filepath: str):
    """
    Load offline I–V data from a .dat file.

    Expected format:
    Column 0 -> Voltage (V)
    Column 1 -> Current (A)

    Args:
        filepath: Path to .dat file

    Returns:
        Tuple of (voltage, current) arrays
    """
    data = np.loadtxt(filepath)

    if data.shape[1] < 2:
        raise ValueError("DAT file must contain at least 2 columns: V and I")

    voltage = data[:, 0]
    current = data[:, 1]

    return voltage, current