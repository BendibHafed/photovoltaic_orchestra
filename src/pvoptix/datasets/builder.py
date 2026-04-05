"""
Dataset builder for experimental I-V curves from .dat files.
File format: <model>_<T_C>_<G>.dat
Example: S75_25_1000.dat
"""

import os
import numpy as np


def build_dataset(data_dir: str) -> list[dict]:
    """
    Load all experimental I–V datasets from .dat files.

    Assumes files are named: <model>_<T_C>_<G>.dat

    Args:
        data_dir: Directory containing .dat files

    Returns:
        List of datasets, each containing: model, T, G, V, I
    """
    datasets = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".dat"):
            continue

        # Parse filename: model_T_G.dat
        parts = fname.replace(".dat", "").split("_")
        if len(parts) != 3:
            raise ValueError(
                f"Filename {fname} does not match format <model>_<T>_<G>.dat"
            )

        model_name = parts[0]
        T_c = float(parts[1])  # temperature in Celsius
        T = 273.15 + T_c       # convert to Kelvin
        G = float(parts[2])    # irradiance [W/m^2]

        # Load I–V data
        data = np.loadtxt(os.path.join(data_dir, fname))
        if data.shape[1] != 2:
            raise ValueError(f"Data file {fname} must have 2 columns: V and I")

        dataset = {
            "model": model_name,
            "T": T,
            "G": G,
            "V": data[:, 0],
            "I": data[:, 1],
        }

        datasets.append(dataset)

    return datasets