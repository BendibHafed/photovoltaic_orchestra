"""
Power analysis module for PV modules.

Functions compute module power, P-V curve, and MPP (Maximum Power Point)
from voltage-current (V-I) data or a single-diode model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from pvoptix.pvoptix.solvers.single import solve_current_single


def compute_power(V: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous power from voltage and current arrays.

    Args:
        V: Voltage array [V]
        I: Current array [A]

    Returns:
        Power array [W]
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    return V * I


def find_mpp(
    V: np.ndarray,
    I: np.ndarray,
    interpolate: bool = True,
    n_fine: int = 1000,
):
    """
    Find Maximum Power Point (MPP) from V-I data.

    Args:
        V: Voltage array [V]
        I: Current array [A]
        interpolate: Whether to interpolate for finer MPP estimation
        n_fine: Number of interpolation points

    Returns:
        Tuple of (V_mpp, I_mpp, P_mpp)
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    P = compute_power(V, I)

    if not interpolate or len(V) < 4:
        idx = np.argmax(P)
        return V[idx], I[idx], P[idx]

    # Cubic interpolation
    k = min(3, len(V) - 1)
    f_interp = interp1d(V, P, kind=k)

    V_fine = np.linspace(V[0], V[-1], n_fine)
    P_fine = f_interp(V_fine)

    idx = np.argmax(P_fine)
    V_mpp = V_fine[idx]
    I_mpp = np.interp(V_mpp, V, I)
    P_mpp = P_fine[idx]

    return V_mpp, I_mpp, P_mpp


def simulate_iv_curve(
    params: dict,
    T: float,
    Ns: int,
    n_points: int = 100,
):
    """
    Simulate full IV curve using single-diode model.

    Args:
        params: PV module parameters (Rs, Rsh, I0, Iph, n)
        T: Temperature [K]
        Ns: Number of cells in series
        n_points: Number of voltage points

    Returns:
        Tuple of (V, I, P) arrays
    """
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = (k / q) * T

    Voc_guess = params["n"] * Vt * Ns * np.log1p(
        params["Iph"] / max(params["I0"], 1e-12)
    )

    V = np.linspace(0.0, Voc_guess * 1.05, n_points)
    I = np.array([solve_current_single(v, T, params, Ns) for v in V])
    P = compute_power(V, I)

    return V, I, P


def compute_global_power(
    datasets: list[dict],
    interpolate: bool = True,
    n_fine: int = 1000,
) -> list[dict]:
    """
    Compute power curves and MPP for multiple datasets.

    Args:
        datasets: List of datasets with V, I, G, T
        interpolate: Whether to interpolate
        n_fine: Number of interpolation points

    Returns:
        List of results with V, I, P, V_mpp, I_mpp, P_mpp, G, T
    """
    results = []

    for ds in datasets:
        V = ds["V"]
        I = ds["I"]

        P = compute_power(V, I)
        V_mpp, I_mpp, P_mpp = find_mpp(V, I, interpolate=interpolate, n_fine=n_fine)

        results.append({
            "V": V,
            "I": I,
            "P": P,
            "V_mpp": V_mpp,
            "I_mpp": I_mpp,
            "P_mpp": P_mpp,
            "G": ds.get("G", None),
            "T": ds.get("T", None),
        })

    return results


def plot_power_curve(
    V: np.ndarray,
    P: np.ndarray,
    V_mpp: float = None,
    P_mpp: float = None,
    title: str = "P-V Curve",
):
    """
    Plot power-voltage curve.

    Args:
        V: Voltage array
        P: Power array
        V_mpp: Voltage at MPP (optional)
        P_mpp: Power at MPP (optional)
        title: Plot title
    """
    plt.figure()
    plt.plot(V, P, label="Power curve")

    if V_mpp is not None and P_mpp is not None:
        plt.scatter([V_mpp], [P_mpp], label="MPP")
        plt.axvline(V_mpp, linestyle="--")
        plt.axhline(P_mpp, linestyle="--")

    plt.xlabel("Voltage [V]")
    plt.ylabel("Power [W]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()