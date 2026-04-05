# pvoptix/pvoptix/analysis/power.py

"""
Power analysis module for PV modules.

Functions compute module power, P-V curve, and MPP (Maximum Power Point)
from voltage-current (V-I) data using the double-diode model.

Compatible with the embedded multi-condition optimization strategy.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Use relative imports for double-diode solver
from ..solvers.double import solve_current_double


# =============================================================================
# Basic power computation
# =============================================================================

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


# =============================================================================
# Maximum Power Point
# =============================================================================

def find_mpp(V: np.ndarray, I: np.ndarray, interpolate: bool = True, n_fine: int = 1000):
    """
    Find Maximum Power Point (MPP) from V-I data.

    Args:
        V: Voltage array [V]
        I: Current array [A]
        interpolate: Whether to interpolate power curve for finer MPP estimation
        n_fine: Number of interpolation points

    Returns:
        Tuple of (V_mpp, I_mpp, P_mpp)
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    P = compute_power(V, I)

    # Fallback if too few points
    if not interpolate or len(V) < 4:
        idx = np.argmax(P)
        return V[idx], I[idx], P[idx]

    # Cubic interpolation when possible
    k = min(3, len(V) - 1)
    f_interp = interp1d(V, P, kind=k)

    V_fine = np.linspace(V[0], V[-1], n_fine)
    P_fine = f_interp(V_fine)

    idx = np.argmax(P_fine)
    V_mpp = V_fine[idx]
    I_mpp = np.interp(V_mpp, V, I)
    P_mpp = P_fine[idx]

    return V_mpp, I_mpp, P_mpp


# =============================================================================
# IV curve simulation with double-diode model
# =============================================================================

def simulate_iv_curve(
    params: dict,
    T: float,
    Ns: int,
    n_points: int = 100
):
    """
    Simulate full IV curve using double-diode model.

    Args:
        params: PV module parameters with keys: Rs, Rsh, I01, I02, Iph, n1, n2
        T: Temperature [K]
        Ns: Number of cells in series
        n_points: Number of voltage points

    Returns:
        Tuple of (V, I, P) arrays
    """
    # Physical constants
    k = 1.380649e-23
    q = 1.602176634e-19
    Vt = (k / q) * T

    # Estimate Voc using dominant diode
    a1 = params["n1"] * Ns * Vt
    Voc_est = a1 * np.log(params["Iph"] / max(params["I01"], 1e-12))
    Voc_est = min(Voc_est, 50.0)

    V = np.linspace(0.0, Voc_est * 1.05, n_points)
    I = np.array([solve_current_double(v, T, params, Ns) for v in V])
    P = compute_power(V, I)

    return V, I, P


# =============================================================================
# Global analysis over multiple datasets
# =============================================================================

def compute_global_power(datasets: list, interpolate: bool = True, n_fine: int = 1000) -> list:
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


# =============================================================================
# Plotting utilities
# =============================================================================

def plot_power_curve(
    V: np.ndarray,
    P: np.ndarray,
    V_mpp: float = None,
    P_mpp: float = None,
    title: str = "P-V Curve",
    show: bool = True,
    save_path: str = None
):
    """
    Plot power-voltage curve.

    Args:
        V: Voltage array
        P: Power array
        V_mpp: Voltage at MPP (optional)
        P_mpp: Power at MPP (optional)
        title: Plot title
        show: Whether to display the plot
        save_path: If provided, save plot to this path
    """
    plt.figure(figsize=(10, 6))
    plt.plot(V, P, 'b-', linewidth=2, label="Power curve")

    if V_mpp is not None and P_mpp is not None:
        plt.scatter([V_mpp], [P_mpp], color='red', s=100, zorder=5, label=f"MPP ({P_mpp:.2f} W)")
        plt.axvline(V_mpp, linestyle='--', color='gray', alpha=0.5)
        plt.axhline(P_mpp, linestyle='--', color='gray', alpha=0.5)

    plt.xlabel("Voltage [V]")
    plt.ylabel("Power [W]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# Multi-condition power analysis (for embedded strategy)
# =============================================================================

def analyze_power_across_conditions(
    datasets: list,
    stc_params: dict,
    ns: int = 36,
    coefficients=None
) -> dict:
    """
    Analyze power and MPP across multiple operating conditions.

    This is useful for evaluating the embedded multi-condition strategy.

    Args:
        datasets: List of datasets with V, I, T, G
        stc_params: STC parameters to evaluate
        ns: Number of cells in series
        coefficients: Model coefficients

    Returns:
        Dictionary with results per condition
    """
    from ..api import simulate_iv_curve_double

    results = {
        "conditions": [],
        "mpp_power_exp": [],
        "mpp_power_sim": [],
        "mpp_voltage_exp": [],
        "mpp_voltage_sim": [],
        "rmse": []
    }

    for ds in datasets:
        V = ds["V"]
        I_exp = ds["I"]
        T = ds["T"]
        G = ds["G"]

        # Simulate with given STC parameters
        I_sim = simulate_iv_curve_double(
            V,
            stc_params=stc_params,
            temperature_k=T,
            irradiance_w_m2=G,
            ns=ns,
            coefficients=coefficients
        )

        # Find MPP for experimental and simulated
        V_mpp_exp, I_mpp_exp, P_mpp_exp = find_mpp(V, I_exp)
        V_mpp_sim, I_mpp_sim, P_mpp_sim = find_mpp(V, I_sim)

        # Calculate RMSE for this condition
        rmse = np.sqrt(np.mean((I_exp - I_sim) ** 2))

        results["conditions"].append({"G": G, "T": T})
        results["mpp_power_exp"].append(P_mpp_exp)
        results["mpp_power_sim"].append(P_mpp_sim)
        results["mpp_voltage_exp"].append(V_mpp_exp)
        results["mpp_voltage_sim"].append(V_mpp_sim)
        results["rmse"].append(rmse)

    return results


__all__ = [
    "compute_power",
    "find_mpp",
    "simulate_iv_curve",
    "compute_global_power",
    "plot_power_curve",
    "analyze_power_across_conditions",
]