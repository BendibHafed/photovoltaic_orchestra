# examples/demo_degradation_tracking.py
"""
Demonstration 5: Degradation Tracking Over Time
Shows how the library can track parameter evolution for degradation monitoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from pvoptix import (
    optimize_double_progressive,
    simulate_iv_curve_double
)


def generate_degraded_scans(month, degradation_factor=0.01):
    """Generate scans with simulated degradation."""
    # Initial healthy parameters
    healthy_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }

    # Apply degradation (Rs increases, Iph decreases, Rsh decreases)
    degraded_params = {
        "Rs": healthy_params["Rs"] * (1 + degradation_factor * month),
        "Rsh": healthy_params["Rsh"] * (1 - degradation_factor * month),
        "I01": healthy_params["I01"] * (1 + degradation_factor * month),
        "I02": healthy_params["I02"] * (1 + degradation_factor * month),
        "Iph": healthy_params["Iph"] * (1 - degradation_factor * month),
        "n1": healthy_params["n1"] + degradation_factor * month * 0.1,
        "n2": healthy_params["n2"] + degradation_factor * month * 0.1,
    }

    ns = 36
    scans = []

    # Generate scans at different conditions throughout a day
    hours = [8, 10, 12, 14, 16]
    for hour in hours:
        G = 200 + 800 * np.sin((hour - 8) * np.pi / 9)
        G = max(100, min(1000, G))
        T = 20 + 15 * np.sin((hour - 8) * np.pi / 9) + 273.15

        V = np.linspace(0, 21.6, 50)
        I = simulate_iv_curve_double(
            V, stc_params=degraded_params,
            temperature_k=T, irradiance_w_m2=G, ns=ns
        )
        I += np.random.randn(len(I)) * 0.005
        I = np.clip(I, 0, degraded_params["Iph"])

        scans.append({"V": V, "I": I, "T": T, "G": G, "hour": hour})

    return scans, degraded_params


def main():
    print("=" * 70)
    print("DEMONSTRATION 5: DEGRADATION TRACKING OVER TIME")
    print("(Parameter evolution for health monitoring)")
    print("=" * 70)

    # Simulate monthly measurements for 24 months (2 years)
    months = range(1, 25)
    degradation_rate = 0.008  # 0.8% degradation per month

    # Store results
    parameter_history = {key: [] for key in ["Rs", "Rsh", "Iph", "n1", "n2"]}
    rmse_history = []

    print("\nSimulating degradation over 24 months...")
    print(f"Monthly degradation rate: {degradation_rate*100:.1f} percent")

    for month in months:
        print(f"\nProcessing month {month}...")

        # Generate scans for this month
        scans, true_degraded = generate_degraded_scans(month, degradation_rate)

        # Run progressive optimization
        result = optimize_double_progressive(
            scan_stream=iter(scans),
            ns=36,
            include_virtual_stc=True,
            ga_kwargs={
                "pop_size": 40,
                "generations": 40,
                "verbose": False
            },
            verbose=False
        )

        # Store results
        for key in parameter_history:
            parameter_history[key].append(result.best_params[key])
        rmse_history.append(result.best_fitness)

        if month % 6 == 0:
            print(f"   Month {month}: Rs={result.best_params['Rs']:.4f}, Iph={result.best_params['Iph']:.4f}")

    # =========================================================
    # DEGRADATION ANALYSIS
    # =========================================================
    print("\n" + "-" * 50)
    print("DEGRADATION ANALYSIS")
    print("-" * 50)

    # Calculate degradation rates from optimization results
    initial_rs = parameter_history["Rs"][0]
    final_rs = parameter_history["Rs"][-1]
    rs_degradation = (final_rs - initial_rs) / initial_rs * 100

    initial_ip