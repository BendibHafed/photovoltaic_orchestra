#!/usr/bin/env python3
"""Demo: Progressive optimization (friend's idea)."""

import numpy as np
from pvoptix import optimize_double_progressive, simulate_iv_curve_double


def generate_synthetic_scans(n_scans=10):
    """Generate synthetic I-V scans for demonstration."""
    # True STC parameters (unknown in reality)
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }
    ns = 36

    for i in range(n_scans):
        # Simulate changing conditions throughout the day
        hour = 8 + i * 0.5  # 8:00 to 12:30
        G = 200 + 800 * np.sin((hour - 8) * np.pi / 9)  # Irradiance
        T = 20 + 15 * np.sin((hour - 8) * np.pi / 9) + 273.15  # Temperature

        # Generate voltage points
        V = np.linspace(0, 21.6, 50)

        # Simulate I-V curve with noise
        I = simulate_iv_curve_double(
            V, stc_params=true_params,
            temperature_k=T, irradiance_w_m2=G, ns=ns
        )
        # Add measurement noise
        I += np.random.randn(len(I)) * 0.01
        I = np.clip(I, 0, true_params["Iph"])

        yield {"V": V, "I": I, "T": T, "G": G}


def main():
    print("=" * 60)
    print("PvOptiX Demo: Progressive Optimization")
    print("=" * 60)

    # Generate synthetic scans
    scans = list(generate_synthetic_scans(n_scans=10))

    print(f"\nGenerated {len(scans)} synthetic I-V curves")
    print("First scan conditions:")
    print(f"  G = {scans[0]['G']:.0f} W/m²")
    print(f"  T = {scans[0]['T'] - 273.15:.1f} °C")

    # Run progressive optimization
    print("\nRunning progressive optimization...")
    result = optimize_double_progressive(
        scan_stream=iter(scans),
        ns=36,
        pop_size=40,
        generations=30,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBest STC parameters:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v:.6e}")

    print(f"\nFinal RMSE: {result.best_fitness:.6f}")


if __name__ == "__main__":
    main()