#!/usr/bin/env python3
"""Demo: Multi-condition optimization (paper method)."""

import numpy as np
from pvoptix import optimize_double_multicondition, simulate_iv_curve_double


def generate_synthetic_datasets():
    """Generate synthetic datasets for different conditions."""
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }
    ns = 36
    
    conditions = [
        (1000, 25, "STC"),
        (800, 25, "LowG"),
        (600, 25, "VeryLowG"),
        (400, 25, "MinG"),
        (1000, 40, "HighT"),
        (1000, 20, "LowT"),
    ]
    
    datasets = []
    for G, T_c, label in conditions:
        T = T_c + 273.15
        V = np.linspace(0, 21.6, 50)
        I = simulate_iv_curve_double(
            V, stc_params=true_params,
            temperature_k=T, irradiance_w_m2=G, ns=ns
        )
        # Add noise
        I += np.random.randn(len(I)) * 0.01
        I = np.clip(I, 0, true_params["Iph"])
        
        datasets.append({
            "model": label,
            "T": T,
            "G": G,
            "V": V,
            "I": I
        })
        print(f"  Created {label}: G={G} W/m², T={T_c}°C")
    
    return datasets


def main():
    print("=" * 60)
    print("PvOptiX Demo: Multi-Condition Optimization")
    print("=" * 60)
    
    print("\nGenerating synthetic datasets...")
    datasets = generate_synthetic_datasets()
    
    print(f"\nOptimizing on {len(datasets)} conditions simultaneously...")
    result = optimize_double_multicondition(
        datasets=datasets,
        ns=36,
        pop_size=60,
        generations=50,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBest STC parameters:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v:.6e}")
    
    print(f"\nGlobal RMSE: {result.best_fitness:.6f}")


if __name__ == "__main__":
    main()