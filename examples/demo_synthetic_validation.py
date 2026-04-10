# examples/demo_synthetic_validation.py
"""
Demonstration 4: Synthetic Data Validation
Validates the progressive algorithm with known ground truth.

VALIDATION FRAMEWORK:
- Generate synthetic I-V curves using known true STC parameters
- Add controlled noise to simulate real measurements
- Run progressive optimization on the noisy data
- Compare estimated parameters with ground truth
- Quantify recovery accuracy and convergence properties

EXPECTED OUTCOME:
- Progressive algorithm should recover parameters close to true values
- Average error should be within acceptable bounds (< 10%)
- RMSE of estimated parameters should approach RMSE of true parameters
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pvoptix import (
    optimize_double_progressive,
    simulate_iv_curve_double,
    evaluate_double_parameters
)


def compute_power(V, I):
    """Compute power from voltage and current arrays: P = V * I."""
    return V * I


def find_mpp(V, I):
    """Find Maximum Power Point from V-I data."""
    P = compute_power(V, I)
    idx = np.argmax(P)
    return V[idx], I[idx], P[idx]


def generate_synthetic_scans(n_scans=10, add_noise=True, noise_level=0.01, random_seed=42):
    """
    Generate synthetic I-V scans with known true parameters.
    
    Args:
        n_scans: Number of scans to generate
        add_noise: Whether to add Gaussian noise
        noise_level: Standard deviation of noise (relative to Iph)
        random_seed: Random seed for reproducibility
    
    Returns:
        scans: List of synthetic scan dictionaries
        true_params: Ground truth STC parameters
    """
    np.random.seed(random_seed)
    
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }
    ns = 36

    scans = []
    for i in range(n_scans):
        # Simulate diurnal variation
        hour = 8 + i * (9 / n_scans)  # 8:00 to 17:00
        G = 200 + 800 * np.sin((hour - 8) * np.pi / 9)
        G = max(100, min(1000, G))
        T = 20 + 15 * np.sin((hour - 8) * np.pi / 9) + 273.15

        V = np.linspace(0, 21.6, 50)
        I = simulate_iv_curve_double(
            V, stc_params=true_params,
            temperature_k=T, irradiance_w_m2=G, ns=ns
        )

        if add_noise:
            # Add Gaussian noise proportional to Iph
            noise = np.random.randn(len(I)) * noise_level * true_params["Iph"]
            I += noise
            I = np.clip(I, 0, true_params["Iph"])

        scans.append({
            "V": V, "I": I, "T": T, "G": G, 
            "hour": hour, "scan_id": i + 1
        })

    return scans, true_params


def calculate_parameter_errors(estimated, true):
    """Calculate individual and average parameter errors."""
    errors = {}
    total_error = 0.0
    for k in true:
        if k in estimated:
            error = abs(estimated[k] - true[k]) / abs(true[k]) * 100
            errors[k] = error
            total_error += error
    avg_error = total_error / len(true)
    return errors, avg_error


def main():
    print("=" * 70)
    print("DEMONSTRATION 4: SYNTHETIC DATA VALIDATION")
    print("(Known ground truth to verify algorithm correctness)")
    print("=" * 70)
    print("\nVALIDATION FRAMEWORK:")
    print("  - Generate I-V curves from known true parameters")
    print("  - Add controlled noise to simulate real measurements")
    print("  - Run progressive optimization on noisy data")
    print("  - Compare estimated parameters with ground truth")
    print("  - Quantify recovery accuracy")

    # Generate synthetic scans
    print("\n[1] GENERATING SYNTHETIC DATA")
    print("-" * 50)
    
    n_scans = 12
    noise_level = 0.01  # 1% noise relative to Iph
    
    print(f"Generating {n_scans} synthetic scans with {noise_level*100:.1f}% noise...")
    scans, true_params = generate_synthetic_scans(
        n_scans=n_scans, 
        add_noise=True, 
        noise_level=noise_level,
        random_seed=42
    )
    
    print(f"\nGenerated {len(scans)} synthetic scans across diurnal cycle:")
    for i, scan in enumerate(scans[:5]):
        print(f"   Scan {i+1}: Hour={scan['hour']:.1f}h, G={scan['G']:.0f} W/m², T={scan['T']-273.15:.1f} deg C")
    if n_scans > 5:
        print(f"   ... and {n_scans - 5} more scans")
    
    print("\nGround truth STC parameters (target to recover):")
    for k, v in true_params.items():
        print(f"   {k}: {v:.6e}")

    # =========================================================
    # PROGRESSIVE OPTIMIZATION
    # =========================================================
    print("\n[2] PROGRESSIVE OPTIMIZATION")
    print("-" * 50)
    print("Running progressive algorithm on noisy synthetic data...")
    print("GA convergence plots will appear for each scan.\n")

    result = optimize_double_progressive(
        scan_stream=iter(scans),
        ns=36,
        include_virtual_stc=True,
        ga_kwargs={
            "pop_size": 40,
            "generations": 60,
            "crossover_rate": 0.85,
            "mutation_rate": 0.12,
            "verbose": True,
            "live_plot": True,
            "figsize": (10, 5),
            "auto_close_plot": True,
            "plot_display_seconds": 4
        },
        verbose=True
    )

    # =========================================================
    # RECOVERY ACCURACY ANALYSIS
    # =========================================================
    print("\n[3] RECOVERY ACCURACY ANALYSIS")
    print("-" * 50)

    errors, avg_error = calculate_parameter_errors(result.best_params, true_params)

    print(f"\n{'Parameter':<8} {'True':<14} {'Estimated':<14} {'Error (%)':<12} {'Status':<10}")
    print("-" * 60)

    for k in true_params:
        true_v = true_params[k]
        est_v = result.best_params.get(k, 0)
        error = errors.get(k, 0)
        
        if error < 5:
            status = "EXCELLENT"
        elif error < 10:
            status = "GOOD"
        elif error < 20:
            status = "FAIR"
        else:
            status = "POOR"
        
        print(f"{k:<8} {true_v:<14.6e} {est_v:<14.6e} {error:<12.2f} {status:<10}")

    print(f"\n{'AVERAGE':<8} {'':<14} {'':<14} {avg_error:<12.2f} {'':<10}")

    # =========================================================
    # RMSE COMPARISON
    # =========================================================
    print("\n[4] RMSE COMPARISON")
    print("-" * 50)

    rmse_optimized = evaluate_double_parameters(result.best_params, scans, ns=36)
    rmse_true = evaluate_double_parameters(true_params, scans, ns=36)

    print(f"RMSE of optimized parameters: {rmse_optimized:.6f}")
    print(f"RMSE of true parameters: {rmse_true:.6f}")
    print(f"Difference: {abs(rmse_optimized - rmse_true):.6f}")

    if rmse_optimized <= rmse_true * 1.1:
        print("\nVERDICT: Progressive algorithm successfully recovered the true parameters")
        print("         (optimized RMSE within 10% of true RMSE)")
    elif rmse_optimized <= rmse_true * 1.2:
        print("\nVERDICT: Progressive algorithm reasonably recovered the true parameters")
        print("         (optimized RMSE within 20% of true RMSE)")
    else:
        print("\nVERDICT: Progressive algorithm partially recovered the true parameters")
        print("         (optimized RMSE higher than expected)")

    # =========================================================
    # CONVERGENCE ANALYSIS
    # =========================================================
    print("\n[5] CONVERGENCE ANALYSIS")
    print("-" * 50)

    # Extract RMSE progression from history
    rmse_progression = [entry['rmse_best'] for entry in result.history]
    scans_processed = [entry['scan_id'] for entry in result.history]
    improvements = [entry['improved'] for entry in result.history]

    print(f"\n{'Scan':<6} {'RMSE Best':<12} {'Improved':<10} {'Cumulative Improvement':<20}")
    print("-" * 50)

    cumulative_imp = 0
    for i, (scan, rmse, improved) in enumerate(zip(scans_processed, rmse_progression, improvements)):
        if i > 0 and rmse < rmse_progression[i-1]:
            imp = (rmse_progression[i-1] - rmse) / rmse_progression[i-1] * 100
            cumulative_imp += imp
            imp_str = f"Yes (+{imp:.1f}%)"
        elif i == 0:
            imp_str = "Initial"
            cumulative_imp = 0
        else:
            imp_str = "No"
        
        print(f"{scan:<6} {rmse:<12.6f} {imp_str:<10} {cumulative_imp:<20.1f}%")

    total_improvement = (rmse_progression[0] - rmse_progression[-1]) / rmse_progression[0] * 100
    print(f"\nTotal RMSE improvement from first to last scan: {total_improvement:.1f}%")

    # =========================================================
    # MPP ACCURACY
    # =========================================================
    print("\n[6] MAXIMUM POWER POINT (MPP) ACCURACY")
    print("-" * 50)

    print(f"\n{'Condition':<35} {'True P_mpp (W)':<18} {'Est P_mpp (W)':<18} {'Error (%)':<12}")
    print("-" * 85)

    mpp_errors = []
    for idx, scan in enumerate(scans[:6]):
        V = scan["V"]
        I_true = simulate_iv_curve_double(
            V, stc_params=true_params,
            temperature_k=scan["T"], irradiance_w_m2=scan["G"], ns=36
        )
        I_est = simulate_iv_curve_double(
            V, stc_params=result.best_params,
            temperature_k=scan["T"], irradiance_w_m2=scan["G"], ns=36
        )

        _, _, P_mpp_true = find_mpp(V, I_true)
        _, _, P_mpp_est = find_mpp(V, I_est)

        error = abs(P_mpp_est - P_mpp_true) / P_mpp_true * 100 if P_mpp_true > 0 else 0
        mpp_errors.append(error)

        condition = f"G={scan['G']:.0f} W/m², T={scan['T']-273.15:.0f} deg C"
        print(f"{condition:<35} {P_mpp_true:<18.2f} {P_mpp_est:<18.2f} {error:<12.1f}")

    avg_mpp_error = np.mean(mpp_errors)
    print("-" * 85)
    print(f"{'AVERAGE':<35} {'':<18} {'':<18} {avg_mpp_error:<12.1f}")

    # =========================================================
    # VISUALIZATION
    # =========================================================
    print("\n[7] GENERATING COMPARISON PLOTS")
    print("-" * 50)

    n_plots = min(len(scans), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    axes = axes.flatten()

    for idx in range(n_plots):
        scan = scans[idx]
        ax = axes[idx]
        V = scan["V"]
        I_noisy = scan["I"]
        T = scan["T"]
        G = scan["G"]
        T_c = T - 273.15

        I_true = simulate_iv_curve_double(
            V, stc_params=true_params,
            temperature_k=T, irradiance_w_m2=G, ns=36
        )
        I_est = simulate_iv_curve_double(
            V, stc_params=result.best_params,
            temperature_k=T, irradiance_w_m2=G, ns=36
        )

        rmse_noisy = np.sqrt(np.mean((I_noisy - I_true) ** 2))
        rmse_est = np.sqrt(np.mean((I_est - I_true) ** 2))

        ax.plot(V, I_true, 'b-', linewidth=2, label='True (noise-free)')
        ax.plot(V, I_noisy, 'ko', markersize=2, label=f'Measured (noisy, RMSE={rmse_noisy:.4f})', alpha=0.5)
        ax.plot(V, I_est, 'r--', linewidth=2, label=f'Estimated (RMSE={rmse_est:.4f})')
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('Current [A]')
        ax.set_title(f'G={G:.0f} W/m², T={T_c:.0f} deg C')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle('Synthetic Data Validation: True vs Estimated I-V Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\nQuantitative Results:")
    print(f"  - Number of synthetic scans: {n_scans}")
    print(f"  - Noise level: {noise_level*100:.1f}%")
    print(f"  - Average parameter recovery error: {avg_error:.2f}%")
    print(f"  - Total RMSE improvement: {total_improvement:.1f}%")
    print(f"  - Average MPP prediction error: {avg_mpp_error:.1f}%")

    print("\nParameter Recovery Quality:")
    excellent_count = sum(1 for e in errors.values() if e < 5)
    good_count = sum(1 for e in errors.values() if 5 <= e < 10)
    fair_count = sum(1 for e in errors.values() if 10 <= e < 20)
    poor_count = sum(1 for e in errors.values() if e >= 20)
    
    print(f"  - Excellent (error < 5%): {excellent_count}/7 parameters")
    print(f"  - Good (error 5-10%): {good_count}/7 parameters")
    print(f"  - Fair (error 10-20%): {fair_count}/7 parameters")
    print(f"  - Poor (error > 20%): {poor_count}/7 parameters")

    print("\nCONCLUSION:")
    if avg_error < 10:
        print("  The progressive algorithm successfully recovers the true STC parameters")
        print("  with high accuracy from noisy measurements across varying conditions.")
    elif avg_error < 20:
        print("  The progressive algorithm reasonably recovers the true STC parameters")
        print("  from noisy measurements. Performance is acceptable for practical use.")
    else:
        print("  The progressive algorithm provides useful estimates but may require")
        print("  more scans or lower noise levels for high-precision applications.")


if __name__ == "__main__":
    main()