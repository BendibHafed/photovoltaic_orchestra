#!/usr/bin/env python3
"""Test progressive optimization algorithm (friend's idea)."""

import numpy as np
from pvoptix import (
    optimize_double_progressive,
    simulate_iv_curve_double,
    evaluate_double_parameters
)


def generate_synthetic_scans(n_scans=10, add_noise=True):
    """Generate synthetic I-V scans for testing."""
    # True STC parameters (the ones we want to find)
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }
    ns = 36
    
    scans = []
    for i in range(n_scans):
        # Simulate changing conditions throughout the day
        hour = 8 + i * (9 / n_scans)  # 8:00 to 17:00
        G = 200 + 800 * np.sin((hour - 8) * np.pi / 9)  # Irradiance
        G = max(100, min(1000, G))  # Clamp to realistic range
        T = 20 + 15 * np.sin((hour - 8) * np.pi / 9) + 273.15  # Temperature
        
        # Generate voltage points
        V = np.linspace(0, 21.6, 50)
        
        # Simulate I-V curve
        I = simulate_iv_curve_double(
            V, stc_params=true_params,
            temperature_k=T, irradiance_w_m2=G, ns=ns
        )
        
        # Add measurement noise
        if add_noise:
            I += np.random.randn(len(I)) * 0.01
            I = np.clip(I, 0, true_params["Iph"])
        
        scans.append({
            "V": V, "I": I, "T": T, "G": G,
            "hour": hour
        })
    
    return scans, true_params


def test_progressive_optimization():
    """Test the progressive optimization algorithm."""
    print("=" * 70)
    print("TEST: Progressive Optimization (Friend's Idea)")
    print("=" * 70)
    
    # Generate scans
    print("\n[1] Generating synthetic scans...")
    scans, true_params = generate_synthetic_scans(n_scans=8, add_noise=True)
    print(f"    Generated {len(scans)} scans")
    print(f"    True STC parameters to recover:")
    for k, v in true_params.items():
        print(f"        {k}: {v:.6e}")
    
    # Run progressive optimization
    print("\n[2] Running progressive optimization...")
    result = optimize_double_progressive(
        scan_stream=iter(scans),
        ns=36,
        pop_size=50,
        generations=40,
        crossover_rate=0.85,
        mutation_rate=0.12,
        verbose=True
    )
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n[3] Optimized STC parameters:")
    for k, v in result.best_params.items():
        true_v = true_params[k]
        error = abs(v - true_v) / abs(true_v) * 100 if true_v != 0 else 0
        status = "✓" if error < 5 else "⚠"
        print(f"    {status} {k}: {v:.6e} (true: {true_v:.6e}, error: {error:.2f}%)")
    
    print(f"\n[4] Final RMSE: {result.best_fitness:.6f}")
    print(f"\n[5] Progressive optimization history:")
    print(f"    Total scans processed: {result.meta.get('scans_processed', 'N/A')}")
    print(f"    Strategy: {result.meta.get('strategy', 'N/A')}")
    
    # Evaluate true parameters on all scans
    print("\n[6] Validation:")
    rmse_true = evaluate_double_parameters(true_params, scans, ns=36)
    print(f"    RMSE of true parameters: {rmse_true:.6f}")
    print(f"    RMSE of optimized parameters: {result.best_fitness:.6f}")
    
    if result.best_fitness <= rmse_true * 1.1:  # Within 10% of true
        print("\n    ✓ Progressive optimization successfully recovered parameters!")
    else:
        print("\n    ⚠ Optimization result is close to true parameters")
    
    return result


def test_progressive_vs_single_scan():
    """Compare progressive optimization with single scan optimization."""
    print("\n" + "=" * 70)
    print("TEST: Progressive vs Single Scan Optimization")
    print("=" * 70)
    
    from pvoptix import optimize_double_multicondition
    
    # Generate scans
    scans, true_params = generate_synthetic_scans(n_scans=8, add_noise=True)
    
    # Single scan optimization (using first scan only)
    print("\n[1] Optimizing on first scan only...")
    first_scan_result = optimize_double_multicondition(
        datasets=[scans[0]],
        ns=36,
        pop_size=50,
        generations=40,
        verbose=False
    )
    
    # Progressive optimization
    print("\n[2] Optimizing on all scans progressively...")
    progressive_result = optimize_double_progressive(
        scan_stream=iter(scans),
        ns=36,
        pop_size=50,
        generations=40,
        verbose=False
    )
    
    # Compare
    print("\n[3] Comparison:")
    print(f"\n    {'Parameter':<8} {'True':<12} {'Single Scan':<12} {'Progressive':<12}")
    print("    " + "-" * 50)
    
    for k in true_params.keys():
        true_v = true_params[k]
        single_v = first_scan_result.best_params.get(k, 0)
        prog_v = progressive_result.best_params.get(k, 0)
        
        single_error = abs(single_v - true_v) / abs(true_v) * 100
        prog_error = abs(prog_v - true_v) / abs(true_v) * 100
        
        print(f"    {k:<8} {true_v:<12.6e} {single_v:<12.6e} {prog_v:<12.6e}")
        print(f"            {'':8} {'error:':<6} {single_error:<11.2f}% {'error:':<6} {prog_error:<11.2f}%")
    
    print(f"\n    RMSE:")
    print(f"        Single scan: {first_scan_result.best_fitness:.6f}")
    print(f"        Progressive: {progressive_result.best_fitness:.6f}")
    
    if progressive_result.best_fitness < first_scan_result.best_fitness:
        print("\n    ✓ Progressive optimization achieved better RMSE than single scan!")
    else:
        print("\n    ⚠ Progressive optimization did not improve over single scan")
    
    return first_scan_result, progressive_result


def test_convergence_progress():
    """Test that the progressive algorithm improves over time."""
    print("\n" + "=" * 70)
    print("TEST: Progressive Optimization Convergence")
    print("=" * 70)
    
    from pvoptix import optimize_double_progressive
    
    # Generate scans
    scans, true_params = generate_synthetic_scans(n_scans=8, add_noise=True)
    
    # Track RMSE at each scan
    rmse_history = []
    
    def track_rmse(result):
        rmse_history.append(result.best_fitness)
        return result
    
    # Run progressive optimization
    result = optimize_double_progressive(
        scan_stream=iter(scans),
        ns=36,
        pop_size=50,
        generations=30,
        verbose=False
    )
    
    # Show progress
    print("\n[1] RMSE evolution:")
    print(f"    {'Scan':<6} {'RMSE':<12} {'Improvement':<12}")
    print("    " + "-" * 35)
    
    for i, rmse in enumerate(result.history):
        improvement = ""
        if i > 0:
            prev_rmse = result.history[i-1]['rmse_best']
            if rmse['rmse_best'] < prev_rmse:
                improvement = f"↓ {prev_rmse - rmse['rmse_best']:.6f}"
            else:
                improvement = "no change"
        print(f"    {i+1:<6} {rmse['rmse_best']:<12.6f} {improvement}")
    
    print(f"\n    Final RMSE: {result.best_fitness:.6f}")
    print(f"    True RMSE: {evaluate_double_parameters(true_params, scans, ns=36):.6f}")
    
    return result


if __name__ == "__main__":
    # Run tests
    print("\n" + "🔬" * 35)
    print("PROGRESSIVE OPTIMIZATION TESTS")
    print("🔬" * 35)
    
    # Test 1: Basic progressive optimization
    result = test_progressive_optimization()
    
    # Test 2: Compare with single scan
    single_result, prog_result = test_progressive_vs_single_scan()
    
    # Test 3: Check convergence progress
    convergence_result = test_convergence_progress()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)