# tests/test_progressive_flowchart.py
"""Test progressive optimization exactly as per flowchart."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pvoptix.pvoptix.api import (
    optimize_double_multicondition,
    evaluate_double_parameters,
    create_virtual_stc_curve_double
)
from pvoptix.pvoptix.api import simulate_iv_curve_double


def create_test_scans():
    """Create scans matching flowchart example."""
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }
    ns = 36
    
    # Flowchart example conditions
    conditions = [
        (733, 23, "08h10"),
        (810, 24.2, "08h20"),
        (650, 22.1, "08h30"),
        (550, 21.5, "08h40"),
        (450, 20.8, "08h50"),
    ]
    
    scans = []
    for G, T_c, label in conditions:
        T = T_c + 273.15
        V = np.linspace(0, 21.6, 50)
        I = simulate_iv_curve_double(
            V, stc_params=true_params,
            temperature_k=T, irradiance_w_m2=G, ns=ns
        )
        scans.append({
            "V": V, "I": I, "T": T, "G": G,
            "model": label,
            "scan_id": label
        })
    
    return scans, true_params


def test_flowchart_step_by_step():
    """Test each step of the flowchart."""
    print("=" * 70)
    print("TEST: Flowchart Step-by-Step Validation")
    print("=" * 70)
    
    scans, true_params = create_test_scans()
    ns = 36
    
    # Step 1: Virtual STC curve
    print("\n[Step 1] Initialization:")
    stc_virtual = create_virtual_stc_curve_double(ns=ns)
    print(f"  ✓ Virtual STC curve created ({len(stc_virtual['V'])} points)")
    
    # Step 2: First scan (08h10)
    print("\n[Step 2] First Scan (08h10):")
    datasets = [scans[0], stc_virtual]
    print(f"  Conditions: M=2 (STC + 1 measured)")
    print(f"  G={scans[0]['G']:.0f} W/m², T={scans[0]['T']-273.15:.1f}°C")
    
    result1 = optimize_double_multicondition(
        datasets=datasets,
        ns=ns,
        pop_size=30,
        generations=20,
        verbose=False
    )
    print(f"  → θ₁ RMSE: {result1.best_fitness:.6f}")
    
    # Step 3: Second scan (08h20)
    print("\n[Step 3] Second Scan (08h20):")
    datasets = scans[:2] + [stc_virtual]
    print(f"  Conditions: M=3 (STC + 2 measured)")
    print(f"  New: G={scans[1]['G']:.0f} W/m², T={scans[1]['T']-273.15:.1f}°C")
    
    result2 = optimize_double_multicondition(
        datasets=datasets,
        ns=ns,
        pop_size=30,
        generations=20,
        verbose=False
    )
    print(f"  → θ₂ RMSE: {result2.best_fitness:.6f}")
    
    # Compare and keep best
    rmse1 = evaluate_double_parameters(result1.best_params, datasets, ns)
    rmse2 = evaluate_double_parameters(result2.best_params, datasets, ns)
    print(f"\n  Comparison:")
    print(f"    RMSE(θ₁) = {rmse1:.6f}")
    print(f"    RMSE(θ₂) = {rmse2:.6f}")
    
    best_theta = result1.best_params if rmse1 < rmse2 else result2.best_params
    print(f"  → Best θ = θ₁" if rmse1 < rmse2 else "  → Best θ = θ₂")
    
    # Step 4: Successive scans
    print("\n[Step 4] Successive Scans:")
    best_rmse = min(rmse1, rmse2)
    
    for i in range(2, len(scans)):
        datasets = scans[:i+1] + [stc_virtual]
        result_i = optimize_double_multicondition(
            datasets=datasets,
            ns=ns,
            pop_size=30,
            generations=20,
            verbose=False
        )
        rmse_i = evaluate_double_parameters(result_i.best_params, datasets, ns)
        rmse_best = evaluate_double_parameters(best_theta, datasets, ns)
        
        print(f"\n  Scan {i+1}: G={scans[i]['G']:.0f} W/m²")
        print(f"    RMSE(θ_new) = {rmse_i:.6f}")
        print(f"    RMSE(θ_best) = {rmse_best:.6f}")
        
        if rmse_i < rmse_best:
            best_theta = result_i.best_params
            best_rmse = rmse_i
            print(f"    → NEW BEST θ!")
        else:
            print(f"    → Previous θ remains best")
    
    # Step 5: End of day
    print("\n[Step 5] End of Day Results:")
    print(f"  Final RMSE: {best_rmse:.6f}")
    print("\n  Final STC Parameters:")
    for k, v in best_theta.items():
        true_v = true_params[k]
        error = abs(v - true_v) / true_v * 100
        status = "✓" if error < 10 else "⚠"
        print(f"    {status} {k}: {v:.6e} (true: {true_v:.6e}, error: {error:.2f}%)")
    
    return best_theta, best_rmse


if __name__ == "__main__":
    best_theta, best_rmse = test_flowchart_step_by_step()
    print("\n" + "=" * 70)
    print("✅ FLOWCHART VALIDATION COMPLETE")
    print("=" * 70)