# examples/demo_progressive_flowchart.py
"""
Demonstration 2: Progressive Optimization Algorithm (Flowchart)
Corresponds exactly to the flowchart from the paper.

ALGORITHM STEPS:
1. Start with first scan + virtual STC condition -> theta1
2. For each new scan i from 2 to N:
   a. Add the new scan to the accumulated dataset
   b. Optimize theta_new on ALL accumulated scans + virtual STC
   c. Compute RMSE(theta_new) on current dataset
   d. Compute RMSE(theta_best) on current dataset
   e. If RMSE(theta_new) < RMSE(theta_best), update theta_best = theta_new
   f. Otherwise, keep theta_best
3. Return theta_best after processing all scans

Includes I-V curves, P-V curves, MPP analysis, and RMSE evolution tracking.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pvoptix import (
    load_datasets_from_dir,
    optimize_double_progressive,
    simulate_iv_curve_double
)


def compute_power(V, I):
    """Compute power from voltage and current arrays: P = V * I."""
    return V * I


def find_mpp(V, I):
    """Find Maximum Power Point from V-I data."""
    P = compute_power(V, I)
    idx = np.argmax(P)
    return V[idx], I[idx], P[idx]


def plot_case_constant_temperature(datasets, best_params, ns=36, title_suffix=""):
    """
    Plot I-V and P-V curves for constant temperature (25°C) with varying irradiance.
    Conditions: G = 200, 400, 600, 800, 1000 W/m², T = 25°C
    Layout: Column-major order (left to right, then top to bottom)
    """
    # Filter datasets for T=25°C
    const_temp_datasets = []
    for ds in datasets:
        T_c = ds["T"] - 273.15
        if abs(T_c - 25.0) < 0.5:
            const_temp_datasets.append(ds)
    
    const_temp_datasets.sort(key=lambda x: x["G"])
    
    if not const_temp_datasets:
        print("No datasets found for constant temperature (25°C)")
        return [], []
    
    n_plots = len(const_temp_datasets)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure with 2 * n_rows rows and n_cols columns
    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * 2 * n_rows))
    fig.suptitle(f'Constant Temperature (25°C) - Varying Irradiance{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92)
    
    mpp_errors = []
    rmse_values = []
    
    # Column-major order: fill columns first
    for plot_idx in range(n_plots):
        col = plot_idx % n_cols
        row = plot_idx // n_cols
        
        ds = const_temp_datasets[plot_idx]
        V = ds["V"]
        I_exp = ds["I"]
        T = ds["T"]
        G = ds["G"]
        T_c = T - 273.15
        
        I_sim = simulate_iv_curve_double(
            V, stc_params=best_params,
            temperature_k=T, irradiance_w_m2=G, ns=ns
        )
        
        P_exp = compute_power(V, I_exp)
        P_sim = compute_power(V, I_sim)
        
        V_mpp_exp, _, P_mpp_exp = find_mpp(V, I_exp)
        V_mpp_sim, _, P_mpp_sim = find_mpp(V, I_sim)
        
        rmse = np.sqrt(np.mean((I_exp - I_sim) ** 2))
        rmse_values.append(rmse)
        mpp_error = abs(P_mpp_sim - P_mpp_exp) / P_mpp_exp * 100 if P_mpp_exp > 0 else 0
        mpp_errors.append(mpp_error)
        
        # I-V curve
        ax_iv = axes[2 * row, col]
        ax_iv.plot(V, I_exp, 'o', markersize=2, label='Measured', alpha=0.7)
        ax_iv.plot(V, I_sim, '-', linewidth=1.5, label='Simulated')
        ax_iv.set_xlabel('Voltage [V]', fontsize=9)
        ax_iv.set_ylabel('Current [A]', fontsize=9)
        ax_iv.set_title(f'G={G:.0f} W/m², T={T_c:.0f}°C\nRMSE={rmse:.4f} A', fontsize=9)
        ax_iv.grid(True, alpha=0.3)
        ax_iv.legend(fontsize=7, loc='upper right')
        ax_iv.tick_params(labelsize=8)
        
        # P-V curve
        ax_pv = axes[2 * row + 1, col]
        ax_pv.plot(V, P_exp, 'o', markersize=2, label='Measured', alpha=0.7)
        ax_pv.plot(V, P_sim, '-', linewidth=1.5, label='Simulated')
        ax_pv.plot(V_mpp_exp, P_mpp_exp, 'r*', markersize=10, 
                   label=f'MPP Exp: {P_mpp_exp:.1f}W')
        ax_pv.plot(V_mpp_sim, P_mpp_sim, 'g*', markersize=10, 
                   label=f'MPP Sim: {P_mpp_sim:.1f}W')
        ax_pv.set_xlabel('Voltage [V]', fontsize=9)
        ax_pv.set_ylabel('Power [W]', fontsize=9)
        ax_pv.set_title(f'MPP Error: {mpp_error:.1f}%', fontsize=9)
        ax_pv.grid(True, alpha=0.3)
        ax_pv.legend(fontsize=6, loc='upper right')
        ax_pv.tick_params(labelsize=8)
    
    # Hide unused subplots
    for plot_idx in range(n_plots, n_cols * n_rows):
        col = plot_idx % n_cols
        row = plot_idx // n_cols
        if 2 * row < len(axes):
            axes[2 * row, col].set_visible(False)
            axes[2 * row + 1, col].set_visible(False)
    
    plt.show(block=False)
    return mpp_errors, rmse_values


def plot_case_constant_irradiance(datasets, best_params, ns=36, title_suffix=""):
    """
    Plot I-V and P-V curves for constant irradiance (1000 W/m²) with varying temperature.
    Conditions: G = 1000 W/m², T = 20, 25, 40, 60°C
    Layout: Column-major order (left to right, then top to bottom)
    """
    # Filter datasets for G=1000 W/m²
    const_irr_datasets = []
    for ds in datasets:
        if abs(ds["G"] - 1000) < 10:
            const_irr_datasets.append(ds)
    
    const_irr_datasets.sort(key=lambda x: x["T"])
    
    if not const_irr_datasets:
        print("No datasets found for constant irradiance (1000 W/m²)")
        return [], []
    
    n_plots = len(const_irr_datasets)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * 2 * n_rows))
    fig.suptitle(f'Constant Irradiance (1000 W/m²) - Varying Temperature{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92)
    
    mpp_errors = []
    rmse_values = []
    
    # Column-major order: fill columns first
    for plot_idx in range(n_plots):
        col = plot_idx % n_cols
        row = plot_idx // n_cols
        
        ds = const_irr_datasets[plot_idx]
        V = ds["V"]
        I_exp = ds["I"]
        T = ds["T"]
        G = ds["G"]
        T_c = T - 273.15
        
        I_sim = simulate_iv_curve_double(
            V, stc_params=best_params,
            temperature_k=T, irradiance_w_m2=G, ns=ns
        )
        
        P_exp = compute_power(V, I_exp)
        P_sim = compute_power(V, I_sim)
        
        V_mpp_exp, _, P_mpp_exp = find_mpp(V, I_exp)
        V_mpp_sim, _, P_mpp_sim = find_mpp(V, I_sim)
        
        rmse = np.sqrt(np.mean((I_exp - I_sim) ** 2))
        rmse_values.append(rmse)
        mpp_error = abs(P_mpp_sim - P_mpp_exp) / P_mpp_exp * 100 if P_mpp_exp > 0 else 0
        mpp_errors.append(mpp_error)
        
        ax_iv = axes[2 * row, col]
        ax_iv.plot(V, I_exp, 'o', markersize=2, label='Measured', alpha=0.7)
        ax_iv.plot(V, I_sim, '-', linewidth=1.5, label='Simulated')
        ax_iv.set_xlabel('Voltage [V]', fontsize=9)
        ax_iv.set_ylabel('Current [A]', fontsize=9)
        ax_iv.set_title(f'G={G:.0f} W/m², T={T_c:.0f}°C\nRMSE={rmse:.4f} A', fontsize=9)
        ax_iv.grid(True, alpha=0.3)
        ax_iv.legend(fontsize=7, loc='upper right')
        ax_iv.tick_params(labelsize=8)
        
        ax_pv = axes[2 * row + 1, col]
        ax_pv.plot(V, P_exp, 'o', markersize=2, label='Measured', alpha=0.7)
        ax_pv.plot(V, P_sim, '-', linewidth=1.5, label='Simulated')
        ax_pv.plot(V_mpp_exp, P_mpp_exp, 'r*', markersize=10, 
                   label=f'MPP Exp: {P_mpp_exp:.1f}W')
        ax_pv.plot(V_mpp_sim, P_mpp_sim, 'g*', markersize=10, 
                   label=f'MPP Sim: {P_mpp_sim:.1f}W')
        ax_pv.set_xlabel('Voltage [V]', fontsize=9)
        ax_pv.set_ylabel('Power [W]', fontsize=9)
        ax_pv.set_title(f'MPP Error: {mpp_error:.1f}%', fontsize=9)
        ax_pv.grid(True, alpha=0.3)
        ax_pv.legend(fontsize=6, loc='upper right')
        ax_pv.tick_params(labelsize=8)
    
    # Hide unused subplots
    for plot_idx in range(n_plots, n_cols * n_rows):
        col = plot_idx % n_cols
        row = plot_idx // n_cols
        if 2 * row < len(axes):
            axes[2 * row, col].set_visible(False)
            axes[2 * row + 1, col].set_visible(False)
    
    plt.show(block=False)
    return mpp_errors, rmse_values


def print_rmse_table(rmse_by_scan, datasets):
    """
    Print a formatted RMSE comparison table similar to Table 2 in the paper.
    """
    print("\n" + "=" * 70)
    print("TABLE: RMSE for Progressive Algorithm Across All Conditions")
    print("=" * 70)
    
    print(f"\n{'Operating Condition':<35} {'RMSE (A)':<15}")
    print("-" * 50)
    
    all_rmse = []
    for ds, rmse in zip(datasets, rmse_by_scan):
        G = ds["G"]
        T_c = ds["T"] - 273.15
        condition = f"G={G:.0f} W/m², T={T_c:.0f}°C"
        print(f"{condition:<35} {rmse:<15.6f}")
        all_rmse.append(rmse)
    
    print("-" * 50)
    print(f"{'Overall RMSE (average)':<35} {np.mean(all_rmse):<15.6f}")
    print(f"{'Standard Deviation':<35} {np.std(all_rmse):<15.6f}")
    print(f"{'Minimum RMSE':<35} {np.min(all_rmse):<15.6f}")
    print(f"{'Maximum RMSE':<35} {np.max(all_rmse):<15.6f}")


def print_algorithm_flow_summary(result):
    """
    Print a summary of the algorithm's step-by-step execution.
    """
    print("\n" + "=" * 70)
    print("ALGORITHM FLOWCHART EXECUTION SUMMARY")
    print("=" * 70)
    
    print("\nStep 1: Initialization")
    print("  - Virtual STC condition created")
    print("  - First scan added to dataset")
    
    print("\nStep 2: Progressive Processing")
    print(f"  - Total scans processed: {len(result.history)}")
    
    print("\n  Scan-by-scan decisions:")
    print(f"  {'Scan':<6} {'Decision':<25} {'RMSE_best (A)':<15}")
    print("  " + "-" * 50)
    
    for entry in result.history:
        scan_id = entry['scan_id']
        rmse_best = entry['rmse_best']
        if entry['improved']:
            decision = "✓ NEW BEST - UPDATED"
        else:
            decision = "○ KEPT PREVIOUS"
        print(f"  {scan_id:<6} {decision:<25} {rmse_best:<15.6f}")
    
    print("\nStep 3: Final Result")
    print(f"  - Final theta* returned after processing all {len(result.history)} scans")
    print(f"  - Final RMSE: {result.best_fitness:.6f} A")


def main():
    print("=" * 70)
    print("DEMONSTRATION 2: PROGRESSIVE OPTIMIZATION ALGORITHM (FLOWCHART)")
    print("=" * 70)

    # Load real S75 data
    data_dir = Path(__file__).parent.parent / "src" / "pvoptix" / "datasets" / "data"
    
    if not data_dir.exists():
        print(f"\nData directory not found: {data_dir}")
        return
    
    datasets = load_datasets_from_dir(str(data_dir))
    n_datasets = len(datasets)

    print(f"\nData loaded: {n_datasets} operating conditions")
    for i, ds in enumerate(datasets):
        print(f"   Condition {i+1}: G={ds['G']:.0f} W/m², T={ds['T']-273.15:.0f} °C")

    # Run progressive optimization
    print("\n" + "-" * 50)
    print("RUNNING PROGRESSIVE OPTIMIZATION (FLOWCHART)")
    print("-" * 50)
    
    result = optimize_double_progressive(
        scan_stream=iter(datasets),
        ns=36,
        include_virtual_stc=True,
        ga_kwargs={
            "pop_size": 100,
            "generations": 50,
            "crossover_rate": 0.85,
            "mutation_rate": 0.12,
            "verbose": True,
            "live_plot": True,
            "figsize": (10, 5),
            "auto_close_plot": True,
            "plot_display_seconds": 3
        },
        verbose=True
    )

    print("\n" + "-" * 50)
    print("PROGRESSIVE ALGORITHM RESULTS")
    print("-" * 50)

    print(f"\nFinal STC parameters after all scans (theta*):")
    for k, v in result.best_params.items():
        print(f"   {k}: {v:.6e}")

    print(f"\nFinal global RMSE: {result.best_fitness:.6f} A")

    # =========================================================
    # FIGURE 1: RMSE EVOLUTION
    # =========================================================
    print("\nGenerating RMSE evolution plot...")
    
    scans = [entry['scan_id'] for entry in result.history]
    rmse_values = [entry['rmse_best'] for entry in result.history]
    
    min_rmse_idx = np.argmin(rmse_values)
    min_rmse_scan = scans[min_rmse_idx]
    min_rmse_value = rmse_values[min_rmse_idx]
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(scans, rmse_values, 'bo-', linewidth=2, markersize=8)
    ax1.plot(min_rmse_scan, min_rmse_value, 'r*', markersize=15, 
             label=f'Best RMSE: {min_rmse_value:.6f} A (Scan {min_rmse_scan})')
    ax1.set_xlabel('Scan Number', fontsize=12)
    ax1.set_ylabel('RMSE [A]', fontsize=12)
    ax1.set_title('RMSE Evolution - Progressive Algorithm (Non-increasing Property)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    plt.tight_layout()
    plt.show(block=False)
    
    # =========================================================
    # FIGURE 2 & 3: I-V AND P-V CURVES
    # =========================================================
    
    print("\nGenerating Figure 2: Constant Temperature (25°C) - Varying Irradiance")
    mpp_errors_temp, rmse_temp = plot_case_constant_temperature(
        datasets, result.best_params, ns=36, title_suffix=" (Progressive Algorithm)"
    )
    
    print("\nGenerating Figure 3: Constant Irradiance (1000 W/m²) - Varying Temperature")
    mpp_errors_irr, rmse_irr = plot_case_constant_irradiance(
        datasets, result.best_params, ns=36, title_suffix=" (Progressive Algorithm)"
    )
    
    # =========================================================
    # RMSE TABLE
    # =========================================================
    
    rmse_by_scan = []
    for ds in datasets:
        V = ds["V"]
        I_exp = ds["I"]
        T = ds["T"]
        G = ds["G"]
        I_sim = simulate_iv_curve_double(
            V, stc_params=result.best_params,
            temperature_k=T, irradiance_w_m2=G, ns=36
        )
        rmse = np.sqrt(np.mean((I_exp - I_sim) ** 2))
        rmse_by_scan.append(rmse)
    
    print_rmse_table(rmse_by_scan, datasets)
    
    # =========================================================
    # ALGORITHM FLOWCHART SUMMARY
    # =========================================================
    
    print_algorithm_flow_summary(result)
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    print(f"\nAlgorithm: Progressive Optimization (Flowchart)")
    print(f"Total scans processed: {result.meta.get('scans_processed', len(datasets))}")
    print(f"Final RMSE (all conditions): {result.best_fitness:.6f} A")
    print(f"Best RMSE achieved: {min_rmse_value:.6f} A at Scan {min_rmse_scan}")
    
    # Improvement analysis
    first_rmse = rmse_values[0]
    last_rmse = rmse_values[-1]
    improvement_pct = (first_rmse - last_rmse) / first_rmse * 100 if first_rmse > 0 else 0
    
    print(f"\nRMSE improvement: {first_rmse:.6f} A -> {last_rmse:.6f} A ({improvement_pct:.1f}% improvement)")
    
    if mpp_errors_temp:
        print(f"\nConstant Temperature (25°C) - Average MPP error: {np.mean(mpp_errors_temp):.1f}%")
    if mpp_errors_irr:
        print(f"Constant Irradiance (1000 W/m²) - Average MPP error: {np.mean(mpp_errors_irr):.1f}%")
    
    # Parameter validation
    print("\nParameter validation:")
    p = result.best_params
    checks = [
        (p.get("Rs", 0) > 0, f"Rs = {p.get('Rs', 0):.6f} Ohm > 0"),
        (p.get("Rsh", 0) > p.get("Rs", 0), f"Rsh = {p.get('Rsh', 0):.2f} Ohm >> Rs"),
        (1.0 < p.get("n1", 0) < 2.0, f"n1 = {p.get('n1', 0):.4f} in [1,2]"),
        (1.0 < p.get("n2", 0) < 2.0, f"n2 = {p.get('n2', 0):.4f} in [1,2]"),
        (abs(p.get("Iph", 0) - 4.68) < 0.5, f"Iph = {p.get('Iph', 0):.3f} A close to datasheet (4.68 A)")
    ]
    for valid, msg in checks:
        status = "VALID" if valid else "INVALID"
        print(f"  - {msg}: {status}")
    
    print("\nRESULT: The progressive algorithm maintains a single STC parameter set")
    print("that is continuously refined as more operating conditions are processed.")
    print("The model predicts I-V and P-V characteristics across:")
    print("  - Irradiance range: 200 to 1000 W/m²")
    print("  - Temperature range: 20 to 60°C")
    
    plt.show(block=True)


if __name__ == "__main__":
    main()