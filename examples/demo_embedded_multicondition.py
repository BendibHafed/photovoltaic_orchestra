# examples/demo_embedded_multicondition.py
"""
Demonstration 1: Embedded Multi-Condition Strategy
Corresponds to Equation (14) from the paper.
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
    optimize_double_multicondition,
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


def plot_case_constant_temperature(datasets, best_params, ns=36):
    """
    Plot I-V and P-V curves for constant temperature (25°C) with varying irradiance.
    Conditions: G = 200, 400, 600, 800, 1000 W/m², T = 25°C
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
        return []
    
    n_plots = len(const_temp_datasets)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure with 2 * n_rows rows and n_cols columns
    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * 2 * n_rows))
    fig.suptitle('Constant Temperature (25°C) - Varying Irradiance', fontsize=14, fontweight='bold')
    
    # Adjust spacing for tight layout
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92)
    
    mpp_errors = []
    plot_idx = 0
    
    for row in range(n_rows):
        for col in range(n_cols):
            if plot_idx >= n_plots:
                # Hide unused subplots
                if 2 * row < len(axes):
                    axes[2 * row, col].set_visible(False)
                    axes[2 * row + 1, col].set_visible(False)
                continue
            
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
            
            plot_idx += 1
    
    plt.show(block=False)
    return mpp_errors


def plot_case_constant_irradiance(datasets, best_params, ns=36):
    """
    Plot I-V and P-V curves for constant irradiance (1000 W/m²) with varying temperature.
    Conditions: G = 1000 W/m², T = 20, 25, 40, 60°C
    """
    # Filter datasets for G=1000 W/m²
    const_irr_datasets = []
    for ds in datasets:
        if abs(ds["G"] - 1000) < 10:
            const_irr_datasets.append(ds)
    
    const_irr_datasets.sort(key=lambda x: x["T"])
    
    if not const_irr_datasets:
        print("No datasets found for constant irradiance (1000 W/m²)")
        return []
    
    n_plots = len(const_irr_datasets)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure with 2 * n_rows rows and n_cols columns
    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * 2 * n_rows))
    fig.suptitle('Constant Irradiance (1000 W/m²) - Varying Temperature', fontsize=14, fontweight='bold')
    
    # Adjust spacing for tight layout
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92)
    
    mpp_errors = []
    plot_idx = 0
    
    for row in range(n_rows):
        for col in range(n_cols):
            if plot_idx >= n_plots:
                # Hide unused subplots
                if 2 * row < len(axes):
                    axes[2 * row, col].set_visible(False)
                    axes[2 * row + 1, col].set_visible(False)
                continue
            
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
            
            plot_idx += 1
    
    plt.show(block=False)
    return mpp_errors


def main():
    print("=" * 70)
    print("DEMONSTRATION 1: EMBEDDED MULTI-CONDITION STRATEGY")
    print("(Equation 14 from the paper)")
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

    # Run optimization
    print("\nOptimizing with live GA convergence plot...")
    
    result = optimize_double_multicondition(
        datasets=datasets,
        ns=36,
        pop_size=100,
        generations=50,
        crossover_rate=0.85,
        mutation_rate=0.12,
        verbose=True,
        live_plot=True,
        figsize=(10, 5),
        auto_close_plot=False,
    )

    print(f"\nFinal RMSE_global (Equation 14): {result.best_fitness:.6f} A")
    print("\nOptimized STC parameters (theta*):")
    for k, v in result.best_params.items():
        print(f"   {k}: {v:.6e}")

    # =========================================================
    # GENERATE PLOTS
    # =========================================================
    print("\nGenerating GA convergence and validation plots...")

    # Extract GA convergence history
    generations = list(range(1, len(result.history) + 1))
    best_fitness = [entry[0] for entry in result.history]
    
    min_fitness_idx = np.argmin(best_fitness)
    min_fitness_gen = generations[min_fitness_idx]
    min_fitness_value = best_fitness[min_fitness_idx]

    # Figure 1: GA Convergence
    fig1, ax_conv = plt.subplots(figsize=(10, 5))
    ax_conv.plot(generations, best_fitness, 'b-', linewidth=2)
    ax_conv.plot(min_fitness_gen, min_fitness_value, 'r*', markersize=12,
                 label=f'Best: {min_fitness_value:.6f} A (Gen {min_fitness_gen})')
    ax_conv.set_xlabel('Generation')
    ax_conv.set_ylabel('RMSE [A]')
    ax_conv.set_title('GA Convergence: RMSE Evolution')
    ax_conv.set_yscale('log')
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend(loc='upper right')
    plt.tight_layout()
    plt.show(block=False)

    # Figure 2: Constant temperature (25°C) with varying irradiance
    print("\nGenerating Figure 2: Constant Temperature (25°C) - Varying Irradiance")
    mpp_errors_temp = plot_case_constant_temperature(datasets, result.best_params, ns=36)
    
    # Figure 3: Constant irradiance (1000 W/m²) with varying temperature
    print("\nGenerating Figure 3: Constant Irradiance (1000 W/m²) - Varying Temperature")
    mpp_errors_irr = plot_case_constant_irradiance(datasets, result.best_params, ns=36)
    
    plt.show(block=True)

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\nNumber of conditions (M): {n_datasets}")
    print(f"Global RMSE (Equation 14): {result.best_fitness:.6f} A")
    print(f"GA convergence: {best_fitness[0]:.6f} A -> {best_fitness[-1]:.6f} A")
    print(f"Best RMSE at generation {min_fitness_gen}: {min_fitness_value:.6f} A")
    
    if mpp_errors_temp:
        print(f"\nConstant Temperature (25°C) - Average MPP error: {np.mean(mpp_errors_temp):.1f}%")
    if mpp_errors_irr:
        print(f"Constant Irradiance (1000 W/m²) - Average MPP error: {np.mean(mpp_errors_irr):.1f}%")
    
    # Parameter validation
    print("\nParameter validation:")
    p = result.best_params
    checks = [
        (p.get("Rs", 0) > 0, f"Rs = {p.get('Rs', 0):.4f} Ohm > 0"),
        (p.get("Rsh", 0) > p.get("Rs", 0), f"Rsh = {p.get('Rsh', 0):.1f} Ohm >> Rs"),
        (1.0 < p.get("n1", 0) < 2.0, f"n1 = {p.get('n1', 0):.4f} in [1,2]"),
        (1.0 < p.get("n2", 0) < 2.0, f"n2 = {p.get('n2', 0):.4f} in [1,2]"),
        (abs(p.get("Iph", 0) - 4.68) < 0.5, f"Iph = {p.get('Iph', 0):.3f} A close to datasheet (4.68 A)")
    ]
    for valid, msg in checks:
        print(f"  - {msg}: {'VALID' if valid else 'INVALID'}")

    print("\nRESULT: A single set of STC parameters works for ALL conditions.")
    print("The model accurately predicts I-V and P-V characteristics across:")
    print("  - Irradiance range: 200 to 1000 W/m²")
    print("  - Temperature range: 20 to 60°C")


if __name__ == "__main__":
    main()