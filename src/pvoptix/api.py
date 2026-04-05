# -*- coding: utf-8 -*-
"""
PvOptiX - Photovoltaic Parameter Optimization Engine

Public API for PV module parameter extraction using double-diode model
with progressive multi-condition optimization.

This module provides the main entry points for:
1. Double-diode model (7 parameters) with multi-condition optimization
2. Progressive optimization (accumulate and compare scans)
3. Dataset loading from .dat files

Author: BENDIB HE.
Version: 1.0.0
License: MIT

Example:
    >>> from pvoptix import load_datasets_from_dir, optimize_double_progressive
    >>> 
    >>> # Load real I-V curves from your hardware
    >>> datasets = load_datasets_from_dir("path/to/data")
    >>> 
    >>> # Run progressive optimization
    >>> result = optimize_double_progressive(
    ...     scan_stream=iter(datasets),
    ...     ns=36,
    ...     ga_kwargs={"pop_size": 50, "generations": 60}
    ... )
    >>> 
    >>> print(result.best_params)
    >>> print(f"RMSE: {result.best_fitness:.6f}")
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, asdict
from typing import Any, Optional
import numpy as np

# Utiliser des imports relatifs (avec .)
from pvoptix.datasets.builder import build_dataset
from pvoptix.models.parameters import (
    PVModelCoefficients, default_coeffs,
    iph_model, i01_model, i02_model, rs_model, rsh_model_double
)
from pvoptix.objectives.double import pv_rmse_objective_double
from pvoptix.optimization.ga.core import run_ga
from pvoptix.optimization.ga.genome_mapping_double import (
    decode_individual_double, encode_individual_double
)
from pvoptix.solvers.double import iv_model_double


# =============================================================================
# Public Data Classes
# =============================================================================

@dataclass(frozen=True)
class OptimizationResult:
    """
    High-level optimization output returned by the public API.
    
    Attributes:
        best_params: Best STC parameters found {Rs, Rsh, I01, I02, Iph, n1, n2}
        best_fitness: Best RMSE value (lower is better)
        history: List of (fitness, params) per generation
        meta: Additional metadata (settings, model type, etc.)
    """
    best_params: dict[str, float]
    best_fitness: float
    history: list[tuple[float, dict[str, float]]]
    meta: dict[str, Any]


@dataclass
class ModelConfig:
    """
    Complete configuration of the PV model.
    
    Attributes:
        ns: Number of series-connected cells (default: 36 for S75 module)
        coefficients: Model coefficients (alpha_I, alpha_Rs, etc.)
    """
    ns: int = 36
    coefficients: PVModelCoefficients = None

    def __post_init__(self):
        if self.coefficients is None:
            self.coefficients = default_coeffs


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_datasets_from_dir(data_dir: str) -> list[dict[str, Any]]:
    """
    Load experimental I-V datasets from a directory containing .dat files.
    
    File format: <model>_<T_C>_<G>.dat
    Example: "S75_25_1000.dat" (S75 module, 25°C, 1000 W/m²)
    
    Args:
        data_dir: Directory containing .dat files
        
    Returns:
        List of dataset dicts with keys: model, T, G, V, I
        
    Example:
        >>> datasets = load_datasets_from_dir("data/")
        >>> print(len(datasets))
        8
        >>> print(datasets[0]['G'])
        1000.0
    """
    return build_dataset(data_dir)


# =============================================================================
# DOUBLE-DIODE MODEL (7 parameters)
# =============================================================================

def simulate_iv_curve_double(
    voltage: np.ndarray,
    *,
    stc_params: dict[str, float],
    temperature_k: float,
    irradiance_w_m2: float,
    ns: int = 36,
    coefficients: Optional[PVModelCoefficients] = None
) -> np.ndarray:
    """
    Simulate I-V curve using double-diode model with STC parameters.
    
    Implements equation (1) from NCMAI'26 paper with translation equations (4-10).
    
    Args:
        voltage: Voltage array [V]
        stc_params: STC parameters with keys: Rs, Rsh, I01, I02, Iph, n1, n2
        temperature_k: Temperature [K]
        irradiance_w_m2: Irradiance [W/m^2]
        ns: Number of series-connected cells (default: 36)
        coefficients: Model coefficients (alpha_I, etc.)
        
    Returns:
        Current array [A]
        
    Example:
        >>> V = np.linspace(0, 21.6, 50)
        >>> params = {'Rs': 0.28, 'Rsh': 3200.0, 'I01': 6.5e-8, 
        ...           'I02': 1.2e-7, 'Iph': 4.68, 'n1': 1.3, 'n2': 1.8}
        >>> I = simulate_iv_curve_double(V, stc_params=params, 
        ...                              temperature_k=298.15, irradiance_w_m2=1000.0)
        >>> print(f"Isc = {I[0]:.3f} A")
    """
    if coefficients is None:
        coefficients = default_coeffs

    # Extract STC parameters
    Rs_stc = float(stc_params["Rs"])
    Rsh_stc = float(stc_params["Rsh"])
    I01_stc = float(stc_params["I01"])
    I02_stc = float(stc_params["I02"])
    Iph_stc = float(stc_params["Iph"])
    n1 = float(stc_params["n1"])
    n2 = float(stc_params["n2"])

    T = float(temperature_k)
    G = float(irradiance_w_m2)

    # Apply translation equations (4-10) from paper
    Iph = float(iph_model(G, T, Iph_stc, alpha_I=coefficients.alpha_I))
    I01 = float(i01_model(T, I01_stc, n1))
    I02 = float(i02_model(T, I02_stc, n2))
    Rs = float(rs_model(
        G, T, Rs_stc,
        alpha_Rs=coefficients.alpha_Rs,
        beta_Rs=coefficients.beta_Rs,
        Rs_min=coefficients.Rs_min,
        Rs_max=coefficients.Rs_max
    ))
    Rsh = float(rsh_model_double(G, Rsh_stc))

    params_operating = {
        "Rs": Rs, "Rsh": Rsh, "I01": I01, "I02": I02,
        "Iph": Iph, "n1": n1, "n2": n2
    }

    return iv_model_double(
        np.asarray(voltage, dtype=float),
        params=params_operating,
        T=T,
        Ns=int(ns)
    )


def evaluate_double_parameters(
    stc_params: dict[str, float],
    datasets: list[dict[str, Any]],
    ns: int = 36,
    coefficients: Optional[PVModelCoefficients] = None
) -> float:
    """
    Evaluate existing STC parameters on a set of datasets.
    
    This is the key function for comparing θ₁ vs θ₂ in progressive optimization.
    
    Args:
        stc_params: STC parameters to evaluate (Rs, Rsh, I01, I02, Iph, n1, n2)
        datasets: List of I-V datasets (real measurements + optional virtual STC)
        ns: Number of cells in series (default: 36)
        coefficients: Model coefficients
        
    Returns:
        Global RMSE value (lower is better)
        
    Example:
        >>> rmse_theta1 = evaluate_double_parameters(theta1, all_scans, ns=36)
        >>> rmse_theta2 = evaluate_double_parameters(theta2, all_scans, ns=36)
        >>> if rmse_theta2 < rmse_theta1:
        ...     print("θ₂ is better!")
    """
    if coefficients is None:
        coefficients = default_coeffs

    # Validate input parameters
    required_keys = ['Rs', 'Rsh', 'I01', 'I02', 'Iph', 'n1', 'n2']
    missing_keys = [k for k in required_keys if k not in stc_params]
    if missing_keys:
        raise KeyError(f"Missing required parameters: {missing_keys}. Got: {list(stc_params.keys())}")

    # Encode the parameters into a normalized genome
    try:
        individual = encode_individual_double(stc_params)
    except Exception as e:
        raise ValueError(f"Failed to encode parameters: {e}. Parameters: {stc_params}")

    # Call the objective function with the individual
    try:
        rmse = pv_rmse_objective_double(individual, datasets, ns, coefficients)
        return rmse
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate RMSE: {e}")


def optimize_double_multicondition(
    datasets: list[dict[str, Any]],
    *,
    ns: int = 36,
    coefficients: Optional[PVModelCoefficients] = None,
    pop_size: int = 80,
    generations: int = 150,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.12,
    elitism: bool = True,
    tournament_size: int = 3,
    diversity_prob: float = 0.1,
    random_init: bool = True,
    init_params: dict[str, float] | None = None,
    noise: float = 0.05,
    seed: int | None = None,
    verbose: bool = False,
    live_plot: bool = False,
    figsize: tuple = (10, 4),
    auto_close_plot: bool = True,
    plot_display_seconds: float = 3,
    on_progress: Callable[[int, int, float, dict[str, float]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> OptimizationResult:
    """
    Optimize double-diode STC parameters using multi-condition GA.
    
    This implements the embedded multi-condition strategy from NCMAI'26 paper.
    The objective function (Equation 14) minimizes RMSE across ALL datasets simultaneously.
    
    Args:
        datasets: List of I-V datasets (real measurements + optional virtual STC)
        ns: Number of series-connected cells (default: 36)
        coefficients: Model coefficients (alpha_I, alpha_Rs, etc.)
        pop_size: GA population size (default: 80)
        generations: Number of generations (default: 150)
        crossover_rate: Crossover probability (default: 0.85)
        mutation_rate: Mutation probability (default: 0.12)
        elitism: Keep best individual (default: True)
        tournament_size: Tournament selection size (default: 3)
        diversity_prob: Probability to select non-best for diversity (default: 0.1)
        random_init: Random initialization or use init_params (default: True)
        init_params: Initial parameter guess
        noise: Noise for initialization diversity (default: 0.05)
        seed: Random seed for reproducibility
        verbose: Print progress (default: False)
        live_plot: Show live convergence plot (default: False)
        figsize: Figure size for live plot (default: (10,4))
        auto_close_plot: Auto-close plot after GA finishes (default: True)
        plot_display_seconds: Seconds to display plot before closing (default: 3)
        on_progress: Callback for progress updates
        should_cancel: Cancellation callback
        
    Returns:
        OptimizationResult with best_params and best_fitness
        
    Example:
        >>> # Load real datasets
        >>> datasets = load_datasets_from_dir("data/")
        >>> 
        >>> # Run multi-condition optimization
        >>> result = optimize_double_multicondition(
        ...     datasets=datasets,
        ...     ns=36,
        ...     pop_size=60,
        ...     generations=100,
        ...     verbose=True
        ... )
        >>> 
        >>> print("Best parameters:", result.best_params)
        >>> print("RMSE:", result.best_fitness)
    """
    if coefficients is None:
        coefficients = default_coeffs

    def objective_fn(individual: np.ndarray) -> float:
        return float(pv_rmse_objective_double(
            individual, datasets, Ns=int(ns),
            coefficients=coefficients
        ))

    best_params, best_fitness, history = run_ga(
        objective_function=objective_fn,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism=elitism,
        random_init=random_init,
        init_params=init_params,
        noise=noise,
        seed=seed,
        tournament_size=tournament_size,
        diversity_prob=diversity_prob,
        verbose=verbose,
        live_plot=live_plot,
        figsize=figsize,
        auto_close_plot=auto_close_plot,
        plot_display_seconds=plot_display_seconds,
        on_progress=on_progress,
        should_cancel=should_cancel,
        genome_length=7,  # Double-diode has 7 parameters
    )

    meta = {
        "model": "double_diode",
        "strategy": "multi_condition",
        "ns": int(ns),
        "coefficients": asdict(coefficients),
        "pop_size": int(pop_size),
        "generations": int(generations),
        "crossover_rate": float(crossover_rate),
        "mutation_rate": float(mutation_rate),
        "elitism": bool(elitism),
        "tournament_size": int(tournament_size),
        "diversity_prob": float(diversity_prob),
        "random_init": bool(random_init),
        "noise": float(noise),
        "live_plot": live_plot,
        "auto_close_plot": auto_close_plot,
        "plot_display_seconds": plot_display_seconds,
    }

    return OptimizationResult(
        best_params=best_params,
        best_fitness=float(best_fitness),
        history=history,
        meta=meta,
    )


def create_virtual_stc_curve_double(
    reference_params: Optional[dict[str, float]] = None,
    ns: int = 36,
    points: int = 100
) -> dict[str, Any]:
    """
    Create a virtual I-V curve at STC conditions (1000 W/m², 25°C).
    
    This is essential for the progressive optimization algorithm to add
    the STC condition as a virtual constraint.
    
    Args:
        reference_params: Reference STC parameters (optional, uses defaults)
        ns: Number of series-connected cells (default: 36)
        points: Number of voltage points (default: 100)
        
    Returns:
        Dataset dict with V, I, T=298.15, G=1000.0
        
    Example:
        >>> stc_curve = create_virtual_stc_curve_double()
        >>> print(f"STC curve has {len(stc_curve['V'])} points")
        >>> print(f"Isc = {stc_curve['I'][0]:.3f} A")
    """
    if reference_params is None:
        reference_params = {
            "Rs": 0.28, "Rsh": 3200.0,
            "I01": 6.5e-8, "I02": 1.2e-7,
            "Iph": 4.68, "n1": 1.3, "n2": 1.8,
        }

    # Estimate Voc for voltage range
    k, q = 1.380649e-23, 1.602176634e-19
    Vt = (k / q) * 298.15
    a1 = reference_params["n1"] * ns * Vt
    Voc_est = a1 * np.log(reference_params["Iph"] / max(reference_params["I01"], 1e-12))
    Voc_est = min(Voc_est, 50.0)

    V = np.linspace(0, Voc_est * 0.98, points)
    I = simulate_iv_curve_double(
        V,
        stc_params=reference_params,
        temperature_k=298.15,
        irradiance_w_m2=1000.0,
        ns=ns
    )

    return {'V': V, 'I': I, 'T': 298.15, 'G': 1000.0, 'model': 'STC_virtual'}


def optimize_double_progressive(
    scan_stream: Iterator[dict[str, Any]],
    *,
    ns: int = 36,
    coefficients: Optional[PVModelCoefficients] = None,
    include_virtual_stc: bool = True,
    ga_kwargs: dict[str, Any] = None,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Progressive optimization accumulating scans over time.
    
    This implements the progressive strategy from the flowchart:
    - Start with first scan + virtual STC → θ₁
    - For each new scan:
      a. Add scan to accumulated dataset
      b. Optimize θ_new on ALL accumulated scans + virtual STC
      c. Evaluate previous best θ_prev on current dataset
      d. Keep the one with lower RMSE
    - Return best θ after all scans
    
    Args:
        scan_stream: Iterator yielding scan dicts with keys: V, I, T, G
        ns: Number of series-connected cells (default: 36)
        coefficients: Model coefficients
        include_virtual_stc: Whether to include virtual STC curve (default: True)
        ga_kwargs: Additional arguments for optimize_double_multicondition
        verbose: Print progress (default: True)
        
    Returns:
        OptimizationResult with best parameters found
        
    Example:
        >>> # Simulate hardware scans
        >>> def scan_generator():
        ...     for hour in range(8, 17):
        ...         V, I = hardware.scan()
        ...         yield {"V": V, "I": I, "T": T, "G": G}
        >>> 
        >>> # Run progressive optimization
        >>> result = optimize_double_progressive(
        ...     scan_stream=scan_generator(),
        ...     ns=36,
        ...     ga_kwargs={"pop_size": 50, "generations": 60}
        ... )
        >>> 
        >>> print("Final STC parameters:", result.best_params)
    """
    if ga_kwargs is None:
        ga_kwargs = {}
    if coefficients is None:
        coefficients = default_coeffs

    # Parameters that belong to run_ga only
    run_ga_only_params = {'figsize', 'auto_close_plot', 'plot_display_seconds', 
                          'on_progress', 'should_cancel'}
    
    # Remove run_ga_only_params from optimization kwargs
    optimization_kwargs = {k: v for k, v in ga_kwargs.items() 
                           if k not in run_ga_only_params}
    
    all_scans = []
    best_theta = None
    best_rmse = float('inf')
    history = []
    scan_count = 0

    print("=" * 70)
    print("PROGRESSIVE OPTIMIZATION - Accumulating scans")
    print("=" * 70)

    for scan in scan_stream:
        scan_count += 1
        all_scans.append(scan)

        # Build dataset: real scans + virtual STC
        datasets = all_scans.copy()
        if include_virtual_stc:
            virtual_stc = create_virtual_stc_curve_double(ns=ns)
            datasets.append(virtual_stc)

        if verbose:
            print(f"\n[Scan {scan_count}] {len(all_scans)} real scans + "
                  f"{1 if include_virtual_stc else 0} virtual = {len(datasets)} total")

        # Optimize new θ on all accumulated data
        result_new = optimize_double_multicondition(
            datasets=datasets,
            ns=ns,
            coefficients=coefficients,
            **optimization_kwargs
        )
        theta_new = result_new.best_params
        rmse_new = result_new.best_fitness

        if verbose:
            print(f"-> New θ RMSE: {rmse_new:.6f}")

        # Compare with previous best if exists
        if best_theta is not None:
            try:
                rmse_prev = evaluate_double_parameters(
                    best_theta, datasets, ns, coefficients
                )
                if verbose:
                    print(f"-> Previous best RMSE: {rmse_prev:.6f}")

                if rmse_prev < rmse_new:
                    if verbose:
                        print(f"-> Previous θ remains best")
                else:
                    best_theta = theta_new.copy()
                    best_rmse = rmse_new
                    if verbose:
                        print(f"-> NEW BEST θ!")
            except Exception as e:
                if verbose:
                    print(f"-> Evaluation failed ({e}), using new θ")
                best_theta = theta_new.copy()
                best_rmse = rmse_new
        else:
            best_theta = theta_new.copy()
            best_rmse = rmse_new
            if verbose:
                print(f"-> Initial best θ (RMSE: {best_rmse:.6f})")

        history.append({
            'scan_id': scan_count,
            'n_scans': len(all_scans),
            'theta_new': theta_new,
            'rmse_new': rmse_new,
            'theta_best': best_theta.copy() if best_theta else None,
            'rmse_best': best_rmse,
            'improved': best_theta == theta_new if best_theta else True
        })

    print("\n" + "=" * 70)
    print("PROGRESSIVE OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Total scans processed: {scan_count}")
    print(f"Final RMSE: {best_rmse:.6f}")

    return OptimizationResult(
        best_params=best_theta,
        best_fitness=best_rmse,
        history=history,
        meta={
            "model": "double_diode",
            "strategy": "progressive",
            "ns": ns,
            "scans_processed": scan_count,
            "coefficients": asdict(coefficients),
            **ga_kwargs
        }
    )

# =============================================================================
# Public API Exports
# =============================================================================

__all__ = [
    # Data classes
    "OptimizationResult",
    "ModelConfig",
    # Dataset loading
    "load_datasets_from_dir",
    # Core functions
    "simulate_iv_curve_double",
    "evaluate_double_parameters",
    "optimize_double_multicondition",
    "optimize_double_progressive",
    "create_virtual_stc_curve_double",
    # Coefficients
    "PVModelCoefficients",
    "default_coeffs",
]