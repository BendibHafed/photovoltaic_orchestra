#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public API for PvOptiX

This module provides the main entry points for:
1. Double-diode model with multi-condition optimization
2. Progressive optimization (accumulate and compare scans)
3. Legacy single-diode model for backward compatibility
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, asdict
from typing import Any, Optional
import numpy as np

from pvoptix.pvoptix.datasets.builder import build_dataset
from pvoptix.pvoptix.models.parameters import (
    PVModelCoefficients, default_coeffs,
    iph_model, i01_model, i02_model, rs_model, rsh_model_double
)
from pvoptix.pvoptix.objectives.double import pv_rmse_objective_double
from pvoptix.pvoptix.objectives.single import pv_rmse_objective
from pvoptix.pvoptix.optimization.ga.core import run_ga
from pvoptix.pvoptix.optimization.ga.genome_mapping_double import (
    decode_individual_double, encode_individual_double
)
from pvoptix.pvoptix.optimization.ga.genome_mapping import decode_individual
from pvoptix.pvoptix.solvers.double import iv_model_double
from pvoptix.pvoptix.solvers.single import iv_model_single


@dataclass(frozen=True)
class OptimizationResult:
    """High-level optimization output returned by the public API."""
    best_params: dict[str, float]
    best_fitness: float
    history: list[tuple[float, dict[str, float]]]
    meta: dict[str, Any]


@dataclass
class ModelConfig:
    """Complete configuration of the PV model."""
    ns: int = 36
    coefficients: PVModelCoefficients = None

    def __post_init__(self):
        if self.coefficients is None:
            self.coefficients = default_coeffs


# =============================================================================
# DATASET LOADING (Shared)
# =============================================================================

def load_datasets_from_dir(data_dir: str) -> list[dict[str, Any]]:
    """
    Load experimental I-V datasets from a directory containing .dat files.

    Args:
        data_dir: Directory containing files named <model>_<T_C>_<G>.dat

    Returns:
        List of dataset dicts with keys: model, T, G, V, I
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
        stc_params: dict with keys: Rs, Rsh, I01, I02, Iph, n1, n2
        temperature_k: Temperature [K]
        irradiance_w_m2: Irradiance [W/m^2]
        ns: Number of series-connected cells
        coefficients: Model coefficients (alpha_I, etc.)

    Returns:
        Current array [A]
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

    This is the key function for the progressive optimization algorithm.

    Args:
        stc_params: STC parameters to evaluate (Rs, Rsh, I01, I02, Iph, n1, n2)
        datasets: List of I-V datasets
        ns: Number of cells in series
        coefficients: Model coefficients

    Returns:
        Global RMSE value (lower is better)
    """
    if coefficients is None:
        coefficients = default_coeffs

    individual = encode_individual_double(stc_params)
    return pv_rmse_objective_double(individual, datasets, ns, coefficients)


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
    on_progress: Callable[[int, int, float, dict[str, float]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> OptimizationResult:
    """
    Optimize double-diode STC parameters using multi-condition GA.

    This implements the embedded multi-condition strategy from NCMAI'26 paper.
    The objective function (Equation 13) minimizes RMSE across ALL datasets
    simultaneously.

    Args:
        datasets: List of I-V datasets (real measurements + optional virtual STC)
        ns: Number of series-connected cells
        coefficients: Model coefficients
        pop_size: GA population size
        generations: Number of generations
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability
        elitism: Keep best individual
        tournament_size: Tournament selection size
        diversity_prob: Probability to select non-best for diversity
        random_init: Random initialization or use init_params
        init_params: Initial parameter guess
        noise: Noise for initialization diversity
        seed: Random seed for reproducibility
        verbose: Print progress
        live_plot: Show live convergence plot
        on_progress: Callback for progress updates
        should_cancel: Cancellation callback

    Returns:
        OptimizationResult with best_params and best_fitness
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
        on_progress=on_progress,
        should_cancel=should_cancel
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
        ns: Number of series-connected cells
        points: Number of voltage points

    Returns:
        Dataset dict with V, I, T=298.15, G=1000.0
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
    Progressive optimization accumulating scans over time (Friend's idea).

    Algorithm:
    1. Start with first scan + virtual STC → θ₁
    2. For each new scan:
       a. Add scan to accumulated dataset
       b. Optimize θ_new on ALL accumulated scans + virtual STC
       c. Evaluate previous best θ_prev on current dataset
       d. Keep the one with lower RMSE
    3. Return best θ after all scans

    This ensures the final parameters work for ALL conditions encountered.

    Args:
        scan_stream: Iterator yielding scan dicts with keys: V, I, T, G
        ns: Number of series-connected cells
        coefficients: Model coefficients
        include_virtual_stc: Whether to include virtual STC curve
        ga_kwargs: Additional arguments for optimize_double_multicondition
        verbose: Print progress

    Returns:
        OptimizationResult with best parameters found
    """
    if ga_kwargs is None:
        ga_kwargs = {}
    if coefficients is None:
        coefficients = default_coeffs

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
            datasets.append(create_virtual_stc_curve_double(ns=ns))

        if verbose:
            print(f"\n[Scan {scan_count}] {len(all_scans)} real scans + "
                  f"{1 if include_virtual_stc else 0} virtual = {len(datasets)} total")

        # Optimize new θ on all accumulated data
        result_new = optimize_double_multicondition(
            datasets=datasets,
            ns=ns,
            coefficients=coefficients,
            **ga_kwargs
        )
        theta_new = result_new.best_params
        rmse_new = result_new.best_fitness

        if verbose:
            print(f"  → New θ RMSE: {rmse_new:.6f}")

        # Compare with previous best if exists
        if best_theta is not None:
            rmse_prev = evaluate_double_parameters(
                best_theta, datasets, ns, coefficients
            )
            if verbose:
                print(f"  → Previous best RMSE: {rmse_prev:.6f}")

            if rmse_prev < rmse_new:
                if verbose:
                    print(f"  → Previous θ remains best")
            else:
                best_theta = theta_new
                best_rmse = rmse_new
                if verbose:
                    print(f"  → NEW BEST θ!")
        else:
            best_theta = theta_new
            best_rmse = rmse_new
            if verbose:
                print(f"  → Initial best θ")

        history.append({
            'scan_id': scan_count,
            'n_scans': len(all_scans),
            'theta_new': theta_new,
            'rmse_new': rmse_new,
            'theta_best': best_theta.copy(),
            'rmse_best': best_rmse,
            'improved': best_theta == theta_new
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
# LEGACY API (Single-diode - kept for backward compatibility)
# =============================================================================

def simulate_iv_curve_stc(
    voltage: np.ndarray,
    *,
    stc_params: dict[str, float],
    temperature_k: float,
    irradiance_w_m2: float,
    ns: int = 36,
    coefficients: Optional[PVModelCoefficients] = None
) -> np.ndarray:
    """Legacy: Single-diode I-V simulation."""
    if coefficients is None:
        coefficients = default_coeffs

    Rs_stc = float(stc_params["Rs"])
    Rsh_stc = float(stc_params["Rsh"])
    I0_stc = float(stc_params["I0"])
    Iph_stc = float(stc_params["Iph"])
    n = float(stc_params["n"])

    T = float(temperature_k)
    G = float(irradiance_w_m2)

    Iph = float(iph_model(G, T, Iph_stc, alpha_I=coefficients.alpha_I))
    I0 = float(i0_model(T, I0_stc, n))
    Rs = float(rs_model(
        G, T, Rs_stc,
        alpha_Rs=coefficients.alpha_Rs,
        beta_Rs=coefficients.beta_Rs,
        Rs_min=coefficients.Rs_min,
        Rs_max=coefficients.Rs_max
    ))
    Rsh = float(rsh_model_double(G, Rsh_stc))

    params_operating = {"Rs": Rs, "Rsh": Rsh, "I0": I0, "Iph": Iph, "n": n}
    return iv_model_single(
        np.asarray(voltage, dtype=float),
        params=params_operating,
        T=T,
        Ns=int(ns)
    )


def optimize_stc_parameters_from_datasets(
    datasets: list[dict[str, Any]],
    *,
    ns: int = 36,
    coefficients: Optional[PVModelCoefficients] = None,
    pop_size: int = 40,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elitism: bool = True,
    tournament_size: int = 3,
    diversity_prob: float = 0.1,
    random_init: bool = True,
    init_params: dict[str, float] | None = None,
    noise: float = 0.0,
    seed: int | None = None,
    verbose: bool = False,
    live_plot: bool = False,
    on_progress: Callable[[int, int, float, dict[str, float]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> OptimizationResult:
    """Legacy: Single-diode optimization."""
    if coefficients is None:
        coefficients = default_coeffs

    def objective_fn(individual: np.ndarray) -> float:
        return float(pv_rmse_objective(
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
        on_progress=on_progress,
        should_cancel=should_cancel
    )

    meta = {
        "model": "single_diode",
        "ns": int(ns),
        "pop_size": int(pop_size),
        "generations": int(generations),
        "crossover_rate": float(crossover_rate),
        "mutation_rate": float(mutation_rate),
        "elitism": bool(elitism),
        "tournament_size": int(tournament_size),
        "diversity_prob": float(diversity_prob),
        "random_init": bool(random_init),
        "noise": float(noise),
    }

    return OptimizationResult(
        best_params=best_params,
        best_fitness=float(best_fitness),
        history=history,
        meta=meta,
    )


def optimize_stc_parameters_from_iv_curve(
    voltage: np.ndarray,
    current: np.ndarray,
    *,
    temperature_k: float = 298.15,
    irradiance_w_m2: float = 1000.0,
    ns: int = 36,
    coefficients: Optional[PVModelCoefficients] = None,
    **ga_kwargs: Any,
) -> OptimizationResult:
    """Legacy: Single-curve optimization."""
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)

    datasets = [{
        "V": voltage, "I": current,
        "T": float(temperature_k), "G": float(irradiance_w_m2),
        "model": "unknown",
    }]

    return optimize_stc_parameters_from_datasets(
        datasets, ns=int(ns), coefficients=coefficients, **ga_kwargs
    )