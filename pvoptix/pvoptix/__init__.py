"""
PvOptiX - Inner package with core implementation
"""

# Import from api
from pvoptix.pvoptix.api import (
    # Double-diode
    optimize_double_multicondition,
    optimize_double_progressive,
    evaluate_double_parameters,
    create_virtual_stc_curve_double,
    simulate_iv_curve_double,
    
    # Legacy
    load_datasets_from_dir,
    optimize_stc_parameters_from_datasets,
    optimize_stc_parameters_from_iv_curve,
    simulate_iv_curve_stc,
    OptimizationResult,
    ModelConfig,
)

# Analysis functions
from pvoptix.pvoptix.analysis.power import (
    compute_power,
    find_mpp,
    simulate_iv_curve,
    plot_power_curve,
    compute_global_power,
)

# Coefficients
from pvoptix.pvoptix.models.parameters import (
    PVModelCoefficients,
    default_coeffs,
)

__all__ = [
    # Double-diode
    "optimize_double_multicondition",
    "optimize_double_progressive",
    "evaluate_double_parameters",
    "create_virtual_stc_curve_double",
    "simulate_iv_curve_double",
    # Legacy
    "load_datasets_from_dir",
    "optimize_stc_parameters_from_datasets",
    "optimize_stc_parameters_from_iv_curve",
    "simulate_iv_curve_stc",
    "OptimizationResult",
    "ModelConfig",
    # Analysis
    "compute_power",
    "find_mpp",
    "simulate_iv_curve",
    "plot_power_curve",
    "compute_global_power",
    # Coefficients
    "PVModelCoefficients",
    "default_coeffs",
]