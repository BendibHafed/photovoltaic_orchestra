
"""
PvOptiX - Photovoltaic Parameter Optimization Engine

Scientific library for PV module parameter extraction using double-diode model
with progressive multi-condition optimization.

Author: BENDIB HE.
Version: 2.0.0
"""

__version__ = "1.0.0"
__author__ = "BENDIB HE."
__license__ = "MIT"

# Re-export from inner package
from pvoptix.pvoptix import (
    # Core functions
    optimize_double_multicondition,
    optimize_double_progressive,
    evaluate_double_parameters,
    create_virtual_stc_curve_double,
    simulate_iv_curve_double,
    
    # Legacy (single-diode)
    load_datasets_from_dir,
    optimize_stc_parameters_from_datasets,
    optimize_stc_parameters_from_iv_curve,
    simulate_iv_curve_stc,
    OptimizationResult,
    ModelConfig,
    
    # Analysis
    compute_power,
    find_mpp,
    simulate_iv_curve,
    plot_power_curve,
    compute_global_power,
    
    # Coefficients
    PVModelCoefficients,
    default_coeffs,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
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