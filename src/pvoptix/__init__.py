"""
PvOptiX - Inner package with core implementation
"""

from importlib.metadata import version

__version__ = version("pvoptix")

# Re-export key functions from submodules
from .api import (
    optimize_double_progressive,
    optimize_double_multicondition,
    evaluate_double_parameters,
    create_virtual_stc_curve_double,
    simulate_iv_curve_double,
    load_datasets_from_dir,
    OptimizationResult,
    ModelConfig,
)

from .models.parameters import (
    PVModelCoefficients,
    default_coeffs,
)

from .analysis.power import (
    compute_power,
    find_mpp,
    plot_power_curve,
)

__all__ = [
    # API
    "optimize_double_progressive",
    "optimize_double_multicondition",
    "evaluate_double_parameters",
    "create_virtual_stc_curve_double",
    "simulate_iv_curve_double",
    "load_datasets_from_dir",
    "OptimizationResult",
    "ModelConfig",
    # Models
    "PVModelCoefficients",
    "default_coeffs",
    # Analysis
    "compute_power",
    "find_mpp",
    "plot_power_curve",
]