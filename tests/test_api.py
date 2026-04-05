# tests/test_api.py
"""Test public API."""

import numpy as np
import pytest

from pvoptix import (
    __version__,
    load_datasets_from_dir,
    optimize_double_multicondition,
    optimize_double_progressive,
    simulate_iv_curve_double,
    evaluate_double_parameters,
    create_virtual_stc_curve_double,
    OptimizationResult,
    ModelConfig,
    compute_power,
    find_mpp,
    PVModelCoefficients,
    default_coeffs
)


class TestPublicAPI:
    """Test public API functions."""

    def test_version(self):
        """Check package version string."""
        assert __version__ == "1.0.0"

    def test_optimization_result_import(self):
        """Check OptimizationResult dataclass behavior."""
        result = OptimizationResult(
            best_params={"Rs": 0.28},
            best_fitness=0.01,
            history=[],
            meta={}
        )
        assert result.best_params["Rs"] == 0.28
        assert result.best_fitness == 0.01

    def test_model_config(self):
        """Check ModelConfig dataclass behavior."""
        config = ModelConfig(ns=72)
        assert config.ns == 72
        assert config.coefficients is not None

    def test_pv_model_coefficients(self):
        """Check PVModelCoefficients dataclass behavior."""
        coeffs = PVModelCoefficients(alpha_I=0.0006)
        assert coeffs.alpha_I == 0.0006

    def test_default_coeffs(self):
        """Check default coefficients values."""
        assert default_coeffs.alpha_I == 0.0005
        assert default_coeffs.beta_Rsh == -0.002

    def test_create_virtual_stc_curve(self):
        """Check creation of virtual STC curve."""
        curve = create_virtual_stc_curve_double(ns=36, points=50)

        assert "V" in curve
        assert "I" in curve
        assert curve["T"] == 298.15
        assert curve["G"] == 1000.0
        assert len(curve["V"]) == 50

    def test_simulate_iv_curve(self):
        """Check I-V curve simulation results."""
        V = np.linspace(0, 21.6, 50)
        params = {
            "Rs": 0.28, "Rsh": 3200.0,
            "I01": 6.5e-8, "I02": 1.2e-7,
            "Iph": 4.68, "n1": 1.3, "n2": 1.8
        }

        I = simulate_iv_curve_double(
            V, stc_params=params,
            temperature_k=298.15, irradiance_w_m2=1000.0, ns=36
        )

        assert len(I) == 50
        assert I[0] > 4.0  # Isc close to 4.68
        assert I[-1] < 0.5  # Current near Voc should be small
        assert np.all(I >= 0)  # No negative currents
        assert np.all(I <= params["Iph"] + 0.1)  # No currents above Iph

    def test_compute_power(self):
        """Check power computation from voltage and current arrays."""
        V = np.array([0, 10, 20])
        I = np.array([4.68, 3.5, 0.5])
        P = compute_power(V, I)

        assert P[0] == 0
        assert P[1] == 35
        assert P[2] == 10

    def test_find_mpp(self):
        """Check maximum power point finding with a realistic I-V curve."""
        V = np.linspace(0, 21.6, 200)
        Iph = 4.68
        I0 = 6.5e-8
        n = 1.3
        Ns = 36
        kTq = 0.0257
        Vt = n * Ns * kTq

        I = Iph - I0 * (np.exp(V / Vt) - 1)
        I = np.clip(I, 0, Iph)

        V_mpp, I_mpp, P_mpp = find_mpp(V, I)

        assert 15 < V_mpp < 20, f"V_mpp = {V_mpp} is outside expected range [15, 20]"
        assert 0 < P_mpp < Iph * max(V), "Power should be positive and reasonable"
        assert I_mpp > 0, "Current at MPP should be positive"


if __name__ == "__main__":
    # Run tests manually
    test = TestPublicAPI()
    test.test_version()
    test.test_optimization_result_import()
    test.test_model_config()
    test.test_pv_model_coefficients()
    test.test_default_coeffs()
    test.test_create_virtual_stc_curve()
    test.test_simulate_iv_curve()
    test.test_compute_power()
    test.test_find_mpp()
    print("All API tests passed.")
