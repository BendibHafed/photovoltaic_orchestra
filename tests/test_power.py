# tests/test_power.py
"""Test power analysis functions."""

import numpy as np
import pytest
from pvoptix.analysis.power import (
    compute_power, find_mpp, simulate_iv_curve, analyze_power_across_conditions
)
from pvoptix.api import simulate_iv_curve_double


class TestPowerAnalysis:
    """Test power analysis functions."""

    def test_compute_power(self):
        """Test power computation."""
        V = np.array([0, 10, 20])
        I = np.array([4.68, 3.5, 0.5])
        P = compute_power(V, I)

        assert P[0] == 0
        assert P[1] == 35
        assert P[2] == 10

    def test_find_mpp_on_simulated_curve(self, true_params):
        """Test MPP finding on simulated curve."""
        V, I, P = simulate_iv_curve(true_params, T=298.15, Ns=36)

        V_mpp, I_mpp, P_mpp = find_mpp(V, I)

        assert 15 < V_mpp < 20, f"V_mpp={V_mpp:.2f}V out of range [15,20]"
        assert I_mpp > 0
        assert P_mpp > 0

    def test_simulate_iv_curve_output_shape(self, true_params):
        """Test simulate_iv_curve returns correct shapes."""
        V, I, P = simulate_iv_curve(true_params, T=298.15, Ns=36, n_points=100)

        assert len(V) == 100
        assert len(I) == 100
        assert len(P) == 100

    def test_power_curve_has_single_peak(self, true_params):
        """Test that power curve has a single peak (not multiple local maxima)."""
        V, I, P = simulate_iv_curve(true_params, T=298.15, Ns=36, n_points=200)

        peak_idx = np.argmax(P)
        pre_peak = P[:peak_idx]
        post_peak = P[peak_idx:]

        if len(pre_peak) > 5:
            pre_trend = np.polyfit(range(len(pre_peak)), pre_peak, 1)[0]
            assert pre_trend > -0.01, f"Power trend before peak is decreasing: {pre_trend}"

        if len(post_peak) > 5:
            post_trend = np.polyfit(range(len(post_peak)), post_peak, 1)[0]
            assert post_trend < 0.01, f"Power trend after peak is increasing: {post_trend}"

        assert peak_idx > 10, "Peak too close to start"
        assert peak_idx < len(P) - 10, "Peak too close to end"

    def test_analyze_power_across_conditions(self, true_params):
        """Test power analysis across multiple conditions."""
        datasets = []
        conditions = [(1000, 298.15), (800, 298.15), (600, 298.15)]
        V = np.linspace(0, 21.6, 50)

        for G, T in conditions:
            I = simulate_iv_curve_double(
                V, stc_params=true_params,
                temperature_k=T, irradiance_w_m2=G, ns=36
            )
            datasets.append({"V": V, "I": I, "T": T, "G": G})

        results = analyze_power_across_conditions(datasets, true_params, ns=36)

        assert len(results["conditions"]) == 3
        assert len(results["mpp_power_exp"]) == 3
        assert len(results["mpp_power_sim"]) == 3
        assert all(p > 0 for p in results["mpp_power_sim"])

    def test_mpp_accuracy(self, true_params):
        """Test that MPP is found accurately."""
        V, I, P = simulate_iv_curve(true_params, T=298.15, Ns=36, n_points=500)

        V_mpp, I_mpp, P_mpp = find_mpp(V, I, interpolate=True)

        fine_idx = np.argmax(P)
        V_brute = V[fine_idx]
        P_brute = P[fine_idx]

        assert abs(V_mpp - V_brute) < 0.5, f"V_mpp error: {abs(V_mpp - V_brute):.3f}V"
        assert abs(P_mpp - P_brute) < 1.0, f"P_mpp error: {abs(P_mpp - P_brute):.3f}W"


if __name__ == "__main__":
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }

    test = TestPowerAnalysis()
    test.test_compute_power()
    test.test_find_mpp_on_simulated_curve(true_params)
    test.test_simulate_iv_curve_output_shape(true_params)
    test.test_power_curve_has_single_peak(true_params)
    test.test_analyze_power_across_conditions(true_params)
    test.test_mpp_accuracy(true_params)
    print("All power analysis tests passed.")
