# tests/test_solvers.py
"""Test numerical solvers for double-diode model."""

import numpy as np
import pytest
from pvoptix.solvers.double import solve_current_double, iv_model_double


class TestDoubleDiodeSolvers:
    """Test the numerical solvers."""

    def test_solve_current_returns_finite(self, true_params):
        """Current solver should return finite value within physical bounds."""
        V = 17.6
        T = 298.15
        Ns = 36

        I = solve_current_double(V, T, true_params, Ns)

        assert np.isfinite(I)
        assert 0 <= I <= true_params["Iph"]

    def test_solve_current_short_circuit(self, true_params):
        """Short circuit current (V=0) should be close to Iph."""
        V = 0
        T = 298.15
        Ns = 36

        Isc = solve_current_double(V, T, true_params, Ns)

        assert abs(Isc - true_params["Iph"]) < 0.01

    def test_solve_current_open_circuit(self, true_params):
        """Open circuit voltage (I≈0) should be reasonable."""
        V_range = np.linspace(0, 22, 100)
        T = 298.15
        Ns = 36

        for V in V_range:
            I = solve_current_double(V, T, true_params, Ns)
            if I < 0.01:
                Voc = V
                break
        else:
            Voc = V_range[-1]

        assert 20 < Voc < 23, f"Voc out of range: {Voc}V"

    def test_iv_model_monotonic(self, true_params, sample_voltage):
        """I-V curve should be monotonic decreasing."""
        T = 298.15
        Ns = 36

        I = iv_model_double(sample_voltage, true_params, T, Ns)

        assert np.all(np.diff(I) <= 0), "I-V curve not monotonic decreasing"

    def test_iv_model_bounds(self, true_params, sample_voltage):
        """I-V curve should stay within physical bounds."""
        T = 298.15
        Ns = 36

        I = iv_model_double(sample_voltage, true_params, T, Ns)

        assert np.all(I >= 0), "Negative currents found"
        assert np.all(I <= true_params["Iph"] + 0.1), "Currents above Iph"

    def test_voc_estimation(self, true_params):
        """Voc estimation should be reasonable."""
        T = 298.15
        Ns = 36

        V_range = np.linspace(0, 22, 100)
        I_range = iv_model_double(V_range, true_params, T, Ns)

        voc_idx = np.where(I_range < 0.01)[0]
        if len(voc_idx) > 0:
            Voc = V_range[voc_idx[0]]
        else:
            Voc = V_range[-1]

        assert 20 < Voc < 22, f"Voc={Voc:.2f}V out of range"


if __name__ == "__main__":
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }
    sample_voltage = np.linspace(0, 21.6, 50)

    test = TestDoubleDiodeSolvers()
    test.test_solve_current_returns_finite(true_params)
    test.test_solve_current_short_circuit(true_params)
    test.test_solve_current_open_circuit(true_params)
    test.test_iv_model_monotonic(true_params, sample_voltage)
    test.test_iv_model_bounds(true_params, sample_voltage)
    test.test_voc_estimation(true_params)
    print("All solver tests passed.")
