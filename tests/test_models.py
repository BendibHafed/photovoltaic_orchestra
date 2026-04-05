"""
Test double-diode model equations.
"""

import numpy as np
import pytest
from pvoptix.models.double_diode import double_diode_equation


class TestDoubleDiodeEquation:
    """Test the double-diode equation."""

    def test_equation_returns_finite_value(self, true_params):
        """Equation should return a finite number for given inputs."""
        I = 4.0
        V = 10.0
        T = 298.15

        result = double_diode_equation(
            I, V, T,
            true_params["Rs"], true_params["Rsh"],
            true_params["I01"], true_params["I02"],
            true_params["Iph"], true_params["n1"], true_params["n2"],
            Ns=36
        )

        assert np.isfinite(result), "Equation returned non-finite value"

    def test_short_circuit_condition(self, true_params):
        """At short circuit (V=0, I=Iph), the equation should be close to zero."""
        I = true_params["Iph"]
        V = 0
        T = 298.15

        result = double_diode_equation(
            I, V, T,
            true_params["Rs"], true_params["Rsh"],
            true_params["I01"], true_params["I02"],
            true_params["Iph"], true_params["n1"], true_params["n2"],
            Ns=36
        )

        assert abs(result) < 0.01, f"Short circuit error: {result}"

    def test_open_circuit_condition(self, true_params):
        """At open circuit (I=0), the equation should be close to zero at Voc."""
        from pvoptix.solvers.double import solve_current_double

        T = 298.15
        Ns = 36

        # Find Voc by scanning voltages
        V_range = np.linspace(0, 22, 100)
        for V in V_range:
            I = solve_current_double(V, T, true_params, Ns)
            if I < 0.01:
                Voc = V
                break
        else:
            Voc = V_range[-1]

        result = double_diode_equation(
            0, Voc, T,
            true_params["Rs"], true_params["Rsh"],
            true_params["I01"], true_params["I02"],
            true_params["Iph"], true_params["n1"], true_params["n2"],
            Ns=36
        )

        assert abs(result) < 0.5, f"Open circuit error at Voc={Voc:.2f}V: {result}"
        assert 20 < Voc < 22, f"Voc={Voc:.2f}V is outside expected range [20,22]"


if __name__ == "__main__":
    # Run tests manually
    true_params = {
        "Rs": 0.28,
        "Rsh": 3200.0,
        "I01": 6.5e-8,
        "I02": 1.2e-7,
        "Iph": 4.68,
        "n1": 1.3,
        "n2": 1.8,
    }

    test = TestDoubleDiodeEquation()
    test.test_equation_returns_finite_value(true_params)
    test.test_short_circuit_condition(true_params)
    test.test_open_circuit_condition(true_params)
    print("All double-diode model tests passed.")
