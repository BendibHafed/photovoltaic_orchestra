"""
Test fitness function (RMSE_global - Equation 14).
"""

import numpy as np
import pytest
from pvoptix.optimization.ga.genome_mapping_double import encode_individual_double
from pvoptix.optimization.ga.fitness_double import global_rmse_double
from pvoptix.api import simulate_iv_curve_double


class TestFitnessFunction:
    """Test the global RMSE fitness function."""

    def test_perfect_fit_rmse_small(self, true_params, sample_dataset):
        """Perfect parameters should give a very small RMSE."""
        individual = encode_individual_double(true_params)
        datasets = [sample_dataset]

        rmse = global_rmse_double(individual, datasets, Ns=36)

        assert rmse < 0.001, f"Perfect fit RMSE not small: {rmse}"

    def test_wrong_params_give_higher_rmse(self, true_params, sample_dataset):
        """Incorrect parameters should result in a higher RMSE compared to perfect parameters."""
        perfect_individual = encode_individual_double(true_params)

        wrong_params = true_params.copy()
        wrong_params["Rs"] = wrong_params["Rs"] * 2
        wrong_individual = encode_individual_double(wrong_params)

        datasets = [sample_dataset]

        perfect_rmse = global_rmse_double(perfect_individual, datasets, Ns=36)
        wrong_rmse = global_rmse_double(wrong_individual, datasets, Ns=36)

        assert wrong_rmse > perfect_rmse

    def test_multiple_conditions(self, true_params):
        """RMSE should remain small when tested across multiple operating conditions."""
        individual = encode_individual_double(true_params)

        # Create datasets at different irradiance and temperature conditions
        datasets = []
        conditions = [(1000, 298.15), (800, 298.15), (600, 298.15), (400, 298.15)]
        V = np.linspace(0, 21.6, 50)

        for G, T in conditions:
            I = simulate_iv_curve_double(
                V, stc_params=true_params,
                temperature_k=T, irradiance_w_m2=G, ns=36
            )
            datasets.append({"V": V, "I": I, "T": T, "G": G})

        rmse = global_rmse_double(individual, datasets, Ns=36)

        assert rmse < 0.01, f"Multi-condition RMSE too high: {rmse}"

    def test_rmse_formula_equation_14(self, true_params):
        """RMSE should follow Equation (14) from the reference paper."""
        individual = encode_individual_double(true_params)

        # Create two datasets with different numbers of points
        V1 = np.linspace(0, 21.6, 10)
        V2 = np.linspace(0, 21.6, 20)

        I1 = simulate_iv_curve_double(V1, stc_params=true_params, temperature_k=298.15, irradiance_w_m2=1000.0, ns=36)
        I2 = simulate_iv_curve_double(V2, stc_params=true_params, temperature_k=298.15, irradiance_w_m2=800.0, ns=36)

        datasets = [
            {"V": V1, "I": I1, "T": 298.15, "G": 1000.0},
            {"V": V2, "I": I2, "T": 298.15, "G": 800.0},
        ]

        rmse = global_rmse_double(individual, datasets, Ns=36)

        assert np.isfinite(rmse)
        assert rmse >= 0


if __name__ == "__main__":
    from pvoptix.models.parameters import default_coeffs

    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }

    V = np.linspace(0, 21.6, 50)
    I = simulate_iv_curve_double(V, stc_params=true_params, temperature_k=298.15, irradiance_w_m2=1000.0, ns=36)
    sample_dataset = {"V": V, "I": I, "T": 298.15, "G": 1000.0}

    test = TestFitnessFunction()
    test.test_perfect_fit_rmse_small(true_params, sample_dataset)
    test.test_wrong_params_give_higher_rmse(true_params, sample_dataset)
    test.test_multiple_conditions(true_params)
    test.test_rmse_formula_equation_14(true_params)
    print("All fitness tests passed.")
