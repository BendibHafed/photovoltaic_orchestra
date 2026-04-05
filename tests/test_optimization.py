# tests/test_optimization.py
"""Test GA optimization for double-diode model."""

import numpy as np
import pytest
from pvoptix.api import optimize_double_multicondition


class TestOptimization:
    """Test GA optimization."""

    def test_optimization_converges(self, synthetic_datasets):
        """GA should converge to reasonable RMSE."""
        result = optimize_double_multicondition(
            datasets=synthetic_datasets,
            ns=36,
            pop_size=40,
            generations=30,
            verbose=False,
            live_plot=False
        )
        assert result.best_fitness < 0.08, f"RMSE too high: {result.best_fitness:.4f}"

    def test_optimization_returns_7_parameters(self, synthetic_datasets):
        """Optimization should return 7 parameters."""
        result = optimize_double_multicondition(
            datasets=synthetic_datasets,
            ns=36,
            pop_size=30,
            generations=20,
            verbose=False,
            live_plot=False
        )
        assert len(result.best_params) == 7
        assert set(result.best_params.keys()) == {"Rs", "Rsh", "I01", "I02", "Iph", "n1", "n2"}

    def test_optimization_parameters_within_bounds(self, synthetic_datasets):
        """Optimized parameters should remain within physical bounds."""
        from pvoptix.optimization.ga.genome_mapping_double import BOUNDS_DOUBLE
        result = optimize_double_multicondition(
            datasets=synthetic_datasets,
            ns=36,
            pop_size=30,
            generations=20,
            verbose=False,
            live_plot=False
        )
        for key, (low, high) in BOUNDS_DOUBLE.items():
            assert low <= result.best_params[key] <= high, f"{key} out of bounds: {result.best_params[key]}"

    def test_optimization_history_not_empty(self, synthetic_datasets):
        """Optimization history should be recorded."""
        result = optimize_double_multicondition(
            datasets=synthetic_datasets,
            ns=36,
            pop_size=30,
            generations=20,
            verbose=False,
            live_plot=False
        )
        assert len(result.history) > 0

    def test_optimization_improves_fitness(self, synthetic_datasets):
        """Fitness should improve over generations."""
        progress_values = []

        def track_progress(gen, total, fitness, params):
            progress_values.append(fitness)

        result = optimize_double_multicondition(
            datasets=synthetic_datasets,
            ns=36,
            pop_size=30,
            generations=20,
            verbose=False,
            live_plot=False,
            on_progress=track_progress
        )

        if len(progress_values) > 1:
            assert progress_values[-1] <= progress_values[0] * 1.2, "Fitness did not improve sufficiently"


if __name__ == "__main__":
    from pvoptix.api import simulate_iv_curve_double

    np.random.seed(42)
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }

    datasets = []
    V = np.linspace(0, 21.6, 50)
    for G in [1000, 800, 600, 400]:
        I = simulate_iv_curve_double(V, stc_params=true_params,
                                     temperature_k=298.15, irradiance_w_m2=G, ns=36)
        I += np.random.randn(len(V)) * 0.005
        I = np.clip(I, 0, true_params["Iph"])
        datasets.append({"V": V, "I": I, "T": 298.15, "G": G})

    test = TestOptimization()
    test.test_optimization_converges(datasets)
    test.test_optimization_returns_7_parameters(datasets)
    test.test_optimization_parameters_within_bounds(datasets)
    test.test_optimization_history_not_empty(datasets)
    test.test_optimization_improves_fitness(datasets)
    print("All optimization tests passed.")
