# tests/test_real_data.py
"""Test with real S75 .dat files."""

import numpy as np
import pytest

from pvoptix.api import (
    load_datasets_from_dir,
    optimize_double_multicondition,
    simulate_iv_curve_double,
    evaluate_double_parameters,
)


class TestRealData:
    """Test on real S75 module data."""

    @pytest.fixture
    def real_datasets(self, data_dir):
        """Load real S75 datasets using shared data_dir fixture."""
        if not data_dir.exists():
            pytest.skip(f"Data directory not found: {data_dir}")

        datasets = load_datasets_from_dir(str(data_dir))

        if len(datasets) == 0:
            pytest.skip("No datasets found in directory")

        return datasets

    def test_load_datasets(self, real_datasets):
        """Test that real datasets load correctly."""
        assert len(real_datasets) > 0, "No datasets loaded"

        ds = real_datasets[0]
        assert "V" in ds
        assert "I" in ds
        assert "T" in ds
        assert "G" in ds
        assert "model" in ds

    def test_dataset_conditions(self, real_datasets):
        """Test dataset physical consistency (not strict matching)."""
        for ds in real_datasets:
            assert ds["G"] > 0
            assert 250 < ds["T"] < 350  # Kelvin
            assert len(ds["V"]) == len(ds["I"])

    def test_simulate_on_real_data(self, real_datasets):
        """Test simulation on real data with dummy parameters."""
        ds = real_datasets[0]

        params = {
            "Rs": 0.28,
            "Rsh": 3200.0,
            "I01": 6.5e-8,
            "I02": 1.2e-7,
            "Iph": 4.68,
            "n1": 1.3,
            "n2": 1.8,
        }

        I_sim = simulate_iv_curve_double(
            ds["V"],
            stc_params=params,
            temperature_k=ds["T"],
            irradiance_w_m2=ds["G"],
            ns=36,
        )

        assert len(I_sim) == len(ds["V"])
        assert np.all(np.isfinite(I_sim))

    def test_optimization_on_real_data(self, real_datasets):
        """Quick optimization test."""
        if len(real_datasets) < 3:
            pytest.skip("Need at least 3 datasets")

        datasets = real_datasets[:3]

        result = optimize_double_multicondition(
            datasets=datasets,
            ns=36,
            pop_size=30,
            generations=30,
            crossover_rate=0.85,
            mutation_rate=0.12,
            verbose=False,
            live_plot=False,
        )

        assert result.best_fitness < 0.5, f"RMSE too high: {result.best_fitness}"
        assert len(result.best_params) == 7

    def test_optimization_on_real_data_full(self, real_datasets):
        """Stronger optimization test."""
        if len(real_datasets) < 3:
            pytest.skip("Need at least 3 datasets")

        datasets = real_datasets[:3]

        result = optimize_double_multicondition(
            datasets=datasets,
            ns=36,
            pop_size=50,
            generations=50,
            crossover_rate=0.85,
            mutation_rate=0.12,
            verbose=False,
            live_plot=False,
        )

        assert result.best_fitness < 0.3, f"RMSE too high: {result.best_fitness}"
        assert len(result.best_params) == 7

    def test_evaluate_on_real_data(self, real_datasets):
        """Test evaluation function."""
        ds = real_datasets[0]

        params = {
            "Rs": 0.28,
            "Rsh": 3200.0,
            "I01": 6.5e-8,
            "I02": 1.2e-7,
            "Iph": 4.68,
            "n1": 1.3,
            "n2": 1.8,
        }

        rmse = evaluate_double_parameters(params, [ds], ns=36)

        assert np.isfinite(rmse)
        assert rmse >= 0


if __name__ == "__main__":
    pytest.main([__file__])
