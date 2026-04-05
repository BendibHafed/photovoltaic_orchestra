# tests/test_integration.py
"""End-to-end integration tests for PVOptix."""

import numpy as np
import pytest
from pathlib import Path

from pvoptix.api import (
    optimize_double_progressive,
    optimize_double_multicondition,
    simulate_iv_curve_double,
    evaluate_double_parameters,
    load_datasets_from_dir
)


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def true_params(self):
        """True STC parameters for S75 module."""
        return {
            "Rs": 0.28,
            "Rsh": 3200.0,
            "I01": 6.5e-8,
            "I02": 1.2e-7,
            "Iph": 4.68,
            "n1": 1.3,
            "n2": 1.8,
        }

    @pytest.fixture
    def synthetic_data_pipeline(self, true_params):
        """Create a complete synthetic data pipeline."""
        scans = []
        conditions = [(1000, 25), (800, 25), (600, 25), (400, 25), (200, 25)]
        V = np.linspace(0, 21.6, 50)

        np.random.seed(42)  # reproducibility

        for G, T_c in conditions:
            T = T_c + 273.15
            I = simulate_iv_curve_double(
                V, stc_params=true_params,
                temperature_k=T, irradiance_w_m2=G, ns=36
            )
            I += np.random.randn(len(V)) * 0.005
            I = np.clip(I, 0, true_params["Iph"])
            scans.append({"V": V, "I": I, "T": T, "G": G})

        return scans, true_params

    @pytest.fixture
    def data_dir(self):
        """Path to real S75 data directory (portable)."""
        # Resolve relative to project root
        return Path(__file__).resolve().parents[1] / "src" / "pvoptix" / "datasets" / "data"

    def test_end_to_end_synthetic(self, synthetic_data_pipeline):
        """Complete end-to-end test with synthetic data."""
        scans, true_params = synthetic_data_pipeline

        result = optimize_double_progressive(
            scan_stream=iter(scans),
            ns=36,
            ga_kwargs={
                "pop_size": 50,
                "generations": 50,
                "verbose": False,
                "live_plot": False
            },
            verbose=False
        )

        assert result.best_params is not None
        assert len(result.best_params) == 7

        final_rmse = evaluate_double_parameters(result.best_params, scans, ns=36)
        assert final_rmse < 0.05

        for scan in scans:
            I_pred = simulate_iv_curve_double(
                scan["V"],
                stc_params=result.best_params,
                temperature_k=scan["T"],
                irradiance_w_m2=scan["G"],
                ns=36
            )
            rmse_curve = np.sqrt(np.mean((scan["I"] - I_pred) ** 2))
            assert rmse_curve < 0.05

        for key in true_params:
            if key in ["Rsh", "I01", "I02", "n1", "n2"]:
                continue
            error = abs(result.best_params[key] - true_params[key]) / true_params[key]
            assert error < 0.3

    def test_multi_condition_vs_single_condition(self, true_params):
        """Compare multi-condition optimization with single-condition optimization."""
        datasets = []
        conditions = [(1000, 298.15), (800, 298.15), (600, 298.15)]
        V = np.linspace(0, 21.6, 50)
        np.random.seed(42)

        for G, T in conditions:
            I = simulate_iv_curve_double(V, stc_params=true_params,
                                         temperature_k=T, irradiance_w_m2=G, ns=36)
            I += np.random.randn(len(V)) * 0.005
            datasets.append({"V": V, "I": I, "T": T, "G": G})

        result_single = optimize_double_multicondition(
            datasets=[datasets[0]],
            ns=36,
            pop_size=30,
            generations=30,
            verbose=False,
            live_plot=False
        )

        result_multi = optimize_double_multicondition(
            datasets=datasets,
            ns=36,
            pop_size=30,
            generations=30,
            verbose=False,
            live_plot=False
        )

        rmse_multi_all = evaluate_double_parameters(result_multi.best_params, datasets, ns=36)
        assert rmse_multi_all < 0.1

    @pytest.mark.slow
    def test_full_progressive_with_real_data(self, data_dir):
        """Complete progressive optimization test with real S75 data."""
        if not data_dir.exists():
            pytest.skip(f"Data directory not found: {data_dir}")

        datasets = load_datasets_from_dir(str(data_dir))
        if len(datasets) < 3:
            pytest.skip("Need at least 3 datasets")

        result = optimize_double_progressive(
            scan_stream=iter(datasets),
            ns=36,
            ga_kwargs={
                "pop_size": 30,
                "generations": 25,
                "verbose": False,
                "live_plot": False
            },
            verbose=False
        )

        assert result.best_fitness < 0.15
        assert len(result.best_params) == 7
        assert result.meta["scans_processed"] == len(datasets)


if __name__ == "__main__":
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }

    test = TestIntegration()
    scans, _ = test.synthetic_data_pipeline(true_params)
    test.test_end_to_end_synthetic((scans, true_params))
    test.test_multi_condition_vs_single_condition(true_params)
    print("All integration tests passed.")
