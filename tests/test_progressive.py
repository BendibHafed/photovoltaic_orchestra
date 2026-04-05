# tests/test_progressive.py
"""Test progressive optimization (flowchart algorithm)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from pvoptix.api import (
    optimize_double_progressive,
    simulate_iv_curve_double,
    evaluate_double_parameters
)


class TestProgressiveOptimization:
    """Test progressive optimization (friend's idea / flowchart)."""

    @pytest.fixture
    def scan_generator(self, true_params):
        """Generate progressive scans throughout the day."""
        scans = []
        conditions = [
            (733, 23, "08h10"),
            (810, 24.2, "08h20"),
            (650, 22.1, "08h30"),
            (550, 21.5, "08h40"),
            (450, 20.8, "08h50"),
        ]
        V = np.linspace(0, 21.6, 50)

        for G, T_c, label in conditions:
            T = T_c + 273.15
            I = simulate_iv_curve_double(
                V, stc_params=true_params,
                temperature_k=T, irradiance_w_m2=G, ns=36
            )
            I += np.random.randn(len(V)) * 0.005
            I = np.clip(I, 0, true_params["Iph"])
            scans.append({"V": V, "I": I, "T": T, "G": G, "model": label})

        return scans

    def test_progressive_returns_results(self, scan_generator):
        """Test that progressive optimization returns results."""
        result = optimize_double_progressive(
            scan_stream=iter(scan_generator),
            ns=36,
            ga_kwargs={
                "pop_size": 20,
                "generations": 15,
                "verbose": False,
                "live_plot": False
            },
            verbose=False
        )

        assert result.best_params is not None
        assert len(result.best_params) == 7
        assert result.best_fitness > 0

    def test_progressive_history_length(self, scan_generator):
        """Test that history records all scans."""
        n_scans = len(scan_generator)

        result = optimize_double_progressive(
            scan_stream=iter(scan_generator),
            ns=36,
            ga_kwargs={
                "pop_size": 20,
                "generations": 15,
                "verbose": False,
                "live_plot": False
            },
            verbose=False
        )

        assert len(result.history) == n_scans

    def test_progressive_meta_information(self, scan_generator):
        """Test that meta contains strategy information."""
        result = optimize_double_progressive(
            scan_stream=iter(scan_generator),
            ns=36,
            ga_kwargs={
                "pop_size": 20,
                "generations": 15,
                "verbose": False,
                "live_plot": False
            },
            verbose=False
        )

        assert result.meta["strategy"] == "progressive"
        assert result.meta["model"] == "double_diode"
        assert result.meta["scans_processed"] == len(scan_generator)

    def test_progressive_without_virtual_stc(self, scan_generator):
        """Test progressive without virtual STC curve."""
        result = optimize_double_progressive(
            scan_stream=iter(scan_generator),
            ns=36,
            include_virtual_stc=False,
            ga_kwargs={
                "pop_size": 20,
                "generations": 15,
                "verbose": False,
                "live_plot": False
            },
            verbose=False
        )

        assert result.best_params is not None

    def test_progressive_improves_or_maintains(self, scan_generator):
        """Test that best RMSE never increases."""
        result = optimize_double_progressive(
            scan_stream=iter(scan_generator),
            ns=36,
            ga_kwargs={
                "pop_size": 20,
                "generations": 15,
                "verbose": False,
                "live_plot": False
            },
            verbose=False
        )

        rmse_values = [entry['rmse_best'] for entry in result.history]

        # RMSE should be non-increasing (allow small floating point errors)
        for i in range(1, len(rmse_values)):
            assert rmse_values[i] <= rmse_values[i-1] + 0.001


if __name__ == "__main__":
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }

    test = TestProgressiveOptimization()
    scans = test.scan_generator(true_params)
    test.test_progressive_returns_results(scans)
    test.test_progressive_history_length(scans)
    test.test_progressive_meta_information(scans)
    test.test_progressive_without_virtual_stc(scans)
    test.test_progressive_improves_or_maintains(scans)
    print("\nAll progressive optimization tests passed!")