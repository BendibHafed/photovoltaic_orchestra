"""
Test genome encoding and decoding for the double-diode model.
"""

import numpy as np
import pytest
from pvoptix.optimization.ga.genome_mapping_double import (
    decode_individual_double, encode_individual_double, BOUNDS_DOUBLE
)


class TestGenomeMapping:
    """Test encoding and decoding of parameters."""

    def test_decode_individual_length(self):
        """Decoded individual should contain 7 parameters."""
        individual = np.random.rand(7)
        params = decode_individual_double(individual)
        
        assert len(params) == 7, f"Expected 7 parameters, got {len(params)}"
        assert set(params.keys()) == {"Rs", "Rsh", "I01", "I02", "Iph", "n1", "n2"}

    def test_decode_individual_within_bounds(self):
        """Decoded parameters should remain within defined bounds."""
        individual = np.random.rand(7)
        params = decode_individual_double(individual)
        
        for key, (low, high) in BOUNDS_DOUBLE.items():
            assert low <= params[key] <= high, f"{key} out of bounds: {params[key]}"

    def test_encode_decode_roundtrip(self, true_params):
        """Encoding then decoding should return parameters close to the original."""
        individual = encode_individual_double(true_params)
        decoded = decode_individual_double(individual)
        
        for key in true_params:
            # Allow small tolerance for floating point differences
            assert abs(true_params[key] - decoded[key]) / true_params[key] < 0.01, \
                f"{key} mismatch: {true_params[key]} vs {decoded[key]}"

    def test_logarithmic_mapping_for_I01_I02(self):
        """I01 and I02 should use logarithmic mapping."""
        # Test minimum values
        individual_min = np.zeros(7)
        params_min = decode_individual_double(individual_min)
        assert params_min["I01"] == 1e-12
        assert params_min["I02"] == 1e-12

        # Test maximum values
        individual_max = np.ones(7)
        params_max = decode_individual_double(individual_max)
        assert params_max["I01"] == 1e-6
        assert params_max["I02"] == 1e-6

    def test_n2_constraint(self, true_params):
        """Constraint n2 >= n1 should be enforced."""
        params = true_params.copy()
        params["n2"] = params["n1"] - 0.2
        
        individual = encode_individual_double(params)
        decoded = decode_individual_double(individual)
        
        assert decoded["n2"] >= decoded["n1"], "Constraint n2 >= n1 not enforced"


if __name__ == "__main__":
    true_params = {
        "Rs": 0.28, "Rsh": 3200.0,
        "I01": 6.5e-8, "I02": 1.2e-7,
        "Iph": 4.68, "n1": 1.3, "n2": 1.8,
    }

    test = TestGenomeMapping()
    test.test_decode_individual_length()
    test.test_decode_individual_within_bounds()
    test.test_encode_decode_roundtrip(true_params)
    test.test_logarithmic_mapping_for_I01_I02()
    test.test_n2_constraint(true_params)
    print("All genome mapping tests passed.")
