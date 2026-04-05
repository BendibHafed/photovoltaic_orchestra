# tests/test_translation.py
"""Test translation equations (4-11) from paper."""

import numpy as np
import pytest
from pvoptix.models.parameters import (
    iph_model, i01_model, i02_model, rs_model, rsh_model_double,
    default_coeffs
)


class TestTranslationEquations:
    """Test all translation equations."""

    def test_iph_model_equation_4(self):
        """Photocurrent model: Iph(G,T) = Iph_ref * (G/G_ref) * [1 + α_I*(T - T_ref)]"""
        Iph_stc = 4.68
        G = 800
        T = 313.15  # 40°C

        Iph = iph_model(G, T, Iph_stc)
        expected = Iph_stc * (800 / 1000) * (1 + 0.0005 * (40 - 25))
        assert abs(Iph - expected) < 1e-6

    def test_iph_model_scales_with_G(self):
        """Iph should be proportional to irradiance G."""
        Iph_stc = 4.68
        T = 298.15

        Iph_800 = iph_model(800, T, Iph_stc)
        Iph_400 = iph_model(400, T, Iph_stc)

        ratio = Iph_800 / Iph_400
        assert abs(ratio - 2.0) < 0.01

    def test_i01_model_increases_with_temperature(self):
        """I01 should increase with temperature (Equation 5)."""
        I01_stc = 6.5e-8
        n1 = 1.3

        I01_298 = i01_model(298.15, I01_stc, n1)
        I01_313 = i01_model(313.15, I01_stc, n1)

        assert I01_313 > I01_298

    def test_i02_model_increases_with_temperature(self):
        """I02 should increase with temperature (Equation 6)."""
        I02_stc = 1.2e-7
        n2 = 1.8

        I02_298 = i02_model(298.15, I02_stc, n2)
        I02_313 = i02_model(313.15, I02_stc, n2)

        assert I02_313 > I02_298

    def test_rsh_model_inverse_with_G_equation_10(self):
        """Rsh should be inversely proportional to irradiance G (Equation 10)."""
        Rsh_stc = 3200.0

        Rsh_800 = rsh_model_double(800, Rsh_stc)
        Rsh_400 = rsh_model_double(400, Rsh_stc)

        ratio = Rsh_400 / Rsh_800
        assert abs(ratio - 2.0) < 0.01

    def test_rs_model_constant_equation_11(self):
        """Rs should remain constant (Equation 11)."""
        Rs_stc = 0.28

        Rs_800_298 = rs_model(800, 298.15, Rs_stc)
        Rs_400_313 = rs_model(400, 313.15, Rs_stc)

        assert abs(Rs_800_298 - Rs_stc) < 1e-6
        assert abs(Rs_400_313 - Rs_stc) < 1e-6


if __name__ == "__main__":
    test = TestTranslationEquations()
    test.test_iph_model_equation_4()
    test.test_iph_model_scales_with_G()
    test.test_i01_model_increases_with_temperature()
    test.test_i02_model_increases_with_temperature()
    test.test_rsh_model_inverse_with_G_equation_10()
    test.test_rs_model_constant_equation_11()
    print("All translation tests passed.")
