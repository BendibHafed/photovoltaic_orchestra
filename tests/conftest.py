"""
Pytest fixtures for shared test data.
Centralized fixtures to avoid duplication and ensure reproducibility.
"""

import pytest
import numpy as np
from pathlib import Path


# ============================================================
#  GLOBAL SEED (VERY IMPORTANT for stable tests)
# ============================================================

@pytest.fixture(scope="session", autouse=True)
def set_random_seed():
    """Ensure all tests are reproducible."""
    np.random.seed(42)


# ============================================================
#  BASIC FIXTURES
# ============================================================

@pytest.fixture
def true_params():
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
def sample_voltage():
    """Sample voltage array."""
    return np.linspace(0, 21.6, 50)


@pytest.fixture
def stc_conditions():
    """STC conditions."""
    return {"G": 1000.0, "T": 298.15}


# ============================================================
#  PATH FIXTURE (FIXED FOR src/ LAYOUT)
# ============================================================

@pytest.fixture
def data_dir():
    root = Path(__file__).resolve().parents[1]  # project_root
    return root / "src" / "pvoptix" / "datasets" / "data"


# ============================================================
#  SYNTHETIC SINGLE DATASET
# ============================================================

@pytest.fixture
def sample_dataset(true_params, sample_voltage, stc_conditions):
    """Create a single synthetic dataset."""
    from pvoptix.api import simulate_iv_curve_double

    I = simulate_iv_curve_double(
        sample_voltage,
        stc_params=true_params,
        temperature_k=stc_conditions["T"],
        irradiance_w_m2=stc_conditions["G"],
        ns=36
    )

    return {
        "V": sample_voltage,
        "I": I,
        "T": stc_conditions["T"],
        "G": stc_conditions["G"],
        "model": "Synthetic_STC"
    }


# ============================================================
#   SYNTHETIC MULTI-CONDITION DATASETS (IMPORTANT)
# ============================================================

@pytest.fixture
def synthetic_datasets(true_params):
    """Create synthetic datasets for optimization tests."""
    from pvoptix.api import simulate_iv_curve_double

    datasets = []
    conditions = [
        (1000, 298.15),
        (800, 298.15),
        (600, 298.15),
        (400, 298.15),
    ]

    V = np.linspace(0, 21.6, 50)

    for G, T in conditions:
        I = simulate_iv_curve_double(
            V,
            stc_params=true_params,
            temperature_k=T,
            irradiance_w_m2=G,
            ns=36
        )

        # Add controlled noise
        I += np.random.randn(len(V)) * 0.005
        I = np.clip(I, 0, true_params["Iph"])

        datasets.append({
            "V": V,
            "I": I,
            "T": T,
            "G": G,
            "model": "Synthetic"
        })

    return datasets