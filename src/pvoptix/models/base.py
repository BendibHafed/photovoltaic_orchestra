"""
Base abstract class for PV models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePVModel(ABC):
    """Abstract base class for PV models."""

    @abstractmethod
    def parameters(self) -> dict[str, float]:
        """Return model parameters as a dict."""
        raise NotImplementedError

    @abstractmethod
    def iv_curve(
        self, voltage: np.ndarray, temperature_k: float, ns: int
    ) -> np.ndarray:
        """Compute I-V curve currents [A] for a given voltage array [V]."""
        raise NotImplementedError