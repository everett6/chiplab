"""Base simulator backend for ChipLab."""

from __future__ import annotations

import abc
import shutil
from typing import Any

from ..schemas.design import DesignSpec
from ..schemas.simulation import SimulationConfig, SimulationResult
from ..schemas.common import FidelityLevel


class SimulatorBackend(abc.ABC):
    """Abstract base class for all simulator backends."""

    name: str
    fidelity: FidelityLevel

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the external simulator is available/installable."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig) -> SimulationResult:
        """Run a simulation and return the result."""
        raise NotImplementedError