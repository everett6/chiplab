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

class ExternalToolBackend(SimulatorBackend):
    """A third-party simulator ChipLab can detect but has no adapter for yet.

    ``run`` always raises: it must never hand back an empty result marked successful.
    ``is_available`` only reports whether the executable is on PATH; ``integrated`` stays
    False until a real input-generation / output-parsing adapter exists.
    """

    executable: str
    display_name: str
    integrated: bool = False

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def status(self) -> dict[str, Any]:
        return {"name": self.name, "installed": self.is_available(),
                "integrated": self.integrated, "usable": self.integrated and self.is_available()}

    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig) -> SimulationResult:
        if not self.is_available():
            raise NotImplementedError(
                f"{self.display_name} is not installed ('{self.executable}' not on PATH).")
        raise NotImplementedError(
            f"{self.display_name} is installed, but ChipLab has no adapter for it yet "
            "(no config generation or output parsing). Use fidelity='analytical' or "
            "'discrete_event'.")
