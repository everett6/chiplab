"""Ramulator backend for ChipLab simulation runs."""

import shutil
from typing import Any

from ..schemas.design import DesignSpec
from ..schemas.simulation import SimulationConfig, SimulationResult
from ..schemas.common import FidelityLevel
from .base import SimulatorBackend


class RamulatorBackend(SimulatorBackend):
    """Ramulator-based DRAM simulator backend.

    Checks for the ramulator executable via shutil.which or environment variable.
    """

    name: str = "ramulator"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED

    def is_available(self) -> bool:
        """Check if ramulator is available on the system."""
        return shutil.which("ramulator") is not None

    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig) -> SimulationResult:
        """Run a ramulator simulation.

        Raises NotImplementedError if ramulator is not installed.
        """
        if not self.is_available():
            raise NotImplementedError(
                "Ramulator is not installed. Please install ramulator or set up the environment "
                "to use the RamulatorBackend."
            )
        # Placeholder: actual ramulator integration would go here
        from ..schemas.simulation import SimulationMetrics, BottleneckReport, AssumptionRecord
        metrics = SimulationMetrics()
        bottleneck = BottleneckReport(
            primary_bottleneck="memory_bandwidth",
            bound_type="memory_bandwidth",
            severity_score=0.5,
            details=["Ramulator simulation placeholder"],
            recommendations=["Configure ramulator for actual simulation"],
        )
        return SimulationResult(
            run_id="ramulator-placeholder",
            design_id=design.name,
            design_name=design.name,
            fidelity=self.fidelity,
            fidelity_disclaimer="Ramulator placeholder: actual simulation not available",
            metrics=metrics,
            bottleneck=bottleneck,
            assumptions=[],
            warnings=[],
            raw_details={},
            success=True,
        )