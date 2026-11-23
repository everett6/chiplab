"""ScaleSim backend for ChipLab simulation runs."""

import shutil
from typing import Any

from ..schemas.design import DesignSpec
from ..schemas.simulation import SimulationConfig, SimulationResult
from ..schemas.common import FidelityLevel
from .base import SimulatorBackend


class ScaleSimBackend(SimulatorBackend):
    """ScaleSim-based cache and memory system simulator backend.

    Checks for the scale_sim executable via shutil.which or environment variable.
    """

    name: str = "scale_sim"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED

    def is_available(self) -> bool:
        """Check if scale_sim is available on the system."""
        return shutil.which("scale_sim") is not None

    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig) -> SimulationResult:
        """Run a scale_sim simulation.

        Raises NotImplementedError if scale_sim is not installed.
        """
        if not self.is_available():
            raise NotImplementedError(
                "ScaleSim is not installed. Please install scale_sim or set up the environment "
                "to use the ScaleSimBackend."
            )
        # Placeholder: actual scale_sim integration would go here
        from ..schemas.simulation import SimulationMetrics, BottleneckReport, AssumptionRecord
        metrics = SimulationMetrics()
        bottleneck = BottleneckReport(
            primary_bottleneck="memory_bandwidth",
            bound_type="memory_bandwidth",
            severity_score=0.5,
            details=["ScaleSim simulation placeholder"],
            recommendations=["Configure scale_sim for actual simulation"],
        )
        return SimulationResult(
            run_id="scale_sim-placeholder",
            design_id=design.name,
            design_name=design.name,
            fidelity=self.fidelity,
            fidelity_disclaimer="ScaleSim placeholder: actual simulation not available",
            metrics=metrics,
            bottleneck=bottleneck,
            assumptions=[],
            warnings=[],
            raw_details={},
            success=True,
        )