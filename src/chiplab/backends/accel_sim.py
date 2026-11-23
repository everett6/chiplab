"""AccelSim backend for ChipLab simulation runs."""

import shutil
from typing import Any

from ..schemas.design import DesignSpec
from ..schemas.simulation import SimulationConfig, SimulationResult
from ..schemas.common import FidelityLevel
from .base import SimulatorBackend


class AccelSimBackend(SimulatorBackend):
    """AccelSim-based accelerator simulator backend.

    Checks for the accel_sim executable via shutil.which or environment variable.
    """

    name: str = "accel_sim"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED

    def is_available(self) -> bool:
        """Check if accel_sim is available on the system."""
        return shutil.which("accel_sim") is not None

    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig) -> SimulationResult:
        """Run an accel_sim simulation.

        Raises NotImplementedError if accel_sim is not installed.
        """
        if not self.is_available():
            raise NotImplementedError(
                "AccelSim is not installed. Please install accel_sim or set up the environment "
                "to use the AccelSimBackend."
            )
        # Placeholder: actual accel_sim integration would go here
        from ..schemas.simulation import SimulationMetrics, BottleneckReport, AssumptionRecord
        metrics = SimulationMetrics()
        bottleneck = BottleneckReport(
            primary_bottleneck="compute",
            bound_type="compute",
            severity_score=0.5,
            details=["AccelSim simulation placeholder"],
            recommendations=["Configure accel_sim for actual simulation"],
        )
        return SimulationResult(
            run_id="accel_sim-placeholder",
            design_id=design.name,
            design_name=design.name,
            fidelity=self.fidelity,
            fidelity_disclaimer="AccelSim placeholder: actual simulation not available",
            metrics=metrics,
            bottleneck=bottleneck,
            assumptions=[],
            warnings=[],
            raw_details={},
            success=True,
        )