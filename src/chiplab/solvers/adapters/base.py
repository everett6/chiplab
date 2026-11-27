"""Solver adapter base classes for ChipLab."""

from __future__ import annotations

import abc
import subprocess
from typing import Any, Dict, Optional, Tuple

from chiplab.schemas.common import TrackedValue, Provenance
from chiplab.schemas.simulation import SimulationConfig, SimulationResult
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.common import FidelityLevel


class SolverAdapter(abc.ABC):
    """Abstract base class for all solver adapters.

    Each adapter implements the interface between ChipLab and a specific
    simulation engine (LAMMPS, MOOSE, openCFS, SPICE, Verilator, OpenROAD, Ramulator).
    """

    name: str = "base"
    use_gpu: bool = True
    fidelity: FidelityLevel = FidelityLevel.ANALYTICAL
    executable: Optional[str] = None
    docker_image: Optional[str] = None

    @abc.abstractmethod
    def prepare_input(
        self, design: DesignSpec, workload: Any, config: SimulationConfig
    ) -> Tuple[str, Dict[str, TrackedValue]]:
        """Prepare solver input file(s) and return (input_path, parameter_map).

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            config: Simulation configuration

        Returns:
            Tuple of (input_file_path, parameter_map) where parameter_map maps
            parameter names to TrackedValue objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(
        self, input_path: str, resources: Optional[Dict[str, Any]] = None
    ) -> subprocess.CompletedProcess:
        """Run the solver with the given input.

        Args:
            input_path: Path to the prepared input file
            resources: Optional resource limits (cpu, memory, gpu, timeout)

        Returns:
            CompletedProcess from subprocess.run
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_output(
        self, process: subprocess.CompletedProcess
    ) -> Dict[str, TrackedValue]:
        """Parse solver output and return tracked values.

        Args:
            process: The CompletedProcess from running the solver

        Returns:
            Dictionary mapping output parameter names to TrackedValue objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cleanup(self, input_path: str) -> None:
        """Clean up temporary files after solver execution.

        Args:
            input_path: Path to the input file to clean up
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if the solver is available for use."""
        if self.executable:
            from shutil import which
            return which(self.executable) is not None
        if self.docker_image:
            # Docker availability check - simplified
            try:
                import subprocess
                result = subprocess.run(
                    ["docker", "ps"], capture_output=True, text=True
                )
                return result.returncode == 0
            except Exception:
                return False
        return False

    def get_capability(self) -> "SolverCapability":
        """Get the capability information for this adapter."""
        from chiplab.solvers.registry import SolverSpec
        return SolverCapability(
            spec=SolverSpec(
                name=self.name,
                executable=self.executable,
                docker_image=self.docker_image,
                supported_physics=self._get_supported_physics(),
                supported_scales=self._get_supported_scales(),
                fidelity=self.fidelity,
            )
        )

    @abc.abstractmethod
    def _get_supported_physics(self) -> List[str]:
        """Return supported physics types as strings."""

    @abc.abstractmethod
    def _get_supported_scales(self) -> List[str]:
        """Return supported scales as strings."""

    def handle_error(self, error: Exception) -> Dict[str, TrackedValue]:
        """Handle solver errors gracefully and return info."""
        return {
            "error": TrackedValue(
                value=str(error),
                unit=None,
                provenance=Provenance.ASSUMED,
                notes=f"Solver {self.name} encountered an error: {error}",
            )
        }

    def run_with_fallback(
        self, input_path: str, resources: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[subprocess.CompletedProcess], Dict[str, TrackedValue]]:
        """Run solver with graceful degradation on failure.

        Returns:
            Tuple of (process, parsed_output) where process may be None if failed
        """
        try:
            process = self.run(input_path, resources)
            parsed = self.parse_output(process)
            return process, parsed
        except Exception as e:
            # Graceful degradation: return error info rather than crashing
            parsed = self.handle_error(e)
            return None, parsed
