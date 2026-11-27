"""Solver registry & capability matching for ChipLab."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..schemas.common import FidelityLevel, Provenance, TrackedValue


class PhysicsType(Enum):
    """Physics domains that solvers can handle."""
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    MAGNETIC = "magnetic"
    PARTICLE = "particle"
    FLUID = "fluid"


class ScaleType(Enum):
    """Scale at which a solver operates."""
    DEVICE = "device"
    CHIPLET = "chiplet"
    SYSTEM = "system"
    BOARD = "board"
    WAFER = "wafer"


@dataclass
class SolverSpec:
    """Specification for a registered solver."""
    name: str
    version: str = "0.1.0"
    executable: Optional[str] = None
    docker_image: Optional[str] = None
    license: Optional[str] = None
    supported_platforms: List[str] = field(default_factory=list)
    supported_physics: List[PhysicsType] = field(default_factory=list)
    supported_scales: List[ScaleType] = field(default_factory=list)
    fidelity: FidelityLevel = FidelityLevel.ANALYTICAL
    input_formats: List[str] = field(default_factory=list)
    output_formats: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class SolverCapability:
    """Capability information for a solver."""
    spec: SolverSpec
    # Geographic/environmental constraints
    requires_gpu: bool = False
    requires_docker: bool = False
    max_particles: Optional[int] = None
    max_mesh_size: Optional[int] = None
    supported_analysis_types: List[str] = field(default_factory=list)
    # Performance characteristics
    typical_runtime_seconds: Optional[Dict[str, float]] = field(default_factory=dict)
    memory_per_gb: Optional[float] = None


class CapabilityMatcher:
    """Given a device + target physics + scale, rank available solvers."""

    def __init__(self, registry: SolverRegistry):
        self.registry = registry

    def match(
        self,
        physics: PhysicsType,
        scale: ScaleType,
        fidelity: Optional[FidelityLevel] = None,
        platform: Optional[str] = None,
    ) -> List[Tuple[SolverSpec, float]]:
        """Return ranked list of (SolverSpec, confidence_score) matching the criteria.

        Args:
            physics: Target physics domain (electrical, thermal, mechanical, etc.)
            scale: Target scale (device, chiplet, system, board, wafer)
            fidelity: Desired fidelity level (analytical, discrete_event, specialized)
            platform: Target platform (linux, windows, macOS, hpc, docker)

        Returns:
            List of (SolverSpec, confidence_score) tuples, sorted by confidence descending.
        """
        candidates: List[Tuple[SolverSpec, float]] = []

        for solver_name, solver_spec in self.registry.solvers.items():
            score = 0.0
            max_score = 0.0

            # Check physics support
            if physics in solver_spec.supported_physics:
                score += 1.0
            max_score += 1.0

            # Check scale support
            if scale in solver_spec.supported_scales:
                score += 1.0
            max_score += 1.0

            # Check fidelity support
            if fidelity and solver_spec.fidelity >= fidelity:
                score += 1.0
            max_score += 1.0

            # Check platform support
            if platform and platform in solver_spec.supported_platforms:
                score += 1.0
            max_score += 1.0

            if max_score > 0:
                confidence = score / max_score
                candidates.append((solver_spec, confidence))

        # Sort by confidence descending, then by name for stability
        candidates.sort(key=lambda x: (-x[1], x[0].name))
        return candidates


@dataclass
class SolverRegistry:
    """Discover, register, and query solvers.

    Scans PATH, conda envs, docker, and modules for available solvers.
    Maintains a catalog of SolverSpec entries keyed by name.
    """

    def __init__(self):
        self.solvers: Dict[str, SolverSpec] = {}
        self._auto_discovery = AutoDiscovery(self)

    def register(self, spec: SolverSpec) -> None:
        """Register a solver specification."""
        self.solvers[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Unregister a solver."""
        self.solvers.pop(name, None)

    def get(self, name: str) -> Optional[SolverSpec]:
        """Get a solver spec by name."""
        return self.solvers.get(name)

    def list_available(self, physics: Optional[PhysicsType] = None,
                       scale: Optional[ScaleType] = None) -> List[SolverSpec]:
        """List solvers, optionally filtered by physics and scale."""
        results = list(self.solvers.values())
        if physics is not None:
            results = [s for s in results if physics in s.supported_physics]
        if scale is not None:
            results = [s for s in results if scale in s.supported_scales]
        return results

    def match(self, physics: PhysicsType, scale: ScaleType,
              fidelity: Optional[FidelityLevel] = None,
              platform: Optional[str] = None) -> List[Tuple[SolverSpec, float]]:
        """Match solvers to physics/scale/fidelity criteria."""
        matcher = CapabilityMatcher(self)
        return matcher.match(physics, scale, fidelity, platform)

    def discover(self) -> Dict[str, SolverSpec]:
        """Run auto-discovery and register found solvers."""
        return self._auto_discovery.discover()


class AutoDiscovery:
    """Scan PATH, conda envs, docker, modules for available solvers."""

    def __init__(self, registry: SolverRegistry):
        self.registry = registry

    def discover(self) -> Dict[str, SolverSpec]:
        """Run all discovery methods and register findings."""
        self._scan_path()
        self._scan_conda_envs()
        self._scan_docker()
        self._scan_modules()
        return self.registry.solvers

    def _scan_path(self) -> None:
        """Scan PATH for solver executables."""
        # Common solver names and their executables
        path_solvers = {
            "lammps": ["lammps"],
            "ngspice": ["ngspice"],
            "spice": ["ngspice"],
            "openroad": ["openroad"],
            "verilator": ["verilator"],
            "moose": ["moose"],
            "opencfs": ["opencfs"],
        }

        # Check if each executable is on PATH
        for name, executables in path_solvers.items():
            for exe in executables:
                exe_path = shutil.which(exe)
                if exe_path:
                    spec = SolverSpec(
                        name=name,
                        version=self._get_version(exe_path),
                        executable=exe_path,
                        supported_physics=self._get_physics_from_name(name),
                        supported_scales=self._get_scales_from_name(name),
                        input_formats=self._get_input_formats(name),
                        output_formats=self._get_output_formats(name),
                    )
                    self.registry.register(spec)
                    break  # Only register once per solver name

    def _scan_conda_envs(self) -> None:
        """Scan conda environments for solver installations."""
        # Placeholder for conda environment scanning
        pass

    def _scan_docker(self) -> None:
        """Scan Docker Hub/registry for solver images."""
        # Placeholder for Docker scanning
        pass

    def _scan_modules(self) -> None:
        """Scan Python modules for solver wrappers."""
        # Placeholder for module scanning
        pass

    @staticmethod
    def _get_version(exe_path: str) -> str:
        """Try to get version from an executable."""
        import subprocess
        result = subprocess.run([exe_path, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract version from first line
            first_line = result.stdout.strip().split("\n")[0]
            # Try to extract version number
            import re
            match = re.search(r"(\d+\.\d+(?:\.\d+)?)", first_line)
            if match:
                return match.group(1)
        return "0.1.0"

    @staticmethod
    def _get_physics_from_name(name: str) -> List[PhysicsType]:
        """Map solver name to physics types."""
        mapping = {
            "lammps": [PhysicsType.MECHANICAL, PhysicsType.THERMAL],
            "ngspice": [PhysicsType.ELECTRICAL],
            "spice": [PhysicsType.ELECTRICAL],
            "openroad": [PhysicsType.ELECTRICAL],
            "verilator": [PhysicsType.ELECTRICAL],
            "moose": [PhysicsType.THERMAL, PhysicsType.MECHANICAL],
            "opencfs": [PhysicsType.THERMAL, PhysicsType.MECHANICAL],
        }
        return mapping.get(name, [])

    @staticmethod
    def _get_scales_from_name(name: str) -> List[ScaleType]:
        """Map solver name to supported scales."""
        mapping = {
            "lammps": [ScaleType.DEVICE, ScaleType.CHIPLET],
            "ngspice": [ScaleType.CHIPLET, ScaleType.SYSTEM],
            "openroad": [ScaleType.SYSTEM, ScaleType.BOARD],
            "verilator": [ScaleType.SYSTEM],
            "moose": [ScaleType.DEVICE, ScaleType.CHIPLET, ScaleType.SYSTEM],
            "opencfs": [ScaleType.DEVICE, ScaleType.CHIPLET, ScaleType.SYSTEM],
        }
        return mapping.get(name, [])

    @staticmethod
    def _get_input_formats(name: str) -> List[str]:
        """Map solver name to supported input formats."""
        mapping = {
            "lammps": ["data", "input script"],
            "ngspice": ["netlist", "spice file"],
            "spice": ["netlist", "spice file"],
            "openroad": ["rtl", "constraints", "lef/def"],
            "verilator": ["systemverilog", "verilog"],
            "moose": ["moose_input"],
            "opencfs": ["opencfs_input"],
        }
        return mapping.get(name, [])

    @staticmethod
    def _get_output_formats(name: str) -> List[str]:
        """Map solver name to supported output formats."""
        mapping = {
            "lammps": ["trajectories", "thermodynamic data", "custom computes"],
            "ngspice": ["waveforms", "csv", "measurements"],
            "spice": ["waveforms", "csv", "measurements"],
            "openroad": ["def", "gds", "timing reports", "power reports"],
            "verilator": ["vcd", "fst", "coverage", "lint"],
            "moose": ["field_data", "mesh_results", "coupled_outputs"],
            "opencfs": ["coupled_field_data", "mesh_results", "solution_outputs"],
        }
        return mapping.get(name, "")