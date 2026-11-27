"""MOOSE adapter for ChipLab solver integration.

MOOSE (Multiphysics Object-Orientated Simulation Environment) adapter for
coupled multiphysics FEM simulations. Supports thermal, structural, and
fluid flow multiphysics simulations.

Input: Input files (ICEM, coupled thermal-structural)
Output: Field data, mesh results, coupled solution outputs
GPU support: optional via Kokkos or other backends
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chiplab.schemas.common import TrackedValue, Provenance
from chiplab.schemas.simulation import SimulationConfig, SimulationResult
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.common import FidelityLevel
from .base import SolverAdapter


class MOOSEAdapter(SolverAdapter):
    """MOOSE adapter for coupled multiphysics FEM simulations.

    MOOSE is a multiphysics finite element framework for coupled simulations
    including thermal-structural, fluid-structural, and electrochemical
    simulations.

    Input: MOOSE input files (.i, .param)
    Output: Field data, mesh results, coupled solution outputs
    GPU support: optional via Kokkos backend
    """

    name: str = "moose"
    use_gpu: bool = True
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
    executable: str = "moose"

    def _get_supported_physics(self) -> List[str]:
        return ["thermal", "structural", "mechanical", "fluid"]

    def _get_supported_scales(self) -> List[str]:
        return ["device", "chiplet", "system"]

    def prepare_input(
        self, design: DesignSpec, workload: Any, config: SimulationConfig
    ) -> Tuple[str, Dict[str, TrackedValue]]:
        """Prepare MOOSE input file and return (input_path, parameter_map).

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            config: Simulation configuration

        Returns:
            Tuple of (input_file_path, parameter_map)
        """
        tmpdir = tempfile.mkdtemp(prefix="chiplab_moose_")
        tmp = Path(tmpdir)

        # Generate MOOSE input file based on design
        input_path = tmp / "simulation.i"
        self._write_moose_input(design, workload, config, input_path)

        # Build parameter map from design/workload
        param_map: Dict[str, TrackedValue] = {}
        if hasattr(design, "chiplets") and design.chiplets:
            for i, chiplet in enumerate(design.chiplets):
                compute = getattr(chiplet, "compute", None)
                if compute:
                    param_map["compute_architecture"] = TrackedValue(
                        value=compute.architecture,
                        unit=None,
                        provenance=Provenance.ASSUMED,
                    )
                    param_map["temperature_tdp"] = TrackedValue(
                        value=chiplet.tdp_watts,
                        unit="W",
                        provenance=Provenance.ASSUMED,
                    )

        # Add workload parameters
        if workload:
            for key, value in workload.items():
                if isinstance(value, (int, float, str, bool)):
                    param_map[key] = TrackedValue(
                        value=value,
                        unit=None,
                        provenance=Provenance.ASSUMED,
                    )

        return str(input_path), param_map

    def _write_moose_input(
        self, design: DesignSpec, workload: Any, config: SimulationConfig,
        path: Path
    ) -> None:
        """Write MOOSE input file based on design.

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            config: Simulation configuration
            path: Path to write the MOOSE input file to
        """
        # Determine physics types from design
        physics_types = self._get_supported_physics()

        with open(path, "w") as f:
            f.write("[Global Parameters]\n")
            f.write("start_time = 0.0\n")
            f.write("stop_time = 100.0\n")
            f.write("dt = 0.1\n")
            f.write("scaling = 1.0\n")
            f.write("\n")

            if "thermal" in physics_types:
                f.write("[./Thermal]\n")
                f.write("type = HeatEquation\n")
                if design.chiplets:
                    for i, chiplet in enumerate(design.chiplets):
                        compute = getattr(chiplet, "compute", None)
                        if compute:
                            f.write(f"material = silicon_{i}\n")
                            f.write(f"heat_source = uniform_{i}\n")
                            f.write(f"heat_power = {chiplet.tdp_watts}\n")
                f.write("\n")

            if "structural" in physics_types:
                f.write("[./Structural]\n")
                f.write("type = LinearElastic\n")
                if design.chiplets:
                    for i, chiplet in enumerate(design.chiplets):
                        compute = getattr(chiplet, "compute", None)
                        if compute:
                            f.write(f"material = silicon_{i}\n")
                            f.write(f"stress_source = thermal_{i}\n")
                f.write("\n")

            if "fluid" in physics_types:
                f.write("[./Fluid]\n")
                f.write("type = NavierStokes\n")
                f.write("\n")

            f.write("[Postprocessor]\n")
            f.write("execute_on = 'timestep end'\n")
            f.write("file_base = 'results'\n")
            f.write("csv = true\n")
            f.write("\n")

            f.write("[./Materials]\n")
            f.write("silicon:\n")
            f.write("type = Isotropic\n")
            f.write("youngs_modulus = 130e9\n")
            f.write("poissons_ratio = 0.28\n")
            f.write("density = 2330\n")
            f.write("\n")

            f.write("[./InitialConditions]\n")
            if "thermal" in physics_types:
                f.write("T0 = 300.0\n")
            if "structural" in physics_types:
                f.write("u0 = 0.0\n")

    def run(
        self, input_path: str, resources: Optional[Dict[str, Any]] = None
    ) -> subprocess.CompletedProcess:
        """Run MOOSE simulation.

        Args:
            input_path: Path to the MOOSE input file
            resources: Optional resource limits (cpu, memory, gpu, timeout)

        Returns:
            CompletedProcess from running MOOSE
        """
        cmd = [self.executable, input_path]

        # Add timeout
        timeout = resources.get("timeout", 300) if resources else 300

        # Add GPU support if requested
        if resources and resources.get("gpu", False):
            cmd.extend(["-g", "1"])

        # Add processor count
        if resources and resources.get("cpu_count"):
            cmd.extend(["-p", str(resources["cpu_count"])])

        try:
            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return process
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"MOOSE simulation timed out after {timeout}s: {e}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"MOOSE executable not found: {self.executable}. "
                "Please install MOOSE."
            )

    def parse_output(
        self, process: subprocess.CompletedProcess
    ) -> Dict[str, TrackedValue]:
        """Parse MOOSE output for simulation results.

        Args:
            process: The CompletedProcess from running MOOSE

        Returns:
            Dictionary mapping output parameter names to TrackedValue objects
        """
        results: Dict[str, TrackedValue] = {}

        if process.returncode != 0:
            return self.handle_error(
                RuntimeError(f"MOOSE failed with return code {process.returncode}")
            )

        output = process.stdout

        # Parse temperature results
        for line in output.split("\n"):
            line = line.strip()

            if "Temperature" in line or "temp" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace(".", "", 1).isdigit() or (
                            part.startswith("-") and part[1:].replace(".", "", 1).isdigit()
                        ):
                            results["temperature_k"] = TrackedValue(
                                value=float(parts[i]),
                                unit="K",
                                provenance=Provenance.MEASURED,
                            )
                            break
                except (ValueError, IndexError):
                    pass

            if "Heat Flux" in line or "heat_flux" in line.lower():
                try:
                    parts = line.split()
                    if len(parts) > 2:
                        results["heat_flux_w_m2"] = TrackedValue(
                            value=float(parts[2]),
                            unit="W/m^2",
                            provenance=Provenance.MEASURED,
                        )
                except (ValueError, IndexError):
                    pass

            if "Displacement" in line or "displacement" in line.lower():
                try:
                    parts = line.split()
                    if len(parts) > 2:
                        results["displacement_m"] = TrackedValue(
                            value=float(parts[2]),
                            unit="m",
                            provenance=Provenance.MEASURED,
                        )
                except (ValueError, IndexError):
                    pass

        # Parse convergence data
        if "converged" in output.lower():
            results["converged"] = TrackedValue(
                value=True,
                unit=None,
                provenance=Provenance.MEASURED,
            )

        if "iterations" in output.lower():
            for line in output.split("\n"):
                line = line.strip()
                if "iter" in line.lower():
                    try:
                        import re
                        match = re.search(r"(\d+)", line)
                        if match:
                            results["num_iterations"] = TrackedValue(
                                value=int(match.group(1)),
                                unit=None,
                                provenance=Provenance.MEASURED,
                            )
                            break
                    except (ValueError, IndexError):
                        pass

        return results

    def cleanup(self, input_path: str) -> None:
        """Clean up MOOSE temporary files.

        Args:
            input_path: Path to the input file to clean up
        """
        base = Path(input_path).parent
        for ext in [".i", ".log", ".csv", ".png", ".vtk"]:
            for f in base.glob(f"*{ext}"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def get_capability(self) -> "SolverCapability":
        """Get MOOSE-specific capability information.

        Returns:
            SolverCapability with MOOSE-specific details
        """
        from chiplab.solvers.registry import SolverCapability, SolverSpec

        return SolverCapability(
            spec=SolverSpec(
                name=self.name,
                version=self._get_version(),
                executable=self.executable,
                fidelity=self.fidelity,
                supported_physics=self._get_supported_physics(),
                supported_scales=self._get_supported_scales(),
                input_formats=["moose_input"],
                output_formats=["field_data", "mesh_results", "coupled_outputs"],
            ),
            requires_gpu=self._requires_gpu(),
            supported_analysis_types=[
                "thermal_structural_coupling",
                "fluid_structural_coupling",
                "electrochemical",
                "multiphysics",
            ],
        )

    def _get_version(self) -> str:
        """Get MOOSE version."""
        try:
            result = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "MOOSE" in line:
                        import re
                        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return "3.18"

    def _requires_gpu(self) -> bool:
        """Check if MOOSE was built with GPU support."""
        try:
            result = subprocess.run(
                [self.executable, "-h"], capture_output=True, text=True
            )
            return "gpu" in result.stdout.lower() or "kokkos" in result.stdout.lower()
        except Exception:
            return False