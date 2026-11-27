"""OpenCFS adapter for ChipLab solver integration.

OpenCFS adapter for coupled multiphysics FEM simulations using the openCFS
framework. Supports structural, thermal, and fluid flow simulations with
coupling between physics domains.

Input: openCFS input files and configurations
Output: Coupled field data, mesh results, solution outputs
GPU support: optional via OpenCL or CUDA acceleration
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


class OpenCFSAdapter(SolverAdapter):
    """OpenCFS adapter for coupled multiphysics FEM simulations.

    OpenCFS (open Coupled Field Simulator) is an open-source framework for
    coupled multiphysics simulations including thermal-structural,
    fluid-structural, and electromagnetics-thermal couplings.

    Input: openCFS XML/JSON input files
    Output: Coupled field data, mesh results, solution outputs
    GPU support: optional via OpenCL/CUDA
    """

    name: str = "opencfs"
    use_gpu: bool = True
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
    executable: str = "opencfs"

    def _get_supported_physics(self) -> List[str]:
        return ["thermal", "structural", "mechanical", "fluid", "electromagnetics"]

    def _get_supported_scales(self) -> List[str]:
        return ["device", "chiplet", "system"]

    def prepare_input(
        self, design: DesignSpec, workload: Any, config: SimulationConfig
    ) -> Tuple[str, Dict[str, TrackedValue]]:
        """Prepare OpenCFS input file and return (input_path, parameter_map).

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            config: Simulation configuration

        Returns:
            Tuple of (input_file_path, parameter_map)
        """
        tmpdir = tempfile.mkdtemp(prefix="chiplab_opencfs_")
        tmp = Path(tmpdir)

        # Generate OpenCFS input file based on design
        input_path = tmp / "simulation.xml"
        self._write_opencfs_input(design, workload, config, input_path)

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
                    param_map["thermal_power"] = TrackedValue(
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

    def _write_opencfs_input(
        self, design: DesignSpec, workload: Any, config: SimulationConfig,
        path: Path
    ) -> None:
        """Write OpenCFS input file based on design.

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            config: Simulation configuration
            path: Path to write the OpenCFS input file to
        """
        physics_types = self._get_supported_physics()

        with open(path, "w") as f:
            f.write("<?xml version=\"1.0\"?>\n")
            f.write("<cfs_solver>\n")
            f.write("  <solver_name>OpenCFS</solver_name>\n")
            f.write("  <start_time>0.0</start_time>\n")
            f.write("  <stop_time>100.0</stop_time>\n")
            f.write("  <time_step>0.1</time_step>\n")
            f.write("  <gravity>0.0 0.0 -9.81</gravity>\n")
            f.write("\n")

            # Physics couplings
            f.write("  <couplings>\n")
            if "thermal" in physics_types:
                f.write("    <thermal>\n")
                f.write("      <type>heat</type>\n")
                if design.chiplets:
                    for i, chiplet in enumerate(design.chiplets):
                        compute = getattr(chiplet, "compute", None)
                        if compute:
                            f.write(f"      <heat_power>{chiplet.tdp_watts}</heat_power>\n")
                f.write("    </thermal>\n")

            if "structural" in physics_types:
                f.write("    <structural>\n")
                f.write("      <type>structural</type>\n")
                if design.chiplets:
                    for i, chiplet in enumerate(design.chiplets):
                        compute = getattr(chiplet, "compute", None)
                        if compute:
                            f.write(f"      <thermal_coupling>true</thermal_coupling>\n")
                f.write("    </structural>\n")

            if "fluid" in physics_types:
                f.write("    <fluid>\n")
                f.write("      <type>fluid</type>\n")
                f.write("    </fluid>\n")

            f.write("  </couplings>\n")
            f.write("\n")

            # Materials section
            f.write("  <materials>\n")
            f.write("    <silicon>\n")
            f.write("      <youngs_modulus>130e9</youngs_modulus>\n")
            f.write("      <poissons_ratio>0.28</poissons_ratio>\n")
            f.write("      <density>2330</density>\n")
            f.write("      <thermal_conductivity>150</thermal_conductivity>\n")
            f.write("    </silicon>\n")
            f.write("  </materials>\n")
            f.write("\n")

            # Mesh and discretization
            f.write("  <mesh>\n")
            f.write("    <type>unstructured</type>\n")
            f.write("    <element_type>tetrahedral</element_type>\n")
            f.write("  </mesh>\n")
            f.write("\n")

            # Output configuration
            f.write("  <output>\n")
            f.write("    <csv>true</csv>\n")
            f.write("    <vtk>true</vtk>\n")
            f.write("    <field_output_interval>10</field_output_interval>\n")
            f.write("  </output>\n")
            f.write("</cfs_solver>\n")

    def run(
        self, input_path: str, resources: Optional[Dict[str, Any]] = None
    ) -> subprocess.CompletedProcess:
        """Run OpenCFS simulation.

        Args:
            input_path: Path to the OpenCFS input file
            resources: Optional resource limits (cpu, memory, gpu, timeout)

        Returns:
            CompletedProcess from running OpenCFS
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
                f"OpenCFS simulation timed out after {timeout}s: {e}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"OpenCFS executable not found: {self.executable}. "
                "Please install OpenCFS."
            )

    def parse_output(
        self, process: subprocess.CompletedProcess
    ) -> Dict[str, TrackedValue]:
        """Parse OpenCFS output for simulation results.

        Args:
            process: The CompletedProcess from running OpenCFS

        Returns:
            Dictionary mapping output parameter names to TrackedValue objects
        """
        results: Dict[str, TrackedValue] = {}

        if process.returncode != 0:
            return self.handle_error(
                RuntimeError(f"OpenCFS failed with return code {process.returncode}")
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
        if "converged" in output.lower() or "convergence" in output.lower():
            results["converged"] = TrackedValue(
                value=True,
                unit=None,
                provenance=Provenance.MEASURED,
            )

        if "iterations" in output.lower():
            for line in output.split("\n"):
                line = line.strip()
                if "iter" in line.lower() or "step" in line.lower():
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

        # Parse stress results
        von_mises_found = False
        for line in output.split("\n"):
            line = line.strip()
            if "von Mises" in line or "vonmises" in line.lower():
                von_mises_found = True
                try:
                    parts = line.split()
                    if len(parts) > 2:
                        results["von_mises_pa"] = TrackedValue(
                            value=float(parts[2]),
                            unit="Pa",
                            provenance=Provenance.MEASURED,
                        )
                except (ValueError, IndexError):
                    pass
        if von_mises_found:
            results["von_mises_found"] = TrackedValue(
                value=True,
                unit=None,
                provenance=Provenance.MEASURED,
            )

        return results

    def cleanup(self, input_path: str) -> None:
        """Clean up OpenCFS temporary files.

        Args:
            input_path: Path to the input file to clean up
        """
        base = Path(input_path).parent
        for ext in [".xml", ".log", ".csv", ".vtk", ".vtu"]:
            for f in base.glob(f"*{ext}"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def get_capability(self) -> "SolverCapability":
        """Get OpenCFS-specific capability information.

        Returns:
            SolverCapability with OpenCFS-specific details
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
                input_formats=["opencfs_input"],
                output_formats=["coupled_field_data", "mesh_results", "solution_outputs"],
            ),
            requires_gpu=self._requires_gpu(),
            supported_analysis_types=[
                "thermal_structural_coupling",
                "fluid_structural_coupling",
                "electromagnetics_thermal",
                "multiphysics",
            ],
        )

    def _get_version(self) -> str:
        """Get OpenCFS version."""
        try:
            result = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "OpenCFS" in line or "cfs" in line.lower():
                        import re
                        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return "1.0"

    def _requires_gpu(self) -> bool:
        """Check if OpenCFS was built with GPU support."""
        try:
            result = subprocess.run(
                [self.executable, "-h"], capture_output=True, text=True
            )
            return "gpu" in result.stdout.lower() or "opencl" in result.stdout.lower() or "cuda" in result.stdout.lower()
        except Exception:
            return False