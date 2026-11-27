"""LAMMPS adapter for ChipLab solver integration."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chiplab.schemas.common import TrackedValue, Provenance
from chiplab.schemas.simulation import SimulationConfig, SimulationResult
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.common import FidelityLevel

from chiplab.solvers.registry import SolverSpec, SolverCapability
from .base import SolverAdapter


class LAMMPSAdapter(SolverAdapter):
    """LAMMPS adapter for particle dynamics and material properties.

    LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) is used for
    molecular dynamics, materials simulation, and particle dynamics.

    Input: data file + input script (particle dynamics, material properties)
    Output: trajectories, thermodynamic data, custom computes
    GPU support: -pk kokkos or -sf gpu
    """

    name: str = "lammps"
    use_gpu: bool = True
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
    executable: str = "lammps"

    def _get_supported_physics(self) -> List[str]:
        return ["mechanical", "thermal", "magnetic", "particle"]

    def _get_supported_scales(self) -> List[str]:
        return ["device", "chiplet"]

    def prepare_input(
        self, design: DesignSpec, workload: Any, config: SimulationConfig
    ) -> Tuple[str, Dict[str, TrackedValue]]:
        """Prepare LAMMPS input data file and script.

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            config: Simulation configuration

        Returns:
            Tuple of (input_file_path, parameter_map)
        """
        # Create temporary directory for LAMMPS files
        with tempfile.TemporaryDirectory(prefix="chiplab_lammps_") as tmpdir:
            tmp = Path(tmpdir)

            # Generate LAMMPS data file based on design
            data_path = tmp / "system.data"
            self._write_lammps_data(design, data_path)

            # Generate LAMMPS input script
            script_path = tmp / "simulation.in"
            self._write_lammps_script(design, workload, config, script_path)

            # Build parameter map from design/workload
            param_map: Dict[str, TrackedValue] = {}
            # Extract key parameters from design
            if hasattr(design, 'chiplets') and design.chiplets:
                for chiplet in design.chiplets:
                    if chiplet.compute:
                        param_map["compute_type"] = TrackedValue(
                            value=chiplet.compute.architecture,
                            unit=None,
                            provenance=Provenance.ASSUMED,
                        )

            return str(script_path), param_map

    def _write_lammps_data(self, design: DesignSpec, path: Path) -> None:
        """Write LAMMPS data file."""
        # Simplified data file generation
        with open(path, "w") as f:
            f.write("LAMMPS Data File\n\n")
            f.write(f"atoms   {len(design.chiplets) if design.chiplets else 1} atom type\n\n")
            f.write("0.0 1.0 xlo xhi\n")
            f.write("0.0 1.0 ylo yhi\n")
            f.write("0.0 1.0 zlo zhi\n\n")
            f.write("Masses\n\n")
            f.write("1 1.0\n\n")  # Placeholder mass
            f.write("Atoms\n\n")
            # Write atom positions based on design
            if design.chiplets:
                for i, chiplet in enumerate(design.chiplets):
                    f.write(f"{i+1} {chiplet.area_mm2 * 100:.1f} {chiplet.tdp_watts * 100:.1f} {chiplet.process_nm if chiplet.process_nm else 7:.1f}\n")

    def _write_lammps_script(
        self, design: DesignSpec, workload: Any, config: SimulationConfig,
        script_path: Path
    ) -> None:
        """Write LAMMPS input script."""
        with open(script_path, "w") as f:
            f.write("units       atomic\n\n")
            f.write("dimension   3\n\n")
            f.write("boundary    p p p\n\n")
            f.write("mass        1 1.0\n\n")
            f.write("timestep    0.001\n\n")
            if config.fidelity == FidelityLevel.SPECIALIZED:
                f.write("run         100\n\n")
            elif config.fidelity == FidelityLevel.DISCRETE_EVENT:
                f.write("run         10\n\n")
            else:
                f.write("run         10\n\n")

    def run(
        self, input_path: str, resources: Optional[Dict[str, Any]] = None
    ) -> subprocess.CompletedProcess:
        """Run LAMMPS simulation.

        Args:
            input_path: Path to the LAMMPS input script
            resources: Optional resource limits (cpu, memory, gpu, timeout)

        Returns:
            CompletedProcess from running LAMMPS
        """
        cmd = [self.executable, input_path]

        # Add GPU support if requested and available
        if resources and resources.get("gpu", False):
            cmd.extend(["-pk", "kokkos", "-sf", "gpu"])

        # Add processor count
        if resources and resources.get("cpu_count"):
            cmd.extend(["-p", str(resources["cpu_count"])])

        # Add timeout
        timeout = resources.get("timeout", 300) if resources else 300

        try:
            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return process
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"LAMMPS simulation timed out after {timeout}s")
        except FileNotFoundError:
            raise RuntimeError(f"LAMMPS executable not found: {self.executable}")

    def parse_output(
        self, process: subprocess.CompletedProcess
    ) -> Dict[str, TrackedValue]:
        """Parse LAMMPS output for thermodynamic data.

        Args:
            process: The CompletedProcess from running LAMMPS

        Returns:
            Dictionary mapping output parameter names to TrackedValue objects
        """
        results: Dict[str, TrackedValue] = {}

        if process.returncode != 0:
            return self.handle_error(RuntimeError(
                f"LAMMPS failed with return code {process.returncode}"
            ))

        output = process.stdout

        # Parse thermodynamic output
        for line in output.split("\n"):
            line = line.strip()
            if "Total energy" in line:
                try:
                    # Format: Step             Energy     ...
                    parts = line.split()
                    energy_val = float(parts[2]) if len(parts) > 2 else 0.0
                    results["total_energy_j"] = TrackedValue(
                        value=energy_val,
                        unit="J",
                        provenance=Provenance.MEASURED,
                    )
                except (ValueError, IndexError):
                    pass
            if "Temperature" in line and "Timestep" not in line:
                try:
                    parts = line.split()
                    temp = float(parts[2]) if len(parts) > 2 else 300.0
                    results["temperature_k"] = TrackedValue(
                        value=temp,
                        unit="K",
                        provenance=Provenance.MEASURED,
                    )
                except (ValueError, IndexError):
                    pass
            if "Pressure" in line:
                try:
                    parts = line.split()
                    pressure = float(parts[2]) if len(parts) > 2 else 0.0
                    results["pressure_pa"] = TrackedValue(
                        value=pressure,
                        unit="Pa",
                        provenance=Provenance.MEASURED,
                    )
                except (ValueError, IndexError):
                    pass

        # Parse atom counts if present
        if "atoms" in output.lower():
            results["atom_count"] = TrackedValue(
                value=len(output.split("\n")),
                unit=None,
                provenance=Provenance.MEASURED,
            )

        return results

    def cleanup(self, input_path: str) -> None:
        """Clean up LAMMPS temporary files.

        Args:
            input_path: Path to the input file to clean up
        """
        import glob
        base = Path(input_path).parent
        for f in base.glob("*.data *.in *.log"):
            try:
                f.unlink()
            except OSError:
                pass

    def get_capability(self) -> SolverCapability:
        """Get LAMMPS-specific capability information."""
        return SolverCapability(
            spec=SolverSpec(
                name=self.name,
                version=self._get_version(),
                executable=self.executable,
                fidelity=self.fidelity,
                supported_physics=self._get_supported_physics(),
                supported_scales=self._get_supported_scales(),
                input_formats=["data file", "input script"],
                output_formats=["trajectories", "thermodynamic data", "custom computes"],
            ),
            requires_gpu=self._requires_gpu(),
            supported_analysis_types=["molecular_dynamics", "materials_properties",
                                      "thermal_conduction", "mechanical_deformation"],
        )

    def _get_version(self) -> str:
        """Get LAMMPS version."""
        try:
            result = subprocess.run(
                [self.executable, "-version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                # Parse version from output
                for line in result.stdout.split("\n"):
                    if "LAMMPS" in line:
                        import re
                        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return "01 Mar 2024"

    def _requires_gpu(self) -> bool:
        """Check if LAMMPS was built with GPU support."""
        try:
            result = subprocess.run(
                [self.executable, "-h"], capture_output=True, text=True
            )
            return "kokkos" in result.stdout.lower() or "gpu" in result.stdout.lower()
        except Exception:
            return False
