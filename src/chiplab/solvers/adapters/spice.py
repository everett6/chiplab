"""SPICE adapter for ChipLab solver integration.

SPICE adapter supports ngspice and Xyce for circuit simulation including
transient and AC analysis.

Input: SPICE netlist (netlist file or spice file)
Output: waveforms, csv, measurements
GPU support: optional acceleration via ngspice/gpu or xyce backends
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


class SpiceAdapter(SolverAdapter):
    """SPICE adapter for ngspice and Xyce circuit simulation.

    Supports transient analysis, AC analysis, and various SPICE analyses.

    Input: SPICE netlist file
    Output: waveforms, csv, measurements
    GPU support: optional backend acceleration
    """

    name: str = "spice"
    use_gpu: bool = True
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
    executable: str = "ngspice"

    def _get_supported_physics(self) -> List[str]:
        return ["electrical"]

    def _get_supported_scales(self) -> List[str]:
        return ["chiplet", "system", "board"]

    def prepare_input(
        self, design: DesignSpec, workload: Any, config: SimulationConfig
    ) -> Tuple[str, Dict[str, TrackedValue]]:
        """Prepare SPICE netlist and return (netlist_path, parameter_map).

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            config: Simulation configuration

        Returns:
            Tuple of (netlist_file_path, parameter_map)
        """
        tmpdir = tempfile.mkdtemp(prefix="chiplab_spice_")
        tmp = Path(tmpdir)

        # Generate SPICE netlist based on design
        netlist_path = tmp / "circuit.net"
        self._write_spice_netlist(design, workload, netlist_path)

        # Build parameter map from design/workload
        param_map: Dict[str, TrackedValue] = {}
        if hasattr(design, "chiplets") and design.chiplets:
            for chiplet in design.chiplets:
                if getattr(chiplet, "compute", None):
                    param_map["compute_type"] = TrackedValue(
                        value=chiplet.compute.architecture,
                        unit=None,
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

        return str(netlist_path), param_map

    def _write_spice_netlist(
        self, design: DesignSpec, workload: Any, path: Path
    ) -> None:
        """Write SPICE netlist file based on design.

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            path: Path to write the netlist to
        """
        with open(path, "w") as f:
            f.write("* ChipLab generated SPICE netlist\n")
            f.write(".title ChipLab Circuit Simulation\n")

            if design.chiplets:
                f.write("* Chiplets and components\n")
                for i, chiplet in enumerate(design.chiplets):
                    f.write(
                        f"Xchiplet{i} chiplet_{i} "
                        f"param={chiplet.area_mm2} "
                        f"tdp={chiplet.tdp_watts}\n"
                    )
            else:
                f.write("* Default inverter\n")
                f.write("Vin in 0 sin(0 1n 1n 0 0)\n")
                f.write("R1 in out 1k\n")
                f.write("C1 out 0 1p\n")
                f.write("Xinv in out inv_1\n")
                f.write(".model inv_1 nmos(Vth=0.7 Rs=10)\n")

            f.write(".analysis transient 0 1n 1n\n")
            f.write(".analysis ac decade 1Hz 1GHz\n")
            f.write(".probe V(out) I(Vin)\n")
            f.write(".end\n")

    def run(
        self, input_path: str, resources: Optional[Dict[str, Any]] = None
    ) -> subprocess.CompletedProcess:
        """Run SPICE simulation.

        Args:
            input_path: Path to the SPICE netlist file
            resources: Optional resource limits (cpu, memory, gpu, timeout)

        Returns:
            CompletedProcess from running the SPICE simulator
        """
        cmd = [self.executable, input_path]

        # Add timeout
        timeout = resources.get("timeout", 60) if resources else 60

        # Add GPU support if requested
        if resources and resources.get("gpu", False):
            if self.executable == "ngspice":
                cmd.extend(["-gpu", "1"])
            elif self.executable == "xyce":
                cmd.extend(["-backend", "openmp"])

        # Add processor count
        if resources and resources.get("cpu_count"):
            cmd.extend(["-cpu", str(resources["cpu_count"])])

        try:
            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return process
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"SPICE simulation timed out after {timeout}s: {e}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"SPICE executable not found: {self.executable}. "
                "Please install ngspice or xyce."
            )

    def parse_output(
        self, process: subprocess.CompletedProcess
    ) -> Dict[str, TrackedValue]:
        """Parse SPICE output for simulation results.

        Args:
            process: The CompletedProcess from running SPICE

        Returns:
            Dictionary mapping output parameter names to TrackedValue objects
        """
        results: Dict[str, TrackedValue] = {}

        if process.returncode != 0:
            return self.handle_error(
                RuntimeError(f"SPICE failed with return code {process.returncode}")
            )

        output = process.stdout

        # Parse transient analysis results
        for line in output.split("\n"):
            line = line.strip()

            # Parse voltage/current measurements
            if "V(" in line and "I(" not in line:
                try:
                    # Format might include time and value
                    parts = line.split()
                    if len(parts) > 2:
                        val = float(parts[-1])
                        # Extract node name from V(node)
                        if "V(" in line:
                            node = line.split("V(")[1].split(")")[0]
                            results[f"voltage_{node}"] = TrackedValue(
                                value=val,
                                unit="V",
                                provenance=Provenance.MEASURED,
                            )
                except (ValueError, IndexError):
                    pass

            if "I(" in line:
                try:
                    parts = line.split()
                    if len(parts) > 2:
                        val = float(parts[-1])
                        results["current_measured"] = TrackedValue(
                            value=val,
                            unit="A",
                            provenance=Provenance.MEASURED,
                        )
                except (ValueError, IndexError):
                    pass

            # Parse AC analysis data
            if "AC" in line.upper() and "|" in line:
                try:
                    parts = line.split()
                    if len(parts) > 2:
                        mag = float(parts[1]) if len(parts) > 1 else 0.0
                        results["ac_magnitude"] = TrackedValue(
                            value=mag,
                            unit="dB",
                            provenance=Provenance.MEASURED,
                        )
                except (ValueError, IndexError):
                    pass

        # Parse CSV-like output
        if "time" in output.lower() and "voltage" in output.lower():
            # Look for data lines
            for line in output.split("\n"):
                line = line.strip()
                if line and not line.startswith(".") and not line.startswith(
                    "*"
                ):
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            freq = float(parts[0]) if "GHz" not in line else float(
                                parts[0].replace("GHz", "")
                            )
                            results["frequency_ghz"] = TrackedValue(
                                value=freq,
                                unit="GHz",
                                provenance=Provenance.MEASURED,
                            )
                        except ValueError:
                            pass
                        break

        return results

    def cleanup(self, input_path: str) -> None:
        """Clean up SPICE temporary files.

        Args:
            input_path: Path to the netfile to clean up
        """
        base = Path(input_path).parent
        for f in base.glob("*.net *.lis *.raw"):
            try:
                f.unlink()
            except OSError:
                pass

    def get_capability(self) -> "SolverCapability":
        """Get SPICE-specific capability information.

        Returns:
            SolverCapability with SPICE-specific details
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
                input_formats=["netlist", "spice file"],
                output_formats=["waveforms", "csv", "measurements"],
            ),
            requires_gpu=self._requires_gpu(),
            supported_analysis_types=[
                "transient_analysis",
                "ac_analysis",
                "dc_analysis",
                "sweep_analysis",
            ],
        )

    def _get_version(self) -> str:
        """Get SPICE simulator version."""
        try:
            result = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "ngspice" in line.lower() or "xyce" in line.lower():
                        import re
                        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return "3.5"

    def _requires_gpu(self) -> bool:
        """Check if SPICE simulator was built with GPU support."""
        try:
            result = subprocess.run(
                [self.executable, "-h"], capture_output=True, text=True
            )
            return "gpu" in result.stdout.lower() or "opencl" in result.stdout.lower()
        except Exception:
            return False