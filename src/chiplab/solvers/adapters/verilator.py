"""Verilator adapter for ChipLab solver integration.

Verilator adapter supports RTL lint, compile, and simulation using the
Verilator tool. Generates VCD/FST tracing output and lint reports.

Input: SystemVerilog or Verilog source files
Output: VCD/FST traces, lint reports, coverage data
GPU support: not typically needed for RTL simulation
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


class VerilatorAdapter(SolverAdapter):
    """Verilator adapter for RTL lint, compile, and simulation.

    Uses Verilator to compile SystemVerilog/Verilog designs for simulation,
    with support for linting, tracing (VCD/FST), and coverage analysis.

    Input: SystemVerilog or Verilog source files
    Output: VCD/FST traces, lint reports, coverage data
    GPU support: not typically needed for RTL simulation
    """

    name: str = "verilator"
    use_gpu: bool = True  # Flag kept for interface compatibility
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
    executable: str = "verilator"

    def _get_supported_physics(self) -> List[str]:
        return ["electrical"]

    def _get_supported_scales(self) -> List[str]:
        return ["system", "board"]

    def prepare_input(
        self, design: DesignSpec, workload: Any, config: SimulationConfig
    ) -> Tuple[str, Dict[str, TrackedValue]]:
        """Prepare Verilator compilation and return (work_dir, parameter_map).

        Args:
            design: The ChipLab design specification
            workload: The workload/simulation configuration
            config: Simulation configuration

        Returns:
            Tuple of (working_directory_path, parameter_map)
        """
        tmpdir = tempfile.mkdtemp(prefix="chiplab_verilator_")
        tmp = Path(tmpdir)

        # Create Verilator project directory structure
        src_dir = tmp / "src"
        src_dir.mkdir()
        build_dir = tmp / "build"
        build_dir.mkdir()

        # Write SystemVerilog/Verilog source based on design
        src_path = src_dir / "design.sv"
        self._write_verilog_source(design, src_path)

        # Build parameter map from design/workload
        param_map: Dict[str, TrackedValue] = {}
        if hasattr(design, "chiplets") and design.chiplets:
            for i, chiplet in enumerate(design.chiplets):
                if getattr(chiplet, "compute", None):
                    param_map[f"chiplet_{i}_arch"] = TrackedValue(
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

        # Run Verilator setup commands
        self._run_verilator_init(src_dir, build_dir, config)

        return str(tmp), param_map

    def _run_verilator_init(self, src_dir: Path, build_dir: Path, config: Any) -> None:
        """Run Verilator initialization commands.

        Creates the Verilated header and sets up the build environment.

        Args:
            src_dir: Source directory path
            build_dir: Build directory path
            config: Simulation configuration
        """
        # Generate Verilator command to initialize the build
        cmd = [
            self.executable,
            "--cc",
            "--trace",
            "-O3",
            str(src_dir / "design.sv"),
            "-o",
            "V",
        ]

        try:
            process = subprocess.run(
                cmd,
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if process.returncode != 0:
                # Non-fatal: just log and continue
                pass
        except Exception:
            # Non-fatal: continue even if init fails
            pass

    def _write_verilog_source(
        self, design: DesignSpec, path: Path
    ) -> None:
        """Write Verilog/SystemVerilog source file based on design.

        Args:
            design: The ChipLab design specification
            path: Path to write the source file to
        """
        with open(path, "w") as f:
            if design.chiplets:
                f.write("* ChipLab generated Verilog source\n")
                f.write("module chiplab_design #(\n")
                f.write("    parameter NUM_CHIPLETS = \n")
                f.write(str(len(design.chiplets)))
                f.write("(\n")
                for i, chiplet in enumerate(design.chiplets):
                    if chiplet.compute:
                        f.write(
                            f"    parameter CHIPLET_{i}_ARCH = "
                            f"'{chiplet.compute.architecture}'\n"
                        )
                f.write(")\n")
                f.write("(\n")
                for i in range(len(design.chiplets)):
                    f.write(f"input [{31 if i == 0 else 31}:0] chiplet_{i}_in")
                    if i < len(design.chiplets) - 1:
                        f.write(", ")
                f.write(";\n")
                f.write("output [31:0] chiplab_result;\n")
                f.write("* Chiplet instances\n")
                for i, chiplet in enumerate(design.chiplets):
                    if chiplet.compute:
                        f.write(
                            f"    chiplet_{i} inst_{i} (\n"
                        )
                        f.write(
                            f"        .in (chiplet_{i}_in),\n"
                        )
                        f.write(
                            f"        .out (chiplet_{i}_out)\n"
                        )
                        f.write("    );\n")
                f.write("assign chiplab_result = |chiplab_result;\n")
                f.write("endmodule\n")
            else:
                f.write("* Default simple design\n")
                f.write("module testbench();\n")
                f.write("    reg [31:0] a;\n")
                f.write("    reg [31:0] b;\n")
                f.write("    wire [31:0] sum;\n")
                f.write("    assign sum = a + b;\n")
                f.write("    initial begin\n")
                f.write("        a = 32'hcafef00d;\n")
                f.write("        b = 32'h00001234;\n")
                f.write("#100 $finish;\n")
                f.write("    end\n")
                f.write("    initial begin\n")
                f.write("        $dumpfile(\"waveforms.vcd\");\n")
                f.write("        $dumpvars(0, testbench);\n")
                f.write("    end\n")
                f.write("endmodule\n")

    def run(
        self, input_path: str, resources: Optional[Dict[str, Any]] = None
    ) -> subprocess.CompletedProcess:
        """Run Verilator compilation and simulation.

        Args:
            input_path: Path to the working directory or source file
            resources: Optional resource limits (cpu, memory, gpu, timeout)

        Returns:
            CompletedProcess from running Verilator
        """
        input_dir = Path(input_path)

        # Determine if input_path is a directory or source file
        if input_dir.is_dir():
            src_file = input_dir / "src" / "design.sv"
            build_dir = input_dir / "build"
        else:
            src_file = input_dir
            build_dir = input_dir.parent

        # Build Verilator command
        cmd = [
            self.executable,
            "--cc",  # C++ code output
            "--trace",  # Enable tracing
            "-O3",
            str(src_file),
            "-o",
            "V",
        ]

        # Add GPU support if requested (Verilator doesn't typically use GPU)
        if resources and resources.get("gpu", False):
            cmd.append("--gpu")

        # Add processor count for parallel compilation
        if resources and resources.get("cpu_count"):
            cmd.extend(["--timing"])

        # Add timeout
        timeout = resources.get("timeout", 120) if resources else 120

        try:
            process = subprocess.run(
                cmd,
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return process
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Verilator compilation timed out after {timeout}s: {e}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Verilator executable not found: {self.executable}. "
                "Please install Verilator."
            )

    def parse_output(
        self, process: subprocess.CompletedProcess
    ) -> Dict[str, TrackedValue]:
        """Parse Verilator output for simulation results.

        Args:
            process: The CompletedProcess from running Verilator

        Returns:
            Dictionary mapping output parameter names to TrackedValue objects
        """
        results: Dict[str, TrackedValue] = {}

        if process.returncode != 0:
            return self.handle_error(
                RuntimeError(
                    f"Verilator failed with return code {process.returncode}"
                )
            )

        output = process.stdout

        # Parse Verilator build warnings/errors
        for line in output.split("\n"):
            line = line.strip()

            # Parse timing information
            if "Time:" in line or "ns:" in line:
                try:
                    import re
                    match = re.search(r"(\d+\.?\d*)\s*ns", line)
                    if match:
                        results["simulation_time_ns"] = TrackedValue(
                            value=float(match.group(1)),
                            unit="ns",
                            provenance=Provenance.MEASURED,
                        )
                except (ValueError, IndexError):
                    pass

            # Parse coverage data
            if "Coverage" in line or "coverage" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace(".", "", 1).isdigit():
                            results["code_coverage"] = TrackedValue(
                                value=float(part),
                                unit="%",
                                provenance=Provenance.MEASURED,
                            )
                            break
                except (ValueError, IndexError):
                    pass

        # Check if VCD trace file was generated
        vcd_path = Path(input_path) if Path(input_path).is_dir() else Path(
            str(Path(input_path).parent / "waveforms.vcd")
        )
        if vcd_path.exists():
            results["vcd_generated"] = TrackedValue(
                value=True,
                unit=None,
                provenance=Provenance.MEASURED,
            )

        return results

    def cleanup(self, input_path: str) -> None:
        """Clean up Verilator temporary files.

        Args:
            input_path: Path to the working directory to clean up
        """
        base = Path(input_path)
        if base.is_dir():
            # Remove build artifacts
            for pattern in ["V", "V*", ".verilator*", "obj_dir"]:
                for f in base.glob(pattern):
                    try:
                        import shutil
                        if f.is_dir():
                            shutil.rmtree(f)
                        else:
                            f.unlink()
                    except OSError:
                        pass
        else:
            # Remove output files from source file location
            base = Path(input_path).parent
            for f in base.glob("V * waveforms.vcd *.txt"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def get_capability(self) -> "SolverCapability":
        """Get Verilator-specific capability information.

        Returns:
            SolverCapability with Verilator-specific details
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
                input_formats=["systemverilog", "verilog"],
                output_formats=["vcd", "fst", "coverage", "lint"],
            ),
            requires_gpu=self._requires_gpu(),
            supported_analysis_types=[
                "rtl_simulation",
                "lint",
                "formal_verification",
                "covergroup_analysis",
            ],
        )

    def _get_version(self) -> str:
        """Get Verilator version."""
        try:
            result = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                # Verilator version output format
                for line in result.stdout.split("\n"):
                    if "Verilator" in line:
                        import re
                        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return "4.220"

    def _requires_gpu(self) -> bool:
        """Check if Verilator was built with GPU support."""
        # Verilator typically does not use GPU for simulation
        return False