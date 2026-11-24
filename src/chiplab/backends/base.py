"""Base simulator backend for ChipLab."""

from __future__ import annotations

import abc
import shutil
from typing import Any

from ..schemas.design import DesignSpec
from ..schemas.simulation import SimulationConfig, SimulationResult
from ..schemas.common import FidelityLevel


class SimulatorBackend(abc.ABC):
    """Abstract base class for all simulator backends."""

    name: str
    fidelity: FidelityLevel

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the external simulator is available/installable."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig) -> SimulationResult:
        """Run a simulation and return the result."""
        raise NotImplementedError

class ExternalToolBackend(SimulatorBackend):
    """A third-party simulator ChipLab can detect but has no adapter for yet.

    ``run`` always raises: it must never hand back an empty result marked successful.
    ``is_available`` only reports whether the executable is on PATH; ``integrated`` stays
    False until a real input-generation / output-parsing adapter exists.
    """

    executable: str
    display_name: str
    integrated: bool = False

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def status(self) -> dict[str, Any]:
        return {"name": self.name, "installed": self.is_available(),
                "integrated": self.integrated, "usable": self.integrated and self.is_available()}

    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig) -> SimulationResult:
        if not self.is_available():
            raise NotImplementedError(
                f"{self.display_name} is not installed ('{self.executable}' not on PATH).")
        raise NotImplementedError(
            f"{self.display_name} is installed, but ChipLab has no adapter for it yet "
            "(no config generation or output parsing). Use fidelity='analytical' or "
            "'discrete_event'.")


class RunnerBackend(SimulatorBackend):
    """A third-party simulator ChipLab drives through a fixed runner script.

    The runner (``_<name>_runner.py`` next to this file) is executed with the tool's own
    Python interpreter as ``[python, "-I", runner, request.json, result.json]``: an argument
    list, never a shell, and no MCP argument can choose the interpreter, the script or a
    path. The interpreter location is fixed by the server operator (environment variable
    ``env_var``, else ``default_python``).
    """

    display_name: str
    runner: str
    env_var: str
    default_python: str
    probe_module: str
    integrated: bool = True
    timeout_s: float = 300.0

    def python(self) -> str:
        import os
        return os.path.expanduser(os.environ.get(self.env_var) or self.default_python)

    def unavailable_reason(self) -> str | None:
        return _probe(self.python(), self.probe_module)

    def is_available(self) -> bool:
        return self.unavailable_reason() is None

    def status(self) -> dict[str, Any]:
        reason = self.unavailable_reason()
        st: dict[str, Any] = {"name": self.name, "installed": reason is None,
                              "integrated": self.integrated, "usable": reason is None}
        if reason:
            st["reason"] = reason
        return st

    def call_runner(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run the fixed runner on ``request``; raise RuntimeError on any failure."""
        import json
        import subprocess
        import tempfile
        from pathlib import Path

        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"{self.display_name} unavailable: {reason}")
        runner = Path(__file__).with_name(self.runner)
        with tempfile.TemporaryDirectory(prefix=f"chiplab_{self.name}_") as tmp:
            req, out = Path(tmp, "request.json"), Path(tmp, "result.json")
            req.write_text(json.dumps(request))
            proc = subprocess.run(
                [self.python(), "-I", str(runner), str(req), str(out)],
                cwd=tmp, capture_output=True, text=True, timeout=self.timeout_s,
                check=False, shell=False)
            if proc.returncode != 0 or not out.is_file():
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
                raise RuntimeError(f"{self.display_name} failed (exit {proc.returncode}): "
                                   + " | ".join(tail))
            return json.loads(out.read_text())

    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig) -> SimulationResult:
        """Whole-design run: the specialized engine combines this tool with the others."""
        from ..engine.specialized import run_specialized_simulation
        return run_specialized_simulation(design, workload, config)


_PROBE_CACHE: dict[tuple[str, str], str | None] = {}


def _probe(python: str, module: str) -> str | None:
    """None if ``python -I -c 'import <module>'`` works, else the reason (cached)."""
    import os
    import subprocess

    key = (python, module)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    if not os.path.isfile(python) or not os.access(python, os.X_OK):
        reason: str | None = f"interpreter not found: {python}"
    else:
        try:
            p = subprocess.run([python, "-I", "-c", f"import {module}"], capture_output=True,
                               text=True, timeout=60, check=False, shell=False)
            reason = None if p.returncode == 0 else (
                f"'import {module}' failed: "
                + ((p.stderr or "").strip().splitlines() or ["?"])[-1])
        except (OSError, subprocess.TimeoutExpired) as e:
            reason = f"probe failed: {e}"
    _PROBE_CACHE[key] = reason
    return reason
