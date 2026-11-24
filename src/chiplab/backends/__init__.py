"""Backends for ChipLab simulation runs."""

from .base import ExternalToolBackend, RunnerBackend, SimulatorBackend
from .ramulator import RamulatorBackend
from .scale_sim import ScaleSimBackend
from .accel_sim import AccelSimBackend
from .ecore import EcoreBackend, EcoreUnavailableError, apply_ecore_to_design

__all__ = ["ExternalToolBackend", "RunnerBackend", "SimulatorBackend", "RamulatorBackend",
           "ScaleSimBackend",
           "AccelSimBackend", "EcoreBackend", "EcoreUnavailableError", "apply_ecore_to_design",
           "list_backends", "backend_status"]


def list_backends() -> list[str]:
    """Names of the known backends. ramulator, scale_sim and ecore are integrated (usable when
    installed); accel_sim is detected only."""
    return [
        "ramulator",
        "scale_sim",
        "accel_sim",
        "ecore",
    ]

def backend_status() -> list[dict[str, object]]:
    """Installed / integrated / usable flags for every backend."""
    return [b().status() for b in (RamulatorBackend, ScaleSimBackend, AccelSimBackend,
                                   EcoreBackend)]
