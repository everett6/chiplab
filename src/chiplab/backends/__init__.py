"""Backends for ChipLab simulation runs."""

from .base import ExternalToolBackend, SimulatorBackend
from .ramulator import RamulatorBackend
from .scale_sim import ScaleSimBackend
from .accel_sim import AccelSimBackend
from .ecore import EcoreBackend, EcoreUnavailableError, apply_ecore_to_design

__all__ = ["ExternalToolBackend", "SimulatorBackend", "RamulatorBackend", "ScaleSimBackend",
           "AccelSimBackend", "EcoreBackend", "EcoreUnavailableError", "apply_ecore_to_design",
           "list_backends", "backend_status"]


def list_backends() -> list[str]:
    """Names of the known backends. Only ``ecore`` is integrated (array-level PIM only)."""
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
