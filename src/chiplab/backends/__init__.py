"""Backends for ChipLab simulation runs."""

from .base import SimulatorBackend
from .ramulator import RamulatorBackend
from .scale_sim import ScaleSimBackend
from .accel_sim import AccelSimBackend


def list_backends() -> list[str]:
    """Return a list of available backend names."""
    return [
        "ramulator",
        "scale_sim",
        "accel_sim",
    ]