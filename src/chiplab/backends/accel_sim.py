"""Accel-Sim backend (GPU simulator). Detected only; not integrated yet."""

from ..schemas.common import FidelityLevel
from .base import ExternalToolBackend


class AccelSimBackend(ExternalToolBackend):
    """Accel-Sim: `run` raises NotImplementedError until an adapter exists."""

    name: str = "accel_sim"
    executable: str = "accel_sim"
    display_name: str = "Accel-Sim"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
