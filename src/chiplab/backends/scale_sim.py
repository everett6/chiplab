"""SCALE-Sim backend (systolic-array accelerator simulator). Detected only; not integrated yet."""

from ..schemas.common import FidelityLevel
from .base import ExternalToolBackend


class ScaleSimBackend(ExternalToolBackend):
    """SCALE-Sim: `run` raises NotImplementedError until an adapter exists."""

    name: str = "scale_sim"
    executable: str = "scale_sim"
    display_name: str = "SCALE-Sim"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
