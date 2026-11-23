"""Ramulator backend (DRAM memory-system simulator). Detected only; not integrated yet."""

from ..schemas.common import FidelityLevel
from .base import ExternalToolBackend


class RamulatorBackend(ExternalToolBackend):
    """Ramulator: `run` raises NotImplementedError until an adapter exists."""

    name: str = "ramulator"
    executable: str = "ramulator"
    display_name: str = "Ramulator"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
