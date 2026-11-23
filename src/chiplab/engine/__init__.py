"""ChipLab engine module - simulation and analysis functions."""

from __future__ import annotations

from . import sweep
from . import analyzer
from ..backends import list_backends, SimulatorBackend, RamulatorBackend, ScaleSimBackend, AccelSimBackend

from .sweep import run_parameter_sweep
from .analyzer import analyze_bottleneck, compare_results

from ..backends.base import SimulatorBackend
from ..backends.ramulator import RamulatorBackend
from ..backends.scale_sim import ScaleSimBackend
from ..backends.accel_sim import AccelSimBackend

from ..schemas.design import DesignSpec
from ..schemas.simulation import (
    SimulationConfig,
    SimulationResult,
    SweepResult,
    ComparisonResult,
    SweepPoint,
    BottleneckReport,
)
from ..schemas.workload import WorkloadSpec


from .validator import validate_design
from .scheduler import schedule_operators
from .analytical import run_analytical_simulation
from .discrete_event import run_discrete_event_simulation