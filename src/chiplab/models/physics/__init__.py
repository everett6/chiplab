"""Physics models module for chiplab.

This module provides industrial-grade physics models for RC delay,
IR-drop analysis, clock tree synthesis, 3D thermal analysis, and
SRAM yield/soft-error rate modeling.

Exports:
    rc_delay, insert_repeaters, wire_capacitance
    ir_drop_grid, static_ir_drop, dynamic_ir_drop
    clock_tree_power, clock_skew, ccsr_insertion_delay
    thermal_gradient_3d, inter_die_coupling, hotspot_analysis
    sram_yield, soft_error_rate, chip_yield
"""

from __future__ import annotations

from chiplab.schemas.common import TrackedValue, Provenance

__all__ = [
    "rc_delay",
    "insert_repeaters",
    "wire_capacitance",
    "ir_drop_grid",
    "static_ir_drop",
    "dynamic_ir_drop",
    "clock_tree_power",
    "clock_skew",
    "ccsr_insertion_delay",
    "thermal_gradient_3d",
    "inter_die_coupling",
    "hotspot_analysis",
    "sram_yield",
    "soft_error_rate",
    "chip_yield",
]

# Import functions from rc_extraction module
from chiplab.models.physics.rc_extraction import (
    rc_delay,
    insert_repeaters,
    wire_capacitance,
)

# Import functions from ir_drop module
from chiplab.models.physics.ir_drop import (
    ir_drop_grid,
    static_ir_drop,
    dynamic_ir_drop,
)

# Import functions from clock_tree module
from chiplab.models.physics.clock_tree import (
    clock_tree_power,
    clock_skew,
    ccsr_insertion_delay,
)

# Import functions from thermal_3d module
from chiplab.models.physics.thermal_3d import (
    thermal_gradient_3d,
    inter_die_coupling,
    hotspot_analysis,
)

# Import functions from yield_model module
from chiplab.models.physics.yield_model import (
    sram_yield,
    soft_error_rate,
    chip_yield,
)
