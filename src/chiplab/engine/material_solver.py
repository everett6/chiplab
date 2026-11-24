"""Level 1: Reduced-order mathematical model for magnetic device simulation.

Re-exports from chiplab.engines.material_solver for compatibility
with the chiplab.engine package namespace.
"""

from __future__ import annotations

from chiplab.engines.material_solver import ReducedOrderSolver  # noqa: F401

__all__ = ["ReducedOrderSolver"]