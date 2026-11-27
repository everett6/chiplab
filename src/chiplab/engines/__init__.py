"""Engines for multiphysics simulation at three fidelity levels.

Exports:
    - ReducedOrderSolver       : Level 1 (analytical, fast)
    - SpatialMultiphysicsSolver: Level 2 (spatial grid, coupled PDEs)
    - ParticleScaleSolver      : Level 3 (particle-scale reference)
    - get_solver(fidelity_level): Factory function
"""

from __future__ import annotations

from typing import Literal

from chiplab.credibility import refuse_ferrofluid

from .material_solver import ReducedOrderSolver
from .reduced_order_solver import SpatialMultiphysicsSolver
from .multiphysics_solver import ParticleScaleSolver


def get_solver(
    fidelity_level: Literal["analytical", "discrete_event", "specialized"] | None = None,
    material: str | None = None,
) -> ReducedOrderSolver | SpatialMultiphysicsSolver | ParticleScaleSolver:
    """Factory function to get a solver for the given fidelity level.

    Parameters
    ----------
    fidelity_level : {"analytical", "discrete_event", "specialized"}, optional
        The desired fidelity level. If None, returns a ReducedOrderSolver
        (Level 1) by default.

    Returns
    -------
    ReducedOrderSolver | SpatialMultiphysicsSolver | ParticleScaleSolver
        An instance of the appropriate solver class.
    material : str, optional
        Ferrofluid is refused: these generic solvers have no ferrofluid physics.
    """
    refuse_ferrofluid("get_solver", material)
    if fidelity_level is None or fidelity_level == "analytical":
        return ReducedOrderSolver()
    elif fidelity_level == "discrete_event":
        # Level 2: spatial multiphysics
        return SpatialMultiphysicsSolver()
    elif fidelity_level == "specialized":
        # Level 3: particle-scale
        return ParticleScaleSolver()
    else:
        raise ValueError(
            f"Unknown fidelity level: {fidelity_level}. "
            f"Supported levels: analytical, discrete_event, specialized"
        )