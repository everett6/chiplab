"""Multiphysics coupling module for chip lab.

Implements true multiphysics coupling with feedback loops between
electromagnetic, thermal, particle dynamics, and mechanical fields.
"""

from __future__ import annotations

from typing import Any

# Field equations
from .fields import (
    FieldSystem,
    MaxwellEquations,
    HeatEquation,
    ParticleDynamics,
    FluidDynamics,
    MechanicalEquations,
    ConstitutiveRelations,
)

# Coupling management
from .coupling import (
    CouplingGraph,
    FeedbackLoop,
    CouplingScheme,
    StaggeredScheme,
    MonolithicScheme,
    QuasiStaticScheme,
    ConvergenceMonitor,
)

# Solver
from .solver import MultiphysicsSolver

# Ferrofluid coupling
from .ferrofluid_coupling import (
    FerrofluidCoupling,
    FerrofluidMaterialModel,
)

__all__ = [
    "FieldSystem",
    "MaxwellEquations",
    "HeatEquation",
    "ParticleDynamics",
    "FluidDynamics",
    "MechanicalEquations",
    "ConstitutiveRelations",
    "CouplingGraph",
    "FeedbackLoop",
    "CouplingScheme",
    "StaggeredScheme",
    "MonolithicScheme",
    "QuasiStaticScheme",
    "ConvergenceMonitor",
    "MultiphysicsSolver",
    "FerrofluidCoupling",
    "FerrofluidMaterialModel",
]