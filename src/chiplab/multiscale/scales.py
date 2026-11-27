"""Scale definitions and hierarchy for multiscale modeling.

Defines the different scales in the multiscale hierarchy, from nanoparticle to system,
along with their physical characteristics, degrees of freedom, and physics models.
Provides the ScaleHierarchy class for parent-child relationships and upscaling/downscaling
operators, as well as the RepresentativeVolumeElement (RVE) for material→device upscaling.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Scale(Enum):
    """Enum representing the different scales in the multiscale hierarchy.

    NANOPARTICLE: Individual nanoparticle level, atomistic or particle-based simulations.
    MATERIAL: Material microstructure level, continuum with representative volume element.
    DEVICE: Device level, circuit or field solver on device geometry.
    MEMORY_ARRAY: Memory array level, DRAM/HBM/NAND array organization.
    CHIP: Chip level, full die with mixed-signal and power distribution.
    PACKAGE: Package level, chip-scale packaging with thermal and I/O effects.
    SYSTEM: System level, full system with multiple chips and interconnections.
    """

    NANOPARTICLE = "nanoparticle"
    MATERIAL = "material"
    DEVICE = "device"
    MEMORY_ARRAY = "memory_array"
    CHIP = "chip"
    PACKAGE = "package"
    SYSTEM = "system"


class ScaleDefinition:
    """Definition of a scale with its physical characteristics.

    Parameters
    ----------
    length_scale : float
        Characteristic length scale in meters.
    time_scale : float
        Characteristic time scale in seconds.
    dof_per_entity : int
        Degrees of freedom per entity at this scale.
    physics_models : list[str]
        List of physics models applicable at this scale.
    """

    def __init__(
        self,
        length_scale: float,
        time_scale: float,
        dof_per_entity: int,
        physics_models: List[str],
    ) -> None:
        self.length_scale = length_scale
        self.time_scale = time_scale
        self.dof_per_entity = dof_per_entity
        self.physics_models = physics_models

    def __repr__(self) -> str:
        return f"ScaleDefinition(length_scale={self.length_scale}, time_scale={self.time_scale}, dof={self.dof_per_entity})"


# Scale definitions for each level in the hierarchy
SCALE_DEFINITIONS: Dict[Scale, ScaleDefinition] = {
    Scale.NANOPARTICLE: ScaleDefinition(
        length_scale=1e-9,  # 1 nm
        time_scale=1e-12,  # 1 ps
        dof_per_entity=3,  # 3 positional degrees of freedom
        physics_models=["particle_dynamics", "quantum"],
    ),
    Scale.MATERIAL: ScaleDefinition(
        length_scale=1e-6,  # 1 µm
        time_scale=1e-9,  # 1 ns
        dof_per_entity=6,  # 3 position + 3 momentum/state
        physics_models=["continuum_mechanics", "electrodynamics"],
    ),
    Scale.DEVICE: ScaleDefinition(
        length_scale=1e-3,  # 1 mm
        time_scale=1e-6,  # 1 ms
        dof_per_entity=12,  # spatial + temporal degrees
        physics_models=["heat_equation", "charge_transport", "electromagnetics"],
    ),
    Scale.MEMORY_ARRAY: ScaleDefinition(
        length_scale=1e-2,  # 1 cm
        time_scale=1e-3,  # 1 ms
        dof_per_entity=8,  # array organization + state
        physics_models=["RC_circuit", "thermal"],
    ),
    Scale.CHIP: ScaleDefinition(
        length_scale=1e-2,  # 1 cm (die size)
        time_scale=1e-2,  # 10 ms
        dof_per_entity=20,  # full chip state
        physics_models=["power_distribution", "thermal", "electromagnetics"],
    ),
    Scale.PACKAGE: ScaleDefinition(
        length_scale=1e-1,  # 10 cm
        time_scale=1.0,  # 1 s
        dof_per_entity=30,  # package-level state
        physics_models=["thermal", "mechanical", "power"],
    ),
    Scale.SYSTEM: ScaleDefinition(
        length_scale=1.0,  # 1 m
        time_scale=100.0,  # 100 s
        dof_per_entity=50,  # system-level state
        physics_models=["system_thermal", "power_management", "reliability"],
    ),
}


class ScaleHierarchy:
    """Hierarchy managing parent-child relationships between scales.

    Provides upscaling (fine→coarse) and downscaling (coarse→fine) operators
    for transitioning between scales in multiscale simulation.

    The hierarchy defines:
    - Parent-child relationships between adjacent scales
    - Upscaling operators (projection from fine to coarse)
    - Downscaling operators (interpolation from coarse to fine)
    - RVE (Representative Volume Element) sizes for each transition
    """

    def __init__(self) -> None:
        # Define parent-child relationships: child → parent
        self._parent: Dict[Scale, Scale] = {
            Scale.NANOPARTICLE: Scale.MATERIAL,
            Scale.MATERIAL: Scale.DEVICE,
            Scale.DEVICE: Scale.MEMORY_ARRAY,
            Scale.MEMORY_ARRAY: Scale.CHIP,
            Scale.CHIP: Scale.PACKAGE,
            Scale.PACKAGE: Scale.SYSTEM,
        }

        # Define upscaling operators: fine → coarse mapping
        self._upscaling_operators: Dict[Tuple[Scale, Scale], str] = {
            (Scale.NANOPARTICLE, Scale.MATERIAL): "volume_averaging",
            (Scale.MATERIAL, Scale.DEVICE): "homogenization",
            (Scale.DEVICE, Scale.MEMORY_ARRAY): "effective_property_calculation",
            (Scale.MEMORY_ARRAY, Scale.CHIP): "submodeling",
            (Scale.CHIP, Scale.PACKAGE): "arlequin_method",
            (Scale.PACKAGE, Scale.SYSTEM): "system_decomposition",
        }

        # Define RVE sizes for material→device upscaling
        self._rve_sizes: Dict[Tuple[Scale, Scale], float] = {
            (Scale.MATERIAL, Scale.DEVICE): 1e-6,  # 1 µm RVE
        }

        # Define downscaling operators: coarse → fine mapping
        self._downscaling_operators: Dict[Tuple[Scale, Scale], str] = {
            (Scale.MATERIAL, Scale.NANOPARTICLE): "stochastic_initialization",
            (Scale.DEVICE, Scale.MATERIAL): "cell_problem_solve",
            (Scale.MEMORY_ARRAY, Scale.DEVICE): "subgrid_refinement",
            (Scale.CHIP, Scale.MEMORY_ARRAY): "chiplet_level_detail",
            (Scale.PACKAGE, Scale.CHIP): "chip_level_detail",
            (Scale.SYSTEM, Scale.PACKAGE): "partition_refinement",
        }

    def get_parent(self, scale: Scale) -> Scale:
        """Get the parent scale of the given scale.

        Parameters
        ----------
        scale : Scale
            The scale to find the parent of.

        Returns
        -------
        Scale
            The parent scale.
        """
        return self._parent[scale]

    def get_child(self, scale: Scale) -> Optional[Scale]:
        """Get the child scale of the given scale.

        Parameters
        ----------
        scale : Scale
            The scale to find the child of.

        Returns
        -------
        Optional[Scale]
            The child scale, or None if no child exists.
        """
        # Find reverse mapping
        for child, parent in self._parent.items():
            if parent == scale:
                return child
        return None

    def get_upscaling_operator(self, fine: Scale, coarse: Scale) -> Optional[str]:
        """Get the upscaling operator for transitioning from fine to coarse scale.

        Parameters
        ----------
        fine : Scale
            The finer scale.
        coarse : Scale
            The coarser scale.

        Returns
        -------
        Optional[str]
            The name of the upscaling operator, or None if not defined.
        """
        return self._upscaling_operators.get((fine, coarse))

    def get_downscaling_operator(self, coarse: Scale, fine: Scale) -> Optional[str]:
        """Get the downscaling operator for transitioning from coarse to fine scale.

        Parameters
        ----------
        coarse : Scale
            The coarser scale.
        fine : Scale
            The finer scale.

        Returns
        -------
        Optional[str]
            The name of the downscaling operator, or None if not defined.
        """
        return self._downscaling_operators.get((coarse, fine))

    def get_rve_size(self, from_scale: Scale, to_scale: Scale) -> Optional[float]:
        """Get the RVE size for upscaling from one scale to another.

        Parameters
        ----------
        from_scale : Scale
            The finer scale.
        to_scale : Scale
            The coarser scale.

        Returns
        -------
        Optional[float]
            The RVE size in meters, or None if not defined.
        """
        return self._rve_sizes.get((from_scale, to_scale))

    def get_all_scales(self) -> List[Scale]:
        """Get all scales in the hierarchy from finest to coarsest.

        Returns
        -------
        List[Scale]
            List of all scales ordered from NANOPARTICLE to SYSTEM.
        """
        return list(Scale)


class RepresentativeVolumeElement:
    """Representative Volume Element (RVE) for material→device upscaling.

    The RVE is a representative sample of the microstructure that captures
    the essential statistical properties of the material. When upscaling from
    material to device scale, the RVE is used to compute effective properties
    that are then used in the device-scale simulation.

    Parameters
    ----------
    size : tuple[float, float, float]
        RVE dimensions (length, width, height) in meters.
    volume_fraction : float
        Volume fraction of the phases in the composite.
    boundary_conditions : str
        Boundary condition type for RVE analysis ("periodic", "dirichlet", "neumann").
    """

    def __init__(
        self,
        size: Tuple[float, float, float],
        volume_fraction: float = 0.5,
        boundary_conditions: str = "periodic",
    ) -> None:
        self.size = size
        self.volume_fraction = volume_fraction
        self.boundary_conditions = boundary_conditions
        self.effective_properties: Dict[str, float] = {}

    def compute_effective_properties(self, microstructure_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute effective properties from microstructure data.

        Parameters
        ----------
        microstructure_data : dict
            Microstructure characterization data (phase properties, geometry, etc.).

        Returns
        -------
        dict[str, float]
            Effective properties including conductivity, permittivity, etc.
        """
        # Placeholder: in a full implementation, this would solve cell problems
        # on the RVE mesh to compute effective properties
        return self.effective_properties

    def __repr__(self) -> str:
        return f"RVE(size={self.size}, frac={self.volume_fraction}, bc={self.boundary_conditions})"