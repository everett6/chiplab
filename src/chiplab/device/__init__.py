"""chiplab.device - Universal physical-device designer.

This module provides a universal framework for creating entirely new physical devices
from scratch, not limited to predefined components. It supports arbitrary electrode
shapes, material regions, interfaces, and write/read mechanisms for multi-physics simulation.
"""

# Import submodules to make them accessible
import chiplab.device.geometry
import chiplab.device.material
import chiplab.device.device
import chiplab.device.editor

# Import and re-export geometry module contents
from .geometry import (
    Geometry2D,
    Geometry3D,
    ElectrodeGeometry,
    MaterialRegion,
    Interface,
    MeshGenerator,
    Rectangle2D,
    Circle2D,
    Polygon2D,
    UnionGeometry,
    IntersectionGeometry,
    DifferenceGeometry,
    UnionGeometry3D,
    IntersectionGeometry3D,
    DifferenceGeometry3D,
    Box3D,
    Sphere3D,
)

# Import and re-export material module contents
from .material import (
    MaterialDefinition,
    PropertyModel,
    ElectricalModel,
    ThermalModel,
    MagneticModel,
    MechanicalModel,
    OpticalModel,
    UncertainProperty,
    MaterialLibrary,
    PropertyModelType,
)

# Import and re-export device module contents
from .device import (
    PhysicalDevice,
    WriteMechanism,
    ReadMechanism,
    BoundaryCondition,
    DeviceFactory,
)

# Import and re-export editor module contents
from .editor import (
    DeviceEditor,
)

__all__ = [
    "Geometry2D",
    "Geometry3D",
    "ElectrodeGeometry",
    "MaterialRegion",
    "Interface",
    "MeshGenerator",
    "Rectangle2D",
    "Circle2D",
    "Polygon2D",
    "UnionGeometry",
    "IntersectionGeometry",
    "DifferenceGeometry",
    "UnionGeometry3D",
    "IntersectionGeometry3D",
    "DifferenceGeometry3D",
    "Box3D",
    "Sphere3D",
    "MaterialDefinition",
    "PropertyModel",
    "ElectricalModel",
    "ThermalModel",
    "MagneticModel",
    "MechanicalModel",
    "OpticalModel",
    "UncertainProperty",
    "MaterialLibrary",
    "PropertyModelType",
    "PhysicalDevice",
    "WriteMechanism",
    "ReadMechanism",
    "BoundaryCondition",
    "DeviceFactory",
    "DeviceEditor",
]
