"""chiplab.device.editor - Programmatic device editor (for AI agents).

Provides a high-level API for AI agents to create and modify devices programmatically,
with validation and export capabilities.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from chiplab.device.geometry import (
    Geometry2D, Geometry3D, ElectrodeGeometry, MaterialRegion,
    Interface, MeshGenerator, Rectangle2D, Circle2D, Polygon2D
)
from chiplab.device.material import (
    MaterialDefinition, ElectricalModel, ThermalModel,
    MagneticModel, MechanicalModel, OpticalModel, UncertainProperty,
    MaterialLibrary, PropertyModelType
)
from chiplab.device.device import (
    PhysicalDevice, WriteMechanism, ReadMechanism,
    BoundaryCondition, DeviceFactory
)


class DeviceEditor:
    """Programmatic device editor for AI agents."""

    def __init__(self):
        self._device: Optional[PhysicalDevice] = None

    def create_device(self, name: str, description: str = "") -> PhysicalDevice:
        """Create a new device with the given name and description."""
        self._device = PhysicalDevice(name=name, description=description)
        return self._device

    def add_electrode(
        self,
        name: str,
        geometry: Union[Geometry2D, Geometry3D],
        material: str,
        mechanism: Optional[WriteMechanism] = None,
    ):
        """Add an electrode to the device."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        self._device.add_electrode(name=name, geometry=geometry, material=material,
                                  mechanism=mechanism)

    def add_material_region(
        self,
        name: str,
        geometry: Geometry2D,
        material: str,
        boundary_conditions: Optional[Dict] = None,
    ):
        """Add a material region to the device."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        self._device.add_material_region(
            name=name, geometry=geometry, material=material,
            boundary_conditions=boundary_conditions
        )

    def add_interface(
        self,
        region1: str,
        region2: str,
        contact_resistance: float = 0.0,
        work_function_difference: float = 0.0,
    ):
        """Add an interface between two material regions."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        self._device.add_interface(
            region1_name=region1,
            region2_name=region2,
            contact_resistance=contact_resistance,
            work_function_difference=work_function_difference,
        )

    def set_write_mechanism(self, port: str, mechanism: WriteMechanism):
        """Set the write mechanism for a port."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        self._device.set_write_mechanism(port=port, mechanism=mechanism)

    def set_read_mechanism(self, port: str, mechanism: ReadMechanism):
        """Set the read mechanism for a port."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        self._device.set_read_mechanism(port=port, mechanism=mechanism)

    def set_material_property(
        self,
        material_name: str,
        property_type: PropertyModelType,
        **kwargs,
    ):
        """Set material properties for a material definition."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")

        # Find or create material definition
        material_def = None
        for region in self._device.material_regions:
            if region.name == material_name:
                material_def = region  # simplified - in real code, use MaterialLibrary
                break

        if material_def is None:
            # Create a new material definition
            from chiplab.device.material import MaterialDefinition
            material_def = MaterialDefinition(
                name=material_name,
                provenance=kwargs.pop("provenance", None),
            )

        # Set the appropriate property model
        if property_type == PropertyModelType.ELECTRICAL:
            electrical = ElectricalModel(**kwargs)
            material_def.electrical = electrical
        elif property_type == PropertyModelType.THERMAL:
            thermal = ThermalModel(**kwargs)
            material_def.thermal = thermal
        elif property_type == PropertyModelType.MAGNETIC:
            magnetic = MagneticModel(**kwargs)
            material_def.magnetic = magnetic
        elif property_type == PropertyModelType.MECHANICAL:
            mechanical = MechanicalModel(**kwargs)
            material_def.mechanical = mechanical
        elif property_type == PropertyModelType.OPTICAL:
            optical = OpticalModel(**kwargs)
            material_def.optical = optical

        # Update the material region
        for region in self._device.material_regions:
            if region.name == material_name:
                # Material definition would be stored separately in real implementation
                pass

    def validate(self) -> List[str]:
        """Validate the device for physics consistency."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        return self._device.validate()

    def export_yaml(self) -> str:
        """Export the device as YAML."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        return self._device.export_yaml()

    def export_json(self) -> str:
        """Export the device as JSON."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        return self._device.export_json()

    def generate_mesh(self, max_cell_size: float = 1e-6, min_cell_size: float = 1e-9):
        """Generate a mesh for the device geometry."""
        if self._device is None:
            raise RuntimeError("No device created. Call create_device() first.")
        mesh_gen = MeshGenerator(geometry=self._device.geometry,
                                max_cell_size=max_cell_size,
                                min_cell_size=min_cell_size)
        return mesh_gen.generate()

    @property
    def device(self) -> PhysicalDevice:
        """Get the current device."""
        return self._device
