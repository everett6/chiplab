"""chiplab.device.device - Complete device specification.

Provides the core PhysicalDevice class and supporting classes for device specification,
including write/read mechanisms, boundary conditions, and a device factory.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Union
import numpy as np
from enum import Enum

from chiplab.device.geometry import (
    Geometry2D, Geometry3D, ElectrodeGeometry, MaterialRegion, Interface, MeshGenerator
)
from chiplab.device.material import (
    MaterialDefinition, PropertyModelType, ElectricalModel, ThermalModel,
    MagneticModel, MechanicalModel, OpticalModel, UncertainProperty, MaterialLibrary
)


class BoundaryCondition:
    """Boundary conditions for device simulation."""

    class Type(str, Enum):
        GROUND = "ground"
        VOLTAGE = "voltage"
        FLOATING = "floating"
        THERMAL = "thermal"
        MECHANICAL = "mechanical"

    def __init__(self, bc_type: Type, port: str, value: Any = None):
        self.type = bc_type
        self.port = port
        self.value = value  # voltage value, temperature, etc.


class WriteMechanism:
    """Write mechanism for a device port."""

    class Mode(str, Enum):
        VOLTAGE_PULSE = "voltage_pulse"
        CURRENT_PULSE = "current_pulse"
        FIELD_ASSISTED = "field_assisted"
        THERMAL = "thermal"
        OPTICAL = "optical"
        MECHANICAL = "mechanical"

    def __init__(
        self,
        mode: Mode,
        port: str,
        parameters: Dict[str, Any],
        material_region: str,
    ):
        self.mode = mode
        self.port = port
        self.parameters = parameters
        self.material_region = material_region

    def apply(self, device: "PhysicalDevice", state: Dict) -> Dict:
        """Apply the write mechanism to the device state."""
        # Default implementation - subclasses may override
        return state


class ReadMechanism:
    """Read mechanism for a device port."""

    class Mode(str, Enum):
        IMPEDANCE = "impedance"
        VOLTAGE = "voltage"
        CURRENT = "current"
        OPTICAL = "optical"
        THERMAL = "thermal"
        MAGNETIC = "magnetic"

    def __init__(
        self,
        mode: Mode,
        port: str,
        parameters: Dict[str, Any],
    ):
        self.mode = mode
        self.port = port
        self.parameters = parameters

    def read(self, device: "PhysicalDevice", state: Dict) -> Dict:
        """Read from the device port."""
        # Default implementation - subclasses may override
        return {"port": self.port, "mode": self.mode.value, "value": None}


class PhysicalDevice:
    """Complete device specification with geometry, materials, electrodes, and mechanisms."""

    def __init__(
        self,
        name: str,
        description: str = "",
        geometry: Optional[Union[Geometry2D, Geometry3D]] = None,
    ):
        self.name = name
        self.description = description
        self.geometry = geometry
        self.material_regions: List[MaterialRegion] = []
        self.electrodes: Dict[str, ElectrodeGeometry] = {}
        self.interfaces: List[Interface] = []
        self.write_mechanisms: Dict[str, WriteMechanism] = {}
        self.read_mechanisms: Dict[str, ReadMechanism] = {}
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
        self._material_library = MaterialLibrary()

    def add_material_region(
        self,
        name: str,
        geometry: Geometry2D,
        material: str,
        boundary_conditions: Optional[Dict] = None,
    ):
        """Add a material region to the device."""
        region = MaterialRegion(name=name, geometry=geometry, material=material,
                                boundary_conditions=boundary_conditions)
        self.material_regions.append(region)

    def add_electrode(
        self,
        name: str,
        geometry: Union[Geometry2D, Geometry3D],
        material: str,
        mechanism: Optional[WriteMechanism] = None,
    ):
        """Add an electrode to the device."""
        electrode = ElectrodeGeometry(geometry=geometry, name=name, material=material)
        self.electrodes[name] = electrode

        # Set write mechanism if provided
        if mechanism is not None:
            self.set_write_mechanism(name, mechanism)

    def add_interface(
        self,
        region1_name: str,
        region2_name: str,
        contact_resistance: float = 0.0,
        work_function_difference: float = 0.0,
    ):
        """Add an interface between two material regions."""
        # Find the regions
        region1 = None
        region2 = None
        for r in self.material_regions:
            if r.name == region1_name:
                region1 = r
            if r.name == region2_name:
                region2 = r

        if region1 is None or region2 is None:
            raise ValueError(f"Material regions {region1_name} or {region2_name} not found")

        interface = Interface(region1, region2,
                             contact_resistance=contact_resistance,
                             work_function_difference=work_function_difference)
        self.interfaces.append(interface)

    def set_write_mechanism(self, port: str, mechanism: WriteMechanism):
        """Set the write mechanism for a port."""
        self.write_mechanisms[port] = mechanism

    def set_read_mechanism(self, port: str, mechanism: ReadMechanism):
        """Set the read mechanism for a port."""
        self.read_mechanisms[port] = mechanism

    def add_boundary_condition(self, bc: BoundaryCondition):
        """Add a boundary condition."""
        self.boundary_conditions[bc.port] = bc

    def validate(self) -> List[str]:
        """Validate physics consistency and return list of issues."""
        issues = []

        # Check that all electrodes have materials
        for name, electrode in self.electrodes.items():
            if electrode.material == "default":
                issues.append(f"Electrode '{name}' uses default material")

        # Check that all material regions have geometries
        for i, region in enumerate(self.material_regions):
            if region.geometry is None:
                issues.append(f"Material region '{region.name}' has no geometry")

        # Check that write mechanisms have ports
        for port, mechanism in self.write_mechanisms.items():
            if port not in self.electrodes:
                issues.append(f"Write mechanism port '{port}' not associated with an electrode")

        # Check that read mechanisms have ports
        for port, mechanism in self.read_mechanisms.items():
            if port not in self.electrodes:
                issues.append(f"Read mechanism port '{port}' not associated with an electrode")

        return issues

    def export_yaml(self) -> str:
        """Export device specification as YAML string."""
        import yaml

        data = {
            "name": self.name,
            "description": self.description,
            "material_regions": [],
            "electrodes": [],
            "interfaces": [],
            "write_mechanisms": [],
            "read_mechanisms": [],
            "boundary_conditions": [],
        }

        # Export material regions
        for region in self.material_regions:
            data["material_regions"].append({
                "name": region.name,
                "material": region.material,
            })

        # Export electrodes
        for name, electrode in self.electrodes.items():
            data["electrodes"].append({
                "name": name,
                "material": electrode.material,
                # Geometry would be exported separately
            })

        # Export interfaces
        for interface in self.interfaces:
            data["interfaces"].append({
                "region1": interface.region1.name,
                "region2": interface.region2.name,
                "contact_resistance": interface.contact_resistance,
                "work_function_difference": interface.work_function_difference,
            })

        # Export write mechanisms
        for port, mechanism in self.write_mechanisms.items():
            data["write_mechanisms"].append({
                "port": port,
                "mode": mechanism.mode.value,
                "parameters": mechanism.parameters,
            })

        # Export read mechanisms
        for port, mechanism in self.read_mechanisms.items():
            data["read_mechanisms"].append({
                "port": port,
                "mode": mechanism.mode.value,
                "parameters": mechanism.parameters,
            })

        # Export boundary conditions
        for port, bc in self.boundary_conditions.items():
            data["boundary_conditions"].append({
                "port": port,
                "type": bc.type.value,
                "value": bc.value,
            })

        return yaml.dump(data, default_flow_style=False)

    def export_json(self) -> str:
        """Export device specification as JSON string."""
        import json

        data = {
            "name": self.name,
            "description": self.description,
            "material_regions": [],
            "electrodes": [],
            "interfaces": [],
            "write_mechanisms": [],
            "read_mechanisms": [],
            "boundary_conditions": [],
        }

        # Export material regions
        for region in self.material_regions:
            data["material_regions"].append({
                "name": region.name,
                "material": region.material,
            })

        # Export electrodes
        for name, electrode in self.electrodes.items():
            data["electrodes"].append({
                "name": name,
                "material": electrode.material,
            })

        # Export interfaces
        for interface in self.interfaces:
            data["interfaces"].append({
                "region1": interface.region1.name,
                "region2": interface.region2.name,
                "contact_resistance": interface.contact_resistance,
                "work_function_difference": interface.work_function_difference,
            })

        # Export write mechanisms
        for port, mechanism in self.write_mechanisms.items():
            data["write_mechanisms"].append({
                "port": port,
                "mode": mechanism.mode.value,
                "parameters": mechanism.parameters,
            })

        # Export read mechanisms
        for port, mechanism in self.read_mechanisms.items():
            data["read_mechanisms"].append({
                "port": port,
                "mode": mechanism.mode.value,
                "parameters": mechanism.parameters,
            })

        # Export boundary conditions
        for port, bc in self.boundary_conditions.items():
            data["boundary_conditions"].append({
                "port": port,
                "type": bc.type.value,
                "value": bc.value,
            })

        return json.dumps(data, indent=2)


class DeviceFactory:
    """Factory for creating devices from geometry + materials + mechanisms."""

    @staticmethod
    def from_geometry_materials(
        name: str,
        geometry: Union[Geometry2D, Geometry3D],
        material_defs: Dict[str, MaterialDefinition],
        electrode_configs: List[Dict],
        interface_configs: Optional[List[Dict]] = None,
    ) -> PhysicalDevice:
        """Create a PhysicalDevice from geometry, materials, and configurations."""
        device = PhysicalDevice(name=name)

        # Add material regions
        for mat_name, mat_def in material_defs.items():
            # Create a rectangular region covering the geometry bounding box
            # This is simplified - real implementation would use proper shape
            from chiplab.device.geometry import Rectangle2D
            if hasattr(geometry, 'vertices') and len(geometry.vertices) > 2:
                # Use polygon geometry
                bbox_min = geometry.vertices.min(axis=0)
                bbox_max = geometry.vertices.max(axis=0)
                region_geom = Rectangle2D(
                    xmin=float(bbox_min[0]), xmax=float(bbox_max[0]),
                    ymin=float(bbox_min[1]), ymax=float(bbox_max[1])
                )
            else:
                # Default bounding box
                region_geom = Rectangle2D(xmin=-1, xmax=1, ymin=-1, ymax=1)

            device.add_material_region(
                name=mat_name,
                geometry=region_geom,
                material=mat_name,
            )

        # Add electrodes
        for config in electrode_configs:
            geom = config.get("geometry", Rectangle2D(xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5))
            material = config.get("material", "default")
            mechanism = config.get("mechanism", None)

            device.add_electrode(
                name=config.get("name", "electrode"),
                geometry=geom,
                material=material,
                mechanism=mechanism,
            )

        # Add interfaces
        if interface_configs:
            for config in interface_configs:
                device.add_interface(
                    region1_name=config.get("region1", ""),
                    region2_name=config.get("region2", ""),
                    contact_resistance=config.get("contact_resistance", 0.0),
                    work_function_difference=config.get("work_function_difference", 0.0),
                )

        return device

    @staticmethod
    def from_yaml(yaml_content: str) -> PhysicalDevice:
        """Create a PhysicalDevice from YAML content."""
        import yaml

        data = yaml.safe_load(yaml_content)
        device = PhysicalDevice(name=data.get("name", "device"),
                                description=data.get("description", ""))

        # Material regions
        for mr in data.get("material_regions", []):
            device.add_material_region(
                name=mr["name"],
                geometry=Rectangle2D(xmin=-1, xmax=1, ymin=-1, ymax=1),  # simplified
                material=mr["material"],
            )

        # Electrodes
        for el in data.get("electrodes", []):
            electrode_geom = Rectangle2D(xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5)  # simplified
            device.add_electrode(
                name=el["name"],
                geometry=electrode_geom,
                material=el.get("material", "default"),
            )

        # Interfaces
        for iface in data.get("interfaces", []):
            device.add_interface(
                region1_name=iface["region1"],
                region2_name=iface["region2"],
                contact_resistance=iface.get("contact_resistance", 0.0),
                work_function_difference=iface.get("work_function_difference", 0.0),
            )

        # Write mechanisms
        for wm in data.get("write_mechanisms", []):
            from chiplab.device.device import WriteMechanism
            mode = WriteMechanism.Mode(wm["mode"])
            params = wm.get("parameters", {})
            mechanism = WriteMechanism(
                mode=mode,
                port=wm["port"],
                parameters=params,
                material_region=wm.get("material_region", ""),
            )
            device.set_write_mechanism(wm["port"], mechanism)

        # Read mechanisms
        for rm in data.get("read_mechanisms", []):
            from chiplab.device.device import ReadMechanism
            mode = ReadMechanism.Mode(rm["mode"])
            params = rm.get("parameters", {})
            mechanism = ReadMechanism(
                mode=mode,
                port=rm["port"],
                parameters=params,
            )
            device.set_read_mechanism(rm["port"], mechanism)

        # Boundary conditions
        for bc in data.get("boundary_conditions", []):
            from chiplab.device.device import BoundaryCondition
            bc_type = BoundaryCondition.Type(bc["type"])
            bc_obj = BoundaryCondition(bc_type=bc_type, port=bc["port"], value=bc.get("value"))
            device.add_boundary_condition(bc_obj)

        return device
