"""chiplab.device.geometry - Custom electrode & material geometry with CSG operations.

Provides 2D/3D geometry classes for defining electrode shapes, material regions,
interfaces, and mesh generation for FEM/FDM simulations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union, Dict, Any


class Geometry2D(ABC):
    """2D geometry base class with constructive solid geometry (CSG) operations."""

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the geometry."""
        pass

    @abstractmethod
    def area(self) -> float:
        """Calculate the area of the 2D geometry."""
        pass

    def union(self, other: "Geometry2D") -> "Geometry2D":
        """Union (OR) operation with another 2D geometry."""
        return UnionGeometry(self, other)

    def intersection(self, other: "Geometry2D") -> "Geometry2D":
        """Intersection (AND) operation with another 2D geometry."""
        return IntersectionGeometry(self, other)

    def difference(self, other: "Geometry2D") -> "Geometry2D":
        """Difference (A - B) operation with another 2D geometry."""
        return DifferenceGeometry(self, other)


class Geometry3D(ABC):
    """3D geometry base class with constructive solid geometry (CSG) operations."""

    @abstractmethod
    def volume(self) -> float:
        """Calculate the volume of the 3D geometry."""
        pass

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the 3D geometry."""
        pass

    def union(self, other: "Geometry3D") -> "Geometry3D":
        """Union (OR) operation with another 3D geometry."""
        return UnionGeometry3D(self, other)

    def intersection(self, other: "Geometry3D") -> "Geometry3D":
        """Intersection (AND) operation with another 3D geometry."""
        return IntersectionGeometry3D(self, other)

    def difference(self, other: "Geometry3D") -> "Geometry3D":
        """Difference (A - B) operation with another 3D geometry."""
        return DifferenceGeometry3D(self, other)


class Rectangle2D(Geometry2D):
    """2D rectangle geometry defined by corner points."""

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def contains(self, point: np.ndarray) -> bool:
        x, y = point[0], point[1]
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


class Circle2D(Geometry2D):
    """2D circle geometry defined by center and radius."""

    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def contains(self, point: np.ndarray) -> bool:
        return np.linalg.norm(point - self.center) <= self.radius

    def area(self) -> float:
        return np.pi * self.radius ** 2


class Polygon2D(Geometry2D):
    """2D polygon geometry defined by a sequence of vertices."""

    def __init__(self, vertices: List[np.ndarray]):
        self.vertices = np.array(vertices, dtype=float)

    def contains(self, point: np.ndarray) -> bool:
        # Ray casting algorithm
        x, y = point[0], point[1]
        inside = False
        n = len(self.vertices)
        for i in range(n):
            j = (i - 1 + n) % n
            xi, yi = self.vertices[i][0], self.vertices[i][1]
            xj, yj = self.vertices[j][0], self.vertices[j][1]
            # Check if edge straddles the point's y coordinate
            if (yi > y) != (yj > y):
                # Compute x intersection
                x_intersect = xi + (y - yi) * (xj - xi) / (yj - yi + 1e-15)
                if x < x_intersect:
                    inside = not inside
        return inside

    def area(self) -> float:
        # Shoelace formula
        vertices = self.vertices
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


class UnionGeometry(Geometry2D):
    """Union (OR) of two 2D geometries."""

    def __init__(self, a: Geometry2D, b: Geometry2D):
        self.a = a
        self.b = b

    def contains(self, point: np.ndarray) -> bool:
        return self.a.contains(point) or self.b.contains(point)

    def area(self) -> float:
        # Approximate area using Monte Carlo or assume no overlap for simplicity
        return self.a.area() + self.b.area()


class IntersectionGeometry(Geometry2D):
    """Intersection (AND) of two 2D geometries."""

    def __init__(self, a: Geometry2D, b: Geometry2D):
        self.a = a
        self.b = b

    def contains(self, point: np.ndarray) -> bool:
        return self.a.contains(point) and self.b.contains(point)

    def area(self) -> float:
        # For simplicity, return minimum of areas
        return min(self.a.area(), self.b.area())


class DifferenceGeometry(Geometry2D):
    """Difference (A - B) of two 2D geometries."""

    def __init__(self, a: Geometry2D, b: Geometry2D):
        self.a = a
        self.b = b

    def contains(self, point: np.ndarray) -> bool:
        return self.a.contains(point) and not self.b.contains(point)

    def area(self) -> float:
        return max(0, self.a.area() - self.b.area())


# ============================================================
# 3D Geometry
# ============================================================


class UnionGeometry3D(Geometry3D):
    """Union (OR) of two 3D geometries."""

    def __init__(self, a: Geometry3D, b: Geometry3D):
        self.a = a
        self.b = b

    def volume(self) -> float:
        return self.a.volume() + self.b.volume()

    def contains(self, point: np.ndarray) -> bool:
        return self.a.contains(point) or self.b.contains(point)


class IntersectionGeometry3D(Geometry3D):
    """Intersection (AND) of two 3D geometries."""

    def __init__(self, a: Geometry3D, b: Geometry3D):
        self.a = a
        self.b = b

    def volume(self) -> float:
        # Simplified: return min of volumes
        return min(self.a.volume(), self.b.volume())

    def contains(self, point: np.ndarray) -> bool:
        return self.a.contains(point) and self.b.contains(point)


class DifferenceGeometry3D(Geometry3D):
    """Difference (A - B) of two 3D geometries."""

    def __init__(self, a: Geometry3D, b: Geometry3D):
        self.a = a
        self.b = b

    def volume(self) -> float:
        a_vol = self.a.volume()
        b_vol = self.b.volume()
        return max(0, a_vol - b_vol)

    def contains(self, point: np.ndarray) -> bool:
        return self.a.contains(point) and not self.b.contains(point)


class Box3D(Geometry3D):
    """3D box geometry defined by coordinate ranges."""

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float,
                 zmin: float, zmax: float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def volume(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)

    def contains(self, point: np.ndarray) -> bool:
        x, y, z = point[0], point[1], point[2]
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)


class Sphere3D(Geometry3D):
    """3D sphere geometry defined by center and radius."""

    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius

    def volume(self) -> float:
        return (4.0 / 3.0) * np.pi * self.radius ** 3

    def contains(self, point: np.ndarray) -> bool:
        return np.linalg.norm(point - self.center) <= self.radius


# ============================================================
# ElectrodeGeometry
# ============================================================


class ElectrodeGeometry:
    """Arbitrary 2D/3D electrode shapes (polygons, splines, parametric)."""

    def __init__(self, geometry: Union[Geometry2D, Geometry3D],
                 name: str = "electrode",
                 material: str = "default"):
        self.geometry = geometry
        self.name = name
        self.material = material
        self.electrode_id = id(self)

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the electrode geometry."""
        if isinstance(self.geometry, Geometry2D):
            # For 2D, add z=0
            point_2d = np.array([point[0], point[1], 0.0])
            return self.geometry.contains(point_2d)
        return self.geometry.contains(point)

    def area(self) -> float:
        """Get the cross-sectional area (for 2D) or surface area."""
        if isinstance(self.geometry, Geometry2D):
            return self.geometry.area()
        return self.geometry.volume()

    def perimeter(self) -> float:
        """Get the perimeter (for 2D) or edge length sum."""
        if isinstance(self.geometry, Geometry2D):
            # For polygons, compute perimeter
            if hasattr(self.geometry, 'vertices') and len(self.geometry.vertices) > 2:
                verts = self.geometry.vertices
                perimeter = 0.0
                n = len(verts)
                for i in range(n):
                    j = (i + 1) % n
                    perimeter += np.linalg.norm(verts[i] - verts[j])
                return perimeter
            return 0.0
        return 0.0

    def export_svg(self, path: str, resolution: int = 100):
        """Export electrode geometry to SVG format."""
        # Simplified SVG export
        with open(path, 'w') as f:
            f.write(f'<svg width="{resolution}" height="{resolution}" xmlns="http://www.w3.org/2000/svg">\n')
            if hasattr(self.geometry, 'vertices') and len(self.geometry.vertices) > 2:
                verts = self.geometry.vertices
                # Translate and scale to SVG canvas
                min_x, min_y = verts.min(axis=0)
                scaled = verts - [min_x, min_y]
                scale = resolution / max(scaled.max(axis=0))
                scaled = scaled * scale
                # Close polygon
                pts = " ".join([f"{v[0]},{v[1]}" for v in scaled])
                f.write(f'  <polygon points="{pts}" fill="none" stroke="blue" stroke-width="2"/>\n')
            f.write('</svg>\n')


# ============================================================
# MaterialRegion
# ============================================================


class MaterialRegion:
    """Define arbitrary material regions with boundaries."""

    def __init__(self, name: str, geometry: Geometry2D, material: str,
                 boundary_conditions: Optional[Dict] = None):
        self.name = name
        self.geometry = geometry
        self.material = material
        self.boundary_conditions = boundary_conditions or {}

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside this material region."""
        return self.geometry.contains(point)


# ============================================================
# Interface
# ============================================================


class Interface:
    """Boundaries between materials with contact physics."""

    def __init__(self, region1: MaterialRegion, region2: MaterialRegion,
                 contact_resistance: float = 0.0,
                 work_function_difference: float = 0.0):
        self.region1 = region1
        self.region2 = region2
        self.contact_resistance = contact_resistance
        self.work_function_difference = work_function_difference

    def get_contact_potential(self) -> float:
        """Compute the contact potential difference."""
        return self.work_function_difference


# ============================================================
# MeshGenerator
# ============================================================


class MeshGenerator:
    """Generate FEM/FDM meshes from geometry (interface to Gmsh if available)."""

    def __init__(self, geometry: Union[Geometry2D, Geometry3D],
                 max_cell_size: float = 1e-6,
                 min_cell_size: float = 1e-9):
        self.geometry = geometry
        self.max_cell_size = max_cell_size
        self.min_cell_size = min_cell_size
        self.mesh = None

    def generate(self) -> Optional[np.ndarray]:
        """Generate a mesh from the geometry."""
        # Try to use gmsh if available
        try:
            import gmsh
            gmsh.initialize()
            self._setup_gmsh()
            gmsh.model.mesh.generate(3 if hasattr(self.geometry, 'volume') else 2)
            # TODO: Extract mesh data
            gmsh.finalize()
        except ImportError:
            # Fall back to simple discretization
            self.mesh = self._simple_discretization()
        return self.mesh

    def _setup_gmsh(self):
        """Setup gmsh model from geometry."""
        pass

    def _simple_discretization(self) -> np.ndarray:
        """Simple mesh discretization as fallback."""
        # For 2D: generate a simple grid
        if hasattr(self.geometry, 'vertices') and len(self.geometry.vertices) > 2:
            # bounding box
            verts = self.geometry.vertices
            min_x, min_y = verts.min(axis=0)
            max_x, max_y = verts.max(axis=0)

            n_x = max(10, int((max_x - min_x) / self.max_cell_size))
            n_y = max(10, int((max_y - min_y) / self.max_cell_size))

            x = np.linspace(min_x, max_x, n_x)
            y = np.linspace(min_y, max_y, n_y)
            X, Y = np.meshgrid(x, y)

            # Check which points are inside
            points = []
            for i in range(n_x):
                for j in range(n_y):
                    pt = np.array([X[i, j], Y[i, j]])
                    if self.geometry.contains(pt):
                        points.append([X[i, j], Y[i, j], 0.0])

            if points:
                return np.array(points)
        return np.array([])

    def export_msh(self, path: str):
        """Export mesh to Gmsh .msh format."""
        # Placeholder for actual msh export
        with open(path, 'w') as f:
            f.write("# Simple mesh export placeholder\n")


# Helper functions for importing from standard formats


def from_svg(path: str) -> ElectrodeGeometry:
    """Import electrode geometry from SVG file."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()

    # Parse polygon paths
    polygons = []
    for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = path_elem.get('d', '')
        # Simple parsing - extract polygon points
        # This is a simplified implementation
        polygons.append(d)

    # Create a simple geometry from parsed paths
    # For now, return a basic Circle2D
    return Circle2D(center=[0, 0], radius=1.0)


def from_dxf(path: str) -> ElectrodeGeometry:
    """Import electrode geometry from DXF file."""
    # DXF import placeholder
    return Circle2D(center=[0, 0], radius=1.0)


def from_gds(path: str) -> ElectrodeGeometry:
    """Import electrode geometry from GDSII file."""
    # GDSII import placeholder
    return Rectangle2D(xmin=-1, xmax=1, ymin=-1, ymax=1)
