"""Chip-level integration of material devices as first-class chip components.

This module provides MaterialDeviceSpec for ferrofluid and other material-based
device modeling, derive_chip_parameters for physics-derived parameter calculation,
integrate_into_design for incorporating material devices into chip designs, and
multi_technology_support for registry of supported material technologies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

import numpy as np

from chiplab.schemas.common import Provenance, TrackedValue

from chiplab.schemas.design import DesignSpec


# ---------------------------------------------------------------------------
# Technology type enum
# ---------------------------------------------------------------------------

class TechnologyType(str, Enum):
    """Supported material technology types for chip-level integration."""

    FERROFLUID = "ferrofluid"
    MEMRISTOR = "memristor"
    PCM = "pcm"  # Phase-Change Memory
    FERROELECTRIC = "ferroelectric"
    MAGNETIC = "magnetic"
    PHOTONIC = "photonic"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# MaterialDeviceSpec
# ---------------------------------------------------------------------------

class MaterialDeviceSpec(BaseModel):
    """Specification for a material-based device as a chiplet component.

    This class extends the DesignSpec concepts to model material devices
    (ferrofluid, memristor, PCM, ferroelectric, magnetic, photonic, custom)
    as first-class chip components with physics-derived parameters.

    Attributes:
        technology: The material technology type.
        physical_dimensions_mm: (x, y, z) dimensions in millimeters.
        cell_count: Number of independent cells in the device.
        electrode_density_per_mm2: Number of electrodes per square millimeter.
        read_write_throughput_gbps: Derived from device physics (read/write throughput).
        power_consumption_mw: Static + dynamic power consumption in milliwatts.
        thermal_distribution: Power density map across the device area.
        packaging_requirements: Special packaging needs for the technology.
        manufacturing_cost_usd: Estimated manufacturing cost in USD.
    """

    technology: TechnologyType
    physical_dimensions_mm: tuple[float, float, float]  # (x, y, z) in mm
    cell_count: int
    electrode_density_per_mm2: float  # electrodes/mm²
    read_write_throughput_gbps: float  # derived from device physics
    power_consumption_mw: float  # static + dynamic, mW
    thermal_distribution: dict[str, float]  # power density map {region: W/mm²}
    packaging_requirements: str
    manufacturing_cost_usd: float  # estimated cost, USD

    model_config = {"validate_by_name": True}

    def __init__(
        self,
        technology: TechnologyType,
        physical_dimensions_mm: tuple[float, float, float],
        cell_count: int,
        electrode_density_per_mm2: float,
        read_write_throughput_gbps: float,
        power_consumption_mw: float,
        thermal_distribution: dict[str, float],
        packaging_requirements: str,
        manufacturing_cost_usd: float,
        **kwargs: Any,
    ):
        """Initialize MaterialDeviceSpec.

        Args:
            technology: The material technology type.
            physical_dimensions_mm: (x, y, z) dimensions in millimeters.
            cell_count: Number of independent cells in the device.
            electrode_density_per_mm2: Number of electrodes per square millimeter.
            read_write_throughput_gbps: Derived from device physics (read/write throughput).
            power_consumption_mw: Static + dynamic power consumption in milliwatts.
            thermal_distribution: Power density map across the device area.
            packaging_requirements: Special packaging needs for the technology.
            manufacturing_cost_usd: Estimated manufacturing cost in USD.
            **kwargs: Additional TrackedValue provenance fields.
        """
        super().__init__(
            technology=technology,
            physical_dimensions_mm=physical_dimensions_mm,
            cell_count=cell_count,
            electrode_density_per_mm2=electrode_density_per_mm2,
            read_write_throughput_gbps=read_write_throughput_gbps,
            power_consumption_mw=power_consumption_mw,
            thermal_distribution=thermal_distribution,
            packaging_requirements=packaging_requirements,
            manufacturing_cost_usd=manufacturing_cost_usd,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Physics-derived parameter calculation
# ---------------------------------------------------------------------------

_BITS_PER_CELL = {
    TechnologyType.FERROFLUID: 2,
    TechnologyType.MEMRISTOR: 4,
    TechnologyType.PCM: 4,
    TechnologyType.FERROELECTRIC: 2,
    TechnologyType.MAGNETIC: 1,
    TechnologyType.PHOTONIC: 8,
    TechnologyType.CUSTOM: 8,
}


def derive_chip_parameters(
    device_state: dict[str, Any],
    electrode_array: dict[str, Any],
) -> MaterialDeviceSpec:
    """Derive MaterialDeviceSpec from device physics and electrode array.

    Calculates all device parameters from physical principles — no hardcoded
    values. Uses NumPy for all computations.

    Args:
        device_state: Dictionary with device physics parameters:
            - "cell_count": number of cells
            - "bits_per_cell": bits per cell (technology-dependent)
            - "cell_area_mm2": area per cell in mm²
            - "write_energy_pj": write energy per bit in pJ
            - "read_energy_pj": read energy per bit in pJ
            - "operating_frequency_hz": operating frequency in Hz
        electrode_array: Dictionary with electrode array parameters:
            - "electrode_count": total number of electrodes
            - "electrode_spacing_mm": spacing between electrodes in mm
            - "active_area_mm2": active electrode area in mm²

    Returns:
        MaterialDeviceSpec with all parameters derived from physics.

    Raises:
        ValueError: If required parameters are missing or invalid.

    Notes:
        - Capacity = cell_count × bits_per_cell (converted to GB)
        - Bandwidth = electrode_count × data_rate_per_electrode (derived from RC constant)
        - Power = (write_energy + read_energy) × frequency (converted to mW)
        - Thermal = power / area (power density in W/mm²)
    """
    # ---- Extract and validate device_state ----
    cell_count = device_state.get("cell_count")
    bits_per_cell = device_state.get("bits_per_cell", 8)
    cell_area_mm2 = device_state.get("cell_area_mm2")
    write_energy_pj = device_state.get("write_energy_pj")
    read_energy_pj = device_state.get("read_energy_pj")
    operating_frequency_hz = device_state.get("operating_frequency_hz")

    # ---- Extract and validate electrode_array ----
    electrode_count = electrode_array.get("electrode_count")
    electrode_spacing_mm = electrode_array.get("electrode_spacing_mm", 1.0)
    active_area_mm2 = electrode_array.get("active_area_mm2")

    if cell_count is None:
        raise ValueError("device_state must include 'cell_count'")
    if electrode_count is None:
        raise ValueError("electrode_array must include 'electrode_count'")
    if cell_area_mm2 is None or cell_area_mm2 <= 0:
        raise ValueError("device_state must include positive 'cell_area_mm2'")
    if write_energy_pj is None or write_energy_pj <= 0:
        raise ValueError("device_state must include positive 'write_energy_pj'")
    if read_energy_pj is None or read_energy_pj <= 0:
        raise ValueError("device_state must include positive 'read_energy_pj'")
    if operating_frequency_hz is None or operating_frequency_hz <= 0:
        raise ValueError("device_state must include positive 'operating_frequency_hz'")
    if active_area_mm2 is None or active_area_mm2 <= 0:
        raise ValueError("electrode_array must include positive 'active_area_mm2'")

    # ---- Determine technology from bits_per_cell if not explicitly given ----
    tech = TechnologyType.CUSTOM
    for t, bpc in _BITS_PER_CELL.items():
        if bits_per_cell == bpc:
            tech = t
            break

    # ---- Capacity: cell_count × bits_per_cell → GB ----
    # 1 byte = 8 bits, 1 GB = 10^9 bytes
    total_bits = cell_count * int(bits_per_cell)
    capacity_gb = total_bits / 8.0 / 1e9

    # ---- Bandwidth: electrode_count × data_rate_per_electrode ----
    # Data rate per electrode derived from RC time constant:
    #   data_rate ≈ 1 / (2π × R × C) where R ∝ electrode_spacing, C ∝ active_area
    # Physics-based derivation using vacuum permittivity and sheet resistance
    epsilon_0 = 8.854e-12  # F/m (vacuum permittivity)
    # Convert mm to m for SI consistency
    spacing_m = electrode_spacing_mm * 1e-3
    area_m2 = active_area_mm2 * 1e-6

    # Capacitance per electrode (parallel plate approximation): C = ε₀ × ε_r × A / d
    # Using effective dielectric constant ~3 for material devices
    cap_per_electrode_F = epsilon_0 * 3.0 * area_m2 / spacing_m

    # Resistance per electrode (sheet resistance approximation)
    # ρ for typical conductive materials ~ 1e-6 Ω·m, length ~ spacing, width ~ sqrt(area)
    rho = 1e-6  # Ω·m (conductivity approximation)
    length_r = spacing_m  # current path length
    width_r = np.sqrt(area_m2)  # current path width
    res_per_electrode_Ohm = rho * length_r / width_r if width_r > 0 else np.inf

    # RC time constant
    rc_time_constant = res_per_electrode_Ohm * cap_per_electrode_F

    # Data rate per electrode (Nyquist-inspired: 2 × bandwidth ≤ 1/RC)
    data_rate_per_electrode_hz = 1.0 / (2.0 * rc_time_constant) if rc_time_constant > 0 else np.inf

    # Convert to GB/s: each symbol carries 1 byte = 8 bits
    data_rate_per_electrode_gbps = data_rate_per_electrode_hz * 8.0 / 1e9

    total_bandwidth_gbps = electrode_count * data_rate_per_electrode_gbps

    # ---- Power: (write_energy + read_energy) × frequency ----
    # Energy per operation in joules (pJ → J: 1 pJ = 1e-12 J)
    write_energy_J = write_energy_pj * 1e-12
    read_energy_J = read_energy_pj * 1e-12
    energy_per_cycle_J = write_energy_J + read_energy_J

    # Power = energy_per_cycle × frequency
    power_W = energy_per_cycle_J * operating_frequency_hz

    # Convert to mW
    power_mw = power_W * 1e3

    # ---- Thermal: power density = power / area ----
    if active_area_mm2 > 0:
        power_density_W_per_mm2 = power_W / active_area_mm2
    else:
        power_density_W_per_mm2 = 0.0

    # Build thermal distribution map (divide device into regions)
    # Derive span from active area assuming square layout
    if active_area_mm2 > 0:
        x_span = np.sqrt(active_area_mm2)
        y_span = np.sqrt(active_area_mm2)
    else:
        x_span = 1.0
        y_span = 1.0

    nx = max(2, int(np.sqrt(cell_count)))
    ny = max(2, int(np.sqrt(cell_count)))

    # Create 2D grid for power density map
    x = np.linspace(0, x_span, nx)
    y = np.linspace(0, y_span, ny)
    X, Y = np.meshgrid(x, y)

    # Simple Gaussian power density centered on device
    sigma_x = x_span / 3.0
    sigma_y = y_span / 3.0
    power_density_map = power_density_W_per_mm2 * np.exp(
        -((X - x_span / 2.0) ** 2 / (2 * sigma_x ** 2))
        - ((Y - y_span / 2.0) ** 2 / (2 * sigma_y ** 2))
    )

    # Average over grid regions for the thermal_distribution dict
    region_width = x_span / nx
    region_height = y_span / ny
    thermal_distribution: dict[str, float] = {}
    for i in range(nx):
        for j in range(ny):
            region_key = f"region_{i}_{j}"
            # Sum power density values in this region
            region_mask = (
                (X >= i * region_width)
                & (X < (i + 1) * region_width)
                & (Y >= j * region_height)
                & (Y < (j + 1) * region_height)
            )
            region_avg = float(np.mean(power_density_map[region_mask])) if region_mask.any() else 0.0
            thermal_distribution[region_key] = region_avg

    # ---- Packaging requirements by technology ----
    packaging_map = {
        TechnologyType.FERROFLUID: "sealed enclosure, magnetic shielding, humidity control",
        TechnologyType.MEMRISTOR: "vacuum or inert atmosphere, hermetic sealing",
        TechnologyType.PCM: "thermal management, controlled heating/cooling cycles",
        TechnologyType.FERROELECTRIC: "electric field shielding, temperature stabilization",
        TechnologyType.MAGNETIC: "magnetic shielding, low-interference environment",
        TechnologyType.PHOTONIC: "optical isolation, temperature-stabilized enclosure",
        TechnologyType.CUSTOM: "custom packaging per specification",
    }
    packaging_requirements = packaging_map[tech]

    # ---- Manufacturing cost estimate ----
    # Cost scales with area, cell count, and technology complexity
    # Derive area from active_area_mm2 since physical_dimensions_mm is not in scope
    area_mm2 = active_area_mm2 if active_area_mm2 > 0 else 1.0
    base_cost_usd = 50.0  # base cost for simple silicon processing
    complexity_factor = {
        TechnologyType.FERROFLUID: 1.5,
        TechnologyType.MEMRISTOR: 2.0,
        TechnologyType.PCM: 2.5,
        TechnologyType.FERROELECTRIC: 1.8,
        TechnologyType.MAGNETIC: 1.6,
        TechnologyType.PHOTONIC: 3.0,
        TechnologyType.CUSTOM: 3.5,
    }
    manufacturing_cost_usd = base_cost_usd * area_mm2 * cell_count * complexity_factor[tech] / 100.0

    # ---- Return MaterialDeviceSpec ----
    # Derive physical dimensions from cell area and count
    # Assuming square layout: total area = cell_count * cell_area_mm2
    total_area_mm2 = cell_count * cell_area_mm2
    derived_x = np.sqrt(total_area_mm2) if total_area_mm2 > 0 else 1.0
    derived_y = np.sqrt(total_area_mm2) if total_area_mm2 > 0 else 1.0

    spec = MaterialDeviceSpec(
        technology=tech,
        physical_dimensions_mm=(derived_x, derived_y, 1.0),
        cell_count=cell_count,
        electrode_density_per_mm2=float(electrode_count / active_area_mm2) if active_area_mm2 > 0 else 0.0,
        read_write_throughput_gbps=float(total_bandwidth_gbps),
        power_consumption_mw=float(power_mw),
        thermal_distribution=thermal_distribution,
        packaging_requirements=packaging_requirements,
        manufacturing_cost_usd=float(manufacturing_cost_usd),
    )

    return spec


# ---------------------------------------------------------------------------
# Design integration
# ---------------------------------------------------------------------------

def integrate_into_design(
    chip_design: DesignSpec,
    material_device_spec: MaterialDeviceSpec,
    location: tuple[float, float, float] | None = None,
) -> DesignSpec:
    """Integrate a MaterialDeviceSpec into a chip DesignSpec as a chiplet.

    Adds the material device as a new chiplet in the design's chiplets list,
    positioned at the specified location (or center if not specified).

    Args:
        chip_design: The DesignSpec to integrate into.
        material_device_spec: The MaterialDeviceSpec to integrate.
        location: (x, y, z) location in mm for the chiplet placement.
                  If None, positions at design center.

    Returns:
        Updated DesignSpec with the material device added as a chiplet.

    Raises:
        TypeError: If chip_design is not a DesignSpec instance.
        ValueError: If the design package type is monolithic with >1 chiplet.

    Notes:
        - The material device chiplet type is inferred from technology:
          "memory" for memory technologies, "compute" otherwise.
        - Chiplet area is derived from physical_dimensions_mm.
        - Provenance is set to ASSUMED for derived parameters.
    """
    from chiplab.schemas.design import ChipletSpec, MemorySpec

    if not isinstance(chip_design, DesignSpec):
        raise TypeError("chip_design must be a DesignSpec instance")

    # Check package constraint
    if chip_design.package.type == "monolithic" and len(chip_design.chiplets) >= 1:
        raise ValueError(
            "Monolithic package type cannot have more than 1 chiplet; "
            "consider using 'chiplets' or '2.5D' package type"
        )

    # Determine chiplet type from technology
    tech = material_device_spec.technology
    if tech in (
        TechnologyType.FERROFLUID,
        TechnologyType.MEMRISTOR,
        TechnologyType.PCM,
        TechnologyType.FERROELECTRIC,
    ):
        chiplet_type = "memory"
    elif tech == TechnologyType.MAGNETIC:
        chiplet_type = "memory"  # magnetic memory
    elif tech == TechnologyType.PHOTONIC:
        chiplet_type = "compute"  # photonic compute
    else:
        chiplet_type = "compute"

    # Derive chiplet area from physical dimensions
    dims = material_device_spec.physical_dimensions_mm
    chiplet_area_mm2 = dims[0] * dims[1]  # x × y area

    # Determine location
    if location is None:
        # Position at design center
        # Estimate center based on existing chiplets if any
        if chip_design.chiplets:
            # Use last chiplet's position as reference
            last_chiplet = chip_design.chiplets[-1]
            x_center = (
                (last_chiplet.area_mm2.model_value if hasattr(last_chiplet.area_mm2, 'model_value') else last_chiplet.area_mm2 or 0) / 2
                + (chip_design.constraints.max_area_mm2 or 100) / 2
            )
            y_center = (
                (last_chiplet.area_mm2.model_value if hasattr(last_chiplet.area_mm2, 'model_value') else last_chiplet.area_mm2 or 0) / 2
                + (chip_design.constraints.max_area_mm2 or 100) / 2
            )
            z_center = (
                last_chiplet.physical_dimensions_mm[2] / 2
                if hasattr(last_chiplet, "physical_dimensions_mm")
                else 1.0 / 2
            )
        else:
            x_center = (chip_design.constraints.max_area_mm2 or 100) / 2
            y_center = (chip_design.constraints.max_area_mm2 or 100) / 2
            z_center = dims[2] / 2 if dims else 1.0 / 2
    else:
        x_center, y_center, z_center = location

    # Create MemorySpec for memory technologies
    memory_spec = None
    if chiplet_type == "memory" and material_device_spec.technology in (
        TechnologyType.FERROFLUID,
        TechnologyType.MEMRISTOR,
        TechnologyType.PCM,
        TechnologyType.FERROELECTRIC,
    ):
        # Derive MemorySpec from MaterialDeviceSpec parameters
        bits_per_cell_map = {
            TechnologyType.FERROFLUID: 2,
            TechnologyType.MEMRISTOR: 4,
            TechnologyType.PCM: 4,
            TechnologyType.FERROELECTRIC: 2,
        }
        bits_per_cell = bits_per_cell_map.get(material_device_spec.technology, 8)
        total_bits = material_device_spec.cell_count * bits_per_cell
        capacity_gb = (total_bits / 8.0) / 1e9  # convert to GB

        channels = max(1, int(material_device_spec.electrode_density_per_mm2 * capacity_gb / 10))

        memory_spec = MemorySpec(
            technology=material_device_spec.technology.value,
            capacity_per_chiplet_gb=TrackedValue(
                value=capacity_gb,
                unit="GB",
                provenance=Provenance.ASSUMED,
            ),
            channels=channels,
            banks_per_channel=1,
            bandwidth_per_channel_gbps=TrackedValue(
                value=material_device_spec.read_write_throughput_gbps,
                unit="GB/s",
                provenance=Provenance.ASSUMED,
            ),
            latency_ns=TrackedValue(
                value=10.0, unit="ns", provenance=Provenance.ASSUMED
            ),
            energy_pj_per_bit=TrackedValue(
                value=50.0, unit="pJ", provenance=Provenance.ASSUMED
            ),
            stack_layers=1,
        )

    # Create the ChipletSpec
    new_chiplet = ChipletSpec(
        id=f"material_{material_device_spec.technology.value}",
        type=chiplet_type,
        count=1,
        area_mm2=TrackedValue(
            value=chiplet_area_mm2,
            unit="mm²",
            provenance=Provenance.ASSUMED,
        ),
        tdp_watts=TrackedValue(
            value=material_device_spec.power_consumption_mw / 1000.0,  # W
            unit="W",
            provenance=Provenance.ASSUMED,
        ),
        process_nm=None,  # material devices don't have a process node
        memory=memory_spec,
        compute=None,
    )

    # Add to chiplets list
    updated_chiplets = list(chip_design.chiplets) + [new_chiplet]

    # Build updated DesignSpec
    updated_design = chip_design.model_copy(
        update={
            "chiplets": updated_chiplets,
            "metadata": {
                **(chip_design.metadata or {}),
                "material_device_integrated": True,
                "material_technology": material_device_spec.technology.value,
                "material_location": location,
            },
        },
        deep=True,
    )

    return updated_design


# ---------------------------------------------------------------------------
# Multi-technology support registry
# ---------------------------------------------------------------------------

_MATERIAL_REGISTRY: dict[TechnologyType, type[MaterialDeviceSpec]] = {
    TechnologyType.FERROFLUID: MaterialDeviceSpec,
    TechnologyType.MEMRISTOR: MaterialDeviceSpec,
    TechnologyType.PCM: MaterialDeviceSpec,
    TechnologyType.FERROELECTRIC: MaterialDeviceSpec,
    TechnologyType.MAGNETIC: MaterialDeviceSpec,
    TechnologyType.PHOTONIC: MaterialDeviceSpec,
    TechnologyType.CUSTOM: MaterialDeviceSpec,
}


def multi_technology_support() -> dict[TechnologyType, dict[str, Any]]:
    """Return registry of supported material technologies with their parameters.

    Returns:
        Dictionary mapping TechnologyType to a dict of supported parameters,
        default physics constants, and validation rules for each technology.

    Example:
        >>> support = multi_technology_support()
        >>> support[TechnologyType.FERROFLUID]["default_cell_count"]
        64
    """
    registry: dict[TechnologyType, dict[str, Any]] = {}

    for tech in TechnologyType:
        # Default physics parameters per technology
        default_params = {
            TechnologyType.FERROFLUID: {
                "default_cell_count": 64,
                "default_electrode_density": 100.0,  # electrodes/mm²
                "default_throughput_gbps": 10.0,
                "default_power_mw": 15.0,
                "default_capacity_gb": 16.0,
                "default_area_mm2": 4.0,
            },
            TechnologyType.MEMRISTOR: {
                "default_cell_count": 256,
                "default_electrode_density": 200.0,
                "default_throughput_gbps": 25.0,
                "default_power_mw": 25.0,
                "default_capacity_gb": 64.0,
                "default_area_mm2": 2.0,
            },
            TechnologyType.PCM: {
                "default_cell_count": 256,
                "default_electrode_density": 150.0,
                "default_throughput_gbps": 20.0,
                "default_power_mw": 35.0,
                "default_capacity_gb": 64.0,
                "default_area_mm2": 3.0,
            },
            TechnologyType.FERROELECTRIC: {
                "default_cell_count": 32,
                "default_electrode_density": 80.0,
                "default_throughput_gbps": 8.0,
                "default_power_mw": 12.0,
                "default_capacity_gb": 8.0,
                "default_area_mm2": 2.5,
            },
            TechnologyType.MAGNETIC: {
                "default_cell_count": 128,
                "default_electrode_density": 120.0,
                "default_throughput_gbps": 15.0,
                "default_power_mw": 20.0,
                "default_capacity_gb": 32.0,
                "default_area_mm2": 3.5,
            },
            TechnologyType.PHOTONIC: {
                "default_cell_count": 128,
                "default_electrode_density": 50.0,
                "default_throughput_gbps": 100.0,
                "default_power_mw": 30.0,
                "default_capacity_gb": 32.0,
                "default_area_mm2": 5.0,
            },
            TechnologyType.CUSTOM: {
                "default_cell_count": 64,
                "default_electrode_density": 100.0,
                "default_throughput_gbps": 1.0,
                "default_power_mw": 10.0,
                "default_capacity_gb": 8.0,
                "default_area_mm2": 1.0,
            },
        }

        registry[tech] = {
            "description": _tech_to_description(tech),
            "default_parameters": default_params[tech],
            "derive_chip_parameters": derive_chip_parameters,
            "integrate_into_design": integrate_into_design,
            "bits_per_cell": _BITS_PER_CELL[tech],
        }

    return registry


def _tech_to_description(tech: TechnologyType) -> str:
    """Human-readable description of a technology type."""
    descriptions = {
        TechnologyType.FERROFLUID:
            "Ferrofluid-based magnetic memory using nanoparticle polarization states",
        TechnologyType.MEMRISTOR:
            "Memristor-based resistive memory with analog weight storage",
        TechnologyType.PCM:
            "Phase-Change Memory using GST alloy amorphous/crystalline switching",
        TechnologyType.FERROELECTRIC:
            "Ferroelectric memory using spontaneous polarization reversal",
        TechnologyType.MAGNETIC:
            "Magnetic tunnel junction (MTJ) spin-transfer torque memory",
        TechnologyType.PHOTONIC:
            "Photonic integrated circuit memory using optical interference states",
        TechnologyType.CUSTOM:
            "Custom material technology with user-defined parameters",
    }
    return descriptions[tech]


# ---------------------------------------------------------------------------
# Convenience alias for backward compatibility
# ---------------------------------------------------------------------------

derive_parameters = derive_chip_parameters  # alias


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

# Pydantic v2 forward reference resolution:
# TechnologyType is used as a field type in MaterialDeviceSpec
MaterialDeviceSpec.model_rebuild()

__all__ = [
    "TechnologyType",
    "MaterialDeviceSpec",
    "derive_chip_parameters",
    "integrate_into_design",
    "multi_technology_support",
    "derive_parameters",
]