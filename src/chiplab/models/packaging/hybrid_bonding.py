"""3D hybrid bonding models (SoIC, Foveros, 3D V-Cache).

Provides RC models for hybrid bonds, TSV thermal and electrical models,
and bandwidth/power/latency calculations for 3D stacked packages.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Tuple

from chiplab.schemas.common import Provenance, TrackedValue


def hybrid_bond_rc(
    bond_pitch_um: float,
    num_bonds: int,
    bond_type: Literal["copper_tungsten", "gold", "aluminum"] = "copper_tungsten",
) -> Tuple[TrackedValue, TrackedValue]:
    """Extract R, C parasitics per hybrid bond.

    Args:
        bond_pitch_um: Bond pitch in micrometers.
        num_bonds: Number of bonds.
        bond_type: Metal type for the bond (copper_tungsten, gold, aluminum).

    Returns:
        Tuple of (R, C) per bond as TrackedValue with ohms, farads units
        and provenance ASSUMED.
    """
    # Resistance per bond: depends on metal resistivity and bond dimensions
    # R = rho * L / A, where L ~ bond_pitch, A ~ cross-sectional area

    # Resistivity values at room temperature (Ω·mm)
    resistivities = {
        "copper_tungsten": 0.025,
        "gold": 0.022,
        "aluminum": 0.031,
    }
    rho = resistivities.get(bond_type, 0.025)

    # Bond length approximated as pitch + some overhang
    bond_length_mm = bond_pitch_um / 1e3  # um to mm

    # Cross-sectional area assumed circular with diameter ~ 0.7 * pitch
    # A = π * (d/2)²
    diameter_mm = 0.7 * bond_pitch_um / 1e3  # um to mm
    area_mm2 = np.pi * (diameter_mm / 2) ** 2

    # Resistance per bond
    R = TrackedValue(
        value=rho * bond_length_mm / area_mm2 if area_mm2 > 0 else np.inf,
        unit="Ω",
        provenance=Provenance.ASSUMED,
        source="hybrid_bond_rc_model",
    )

    # Capacitance per bond: parallel plate approximation
    # C = ε * A / d, where d ~ spacing ~ bond_pitch
    # ε_r ~ 3.9 for typical bond line (epoxy/underfill)
    epsilon_r = 3.9
    epsilon0 = 8.854e-14  # F/cm = 8.854e-12 F/m
    # Convert to consistent units: mm
    # C = ε_r * ε0 * A_mm2 / d_mm * 1e3 (to get pF from F*m)
    # Actually let's use: C = ε_r * ε0 * A / d where A in m², d in m
    area_m2 = area_mm2 * 1e-6  # mm² to m²
    d_m = bond_pitch_um / 1e9  # um to m
    C_f = epsilon_r * epsilon0 * area_m2 / d_m  # Farads

    C = TrackedValue(
        value=C_f * 1e12,  # Convert to pF
        unit="F",
        provenance=Provenance.ASSUMED,
        source="hybrid_bond_c_model",
    )

    return R, C


def hybrid_bond_bandwidth(
    bond_pitch_um: float,
    num_bonds: int,
    data_rate_gbps: float,
) -> TrackedValue:
    """Calculate 3D bandwidth for hybrid bonded bump array.

    Args:
        bond_pitch_um: Bond pitch in micrometers.
        num_bonds: Number of bonds.
        data_rate_gbps: Data rate per bond in Gbps.

    Returns:
        TrackedValue with 3D bandwidth in GB/s and provenance ASSUMED.
    """
    # Hybrid bonding provides higher bandwidth density than traditional bumps
    # Efficiency factor higher than interposer (~85% vs ~70%)
    efficiency = 0.85

    # Total bandwidth
    total_bw_gbps = num_bonds * data_rate_gbps * efficiency

    # Convert to GB/s
    total_bw_gb_s = total_bw_gbps / 8.0

    return TrackedValue(
        value=total_bw_gb_s,
        unit="GB/s",
        provenance=Provenance.ASSUMED,
        source="hybrid_bond_bandwidth_model",
    )


def hybrid_bond_power(
    bond_pitch_um: float,
    num_bonds: int,
    data_rate_gbps: float,
    voltage: float = 1.1,
) -> TrackedValue:
    """Calculate power consumption for hybrid bonded bump array.

    Args:
        bond_pitch_um: Bond pitch in micrometers.
        num_bonds: Number of bonds.
        data_rate_gbps: Data rate per bond in Gbps.
        voltage: Supply voltage in Volts (default 1.1V).

    Returns:
        TrackedValue with power in Watts and provenance ASSUMED.
    """
    # Hybrid bond power is typically lower than traditional bump due to
    # shorter interconnects and better thermal management

    # Per-bond power coefficient
    k = 5e-4  # W/μm²/Gbps/V² (lower than traditional bumps)

    # Bond area
    pitch_area_mm2 = (bond_pitch_um ** 2) / 1e6  # μm² to mm²

    power_w = TrackedValue(
        value=num_bonds * k * pitch_area_mm2 * (voltage ** 2) * data_rate_gbps,
        unit="W",
        provenance=Provenance.ASSUMED,
        source="hybrid_bond_power_model",
    )

    return power_w


def tsv_rc(
    tsv_diameter_um: float,
    tsv_depth_um: float,
    tsv_pitch_um: float,
    liner_thickness_um: float,
) -> Tuple[TrackedValue, TrackedValue]:
    """Extract R, C parasitics for a Through-Silicon Via (TSV).

    Args:
        tsv_diameter_um: TSV diameter in micrometers.
        tsv_depth_um: TSV depth (silicon thickness) in micrometers.
        tsv_pitch_um: TSV pitch (center-to-center) in micrometers.
        liner_thickness_um: Oxide liner thickness in micrometers.

    Returns:
        Tuple of (R, C) as TrackedValue with ohms, farads units
        and provenance ASSUMED.
    """
    # Geometry
    diameter_cm = tsv_diameter_um / 1e4  # um to cm
    depth_cm = tsv_depth_um / 1e4  # um to cm
    pitch_cm = tsv_pitch_um / 1e4  # um to cm
    liner_cm = liner_thickness_um / 1e4  # um to cm

    # Silicon resistivity: ~ 1000 Ω·cm (varies by doping)
    rho_si = 1000.0  # Ω·cm

    # Resistance: current flows through the TSV depth
    # R = rho * depth / area
    # Effective area = π * (diameter/2)² - π * ( (diameter/2) - liner )²
    # = π/4 * (diameter² - (diameter - 2*liner)²)
    inner_diameter_cm = diameter_cm - 2 * liner_cm

    if inner_diameter_cm > 0:
        area_cm2 = np.pi / 4 * (diameter_cm ** 2 - inner_diameter_cm ** 2)
    else:
        # Liner covers entire diameter
        area_cm2 = np.pi / 4 * diameter_cm ** 2

    R = TrackedValue(
        value=rho_si * depth_cm / area_cm2 if area_cm2 > 0 else np.inf,
        unit="Ω",
        provenance=Provenance.ASSUMED,
        source="tsv_rc_model",
    )

    # Capacitance: TSV to surrounding substrate (cylindrical capacitor)
    # C = 2π ε L / ln(b/a)
    # ε_r ~ 11.7 for silicon, but effective ε_r depends on surroundings
    # Let's use effective ε_r ~ 4.0 for TSV to substrate capacitance
    epsilon_r_eff = 4.0
    epsilon0 = 8.854e-14  # F/cm

    # Outer radius = pitch/2, inner radius = (diameter/2 + liner)
    outer_radius_cm = pitch_cm / 2
    inner_radius_cm = diameter_cm / 2 + liner_cm

    if outer_radius_cm > inner_radius_cm > 0:
        # C per via = 2π ε L / ln(b/a)
        C_f = 2 * np.pi * epsilon_r_eff * epsilon0 * depth_cm / np.log(outer_radius_cm / inner_radius_cm)
    else:
        # Fallback if geometry is degenerate
        C_f = epsilon_r_eff * epsilon0 * area_cm2 / depth_cm

    C = TrackedValue(
        value=C_f,
        unit="F",
        provenance=Provenance.ASSUMED,
        source="tsv_c_model",
    )

    return R, C


def tsv_thermal(
    tsv_diameter_um: float,
    tsv_depth_um: float,
    tsv_pitch_um: float,
    num_tsvs: int,
) -> TrackedValue:
    """Calculate thermal conductance for an array of TSVs.

    Args:
        tsv_diameter_um: TSV diameter in micrometers.
        tsv_depth_um: TSV depth in micrometers.
        tsv_pitch_um: TSV pitch in micrometers.
        num_tsvs: Number of TSVs in the array.

    Returns:
        TrackedValue with thermal conductance in W/K and provenance ASSUMED.
    """
    # Thermal conductance via TSVs: G = k * A / d
    # k ~ thermal conductivity of silicon ~ 150 W/(m·K) at room temperature

    # Individual TSV conductance
    diameter_m = tsv_diameter_um / 1e6  # um to m
    depth_m = tsv_depth_um / 1e6  # um to m

    # Cross-sectional area of TSV (silicon)
    area_per_tsv_m2 = np.pi * (diameter_m / 2) ** 2

    # Conductance of single TSV: G = k * A / d
    k_si = 150.0  # W/(m·K) - silicon thermal conductivity
    g_single = k_si * area_per_tsv_m2 / depth_m  # W/K

    # Array of TSVs - assume thermal parallel paths
    # But thermal coupling reduces efficiency; use ~80% array efficiency
    array_efficiency = 0.8
    total_g = g_single * num_tsvs * array_efficiency

    return TrackedValue(
        value=total_g,
        unit="W/K",
        provenance=Provenance.ASSUMED,
        source="tsv_thermal_model",
    )


# Convenience functions for specific hybrid bonding types


def soic(
    bond_pitch_um: float = 5.0,
    num_bonds: int = 10000,
    bond_type: Literal["copper_tungsten", "gold", "aluminum"] = "copper_tungsten",
) -> dict[str, TrackedValue]:
    """SoIC (System-on-Interconnect) hybrid bonding.

    Typical parameters: 5 μm pitch, ~10000 bonds, copper-tungsten.
    """
    R, C = hybrid_bond_rc(bond_pitch_um, num_bonds, bond_type)
    bandwidth = hybrid_bond_bandwidth(bond_pitch_um, num_bonds, 100.0)  # 100 Gbps typical
    power = hybrid_bond_power(bond_pitch_um, num_bonds, 100.0, 1.1)
    return {
        "R_per_bond": R,
        "C_per_bond": C,
        "bandwidth_gb_s": bandwidth,
        "power_W": power,
    }


def foveros(
    bond_pitch_um: float = 3.0,
    num_bonds: int = 5000,
    bond_type: Literal["copper_tungsten", "gold", "aluminum"] = "copper_tungsten",
) -> dict[str, TrackedValue]:
    """Foveros hybrid bonding.

    Typical parameters: 3 μm pitch, ~5000 bonds, copper-tungsten.
    """
    R, C = hybrid_bond_rc(bond_pitch_um, num_bonds, bond_type)
    bandwidth = hybrid_bond_bandwidth(bond_pitch_um, num_bonds, 80.0)  # 80 Gbps typical
    power = hybrid_bond_power(bond_pitch_um, num_bonds, 80.0, 1.1)
    return {
        "R_per_bond": R,
        "C_per_bond": C,
        "bandwidth_gb_s": bandwidth,
        "power_W": power,
    }


def v_cache(
    bond_pitch_um: float = 4.0,
    num_bonds: int = 8000,
    bond_type: Literal["copper_tungsten", "gold", "aluminum"] = "copper_tungsten",
) -> dict[str, TrackedValue]:
    """3D V-Cache hybrid bonding.

    Typical parameters: 4 μm pitch, ~8000 bonds, copper-tungsten.
    """
    R, C = hybrid_bond_rc(bond_pitch_um, num_bonds, bond_type)
    bandwidth = hybrid_bond_bandwidth(bond_pitch_um, num_bonds, 120.0)  # 120 Gbps typical
    power = hybrid_bond_power(bond_pitch_um, num_bonds, 120.0, 1.1)
    return {
        "R_per_bond": R,
        "C_per_bond": C,
        "bandwidth_gb_s": bandwidth,
        "power_W": power,
    }