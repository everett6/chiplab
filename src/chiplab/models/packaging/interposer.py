"""2.5D silicon interposer models (CoWoS, EMIB, InFO_oS).

Provides parasitic RC models, bandwidth, power, and latency calculations
for silicon interposers used in advanced IC packaging.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Tuple

from chiplab.schemas.common import Provenance, TrackedValue


def interposer_rc(
    interposer_area_mm2: float,
    metal_layers: int,
    rdl_density: float,
    bump_pitch_um: float,
) -> Tuple[TrackedValue, TrackedValue, TrackedValue]:
    """Extract R, L, C parasitics for a silicon interposer.

    Args:
        interposer_area_mm2: Total interposer area in mm².
        metal_layers: Number of metal layers in the interposer.
        rdl_density: Routing density level (0.0 = sparse, 1.0 = dense).
        bump_pitch_um: Bump pitch in micrometers.

    Returns:
        Tuple of (R, L, C) as TrackedValue with ohms, henries, farads units
        and provenance ASSUMED.
    """
    # Unit conversion
    area_cm2 = interposer_area_mm2 / 100.0  # mm² to cm²
    pitch_cm = bump_pitch_um / 1e4  # um to cm

    # Resistance: sheet resistance of silicon interposer ~ 100 mΩ/sq
    # R scales with area and inversely with number of parallel paths (metal layers * rdl_density)
    sheet_r_ohms_sq = 0.1  # 100 mΩ per square
    # Number of effective squares ~ sqrt(area) / pitch, adjusted for metal layers
    effective_squares = np.sqrt(area_cm2) / pitch_cm / max(metal_layers, 1) / rdl_density
    R = TrackedValue(
        value=sheet_r_ohms_sq * effective_squares,
        unit="Ω",
        provenance=Provenance.ASSUMED,
        source="interposer_rc_model",
    )

    # Inductance: planar spiral inductance approximation
    # L ~ μ0 * length * (ln(length/widh) + 1) ... simplified for interposer
    # L scales with area and metal layer count
    L = TrackedValue(
        value=1e-9 * metal_layers * np.sqrt(area_cm2),  # nH simplified
        unit="H",
        provenance=Provenance.ASSUMED,
        source="interposer_l_model",
    )

    # Capacitance: parallel plate capacitance between metal layers
    # C ~ ε_r * ε0 * area / spacing
    # ε_r ~ 11.7 for silicon, ε0 = 8.854e-14 F/cm
    epsilon_r = 11.7
    epsilon0 = 8.854e-14  # F/cm
    spacing_cm = pitch_cm * 0.5  # assumed spacing between metal layers
    C = TrackedValue(
        value=epsilon_r * epsilon0 * area_cm2 / spacing_cm,
        unit="F",
        provenance=Provenance.ASSUMED,
        source="interposer_c_model",
    )

    return R, L, C


def interposer_bandwidth(
    bump_pitch_um: float,
    num_bumps: int,
    data_rate_gbps: float,
    signaling: Literal["NRZ", "PAM4", "CDR"] = "NRZ",
) -> TrackedValue:
    """Calculate aggregate bandwidth for an interposer bump array.

    Args:
        bump_pitch_um: Bump pitch in micrometers.
        num_bumps: Number of bumps in the array.
        data_rate_gbps: Data rate per bump in Gbps.
        signaling: Signaling scheme (NRZ, PAM4, CDR).

    Returns:
        TrackedValue with aggregate bandwidth in GB/s and provenance ASSUMED.
    """
    # Effective bandwidth per bump depends on signaling
    # NRZ: 1 bit per symbol, PAM4: 2 bits per symbol, CDR: effective gain
    signaling_factor = {"NRZ": 1.0, "PAM4": 2.0, "CDR": 1.2}[signaling]

    # Bump limited bandwidth - not all bumps can switch at full rate simultaneously
    # due to power and thermal constraints. Use ~70% efficiency factor.
    efficiency = 0.7

    # Total aggregate bandwidth
    total_bw_gbps = num_bumps * data_rate_gbps * signaling_factor * efficiency

    # Convert to GB/s (divide by 8 for bytes)
    total_bw_gb_s = total_bw_gbps / 8.0

    return TrackedValue(
        value=total_bw_gb_s,
        unit="GB/s",
        provenance=Provenance.ASSUMED,
        source="interposer_bandwidth_model",
    )


def interposer_power(
    bump_pitch_um: float,
    num_bumps: int,
    data_rate_gbps: float,
    voltage: float = 1.1,
    signaling: Literal["NRZ", "PAM4", "CDR"] = "NRZ",
) -> TrackedValue:
    """Calculate power consumption for interposer bump array.

    Args:
        bump_pitch_um: Bump pitch in micrometers.
        num_bumps: Number of bumps in the array.
        data_rate_gbps: Data rate per bump in Gbps.
        voltage: Supply voltage in Volts (default 1.1V).
        signaling: Signaling scheme (NRZ, PAM4, CDR).

    Returns:
        TrackedValue with power in Watts and provenance ASSUMED.
    """
    # Dynamic power: P = α * C * V^2 * f
    # α ~ activity factor, C ~ capacitance per bump, V ~ voltage, f ~ frequency

    # NRZ: 1 bit/eff, PAM4: 2 bits/eff but higher power per bit
    signaling_factor = {"NRZ": 1.0, "PAM4": 1.8, "CDR": 1.3}[signaling]

    # Approximate power model: P = num_bumps * k * pitch² * V² * data_rate
    # k is a process-dependent constant ~ 1e-3 W/μm²/Gbps/V²
    k = 1e-3  # process-dependent constant

    # Effective pitch area
    pitch_area_mm2 = (bump_pitch_um ** 2) / 1e6  # μm² to mm²

    power_w = TrackedValue(
        value=num_bumps * k * pitch_area_mm2 * (voltage ** 2) * data_rate_gbps * signaling_factor,
        unit="W",
        provenance=Provenance.ASSUMED,
        source="interposer_power_model",
    )

    return power_w


def interposer_latency(
    distance_mm: float, signaling: Literal["NRZ", "PAM4", "CDR"] = "NRZ"
) -> TrackedValue:
    """Calculate signal latency across an interposer.

    Args:
        distance_mm: Signal distance across interposer in mm.
        signaling: Signaling scheme (NRZ, PAM4, CDR).

    Returns:
        TrackedValue with latency in ns and provenance ASSUMED.
    """
    # Signal propagation speed in silicon ~ 0.5 * c (speed of light)
    # c = 3e8 m/s, so v ~ 1.5e8 m/s = 1.5e5 mm/ns
    # Delay = distance / speed

    # NRZ has fastest edge rate, PAM4 slower due to more levels, CDR equalizes
    speed_factors = {"NRZ": 1.0, "PAM4": 0.85, "CDR": 0.95}
    speed_mm_ns = 150.0 * speed_factors[signaling]  # mm/ns in silicon

    latency_ns = TrackedValue(
        value=distance_mm / speed_mm_ns,
        unit="ns",
        provenance=Provenance.ASSUMED,
        source="interposer_latency_model",
    )

    return latency_ns


# Convenience functions for specific interposer types


def coswo_s(
    interposer_area_mm2: float,
    metal_layers: int = 12,
    rdl_density: float = 0.8,
    bump_pitch_um: float = 15.0,
) -> dict[str, TrackedValue]:
    """CoWoS-S (Silicon interposer for high-performance computing).

    Typical parameters: 12 metal layers, dense RDL, 15 μm bump pitch.
    """
    R, L, C = interposer_rc(interposer_area_mm2, metal_layers, rdl_density, bump_pitch_um)
    return {
        "R": R,
        "L": L,
        "C": C,
    }


def coswo_r(
    interposer_area_mm2: float,
    metal_layers: int = 10,
    rdl_density: float = 0.7,
    bump_pitch_um: float = 20.0,
) -> dict[str, TrackedValue]:
    """CoWoS-R (Reduced interposer for I/O oriented applications).

    Typical parameters: 10 metal layers, moderate RDL, 20 μm bump pitch.
    """
    R, L, C = interposer_rc(interposer_area_mm2, metal_layers, rdl_density, bump_pitch_um)
    return {
        "R": R,
        "L": L,
        "C": C,
    }


def emib(
    interposer_area_mm2: float,
    metal_layers: int = 8,
    rdl_density: float = 0.5,
    bump_pitch_um: float = 30.0,
) -> dict[str, TrackedValue]:
    """EMIB (Embedded Multi-die Interconnect Bridge).

    Typical parameters: 8 metal layers, sparse RDL, 30 μm bump pitch.
    """
    R, L, C = interposer_rc(interposer_area_mm2, metal_layers, rdl_density, bump_pitch_um)
    return {
        "R": R,
        "L": L,
        "C": C,
    }


def info_s(
    interposer_area_mm2: float,
    metal_layers: int = 10,
    rdl_density: float = 0.9,
    bump_pitch_um: float = 12.0,
) -> dict[str, TrackedValue]:
    """InFO_oS (Interposer-Free Overhead Stack).

    Typical parameters: 10 metal layers, very dense RDL, 12 μm bump pitch.
    """
    R, L, C = interposer_rc(interposer_area_mm2, metal_layers, rdl_density, bump_pitch_um)
    return {
        "R": R,
        "L": L,
        "C": C,
    }