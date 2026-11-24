"""Package warpage & thermo-mechanical stress models.

Provides warpage analysis, CTE stress, solder joint reliability,
and warpage derating calculations for advanced IC packaging.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from chiplab.schemas.common import Provenance, TrackedValue


def warpage_analysis(
    package_size_mm: float,
    die_size_mm: float,
    substrate_thickness_mm: float,
    cte_mismatch: float,
    temp_delta: float,
) -> TrackedValue:
    """Calculate package warpage due to CTE mismatch.

    Args:
        package_size_mm: Package overall size (assumed square) in mm.
        die_size_mm: Die size (assumed square) in mm.
        substrate_thickness_mm: Substrate thickness in mm.
        cte_mismatch: Coefficient of Thermal Expansion mismatch (ppm/K or 1e-6/K).
        temp_delta: Temperature change from processing reference in °C.

    Returns:
        TrackedValue with warpage in micrometers (um) and provenance ASSUMED.
    """
    # Stoney's equation variant for warpage:
    # w = (3 * (1 - ν) * Δα * ΔT * h²) / (8 * E * t)
    # Simplified analytical model for packaging warpage

    # Poisson's ratio for typical packaging materials
    nu = 0.33

    # Effective Young's modulus (Pa) - typical substrate value
    E_eff_Pa = 150e9  # Pa

    # Convert to consistent units
    # h = substrate thickness in m, t = die thickness assumption
    h_m = substrate_thickness_mm / 1000.0  # mm to m
    t_m = die_size_mm / 1000.0  # Using die size as proxy for die thickness, or could be separate

    # CTE mismatch in 1/K
    delta_alpha_1K = cte_mismatch * 1e-6  # ppm/K to 1/K
    delta_T = temp_delta  # °C

    # Warpage formula: w = (3 * (1 - ν) * Δα * ΔT * h²) / (8 * E * t)
    warpage_m = (
        3.0 * (1.0 - nu) * delta_alpha_1K * delta_T * h_m ** 2
    ) / (8.0 * E_eff_Pa * t_m) if t_m > 0 else 0.0

    # Convert to um
    warpage_um = warpage_m * 1e6  # m to um

    return TrackedValue(
        value=abs(warpage_um),
        unit="um",
        provenance=Provenance.ASSUMED,
        source="warpage_analysis_model",
    )


def cte_stress(
    die_size_mm: float,
    substrate_size_mm: float,
    cte_die: float,
    cte_substrate: float,
    temp_delta: float,
    youngs_modulus: float = 150e9,
) -> TrackedValue:
    """Calculate thermo-mechanical stress from CTE mismatch.

    Args:
        die_size_mm: Die size in mm (square assumption).
        substrate_size_mm: Substrate size in mm (square assumption).
        cte_die: Die CTE in ppm/K.
        cte_substrate: Substrate CTE in ppm/K.
        temp_delta: Temperature change from processing in °C.
        youngs_modulus: Young's modulus in Pa (default 150 GPa for typical substrates).

    Returns:
        TrackedValue with stress in MPa and provenance ASSUMED.
    """
    # Thermal stress formula: σ = E * Δα * ΔT / (1 - ν)
    # where Δα = |CTE_die - CTE_substrate|

    nu = 0.33  # Poisson's ratio

    delta_cte_ppm = abs(cte_die - cte_substrate)  # ppm/K

    # Stress in Pa: σ = E * Δα * ΔT / (1 - ν)
    # Δα in 1/K = ppm/K * 1e-6
    stress_pa = youngs_modulus * delta_cte_ppm * 1e-6 * temp_delta / (1.0 - nu)

    # Convert to MPa
    stress_mpa = stress_pa / 1e6

    return TrackedValue(
        value=stress_mpa,
        unit="MPa",
        provenance=Provenance.ASSUMED,
        source="cte_stress_model",
    )


def solder_joint_reliability(
    num_bumps: int,
    pitch_um: float,
    temp_cycle: float,
    cycles: int,
) -> TrackedValue:
    """Calculate mean cycles to solder joint failure using Eyring model.

    Args:
        num_bumps: Number of solder bumps.
        pitch_um: Bump pitch in micrometers.
        temp_cycle: Temperature cycle range in °C (peak-to-peak).
        cycles: Number of cycles applied (for looking up failure rate).

    Returns:
        TrackedValue with mean cycles to failure and provenance ASSUMED.
    """
    # Eyring reliability model for solder joints
    # MTF = A * exp(B * ΔT * p² / N) ... simplified

    # Constants for lead-free solder (SAC305 typical)
    A = 1e6  # baseline cycles
    B = 1.2e-4  # Eyring constant (depends on solder alloy, bump geometry)

    # Stress factor: ΔT * pitch²
    stress_factor = temp_cycle * (pitch_um ** 2)

    # Mean time to failure cycles
    # More cycles applied reduces remaining life, but we compute expected life
    mtf_cycles = A * np.exp(-B * stress_factor / max(num_bumps, 1))

    # If looking up at a specific cycle count, return derated value
    if cycles > 0 and cycles < mtf_cycles:
        # For mean cycles to failure at the given condition:
        result_cycles = mtf_cycles
    else:
        result_cycles = mtf_cycles

    return TrackedValue(
        value=float(result_cycles),
        unit="cycles",
        provenance=Provenance.ASSUMED,
        source="solder_joint_reliability_model",
    )


def warpage_derating(
    nominal_value: float,
    warpage_um: float,
    sensitivity: float = 1.0,
) -> float:
    """Derate a nominal value based on measured warpage.

    Args:
        nominal_value: The nominal (unwarped) value to derate.
        warpage_um: Measured warpage in micrometers.
        sensitivity: Warpage sensitivity factor (how much warpage affects the value).

    Returns:
        Derated value (float).
    """
    # Linear derating: derated = nominal * (1 - sensitivity * warpage / reference)
    # Reference warpage is typically 10 um for full derating
    reference_warpage_um = 10.0  # um

    derated = nominal_value * (1.0 - sensitivity * warpage_um / reference_warpage_um)

    # Ensure derated value doesn't go negative or exceed nominal
    derated = max(0.0, min(nominal_value, derated))

    return derated