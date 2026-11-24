"""RC wire delay & repeater insertion models.

Provides Elmore delay calculation, repeater insertion optimization,
and wire capacitance extraction for VLSI interconnects.

Metal layer presets are defined for sky130, tsmc_n7, and tsmc_n3 processes.
"""

from __future__ import annotations

from chiplab.schemas.common import TrackedValue, Provenance
import numpy as np


# Metal layer presets for various processes
# Format: {metal_layer: {"resistivity_uc_ohm_cm": float, "dielectric_er": float}}
_METAL_PRESETS = {
    "sky130": {
        "M1": {"resistivity_uc_ohm_cm": 1.95, "dielectric_er": 3.8},
        "M2": {"resistivity_uc_ohm_cm": 1.75, "dielectric_er": 3.8},
        "M3": {"resistivity_uc_ohm_cm": 1.55, "dielectric_er": 3.8},
        "M4": {"resistivity_uc_ohm_cm": 1.35, "dielectric_er": 3.8},
        "M5": {"resistivity_uc_ohm_cm": 1.15, "dielectric_er": 3.8},
        "M6": {"resistivity_uc_ohm_cm": 0.95, "dielectric_er": 3.8},
        "M7": {"resistivity_uc_ohm_cm": 0.75, "dielectric_er": 3.8},
        "M8": {"resistivity_uc_ohm_cm": 0.55, "dielectric_er": 3.8},
    },
    "tsmc_n7": {
        "M1": {"resistivity_uc_ohm_cm": 2.10, "dielectric_er": 3.5},
        "M2": {"resistivity_uc_ohm_cm": 1.90, "dielectric_er": 3.5},
        "M3": {"resistivity_uc_ohm_cm": 1.70, "dielectric_er": 3.5},
        "M4": {"resistivity_uc_ohm_cm": 1.50, "dielectric_er": 3.5},
        "M5": {"resistivity_uc_ohm_cm": 1.30, "dielectric_er": 3.5},
        "M6": {"resistivity_uc_ohm_cm": 1.10, "dielectric_er": 3.5},
        "M7": {"resistivity_uc_ohm_cm": 0.90, "dielectric_er": 3.5},
        "M8": {"resistivity_uc_ohm_cm": 0.70, "dielectric_er": 3.5},
    },
    "tsmc_n3": {
        "M1": {"resistivity_uc_ohm_cm": 2.30, "dielectric_er": 3.4},
        "M2": {"resistivity_uc_ohm_cm": 2.10, "dielectric_er": 3.4},
        "M3": {"resistivity_uc_ohm_cm": 1.90, "dielectric_er": 3.4},
        "M4": {"resistivity_uc_ohm_cm": 1.70, "dielectric_er": 3.4},
        "M5": {"resistivity_uc_ohm_cm": 1.50, "dielectric_er": 3.4},
        "M6": {"resistivity_uc_ohm_cm": 1.30, "dielectric_er": 3.4},
        "M7": {"resistivity_uc_ohm_cm": 1.10, "dielectric_er": 3.4},
        "M8": {"resistivity_uc_ohm_cm": 0.90, "dielectric_er": 3.4},
    },
}


def rc_delay(
    length_mm: float,
    width_um: float,
    metal_layer: str,
    resistivity_uc_ohm_cm: float | None = None,
    dielectric_er: float | None = None,
) -> float:
    """Calculate Elmore RC delay in picoseconds.

    The Elmore delay for a uniform RC line is:
        t_delay = 0.5 * R * C * L^2

    Where R is resistance per unit length, C is capacitance per unit length,
    and L is the wire length.

    Parameters
    ----------
    length_mm : float
        Wire length in millimeters.
    width_um : float
        Wire width in micrometers.
    metal_layer : str
        Metal layer name (e.g., "M1", "M2", ..., "M8").
    resistivity_uc_ohm_cm : float, optional
        Sheet resistivity in ohm-cm. If not provided, uses metal layer preset.
    dielectric_er : float, optional
        Dielectric relative permittivity. If not provided, uses metal layer preset.

    Returns
    -------
    float
        Elmore delay in picoseconds (ps).

    Raises
    ------
    KeyError
        If metal_layer is not found in the presets and no custom values provided.
    """
    # Use metal layer preset if available
    if metal_layer in _METAL_PRESETS.get("sky130", {}):
        preset = _METAL_PRESETS["sky130"][metal_layer]
        if resistivity_uc_ohm_cm is None:
            resistivity_uc_ohm_cm = preset["resistivity_uc_ohm_cm"]
        if dielectric_er is None:
            dielectric_er = preset["dielectric_er"]
    elif metal_layer in _METAL_PRESETS.get("tsmc_n7", {}):
        preset = _METAL_PRESETS["tsmc_n7"][metal_layer]
        if resistivity_uc_ohm_cm is None:
            resistivity_uc_ohm_cm = preset["resistivity_uc_ohm_cm"]
        if dielectric_er is None:
            dielectric_er = preset["dielectric_er"]
    elif metal_layer in _METAL_PRESETS.get("tsmc_n3", {}):
        preset = _METAL_PRESETS["tsmc_n3"][metal_layer]
        if resistivity_uc_ohm_cm is None:
            resistivity_uc_ohm_cm = preset["resistivity_uc_ohm_cm"]
        if dielectric_er is None:
            dielectric_er = preset["dielectric_er"]

    # Constants
    # Resistance per sq: R_sq = resistivity_uc / width (in consistent units)
    # Capacitance per unit length: C ~ epsilon * epsilon_0 * width / spacing
    # Using simplified models for VLSI interconnects

    # Convert units
    length_cm = length_mm * 0.1  # mm to cm
    width_cm = width_um * 1e-4  # um to cm

    # Resistance: R = resistivity * length / (width * thickness)
    # For simplicity, using sheet resistance model
    if resistivity_uc_ohm_cm is not None and resistivity_uc_ohm_cm > 0:
        # Sheet resistance per square: R_sq = resistivity / effective_thickness
        # We approximate: R = resistivity * length / width
        resistance_ohm = resistivity_uc_ohm_cm * length_cm / width_cm if width_cm > 0 else np.inf
    else:
        resistance_ohm = 1e12  # fallback very large value

    # Capacitance per unit length (fF/mm): C ~ epsilon_0 * epsilon_r * width / spacing
    # Assuming spacing ~ width for minimum spacing
    # C_per_mm = epsilon_0 * epsilon_r * (width_mm / spacing_mm) ≈ epsilon_0 * epsilon_r
    # epsilon_0 = 8.854e-12 F/m = 8.854e-3 fF/mm
    # For a rough estimate: C_per_mm ≈ 0.1 * epsilon_r fF/mm for typical VLSI
    capacitance_per_mm_fF = 0.1 * dielectric_er if dielectric_er else 0.1

    # Total capacitance
    total_capacitance_fF = capacitance_per_mm_fF * length_mm

    # Elmore delay: t = 0.5 * R * C
    # But we need consistent units. R in ohms, C in farads.
    # C in fF = C * 1e-15
    # t_ps = 0.5 * R * C * 1e12

    if resistance_ohm > 0 and total_capacitance_fF > 0:
        delay_ps = 0.5 * resistance_ohm * total_capacitance_fF * 1e-15 * 1e12
    else:
        delay_ps = 0.0

    return float(delay_ps)


def wire_capacitance(
    length_mm: float,
    width_um: float,
    thickness_um: float,
    er: float,
) -> float:
    """Calculate wire capacitance in femtofarads (fF).

    Uses parallel-plate capacitance model:
        C = epsilon_0 * epsilon_r * (area / distance)

    Where area = length * width, distance = thickness.

    Parameters
    ----------
    length_mm : float
        Wire length in millimeters.
    width_um : float
        Wire width in micrometers.
    thickness_um : float
        Insulator thickness in micrometers.
    er : float
        Dielectric relative permittivity.

    Returns
    -------
    float
        Capacitance in femtofarads (fF).
    """
    # Constants
    epsilon_0 = 8.854187817e-12  # F/m

    # Convert all to meters
    length_m = length_mm * 1e-3
    width_m = width_um * 1e-6
    thickness_m = thickness_um * 1e-6

    # Parallel-plate capacitance: C = epsilon_0 * er * (width * length) / thickness
    # But for interconnect capacitance, we use a more practical model
    # C_per_mm ≈ 0.1 * er fF/mm for typical VLSI

    # More accurate: C = epsilon_0 * er * width * length / thickness
    area_m2 = width_m * length_m
    capacitance_fF = epsilon_0 * er * area_m2 / thickness_m * 1e15

    return float(capacitance_fF)


def insert_repeaters(
    length_mm: float,
    r_per_um: float,
    c_per_um: float,
    gate_delay_ps: float,
) -> dict:
    """Optimize repeater insertion for minimum delay.

    Uses the classic model for optimal number of repeaters:
        N_opt = sqrt(r * C * L^3 / (2 * t_gate))

    And optimal spacing:
        L_opt = L / (N_opt + 1)

    Where:
        r = resistance per unit length per repeater stage
        c = capacitance per unit length per stage
        L = total wire length
        t_gate = gate delay

    Parameters
    ----------
    length_mm : float
        Total wire length in millimeters.
    r_per_um : float
        Resistance per micrometer in ohms/um.
    c_per_um : float
        Capacitance per micrometer in fF/um.
    gate_delay_ps : float
        Gate delay per buffer in picoseconds.

    Returns
    -------
    dict
        Dictionary with:
        - "num_repeaters": int, optimal number of repeaters
        - "spacing_mm": float, spacing between repeaters in mm
        - "repeater_size": float, relative buffer size
        - "min_delay_ps": float, minimum achievable delay in ps
    """
    if length_mm <= 0 or r_per_um <= 0 or c_per_um <= 0 or gate_delay_ps <= 0:
        return {
            "num_repeaters": 0,
            "spacing_mm": length_mm,
            "repeater_size": 1.0,
            "min_delay_ps": gate_delay_ps * length_mm / 1.0,
        }

    # Convert length to micrometers for consistency
    length_um = length_mm * 1e6

    # Calculate optimal number of repeaters using the Rent's rule / Elmore model
    # N_opt ≈ sqrt(r * c * L^3 / (2 * t_gate))
    # where r and c are per-stage resistance and capacitance

    # Using the formula: N = sqrt(r_per_um * c_per_um * L_um^3 / (2 * gate_delay_ps))
    numerator = r_per_um * c_per_um * (length_um ** 3)
    denominator = 2 * gate_delay_ps

    if numerator > 0 and denominator > 0:
        num_repeaters = int(np.sqrt(numerator / denominator))
    else:
        num_repeaters = 0

    # Check if spacing is reasonable - if spacing between repeaters would be too small,
    # use no repeaters for short wires
    min_spacing_um = 5.0  # minimum spacing between repeaters in micrometers
    if num_repeaters > 0:
        spacing_mm = length_mm / (num_repeaters + 1)
        if spacing_mm * 1e3 < min_spacing_um:
            num_repeaters = 0

    # Ensure at least 0 and reasonable max
    num_repeaters = max(0, min(num_repeaters, 50))

    # If no repeaters, spacing is the full length
    if num_repeaters == 0:
        return {
            "num_repeaters": 0,
            "spacing_mm": length_mm,
            "repeater_size": 1.0,
            "min_delay_ps": gate_delay_ps * (num_repeaters + 1),
        }

    # Spacing between repeaters
    spacing_mm = length_mm / (num_repeaters + 1)

    # Repeater size scaling (larger buffers drive more capacitance)
    # Simple model: size proportional to load per stage
    repeater_size = float(num_repeaters + 1) / max(1, num_repeaters)

    # Calculate minimum delay with repeaters
    # Each segment has RC delay, plus buffer delay
    segment_length_mm = length_mm / (num_repeaters + 1)
    # RC delay per segment
    r_segment = r_per_um * segment_length_mm * 1e3  # convert mm to um
    c_segment = c_per_um * segment_length_mm * 1e3
    rc_delay_ps = 0.5 * r_segment * c_segment
    buf_delay_ps = gate_delay_ps  # one buffer per segment

    # Total delay: N segments * (RC + buf) but with Elmore optimization
    min_delay_ps = (num_repeaters + 1) * (rc_delay_ps + buf_delay_ps) / (num_repeaters + 1)

    return {
        "num_repeaters": num_repeaters,
        "spacing_mm": spacing_mm,
        "repeater_size": repeater_size,
        "min_delay_ps": float(min_delay_ps),
    }