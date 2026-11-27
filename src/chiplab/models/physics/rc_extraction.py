"""RC wire delay & repeater insertion models.

Provides Elmore delay calculation, repeater insertion optimization,
and wire capacitance extraction for VLSI interconnects.

SKY130 wire R and C come from the SkyWater PDK RCX tables (cited in SKY130_SOURCE); other
processes need explicit resistivity, thickness and capacitance.
"""

from __future__ import annotations

from chiplab.schemas.common import TrackedValue, Provenance
import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend


_RCX = "https://foss-eda-tools.googlesource.com/skywater-pdk/+/refs/heads/main/docs/rules/rcx/"

# SKY130 interconnect from the SkyWater PDK RCX tables (values as printed there).
#   r_sq_ohm      resistance-values.tsv, "Layer Resistivity (mohms/sq)" / 1000
#   below         the layer directly underneath (worst case: a dense lower layer)
#   c_area_af_um2 capacitance-parallel.tsv, plate cap to `below` (aF/um^2)
#   c_fringe_af_um capacitance-fringe-downward.tsv, fringe cap to `below` per edge (aF/um)
SKY130_LAYERS: dict[str, dict[str, float | str]] = {
    "li1":  {"r_sq_ohm": 12.8,  "below": "poly", "c_area_af_um2": 94.1644,
             "c_fringe_af_um": 51.846},
    "met1": {"r_sq_ohm": 0.125, "below": "li1",  "c_area_af_um2": 114.1970,
             "c_fringe_af_um": 59.496},
    "met2": {"r_sq_ohm": 0.125, "below": "met1", "c_area_af_um2": 133.8610,
             "c_fringe_af_um": 67.045},
    "met3": {"r_sq_ohm": 0.047, "below": "met2", "c_area_af_um2": 86.1861,
             "c_fringe_af_um": 69.846},
    "met4": {"r_sq_ohm": 0.047, "below": "met3", "c_area_af_um2": 84.0346,
             "c_fringe_af_um": 70.522},
    "met5": {"r_sq_ohm": 0.029, "below": "met4", "c_area_af_um2": 68.3252,
             "c_fringe_af_um": 82.819},
}
SKY130_SOURCE = _RCX + " (resistance-values.tsv, capacitance-parallel.tsv, " \
                       "capacitance-fringe-downward.tsv)"
_ALIASES = {f"M{i}": f"met{i}" for i in range(1, 6)}


def sky130_wire_rc(width_um: float, metal_layer: str) -> dict[str, float | str]:
    """Per-micron R and C of a SKY130 wire over a dense layer below it.

    R/um = R_sq / W.  C/um = c_area W + 2 c_fringe (both edges). Same-layer sidewall
    coupling to neighbours is not in the PDK tables and is left out, so C is a lower bound
    for a wire with close neighbours."""
    layer = _ALIASES.get(metal_layer, metal_layer)
    if layer not in SKY130_LAYERS:
        raise KeyError(f"sky130 layers are li1 and met1-met5 (or M1-M5); got {metal_layer!r}")
    if width_um <= 0:
        raise ValueError("width_um must be positive")
    d = SKY130_LAYERS[layer]
    r = float(d["r_sq_ohm"]) / width_um
    c = (float(d["c_area_af_um2"]) * width_um + 2 * float(d["c_fringe_af_um"])) * 1e-18
    return {"layer": layer, "r_ohm_per_um": r, "c_f_per_um": c, "source": SKY130_SOURCE}


def rc_delay(
    length_mm: float,
    width_um: float,
    metal_layer: str,
    resistivity_uc_ohm_cm: float | None = None,
    dielectric_er: float | None = None,
    device: str = "auto",
    thickness_um: float | None = None,
    c_ff_per_um: float | None = None,
) -> float:
    """Elmore delay of a uniform distributed RC wire, in picoseconds: t = 0.5 R C.

    (The 50 % step-response delay of the same line is ~0.38 R C.)

    By default R and C come from the SKY130 PDK RCX tables (see sky130_wire_rc); metal_layer
    is li1, met1-met5, or M1-M5 as aliases for met1-met5.

    Overrides for other processes: resistivity_uc_ohm_cm (bulk, micro-ohm cm) needs
    thickness_um, and gives R = rho L / (W t); c_ff_per_um sets the capacitance per micron.
    dielectric_er scales the table capacitance by er / 3.9 (assumes the tables reflect an
    SiO2-like effective permittivity). Wire driver and load are not included.

    Before 2026-11 this function used a placeholder (rho in micro-ohm cm read as ohm cm, no
    thickness, 0.1 er fF/mm), which under-estimated a 1 mm met1 wire by ~30x.
    """
    get_gpu_backend(device=device)
    if length_mm < 0 or width_um <= 0:
        raise ValueError("length_mm must be >= 0 and width_um > 0")
    length_um = length_mm * 1e3
    custom_r = resistivity_uc_ohm_cm is not None
    if custom_r:
        if not thickness_um or thickness_um <= 0:
            raise ValueError("resistivity_uc_ohm_cm needs thickness_um (R = rho L / (W t))")
        r_per_um = resistivity_uc_ohm_cm * 1e-8 / (width_um * 1e-6 * thickness_um * 1e-6) * 1e-6
    if c_ff_per_um is not None:
        c_per_um = c_ff_per_um * 1e-15
    if not custom_r or c_ff_per_um is None:
        try:
            tab = sky130_wire_rc(width_um, metal_layer)
        except KeyError:
            raise KeyError(f"no preset for {metal_layer!r}: give resistivity_uc_ohm_cm + "
                           "thickness_um and c_ff_per_um for non-SKY130 processes") from None
        if not custom_r:
            r_per_um = float(tab["r_ohm_per_um"])
        if c_ff_per_um is None:
            c_per_um = float(tab["c_f_per_um"]) * ((dielectric_er / 3.9) if dielectric_er
                                                    else 1.0)
    delay_s = 0.5 * (r_per_um * length_um) * (c_per_um * length_um)
    return float(delay_s * 1e12)


def wire_capacitance(
    length_mm: float,
    width_um: float,
    thickness_um: float,
    er: float,
    device: str = "auto",
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
    device : str, default "auto"
        Device to run solver on. "auto", "cpu", or "cuda".
        If "auto", uses GPU if available, otherwise CPU.
        Note: For wire_capacitance, this parameter is reserved for future GPU offloading;
        the function currently runs on CPU but the parameter is maintained for
        API consistency with other physics models.

    Returns
    -------
    float
        Capacitance in femtofarads (fF).
    """
    backend = get_gpu_backend(device=device)

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
    device: str = "auto",
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
    device : str, default "auto"
        Device to run solver on. "auto", "cpu", or "cuda".
        If "auto", uses GPU if available, otherwise CPU.
        Note: For insert_repeaters, this parameter is reserved for future GPU offloading;
        the function currently runs on CPU but the parameter is maintained for
        API consistency with other physics models.

    Returns
    -------
    dict
        Dictionary with:
        - "num_repeaters": int, optimal number of repeaters
        - "spacing_mm": float, spacing between repeaters in mm
        - "repeater_size": float, relative buffer size
        - "min_delay_ps": float, minimum achievable delay in ps
    """
    backend = get_gpu_backend(device=device)

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