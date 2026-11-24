"""Clock tree synthesis & analysis models.

Provides clock tree power estimation, skew analysis, and insertion delay
calculation for synchronous designs.
"""

from __future__ import annotations

from chiplab.schemas.common import TrackedValue, Provenance
import numpy as np


def clock_tree_power(
    num_ff: int,
    clock_freq_ghz: float,
    clock_cap_ff: float,
    num_sinks_per_buf: int,
    buffer_size: int,
) -> float:
    """Calculate clock tree power consumption in milliwatts (mW).

    Uses the standard model:
        P = alpha * C * V^2 * f

    Where alpha is the activity factor, C is total capacitance,
    V is supply voltage, and f is clock frequency.

    For clock trees, we model:
        - Total load capacitance = num_ff * clock_cap_ff + buffer capacitance
        - Buffer count based on sinks and buffer size
        - Activity factor derived from toggle rate

    Parameters
    ----------
    num_ff : int
        Number of flip-flops in the design.
    clock_freq_ghz : float
        Clock frequency in gigahertz.
    clock_cap_ff : float
        Clock capacitance per flip-flop in femtofarads.
    num_sinks_per_buf : int
        Number of sinks per buffer stage.
    buffer_size : int
        Buffer drive size (relative unit).

    Returns
    -------
    float
        Clock tree power in milliwatts (mW).
    """
    # Supply voltage (typical for modern processes)
    vdd = 1.0  # Voltage in volts, will be scaled per process

    # Total load capacitance (in farads)
    # C_total = num_ff * clock_cap_ff (in fF) + buffer capacitance
    # Convert fF to fF, then to Farads: 1 fF = 1e-15 F
    c_load_ff = float(num_ff) * float(clock_cap_ff)

    # Buffer tree structure
    # Number of buffers needed to drive all sinks
    # Each buffer drives num_sinks_per_buf loads
    # Total buffer count considering the tree structure
    if num_sinks_per_buf > 0:
        # Total sinks = num_ff (each FF is a sink)
        # Buffer tree: first level drives num_sinks_per_buf, etc.
        # Rough estimate: num_bufs ≈ num_ff / num_sinks_per_buf
        num_bufs = max(1, int(np.ceil(float(num_ff) / float(num_sinks_per_buf))))
    else:
        num_bufs = 1

    # Buffer capacitance (approximate, proportional to size)
    # Each buffer adds some capacitance
    buf_cap_per_stage_fF = 0.1 * float(buffer_size)  # fF per stage

    # Total buffer capacitance
    total_buf_cap_fF = num_bufs * buf_cap_per_stage_fF

    # Total load capacitance including buffers
    total_capacitance_fF = c_load_ff + total_buf_cap_fF

    # Activity factor for clock (typically 0.5 for ideal clock)
    activity_factor = 0.5

    # Convert frequency from GHz to Hz
    freq_hz = float(clock_freq_ghz) * 1e9

    # Convert capacitance from fF to Farads
    total_capacitance_fF_total = total_capacitance_fF * 1e-15

    # Dynamic power: P = alpha * C * V^2 * f
    # P in watts, then convert to mW
    power_w = activity_factor * total_capacitance_fF_total * (vdd ** 2) * freq_hz
    power_mw = power_w * 1e3

    return float(power_mw)


def clock_skew(
    num_sinks: int,
    num_levels: int,
    variation_pct: float,
) -> float:
    """Estimate clock skew in picoseconds.

    Clock skew arises from:
    - Buffer delay variations
    - Wire delay variations
    - Process variations
    - Temperature variations

    The skew is modeled as accumulating through the tree levels:
        skew = base_skew * sqrt(levels) * (1 + variation/100)

    Where base_skew depends on the number of sinks and tree topology.

    Parameters
    ----------
    num_sinks : int
        Number of clock sink destinations.
    num_levels : int
        Number of levels in the clock tree.
    variation_pct : float
        Process/temperature variation percentage.

    Returns
    -------
    float
        Estimated clock skew in picoseconds (ps).
    """
    # Base skew model: each level adds some delay variation
    # For a balanced H-tree or mesh, skew grows as sqrt(N)

    # Base delay per stage (approximate)
    base_stage_delay_ps = 10.0  # ps per stage for typical buffer

    # Skew grows with square root of number of levels for H-tree
    # and linearly with number of levels for some topologies
    # Using sqrt model for H-tree which is common

    # sqrt(num_levels) factor for random variation accumulation
    skew_factor = np.sqrt(max(1, num_levels))

    # Variation contribution
    variation_factor = 1.0 + (variation_pct / 100.0)

    # Total skew
    total_skew_ps = base_stage_delay_ps * skew_factor * variation_factor

    # Also account for number of sinks - more sinks means deeper tree
    # Additional skew from sink distribution
    if num_sinks > 1:
        sink_distribution_factor = np.log2(max(2, num_sinks))
        total_skew_ps *= sink_distribution_factor / np.log2(2)

    return float(total_skew_ps)


def ccsr_insertion_delay(
    num_sinks: int,
    wire_length_mm: float,
    buffer_chain: int,
) -> float:
    """Calculate CCRS (Cell-Clock-Reset-Set) insertion delay in picoseconds.

    Models the delay through a buffer chain driving a wire:
        t_insertion = t_buf * ceil(log2(N)) + t_wire

    Where:
        t_buf = buffer delay
        N = number of sinks
        t_wire = wire delay = r_wire * c_wire * L^2 / 2 (Elmore)

    Parameters
    ----------
    num_sinks : int
        Number of clock sinks.
    wire_length_mm : float
        Total wire length in millimeters.
    buffer_chain : int
        Number of buffers in the chain.

    Returns
    -------
    float
        Insertion delay in picoseconds (ps).
    """
    # Buffer delay per stage (typical inverter delay)
    t_buf_ps = 50.0  # ps per buffer stage

    # Number of buffer stages needed for N sinks
    # Binary tree: log2(N) stages
    num_buf_stages = max(1, int(np.ceil(np.log2(max(2, num_sinks)))))

    # Ensure we don't exceed the provided buffer chain length
    actual_buf_stages = min(num_buf_stages, buffer_chain)

    # Buffer chain delay
    buf_delay_ps = actual_buf_stages * t_buf_ps

    # Wire delay using Elmore model
    # For a distributed RC line: t_wire = 0.5 * R * C * L^2
    # We need resistance and capacitance per unit length

    # Typical values for on-chip interconnects
    # R_per_mm ≈ 1 ohm/mm for typical metal layers
    # C_per_mm ≈ 0.1 fF/mm = 0.1e-15 F/mm

    # Convert wire length to consistent units
    length_mm = float(wire_length_mm)

    # Elmore wire delay: t_wire = 0.5 * r * c * L^2
    # r = resistance per mm, c = capacitance per mm
    r_per_mm = 1.0  # ohms/mm (approximate)
    c_per_mm_fF = 0.1  # fF/mm

    # Convert capacitance to Farads
    c_per_mm_F = c_per_mm_fF * 1e-15

    # Elmore delay
    wire_delay_ps = 0.5 * r_per_mm * c_per_mm_F * (length_mm ** 2) * 1e12

    # Total insertion delay
    total_delay_ps = buf_delay_ps + wire_delay_ps

    return float(total_delay_ps)