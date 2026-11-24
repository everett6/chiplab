"""UCIe die-to-die PHY models.

Provides bandwidth, power, latency, and area calculations for
the UCIe (Universal Chiplet Interconnect express) die-to-die interface.
Supports UCIe 1.0, 1.1, and 2.0 standards, standard and advanced packages.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from chiplab.schemas.common import Provenance, TrackedValue


def ucie_bandwidth(
    lane_count: int,
    data_rate_gbps: float,
    width: Literal["standard", "advanced"] = "standard",
) -> TrackedValue:
    """Calculate UCIe bandwidth per direction.

    Args:
        lane_count: Number of UCIe lanes.
        data_rate_gbps: Data rate per lane in Gbps (depends on UCIe version).
        width: Package type ("standard" or "advanced").

    Returns:
        TrackedValue with bandwidth per direction in GB/s and provenance ASSUMED.
    """
    # UCIe data rates by version:
    # UCIe 1.0: 16 Gbps per lane (PAM4)
    # UCIe 1.1: 32 Gbps per lane (PAM4)
    # UCIe 2.0: 64 Gbps per lane (PAM4)
    # For advanced packages, effective rate may be higher due to wider PHY

    # Determine base data rate based on typical UCIe version
    # Since version isn't a parameter, we use a reasonable default
    # and adjust for width and lane count

    # Standard width has 1x effective lanes, advanced has wider PHY
    width_factor = 1.0 if width == "standard" else 1.5  # advanced has 50% wider PHY

    # Base rate per lane - assume UCIe 2.0 speed as baseline, adjustable
    base_rate_per_lane_gbps = 64.0  # UCIe 2.0 PAM4 rate

    # Total bandwidth per direction (lanes are unidirectional in each direction)
    total_bw_gbps = lane_count * base_rate_per_lane_gbps * width_factor

    # Convert to GB/s (bytes per second)
    total_bw_gb_s = total_bw_gbps / 8.0

    return TrackedValue(
        value=total_bw_gb_s,
        unit="GB/s",
        provenance=Provenance.ASSUMED,
        source="ucie_bandwidth_model",
    )


def ucie_power(
    lane_count: int,
    data_rate_gbps: float,
    voltage: float = 1.1,
    width: Literal["standard", "advanced"] = "standard",
) -> TrackedValue:
    """Calculate UCIe power consumption.

    Args:
        lane_count: Number of UCIe lanes.
        data_rate_gbps: Data rate per lane in Gbps.
        voltage: Supply voltage in Volts (default 1.1V).
        width: Package type ("standard" or "advanced").

    Returns:
        TrackedValue with power in Watts and provenance ASSUMED.
    """
    # Dynamic power: P = α * C * V^2 * f
    # Scales with lane count, data rate, and voltage squared

    # Width factor: advanced packages have more capacitance
    width_factor = 1.0 if width == "standard" else 1.3

    # Power per lane: proportional to data rate and V²
    # Typical coefficient: ~1e-3 W per lane at 32 Gbps / 1.1V
    k_per_lane = 1e-3 * (data_rate_gbps / 32.0) * (voltage / 1.1) ** 2

    power_w = TrackedValue(
        value=lane_count * k_per_lane * width_factor,
        unit="W",
        provenance=Provenance.ASSUMED,
        source="ucie_power_model",
    )

    return power_w


def ucie_latency(
    lane_count: int,
    data_rate_gbps: float,
    reach_mm: float,
) -> TrackedValue:
    """Calculate UCIe latency per direction.

    Args:
        lane_count: Number of UCIe lanes.
        data_rate_gbps: Data rate per lane in Gbps.
        reach_mm: Electrical reach in millimeters.

    Returns:
        TrackedValue with latency in ns and provenance ASSUMED.
    """
    # Signal propagation: ~5 ps/mm in PCB/trace, ~2 ps/mm in silicon
    # UCIe uses serialized PAM4, so we account for unit interval

    # Propagation delay dominates for longer reaches
    # Speed ~ 0.65 * c in PCB materials
    speed_mm_ns = 0.65 * 300.0  # ~195 mm/ns (0.65 * speed of light in mm/ns)

    # Propagation delay
    prop_delay_ns = reach_mm / speed_mm_ns

    # Unit interval based on data rate
    # UI = 1 / (data_rate_gbps * 1e9) seconds = 1000 / data_rate_gbps ps
    ui_ps = 1000.0 / data_rate_gbps  # pico-seconds per unit interval

    # Total latency includes propagation + at least 1 UI for handshaking
    # For simplicity, use propagation + 0.5 UI (average)
    total_latency_ns = prop_delay_ns + 0.5 * ui_ps / 1000.0  # convert ps to ns

    return TrackedValue(
        value=total_latency_ns,
        unit="ns",
        provenance=Provenance.ASSUMED,
        source="ucie_latency_model",
    )


def ucie_area(
    lane_count: int,
    width: Literal["standard", "advanced"] = "standard",
) -> TrackedValue:
    """Calculate UCIe PHY area.

    Args:
        lane_count: Number of UCIe lanes.
        width: Package type ("standard" or "advanced").

    Returns:
        TrackedValue with PHY area in mm² and provenance ASSUMED.
    """
    # Area per lane depends on package type and technology node
    # Standard: ~0.5 mm² per lane (includes SerDes, pads, etc.)
    # Advanced: ~0.4 mm² per lane (more compact PHY)

    area_per_lane_mm2 = 0.5 if width == "standard" else 0.4

    total_area_mm2 = lane_count * area_per_lane_mm2

    return TrackedValue(
        value=total_area_mm2,
        unit="mm²",
        provenance=Provenance.ASSUMED,
        source="ucie_area_model",
    )


# Convenience functions for specific UCIe versions


def ucie_1p0(
    lane_count: int,
    width: Literal["standard", "advanced"] = "standard",
) -> dict[str, TrackedValue]:
    """UCIe 1.0: 16 Gbps per lane, standard/advanced package.

    Args:
        lane_count: Number of UCIe lanes.
        width: Package type ("standard" or "advanced").

    Returns:
        Dictionary of TrackedValue metrics.
    """
    # UCIe 1.0: 16 Gbps per lane (PAM4)
    data_rate = 16.0

    bandwidth = ucie_bandwidth(lane_count, data_rate, width)
    power = ucie_power(lane_count, data_rate, 1.1, width)
    latency = ucie_latency(lane_count, data_rate, 10.0)  # 10mm typical reach
    area = ucie_area(lane_count, width)

    return {
        "bandwidth_gb_s": bandwidth,
        "power_W": power,
        "latency_ns": latency,
        "area_mm2": area,
    }


def ucie_1p1(
    lane_count: int,
    width: Literal["standard", "advanced"] = "standard",
) -> dict[str, TrackedValue]:
    """UCIe 1.1: 32 Gbps per lane, standard/advanced package.

    Args:
        lane_count: Number of UCIe lanes.
        width: Package type ("standard" or "advanced").

    Returns:
        Dictionary of TrackedValue metrics.
    """
    # UCIe 1.1: 32 Gbps per lane (PAM4)
    data_rate = 32.0

    bandwidth = ucie_bandwidth(lane_count, data_rate, width)
    power = ucie_power(lane_count, data_rate, 1.1, width)
    latency = ucie_latency(lane_count, data_rate, 10.0)  # 10mm typical reach
    area = ucie_area(lane_count, width)

    return {
        "bandwidth_gb_s": bandwidth,
        "power_W": power,
        "latency_ns": latency,
        "area_mm2": area,
    }


def ucie_2p0(
    lane_count: int,
    width: Literal["standard", "advanced"] = "standard",
) -> dict[str, TrackedValue]:
    """UCIe 2.0: 64 Gbps per lane, standard/advanced package.

    Args:
        lane_count: Number of UCIe lanes.
        width: Package type ("standard" or "advanced").

    Returns:
        Dictionary of TrackedValue metrics.
    """
    # UCIe 2.0: 64 Gbps per lane (PAM4)
    data_rate = 64.0

    bandwidth = ucie_bandwidth(lane_count, data_rate, width)
    power = ucie_power(lane_count, data_rate, 1.1, width)
    latency = ucie_latency(lane_count, data_rate, 10.0)  # 10mm typical reach
    area = ucie_area(lane_count, width)

    return {
        "bandwidth_gb_s": bandwidth,
        "power_W": power,
        "latency_ns": latency,
        "area_mm2": area,
    }