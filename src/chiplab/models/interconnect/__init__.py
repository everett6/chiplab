from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from chiplab.schemas.simulation import AssumptionRecord
from chiplab.schemas.common import Provenance
from chiplab.schemas.design import DesignSpec
from chiplab.library.components import COMPONENT_LIBRARY, get_component_defaults


def estimate_interconnect(
    design: DesignSpec,
    total_chiplets: int,
    allow_defaults: bool = True,
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Estimate interconnect parameters based on design and topology.

    Handles pcie, chiplet_link, mesh, crossbar, ring, bus topologies.

    Computes:
        - bisection_bw_tbps: bisection bandwidth in TB/s
        - per_link_bw_gbps: per-link bandwidth in GB/s
        - hop_latency_ns: average hop latency in ns
        - avg_hops: average number of hops
        - energy_pj_per_bit: energy per bit in pJ
        - power_watts: interconnect power in watts
        - area_mm2: interconnect area in mm²

    Args:
        design: DesignSpec containing interconnect configuration
        total_chiplets: total number of chiplets in the system
        allow_defaults: whether to use component library defaults when params are missing

    Returns:
        tuple of (metrics_dict, assumption_records, warnings)
    """
    assumptions: list[AssumptionRecord] = []
    warnings: list[str] = []

    interconnect_spec = design.interconnect
    topology = interconnect_spec.topology

    # Values written in the design always win; component-library defaults for the topology
    # only fill fields the design leaves unset (they used to replace the design's values).
    comp_defaults = get_component_defaults("interconnect", topology)
    units = {"link_bandwidth_gbps": "GB/s", "latency_ns": "ns", "energy_pj_per_bit": "pJ/bit",
             "lanes_per_chiplet": "count"}
    resolved: dict[str, Any] = {}
    for field_name, unit in units.items():
        value = getattr(interconnect_spec, field_name, None)
        if value is not None:
            resolved[field_name] = value
            assumptions.append(AssumptionRecord(
                parameter=field_name, value=value, unit=unit,
                provenance=Provenance.MEASURED, source="design_spec"))
        elif comp_defaults is not None and field_name in comp_defaults:
            resolved[field_name] = comp_defaults[field_name]
            if not allow_defaults:
                warnings.append(f"{field_name} missing; using the {topology} component default "
                                "despite allow_defaults=False")
            assumptions.append(AssumptionRecord(
                parameter=field_name, value=comp_defaults[field_name], unit=unit,
                provenance=Provenance.ASSUMED, source="component_library"))
        else:
            resolved[field_name] = None
            warnings.append(f"{field_name} not specified")
    link_bw_gbps = resolved["link_bandwidth_gbps"]
    latency_ns = resolved["latency_ns"]
    energy_pj_per_bit = resolved["energy_pj_per_bit"]
    lanes_per_chiplet = resolved["lanes_per_chiplet"] or 1

    # Determine per-link bandwidth
    per_link_bw_gbps = float(link_bw_gbps) if link_bw_gbps is not None else 0.0

    # Compute bisection bandwidth
    # For mesh: bisection ~ (sqrt(n) * link_bw) / 2 where n = total_chiplets
    # For crossbar: full bisection = total_chiplets * link_bw / 2
    # For ring: bisection = link_bw / 2
    # For pcie: point-to-point, bisection depends on link width
    # For chiplet_link: similar to mesh

    if topology == "mesh":
        # Mesh interconnect: sqrt(total_chiplets) links in each dimension
        n_sqrt = total_chiplets ** 0.5
        bisection_bw_tbps = (n_sqrt * per_link_bw_gbps) / 2000.0  # convert GB/s to TB/s
        avg_hops = 2 * n_sqrt  # average hops in 2D mesh
    elif topology == "crossbar":
        # Crossbar: full connectivity, bisection is half the total bandwidth
        total_links_bw = total_chiplets * per_link_bw_gbps
        bisection_bw_tbps = (total_links_bw / 2.0) / 1000.0  # GB/s to TB/s
        avg_hops = 1.0  # crossbar: 1 hop
    elif topology == "ring":
        # Ring: bisection is half the link bandwidth
        bisection_bw_tbps = (per_link_bw_gbps / 2.0) / 1000.0  # GB/s to TB/s
        avg_hops = total_chiplets / 4.0  # average hops in ring
    elif topology == "pcie":
        # PCIe: point-to-point, assume x16 link per chiplet
        # Bisection is roughly total bandwidth / 2
        total_bw = total_chiplets * per_link_bw_gbps
        bisection_bw_tbps = (total_bw / 2.0) / 1000.0  # GB/s to TB/s
        avg_hops = 2.0  # typical for PCIe switching
    elif topology == "chiplet_link":
        # Chiplet link: mesh-like
        n_sqrt = total_chiplets ** 0.5
        bisection_bw_tbps = (n_sqrt * per_link_bw_gbps) / 2000.0  # GB/s to TB/s
        avg_hops = 2 * n_sqrt  # similar to mesh
    elif topology == "bus":
        # Bus: shared bandwidth, bisection = total / 2
        total_bw = per_link_bw_gbps
        bisection_bw_tbps = (total_bw / 2.0) / 1000.0  # GB/s to TB/s
        avg_hops = total_chiplets / 2.0  # worst case for bus
    else:
        # Unknown topology, default to mesh-like calculation
        warnings.append(f"Unknown topology '{topology}', defaulting to mesh-like calculation")
        n_sqrt = total_chiplets ** 0.5
        bisection_bw_tbps = (n_sqrt * per_link_bw_gbps) / 2000.0  # GB/s to TB/s
        avg_hops = 2 * n_sqrt

    # Hop latency
    hop_latency_ns = float(latency_ns) if latency_ns is not None else 0.0
    if latency_ns is None and comp_defaults is not None:
        warnings.append("latency_ns not specified, using component default or 0")

    # Energy per bit
    energy_pj_per_bit_val = float(energy_pj_per_bit) if energy_pj_per_bit is not None else 0.0
    if energy_pj_per_bit is None and comp_defaults is not None:
        warnings.append("energy_pj_per_bit not specified, using component default or 0")

    # Interconnect power estimation
    # Rough estimate: power proportional to bandwidth and number of links
    # P = alpha * BW * V^2 * f, but simplified to:
    # power_watts ~ 0.1 * total_bw_gbps (rough industry estimate)
    total_bw_gbps = per_link_bw_gbps * total_chiplets
    power_watts = 0.1 * total_bw_gbps  # 0.1W per GB/s as rough estimate
    if per_link_bw_gbps is None:
        warnings.append(
            f"power_watts estimated with assumption: 0.1W per GB/s, total_bw_gbps={total_bw_gbps}"
        )

    # Interconnect area estimation
    # Typical area per link: ~1-5 mm² depending on topology
    # For mesh: area scales with number of links
    # For crossbar: area scales with N^2
    if topology == "crossbar":
        area_mm2 = 5.0 * total_chiplets ** 2  # crossbar area grows quadratically
        warnings.append(
            f"area_mm2 for crossbar estimated as {area_mm2:.1f} mm² (N^2 scaling)"
        )
    elif topology == "mesh":
        # Mesh: area proportional to number of links * area per link
        num_links = total_chiplets * 4  # 4 links per chiplet in mesh
        area_mm2 = num_links * 1.0  # 1 mm² per link
    elif topology == "ring":
        area_mm2 = total_chiplets * 0.5  # 0.5 mm² per chiplet in ring
    elif topology == "pcie":
        area_mm2 = total_chiplets * 2.0  # 2 mm² per PCIe interface
    elif topology == "bus":
        area_mm2 = 10.0  # shared bus area
    else:
        area_mm2 = total_chiplets * 1.0  # default: 1 mm² per chiplet

    result: dict[str, float] = {
        "bisection_bw_tbps": float(bisection_bw_tbps),
        "per_link_bw_gbps": float(per_link_bw_gbps),
        "hop_latency_ns": hop_latency_ns,
        "avg_hops": float(avg_hops),
        "energy_pj_per_bit": energy_pj_per_bit_val,
        "power_watts": float(power_watts),
        "area_mm2": float(area_mm2),
    }

    return result, assumptions, warnings