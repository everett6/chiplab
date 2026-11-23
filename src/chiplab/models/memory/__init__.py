from __future__ import annotations

from typing import Any, Dict, List, Optional

from chiplab.schemas.simulation import AssumptionRecord
from chiplab.schemas.common import Provenance
from chiplab.schemas.design import ChipletSpec, DesignSpec
from chiplab.library.components import COMPONENT_LIBRARY, get_component_defaults


def estimate_memory_chiplet(
    chiplet: ChipletSpec,
    allow_defaults: bool = True,
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Estimate memory chiplet parameters using component library defaults.

    Returns:
        tuple of (metrics_dict, assumption_records, warnings)
        - capacity_gb: total capacity in GB for 1 chiplet
        - bandwidth_tbps: aggregate bandwidth in TB/s for 1 chiplet
        - latency_ns: access latency in ns
        - energy_pj_per_bit: energy per bit in pJ
        - area_mm2: chip area in mm²
        - power_watts: power in watts
        - channels: number of I/O channels
        - banks_per_channel: banks per channel
    """
    assumptions: list[AssumptionRecord] = []
    warnings: list[str] = []

    # Start with chiplet memory spec if available
    mem_spec = chiplet.memory

    if mem_spec is not None:
        # Use provided memory spec values
        capacity_gb = float(mem_spec.capacity_per_chiplet_gb) * chiplet.count
        channels = int(mem_spec.channels) * chiplet.count
        banks_per_channel = int(mem_spec.banks_per_channel)  # per channel; not scaled
        bandwidth_per_channel_gbps = (
            float(mem_spec.bandwidth_per_channel_gbps) if mem_spec.bandwidth_per_channel_gbps else None
        )
        latency_ns = float(mem_spec.latency_ns) if mem_spec.latency_ns else None
        energy_pj_per_bit = float(mem_spec.energy_pj_per_bit) if mem_spec.energy_pj_per_bit else None
        area_mm2_raw = float(chiplet.area_mm2) if chiplet.area_mm2 else None
        power_watts_raw = float(chiplet.tdp_watts) if chiplet.tdp_watts else None

        # Record used values as measured/provenance
        if mem_spec.capacity_per_chiplet_gb is not None:
            assumptions.append(
                AssumptionRecord(
                    parameter="capacity_per_chiplet_gb",
                    value=float(mem_spec.capacity_per_chiplet_gb),
                    unit="GB",
                    provenance=Provenance.MEASURED,
                    source="chiplet_spec",
                )
            )
        if mem_spec.bandwidth_per_channel_gbps is not None:
            assumptions.append(
                AssumptionRecord(
                    parameter="bandwidth_per_channel_gbps",
                    value=float(mem_spec.bandwidth_per_channel_gbps),
                    unit="GB/s",
                    provenance=Provenance.MEASURED,
                    source="chiplet_spec",
                )
            )
        if mem_spec.latency_ns is not None:
            assumptions.append(
                AssumptionRecord(
                    parameter="latency_ns",
                    value=float(mem_spec.latency_ns),
                    unit="ns",
                    provenance=Provenance.MEASURED,
                    source="chiplet_spec",
                )
            )
        if mem_spec.energy_pj_per_bit is not None:
            assumptions.append(
                AssumptionRecord(
                    parameter="energy_pj_per_bit",
                    value=float(mem_spec.energy_pj_per_bit),
                    unit="pJ/bit",
                    provenance=Provenance.MEASURED,
                    source="chiplet_spec",
                )
            )
    else:
        # Fall back to component library defaults
        comp_name = chiplet.type or "SRAM"
        comp_defaults = get_component_defaults("memory", comp_name)

        if comp_defaults is not None and allow_defaults:
            capacity_per_gb = comp_defaults["capacity_per_chiplet_gb"]
            channels = comp_defaults["channels"]
            banks_per_channel = comp_defaults["banks_per_channel"]
            bandwidth_per_channel_gbps = comp_defaults["bandwidth_per_channel_gbps"]
            latency_ns = comp_defaults["latency_ns"]
            energy_pj_per_bit = comp_defaults["energy_pj_per_bit"]

            assumptions.append(
                AssumptionRecord(
                    parameter="capacity_per_chiplet_gb",
                    value=capacity_per_gb,
                    unit="GB",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="channels",
                    value=channels,
                    unit="count",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="banks_per_channel",
                    value=banks_per_channel,
                    unit="count",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="bandwidth_per_channel_gbps",
                    value=bandwidth_per_channel_gbps,
                    unit="GB/s",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="latency_ns",
                    value=latency_ns,
                    unit="ns",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
            assumptions.append(
                AssumptionRecord(
                    parameter="energy_pj_per_bit",
                    value=energy_pj_per_bit,
                    unit="pJ/bit",
                    provenance=Provenance.ASSUMED,
                    source="component_library",
                )
            )
        elif comp_defaults is not None and not allow_defaults:
            # User explicitly disallowed defaults, but no chiplet spec provided
            raise ValueError(
                f"No memory specification found for chiplet type '{chiplet.type}' and allow_defaults=False"
            )
            capacity_per_gb = comp_defaults["capacity_per_chiplet_gb"]
            channels = comp_defaults["channels"]
            banks_per_channel = comp_defaults["banks_per_channel"]
            bandwidth_per_channel_gbps = comp_defaults["bandwidth_per_channel_gbps"]
            latency_ns = comp_defaults["latency_ns"]
            energy_pj_per_bit = comp_defaults["energy_pj_per_bit"]
        else:
            # No component found and no chiplet spec
            raise ValueError(
                f"No memory specification found for chiplet type '{chiplet.type}' and no component library defaults available"
            )

        # Set area and power from chiplet if available
        area_mm2_raw = float(chiplet.area_mm2) if chiplet.area_mm2 else capacity_per_gb * 0.5  # rough estimate: 0.5 mm²/GB
        if area_mm2_raw is not None and chiplet.area_mm2 is None:
            warnings.append(
                f"area_mm2 not in chiplet spec, estimated as {area_mm2_raw:.1f} mm² based on capacity"
            )

        power_watts_raw = float(chiplet.tdp_watts) if chiplet.tdp_watts else capacity_per_gb * 5.0  # rough: 5W/GB
        if power_watts_raw is not None and chiplet.tdp_watts is None:
            warnings.append(
                f"power_watts not in chiplet spec, estimated as {power_watts_raw:.1f} W based on capacity"
            )

        # Library defaults are per instance; scale like the explicit-spec path.
        capacity_gb = capacity_per_gb * chiplet.count
        channels = channels * chiplet.count if channels is not None else None

    # Compute aggregate bandwidth in TB/s
    if bandwidth_per_channel_gbps is not None and channels is not None:
        bandwidth_tbps = (bandwidth_per_channel_gbps * channels) / 1000.0
    else:
        bandwidth_tbps = 0.0
        warnings.append("Could not compute bandwidth_tbps: missing bandwidth_per_channel_gbps or channels")

    # Set defaults for area and power if not already set
    if 'area_mm2_raw' not in dir() or area_mm2_raw is None:
        area_mm2_raw = 0.0
    if 'power_watts_raw' not in dir() or power_watts_raw is None:
        power_watts_raw = 0.0

    # Ensure all values are float
    area_mm2 = float(area_mm2_raw) if area_mm2_raw is not None else 0.0
    power_watts = float(power_watts_raw) if power_watts_raw is not None else 0.0

    result: dict[str, float] = {
        "capacity_gb": float(capacity_gb) if 'capacity_gb' in dir() else 0.0,
        "bandwidth_tbps": float(bandwidth_tbps) if 'bandwidth_tbps' in dir() else 0.0,
        "latency_ns": float(latency_ns) if latency_ns is not None else 0.0,
        "energy_pj_per_bit": float(energy_pj_per_bit) if energy_pj_per_bit is not None else 0.0,
        "area_mm2": float(area_mm2) if area_mm2 is not None else 0.0,
        "power_watts": float(power_watts) if power_watts is not None else 0.0,
        "channels": int(channels) if channels is not None else 1,
        "banks_per_channel": int(banks_per_channel) if banks_per_channel is not None else 8,
    }

    # Fill in missing keys with defaults
    for key in ["capacity_gb", "bandwidth_tbps", "latency_ns", "energy_pj_per_bit", "area_mm2", "power_watts"]:
        if key not in result:
            result[key] = 0.0
    for key in ["channels", "banks_per_channel"]:
        if key not in result:
            result[key] = 1 if key == "channels" else 8

    return result, assumptions, warnings