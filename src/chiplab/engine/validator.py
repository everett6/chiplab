"""Design validation for ChipLab."""

from __future__ import annotations

from typing import Any

from ..schemas.design import (
    ChipletSpec,
    DesignSpec,
    InterconnectSpec,
    MemorySpec,
    PackageSpec,
    PhysicalConstraints,
)
from ..schemas.simulation import ValidationResult
from ..schemas.common import Provenance, unwrap_value, get_provenance


def _get_provenance_counts(design: DesignSpec) -> dict[str, int]:
    """Count provenance types across all tracked parameters in the design."""
    counts = {p.value: 0 for p in Provenance}

    def count_in_obj(obj: Any, prefix: str = "") -> None:
        if obj is None:
            return
        if hasattr(obj, "model_dump"):
            for k, v in obj.model_dump().items():
                count_in_obj(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                count_in_obj(v, f"{prefix}[{i}]")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                count_in_obj(v, f"{prefix}.{k}")
        else:
            prov = get_provenance(obj)
            counts[prov.value] += 1

    count_in_obj(design)
    return {k: v for k, v in counts.items() if v > 0}


def _check_chiplet(chiplet: ChipletSpec, allow_defaults: bool, errors: list[str], warnings: list[str], missing: list[str]) -> None:
    """Validate a single chiplet spec."""
    if chiplet.count is not None:
        c = unwrap_value(chiplet.count)
        if c <= 0:
            errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': count must be positive, got {c}")

    if chiplet.area_mm2 is not None:
        a = unwrap_value(chiplet.area_mm2)
        if a <= 0:
            errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': area_mm2 must be positive, got {a}")

    if chiplet.tdp_watts is not None:
        p = unwrap_value(chiplet.tdp_watts)
        if p <= 0:
            errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': tdp_watts must be positive, got {p}")

    if chiplet.process_nm is not None:
        p = unwrap_value(chiplet.process_nm)
        if p <= 0:
            errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': process_nm must be positive, got {p}")

    if chiplet.type == "memory" and chiplet.memory is None:
        if allow_defaults:
            warnings.append(f"Chiplet '{chiplet.id or 'unnamed'}': memory type but no MemorySpec provided; will use defaults")
        else:
            errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': memory type requires MemorySpec")
            missing.append(f"chiplets.{chiplet.id or 'unnamed'}.memory")

    if chiplet.type == "compute" and chiplet.compute is None:
        if allow_defaults:
            warnings.append(f"Chiplet '{chiplet.id or 'unnamed'}': compute type but no ComputeSpec provided; will use defaults")
        else:
            errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': compute type requires ComputeSpec")
            missing.append(f"chiplets.{chiplet.id or 'unnamed'}.compute")

    if chiplet.memory is not None:
        mem = chiplet.memory
        if mem.capacity_per_chiplet_gb is not None:
            cap = unwrap_value(mem.capacity_per_chiplet_gb)
            if cap <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': capacity_per_chiplet_gb must be positive, got {cap}")
        if mem.channels is not None:
            ch = unwrap_value(mem.channels)
            if ch <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': channels must be positive, got {ch}")
        if mem.banks_per_channel is not None:
            b = unwrap_value(mem.banks_per_channel)
            if b <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': banks_per_channel must be positive, got {b}")
        if mem.bandwidth_per_channel_gbps is not None:
            bw = unwrap_value(mem.bandwidth_per_channel_gbps)
            if bw <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': bandwidth_per_channel_gbps must be positive, got {bw}")
        if mem.latency_ns is not None:
            lat = unwrap_value(mem.latency_ns)
            if lat <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': latency_ns must be positive, got {lat}")
        if mem.energy_pj_per_bit is not None:
            e = unwrap_value(mem.energy_pj_per_bit)
            if e <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': energy_pj_per_bit must be positive, got {e}")
        if mem.stack_layers is not None:
            sl = unwrap_value(mem.stack_layers)
            if sl <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': stack_layers must be positive, got {sl}")

    if chiplet.compute is not None:
        comp = chiplet.compute
        if comp.peak_tops is not None:
            pt = unwrap_value(comp.peak_tops)
            if pt <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': peak_tops must be positive, got {pt}")
        if comp.clock_ghz is not None:
            clk = unwrap_value(comp.clock_ghz)
            if clk <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': clock_ghz must be positive, got {clk}")
        if comp.mac_units is not None:
            mac = unwrap_value(comp.mac_units)
            if mac <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': mac_units must be positive, got {mac}")
        if comp.sram_mb is not None:
            sram = unwrap_value(comp.sram_mb)
            if sram <= 0:
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': sram_mb must be positive, got {sram}")
        if comp.efficiency_factor is not None:
            ef = unwrap_value(comp.efficiency_factor)
            if not (0 < ef <= 1):
                errors.append(f"Chiplet '{chiplet.id or 'unnamed'}': efficiency_factor must be in (0, 1], got {ef}")


def validate_design(design: DesignSpec, allow_estimated_defaults: bool = True) -> ValidationResult:
    """Validate a DesignSpec for contradictions and physical limits."""
    errors: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    # Check empty chiplets
    if not design.chiplets:
        errors.append("Design must have at least one chiplet")

    # Validate each chiplet
    for chiplet in design.chiplets:
        _check_chiplet(chiplet, allow_estimated_defaults, errors, warnings, missing)

    # Check package constraints
    if design.package.type == "monolithic" and len(design.chiplets) > 1:
        errors.append("Monolithic package type cannot have more than 1 chiplet")

    # Check physical constraints
    if design.constraints.max_power_watts is not None:
        mp = unwrap_value(design.constraints.max_power_watts)
        if mp <= 0:
            errors.append(f"max_power_watts must be positive, got {mp}")

    if design.constraints.max_area_mm2 is not None:
        ma = unwrap_value(design.constraints.max_area_mm2)
        if ma <= 0:
            errors.append(f"max_area_mm2 must be positive, got {ma}")

    if design.constraints.max_temp_c is not None:
        mt = unwrap_value(design.constraints.max_temp_c)
        if mt <= 0:
            errors.append(f"max_temp_c must be positive, got {mt}")

    if design.constraints.max_cost_usd is not None:
        mc = unwrap_value(design.constraints.max_cost_usd)
        if mc <= 0:
            errors.append(f"max_cost_usd must be positive, got {mc}")

    if design.constraints.thermal_resistance_c_per_w is not None:
        tr = unwrap_value(design.constraints.thermal_resistance_c_per_w)
        if tr <= 0:
            errors.append(f"thermal_resistance_c_per_w must be positive, got {tr}")

    # Check interconnect
    if design.interconnect.link_bandwidth_gbps is not None:
        bw = unwrap_value(design.interconnect.link_bandwidth_gbps)
        if bw <= 0:
            errors.append(f"interconnect.link_bandwidth_gbps must be positive, got {bw}")

    if design.interconnect.latency_ns is not None:
        lat = unwrap_value(design.interconnect.latency_ns)
        if lat <= 0:
            errors.append(f"interconnect.latency_ns must be positive, got {lat}")

    if design.interconnect.energy_pj_per_bit is not None:
        e = unwrap_value(design.interconnect.energy_pj_per_bit)
        if e <= 0:
            errors.append(f"interconnect.energy_pj_per_bit must be positive, got {e}")

    if design.interconnect.lanes_per_chiplet is not None:
        ln = unwrap_value(design.interconnect.lanes_per_chiplet)
        if ln <= 0:
            errors.append(f"interconnect.lanes_per_chiplet must be positive, got {ln}")

    # Check package specs
    if design.package.max_area_mm2 is not None:
        ma = unwrap_value(design.package.max_area_mm2)
        if ma <= 0:
            errors.append(f"package.max_area_mm2 must be positive, got {ma}")

    if design.package.cost_per_mm2_usd is not None:
        cp = unwrap_value(design.package.cost_per_mm2_usd)
        if cp <= 0:
            errors.append(f"package.cost_per_mm2_usd must be positive, got {cp}")

    if design.package.yield_defect_density_per_cm2 is not None:
        yd = unwrap_value(design.package.yield_defect_density_per_cm2)
        if yd < 0:
            errors.append(f"package.yield_defect_density_per_cm2 must be non-negative, got {yd}")

    warnings.extend(_commercial_readiness_warnings(design))

    # Provenance summary
    provenance_summary = _get_provenance_counts(design)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        missing_parameters=missing,
        provenance_summary=provenance_summary,
    )

def _commercial_readiness_warnings(design: DesignSpec) -> list[str]:
    """PDK tape-out / licence and IP licence checks driven by design.metadata.

    metadata keys: ``pdk`` (ladder key, e.g. 'sky130', 'asap7'), ``intended_use``
    ('research' default, or 'commercial'), ``ip`` (catalog names), ``blackbox_licenses``
    ({macro: commercial_use_allowed}). Without ``pdk`` the nearest ladder node to each
    chiplet's process_nm is checked, but only for commercial intent.
    """
    from chiplab.library.pdk_library import commercial_readiness, pdk_for_nm

    md = design.metadata or {}
    use = str(md.get("intended_use", "research"))
    keys: list[str] = []
    if md.get("pdk"):
        keys = [str(md["pdk"])]
    elif use.lower() in ("commercial", "production", "commercial_production"):
        for ch in design.chiplets:
            if ch.process_nm is not None and not (ch.memory and ch.compute is None):
                keys.append(pdk_for_nm(float(unwrap_value(ch.process_nm))).key)
    out: list[str] = []
    for k in dict.fromkeys(keys):
        out.extend(commercial_readiness(k, use))
    out.extend(commercial_readiness(None, use, list(md.get("ip") or []),
                                    dict(md.get("blackbox_licenses") or {})))
    return list(dict.fromkeys(out))
