from __future__ import annotations

import time
from typing import Any

from chiplab.schemas.design import DesignSpec, ChipletSpec
from chiplab.schemas.simulation import SimulationConfig
from chiplab.schemas.workload import WorkloadSpec
from chiplab.schemas.simulation import (
    SimulationResult,
    BottleneckReport,
    SimulationMetrics,
    FidelityLevel,
    AssumptionRecord,
    Provenance,
)


def _estimate_memory_chiplet(
    chiplet: ChipletSpec, allow_defaults: bool = True
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Wrap chiplab.models.estimate_memory_chiplet with error handling."""
    from chiplab.models.memory import estimate_memory_chiplet
    try:
        return estimate_memory_chiplet(chiplet, allow_defaults=allow_defaults)
    except ValueError:
        # Chiplet type doesn't have memory spec; return zeros
        return {
            "capacity_gb": 0.0,
            "bandwidth_tbps": 0.0,
            "latency_ns": 0.0,
            "energy_pj_per_bit": 0.0,
            "area_mm2": float(chiplet.area_mm2) if chiplet.area_mm2 else 0.0,
            "power_watts": float(chiplet.tdp_watts) if chiplet.tdp_watts else 0.0,
            "channels": 1,
            "banks_per_channel": 8,
        }, [], ["memory spec not available for chiplet type", chiplet.type]


def _estimate_compute_chiplet(
    chiplet: ChipletSpec, allow_defaults: bool = True
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Wrap chiplab.models.estimate_compute_chiplet with error handling for non-compute chiplets."""
    from chiplab.models.compute import estimate_compute_chiplet
    if chiplet.compute is None and (
        chiplet.type in ("memory", "io") or chiplet.memory is not None
    ):
        # Memory power is counted by the memory model; avoid counting the TDP twice.
        return {
            "peak_tops": 0.0,
            "efficiency_factor": 1.0,
            "effective_tops": 0.0,
            "sram_mb": 0.0,
            "area_mm2": float(chiplet.area_mm2) if chiplet.area_mm2 else 0.0,
            "power_watts": (
                float(chiplet.tdp_watts) if chiplet.tdp_watts and chiplet.memory is None else 0.0
            ),
            "clock_ghz": 0.0,
        }, [], []
    return estimate_compute_chiplet(chiplet, allow_defaults=allow_defaults)


def _estimate_interconnect(
    design: DesignSpec, total_chiplets: int, allow_defaults: bool = True
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Wrap chiplab.models.estimate_interconnect."""
    from chiplab.models.interconnect import estimate_interconnect
    return estimate_interconnect(design, total_chiplets, allow_defaults=allow_defaults)


def _estimate_packaging(
    design: DesignSpec, chiplet_areas: list[float], allow_defaults: bool = True
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Wrap chiplab.models.estimate_packaging."""
    from chiplab.models.packaging import estimate_packaging
    return estimate_packaging(design, chiplet_areas, allow_defaults=allow_defaults)


def _estimate_power_thermal(
    design: DesignSpec,
    component_power_watts: float,
    total_area_mm2: float,
    allow_defaults: bool = True,
) -> tuple[dict[str, float], list[AssumptionRecord], list[str]]:
    """Wrap chiplab.models.estimate_power_thermal."""
    from chiplab.models.power_thermal import estimate_power_thermal
    return estimate_power_thermal(
        design, component_power_watts, total_area_mm2, allow_defaults=allow_defaults
    )


def _compute_roofline(total_flops: float, total_bytes: float) -> dict[str, Any]:
    """Compute Level 1 roofline inputs (arithmetic intensity in ops/byte).

    The ridge point depends on aggregate memory bandwidth and is computed by the caller.
    """
    ai = total_flops / max(total_bytes, 1.0)
    return {
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "arithmetic_intensity": ai,
    }


def _build_bottleneck_report(
    utilization_breakdown: dict[str, float],
    severity_score: float,
    primary_bottleneck: str,
    bound_type: str,
    details: list[str],
    recommendations: list[str],
) -> BottleneckReport:
    """Build a BottleneckReport."""
    return BottleneckReport(
        primary_bottleneck=primary_bottleneck,
        bound_type=bound_type,
        severity_score=severity_score,
        utilization_breakdown=utilization_breakdown,
        details=details,
        recommendations=recommendations,
    )


def run_analytical_simulation(
    design: DesignSpec,
    workload: WorkloadSpec,
    config: SimulationConfig,
    design_id: str = "unsaved",
) -> SimulationResult:
    """Run Level 1 analytical simulation (roofline model + first-order estimates).

    Calls estimate_memory_chiplet, estimate_compute_chiplet, estimate_interconnect,
    estimate_packaging, estimate_power_thermal from chiplab.models.

    Computes Level 1 roofline: total_flops, total_bytes, arithmetic_intensity,
    ridge_point. Computes estimated_latency_ms, throughput_tokens_per_sec,
    throughput_ops_per_sec, utilization_pct, energy_per_inference_mj.

    Builds BottleneckReport identifying memory_bandwidth, compute, interconnect,
    capacity, power_thermal, or balanced.

    Sets fidelity=FidelityLevel.ANALYTICAL and
    fidelity_disclaimer="Level 1 analytical model: theoretical roofline and first-order
    estimates; not cycle-accurate.".
    """
    run_id = f"analytical_{design_id}_{int(time.time())}"
    assumptions: list[AssumptionRecord] = []
    warnings: list[str] = []

    total_chiplets = sum(c.count for c in design.chiplets)

    # --- Estimate memory chiplet parameters ---
    mem_metrics_list: list[dict[str, float]] = []
    for chiplet in design.chiplets:
        mem_metrics, mem_assumptions, mem_warnings = _estimate_memory_chiplet(chiplet)
        mem_metrics_list.append(mem_metrics)
        assumptions.extend(mem_assumptions)
        warnings.extend(mem_warnings)

    # Aggregate memory bandwidth (TB/s) and capacity (GB)
    agg_mem_bw_tbps = sum(m.get("bandwidth_tbps", 0.0) for m in mem_metrics_list)
    total_mem_cap_gb = sum(m.get("capacity_gb", 0.0) for m in mem_metrics_list)

    # --- Estimate compute chiplet parameters ---
    comp_metrics_list: list[dict[str, float]] = []
    for chiplet in design.chiplets:
        comp_metrics, comp_assumptions, comp_warnings = _estimate_compute_chiplet(chiplet)
        comp_metrics_list.append(comp_metrics)
        assumptions.extend(comp_assumptions)
        warnings.extend(comp_warnings)

    # Aggregate peak and effective TOPS
    # Per-chiplet compute/power values are per instance; scale by count.
    peak_tops_total = sum(
        c.get("peak_tops", 0.0) * ch.count for ch, c in zip(design.chiplets, comp_metrics_list)
    )
    effective_tops = sum(
        c.get("effective_tops", 0.0) * ch.count for ch, c in zip(design.chiplets, comp_metrics_list)
    )

    # --- Estimate interconnect ---
    interconnect_metrics, inter_assumptions, inter_warnings = _estimate_interconnect(
        design, total_chiplets
    )
    assumptions.extend(inter_assumptions)
    warnings.extend(inter_warnings)

    agg_interconnect_bw_tbps = interconnect_metrics.get("bisection_bw_tbps", 0.0)

    # --- Estimate packaging ---
    chiplet_areas = [c.area_mm2 for c in design.chiplets if c.area_mm2 for _ in range(c.count)]
    packaging_metrics, pack_assumptions, pack_warnings = _estimate_packaging(
        design, chiplet_areas
    )
    assumptions.extend(pack_assumptions)
    warnings.extend(pack_warnings)

    total_area_mm2 = packaging_metrics.get("total_area_mm2", 0.0)

    # --- Estimate power/thermal ---
    # Sum total power from compute + memory chiplet estimates
    total_comp_power = sum(
        m.get("power_watts", 0.0) * ch.count for ch, m in zip(design.chiplets, comp_metrics_list)
    )
    # Memory power: only for pure memory chiplets (compute chiplets' TDP is already counted
    # above), per instance like compute.
    total_mem_power = sum(
        m.get("power_watts", 0.0) * ch.count
        for ch, m in zip(design.chiplets, mem_metrics_list)
        if ch.memory is not None and ch.compute is None
    )
    component_power_watts = total_comp_power + total_mem_power

    power_metrics, power_assumptions, power_warnings = _estimate_power_thermal(
        design, component_power_watts, total_area_mm2
    )
    assumptions.extend(power_assumptions)
    warnings.extend(power_warnings)

    # --- Compute workload-level metrics ---
    from chiplab.schemas.workload import expand_workload_operators
    if workload is None:
        workload = WorkloadSpec()
    ops = expand_workload_operators(workload)
    total_flops = sum(op.flops for op in ops)
    if total_flops <= 0:
        # No workload FLOPs: assume one second of effective compute work.
        total_flops = effective_tops * 1e12
    total_bytes = sum(op.bytes_read + op.bytes_write for op in ops)

    # Level 1 roofline
    roofline = _compute_roofline(total_flops, total_bytes)

    # Arithmetic intensity and ridge point
    arithmetic_intensity = roofline["arithmetic_intensity"]
    # Ridge point (ops/byte): peak ops/s / bytes/s = TOPS / TB/s
    ridge_point = (
        (effective_tops * 1e12) / max(agg_mem_bw_tbps * 1e12, 1.0)
        if effective_tops > 0 and agg_mem_bw_tbps > 0
        else 0.0
    )

    # --- Estimated latency (ms) ---
    # Simple roofline-based estimate: time = max(flops/effective_tops_tps, bytes/agg_bw_tps)
    effective_tops_s = effective_tops * 1e12  # convert TOPS to ops/s
    if effective_tops_s > 0:
        compute_time_ms = (total_flops / effective_tops_s) * 1000.0
    else:
        compute_time_ms = 0.0

    if agg_mem_bw_tbps > 0:
        mem_time_ms = (total_bytes / (agg_mem_bw_tbps * 1e12)) * 1000.0
    else:
        mem_time_ms = 0.0

    estimated_latency_ms = max(compute_time_ms, mem_time_ms)

    # --- Throughput ---
    # throughput_tokens_per_sec: for LLM/MoE/SSM workloads
    # Rough estimate: tokens per second proportional to 1 / latency
    if estimated_latency_ms > 0:
        from chiplab.schemas.workload import is_llm_decode

        # Decode emits one token per sequence per step; other workloads process the full
        # sequence per pass.
        tokens = workload.batch_size * (1 if is_llm_decode(workload) else workload.sequence_length)
        throughput_tokens_per_sec = tokens / (estimated_latency_ms / 1000.0)
    else:
        throughput_tokens_per_sec = 0.0

    # throughput_ops_per_sec
    if effective_tops_s > 0:
        throughput_ops_per_sec = effective_tops_s
    else:
        throughput_ops_per_sec = 0.0

    # --- Utilization ---
    # utilization_pct: fraction of peak compute utilized
    if effective_tops_s > 0 and total_flops > 0:
        # ops executed / (ops that would run at peak in the latency time)
        # Simple: utilization = (actual throughput / peak throughput) * 100
        # In the latency time, peak can execute effective_tops_s * (estimated_latency_ms / 1000) ops
        peak_in_latency = effective_tops_s * (estimated_latency_ms / 1000.0)
        utilization_pct = (total_flops / max(peak_in_latency, 1.0)) * 100.0
        utilization_pct = min(utilization_pct, 100.0)
    else:
        utilization_pct = 0.0

    # --- Energy per inference (mJ) ---
    # Rough estimate: energy = power * time
    if estimated_latency_ms > 0:
        # W * ms = mJ
        energy_per_inference_mj = component_power_watts * estimated_latency_ms
    else:
        energy_per_inference_mj = 0.0

    # --- Bottleneck analysis ---
    # Determine primary bottleneck
    bottleneck: str = "balanced"
    severity_score: float = 50.0
    utilization_breakdown: dict[str, float] = {}
    details: list[str] = []
    recommendations: list[str] = []

    # Compare arithmetic intensity to ridge point
    if arithmetic_intensity > 0 and ridge_point > 0:
        if arithmetic_intensity < ridge_point:
            # Memory-bound
            bottleneck = "memory_bandwidth"
            severity_score = min(90.0, 60 + 0.5 * (ridge_point - arithmetic_intensity) / max(ridge_point, 1e-6) * 40)
            details.append(
                f"Arithmetic intensity ({arithmetic_intensity:.2f} ops/byte) is below ridge point "
                f"({ridge_point:.2f} ops/byte). Design is memory-bandwidth bound."
            )
            recommendations.append(
                "Increase compute density or use PIM/memory-compute chiplets to reduce data movement."
            )
            severity_score = max(severity_score, 70.0)
        elif arithmetic_intensity > ridge_point * 1.5:
            # Compute-bound
            bottleneck = "compute"
            severity_score = min(90.0, 60 + 0.5 * (arithmetic_intensity / max(ridge_point, 1e-6) - 1.5) * 40)
            details.append(
                f"Arithmetic intensity ({arithmetic_intensity:.2f} ops/byte) exceeds ridge point "
                f"({ridge_point:.2f} ops/byte). Design is compute-bound."
            )
            recommendations.append(
                "Increase memory bandwidth or use wider memory channels to avoid compute stalls."
            )
            severity_score = max(severity_score, 70.0)
        else:
            bottleneck = "balanced"
            severity_score = 40.0
    else:
        # If we can't compute ridge point, check other factors
        if agg_mem_bw_tbps < 10.0:
            bottleneck = "memory_bandwidth"
            severity_score = 70.0
            details.append("Low aggregate memory bandwidth detected.")
            recommendations.append("Consider HBM or wider memory interfaces.")
        elif effective_tops < 100.0:
            bottleneck = "compute"
            severity_score = 70.0
            details.append("Low peak compute performance detected.")
            recommendations.append("Consider systolic array or tensor engine chiplets.")
        else:
            bottleneck = "balanced"
            severity_score = 50.0

    # Factor in interconnect
    if agg_interconnect_bw_tbps < 10.0 and bottleneck != "memory_bandwidth":
        bottleneck = "interconnect"
        severity_score = max(severity_score, 60.0)
        details.append(
            f"Low interconnect bisection bandwidth ({agg_interconnect_bw_tbps:.2f} TB/s)."
        )
        recommendations.append(
            "Consider mesh/topology upgrade or reduce chiplet count per domain."
        )

    # Factor in power/thermal
    est_temp = power_metrics.get("estimated_temp_c", 25.0)
    max_temp = design.constraints.max_temp_c if design.constraints else 85.0
    if est_temp > max_temp * 0.8 and bottleneck != "power_thermal":
        bottleneck = "power_thermal"
        severity_score = max(severity_score, 65.0)
        details.append(
            f"Estimated temperature {est_temp:.1f}°C approaching limit {max_temp}°C."
        )
        recommendations.append(
            "Improve thermal design or reduce power density."
        )

    # Factor in capacity
    if total_mem_cap_gb < 64.0 and bottleneck != "power_thermal":
        if bottleneck == "balanced":
            bottleneck = "capacity"
        severity_score = max(severity_score, 55.0)
        details.append(
            f"Total memory capacity {total_mem_cap_gb:.1f} GB is modest."
        )
        recommendations.append(
            "Increase memory capacity or use multi-level caching."
        )

    # If still balanced, keep it
    if bottleneck == "balanced":
        severity_score = 45.0

    # Build bottleneck report directly
    from ..schemas.simulation import BottleneckReport

    bottleneck = BottleneckReport(
        primary_bottleneck=bottleneck,
        bound_type=bottleneck,  # bound_type is required and should match primary_bottleneck
        severity_score=severity_score,
        utilization_breakdown=utilization_breakdown,
        details=details,
        recommendations=recommendations,
    )

    # Compute final metrics
    metrics = SimulationMetrics(
        total_memory_capacity_gb=total_mem_cap_gb,
        aggregate_memory_bandwidth_tbps=agg_mem_bw_tbps,
        peak_compute_tops=peak_tops_total,
        effective_compute_tops=effective_tops,
        interconnect_bisection_bw_tbps=agg_interconnect_bw_tbps,
        estimated_latency_ms=estimated_latency_ms,
        throughput_tokens_per_sec=throughput_tokens_per_sec,
        throughput_ops_per_sec=throughput_ops_per_sec,
        total_area_mm2=total_area_mm2,
        total_power_watts=component_power_watts,
        estimated_temp_c=est_temp,
        estimated_cost_usd=packaging_metrics.get("estimated_cost_usd", 0.0),
        arithmetic_intensity=arithmetic_intensity,
        ridge_point_ops_per_byte=ridge_point,
        utilization_pct=utilization_pct,
        energy_per_inference_mj=energy_per_inference_mj,
    )

    fidelity_disclaimer = (
        "Level 1 analytical model: theoretical roofline and first-order estimates; "
        "not cycle-accurate."
    )

    result = SimulationResult(
        run_id=run_id,
        design_id=design_id,
        design_name=design.name,
        design_version=design.version,
        fidelity=FidelityLevel.ANALYTICAL,
        fidelity_disclaimer=fidelity_disclaimer,
        timestamp=time.time(),
        metrics=metrics,
        bottleneck=bottleneck,
        assumptions=assumptions,
        warnings=warnings,
        raw_details={},
        success=True,
    )

    return result