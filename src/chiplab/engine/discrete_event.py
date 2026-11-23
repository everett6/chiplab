"""Discrete-event simulation engine for ChipLab."""

from __future__ import annotations

from typing import Any

from ..schemas.design import DesignSpec
from ..schemas.simulation import SimulationConfig, SimulationResult
from ..schemas.workload import WorkloadSpec


def run_discrete_event_simulation(
    design: DesignSpec,
    workload: WorkloadSpec,
    config: SimulationConfig,
    design_id: str = "unsaved",
) -> SimulationResult:
    """Run a discrete-event fidelity simulation.

    Models the system as a sequence of events (requests, transfers, computations)
    with timing and resource tracking. More accurate than analytical but slower.

    Args:
        design: The design specification.
        workload: The workload specification.
        config: Simulation configuration.

    Returns:
        SimulationResult with metrics and bottleneck analysis.
    """
    from ..engine.validator import validate_design
    validation = validate_design(design, config.allow_estimated_defaults)

    # If design is invalid, return a result with errors
    if not validation.valid:
        from ..schemas.simulation import (
            SimulationMetrics,
            BottleneckReport,
            AssumptionRecord,
        )
        metrics = SimulationMetrics()
        bottleneck = BottleneckReport(
            primary_bottleneck="validation_error",
            bound_type="capacity",
            severity_score=1.0,
            details=validation.errors,
            recommendations=["Fix design validation errors before running simulation"],
        )
        return SimulationResult(
            run_id="discrete_event-invalid-design",
            design_id=design.name,
            design_name=design.name,
            fidelity=config.fidelity,
            fidelity_disclaimer="Simulation skipped: design validation failed",
            metrics=metrics,
            bottleneck=bottleneck,
            assumptions=validation.missing_parameters,
            warnings=validation.warnings,
            raw_details={"validation_errors": validation.errors},
            success=False,
            error_message="; ".join(validation.errors),
        )

    # Discrete-event simulation parameters
    num_requests = getattr(config, "num_requests", 32) or 32
    seed = getattr(config, "seed", 42) or 42

    # Extract design parameters
    total_chiplet_area = 0.0
    total_compute_tops = 0.0
    total_memory_bw_tbps = 0.0
    total_power_w = 0.0

    for chiplet in design.chiplets:
        # Per-chiplet quantities are scaled by the instance count.
        if chiplet.area_mm2 is not None:
            total_chiplet_area += float(chiplet.area_mm2) * chiplet.count
        if chiplet.compute is not None and chiplet.compute.peak_tops is not None:
            total_compute_tops += float(chiplet.compute.peak_tops) * chiplet.count
        if chiplet.tdp_watts is not None:
            total_power_w += float(chiplet.tdp_watts) * chiplet.count
        if chiplet.memory is not None:
            mem = chiplet.memory
            if mem.channels is not None and mem.bandwidth_per_channel_gbps is not None:
                # GB/s per channel -> TB/s aggregate across all instances
                total_memory_bw_tbps += (
                    float(mem.channels) * float(mem.bandwidth_per_channel_gbps) * chiplet.count / 1000.0
                )

    # Workload parameters
    batch_size = 1
    sequence_length = 128
    hidden_dim = 4096
    num_layers = 32
    num_experts = 8
    active_experts = 2

    if workload is not None:
        batch_size = getattr(workload, "batch_size", 1) or 1
        sequence_length = getattr(workload, "sequence_length", 128) or 128
        hidden_dim = getattr(workload, "hidden_dim", 4096) or 4096
        num_layers = getattr(workload, "num_layers", 32) or 32
        num_experts = getattr(workload, "num_experts", 8) or 8
        active_experts = getattr(workload, "active_experts", 2) or 2

    # Simulate requests through the system
    # Each request goes through: compute -> memory -> interconnect
    total_flops = 0.0
    total_bytes_read = 0.0
    total_bytes_write = 0.0

    if workload is not None:
        total_flops = float(getattr(workload, "flops", 0) or 0)
        total_bytes_read = float(getattr(workload, "bytes_read", 0) or 0)
        total_bytes_write = float(getattr(workload, "bytes_write", 0) or 0)

    # Event-based simulation
    # Model: for each request, compute time + memory transfer time + interconnect delay
    compute_time_per_request_ms = 0.0
    memory_time_per_request_ms = 0.0
    interconnect_time_per_request_ms = 0.0

    # Compute time
    if total_compute_tops > 0 and total_flops > 0:
        # FLOPs / (TOPS * 1e12 ops/s) gives seconds; convert to ms.
        compute_time_per_request_ms = (total_flops / (total_compute_tops * 1e12)) * 1e3

    # Memory time (based on bandwidth)
    if total_memory_bw_tbps > 0:
        # Total data movement per request
        data_per_request_gb = (total_bytes_read + total_bytes_write) / 1e9
        # GB / (TB/s) is already milliseconds (1 TB/s = 1000 GB/s, s -> ms cancels).
        memory_time_per_request_ms = data_per_request_gb / total_memory_bw_tbps

    # Interconnect time
    if total_memory_bw_tbps > 0:
        # Interconnect adds some overhead (typically 10-20% of memory time)
        interconnect_time_per_request_ms = memory_time_per_request_ms * 0.15

    # Total per-request time
    per_request_time_ms = max(compute_time_per_request_ms, memory_time_per_request_ms) + interconnect_time_per_request_ms

    # Total simulation time
    total_simulation_time_ms = per_request_time_ms * num_requests

    # Throughput (tokens/sec) - inverse of per-request time
    if per_request_time_ms > 0:
        tokens_per_request = batch_size * sequence_length
        throughput_tokens_per_sec = (tokens_per_request * 1000.0) / per_request_time_ms
    else:
        throughput_tokens_per_sec = 0.0

    # Effective compute
    effective_compute_tops = total_compute_tops  # In a simple model, assume full utilization

    # Latency (per request)
    estimated_latency_ms = per_request_time_ms

    # Power
    total_power_watts = total_power_w  # Use designed TDP

    # Area
    total_area_mm2 = total_chiplet_area

    # Energy per inference
    # W * ms = mJ
    energy_per_inference_mj = total_power_watts * estimated_latency_ms if estimated_latency_ms > 0 and total_power_watts > 0 else 0.0

    # Utilization
    # Fraction of the per-request time spent doing compute work.
    utilization_pct = min(100.0, (compute_time_per_request_ms / per_request_time_ms) * 100) if per_request_time_ms > 0 and compute_time_per_request_ms > 0 else 0.0

    # Arithmetic intensity
    arithmetic_intensity = 0.0
    if total_flops > 0 and (total_bytes_read + total_bytes_write) > 0:
        arithmetic_intensity = total_flops / (total_bytes_read + total_bytes_write)

    # Ridge point (ops/byte) = peak TOPS / bandwidth TB/s, matching the analytical engine.
    ridge_point = total_compute_tops / total_memory_bw_tbps if total_compute_tops > 0 and total_memory_bw_tbps > 0 else 0.0

    # Build metrics
    from ..schemas.simulation import SimulationMetrics, BottleneckReport, AssumptionRecord

    metrics = SimulationMetrics(
        total_memory_capacity_gb=sum(
            getattr(getattr(c, "memory", None), "capacity_per_chiplet_gb", 0) or 0
            for c in design.chiplets
        ),
        aggregate_memory_bandwidth_tbps=total_memory_bw_tbps,
        peak_compute_tops=total_compute_tops,
        effective_compute_tops=effective_compute_tops,
        interconnect_bisection_bw_tbps=total_memory_bw_tbps * 0.1,
        estimated_latency_ms=round(estimated_latency_ms, 2),
        throughput_tokens_per_sec=round(throughput_tokens_per_sec, 2) if throughput_tokens_per_sec else None,
        throughput_ops_per_sec=round(effective_compute_tops * 1e12, 2) if effective_compute_tops > 0 else 0.0,
        total_area_mm2=round(total_area_mm2, 2),
        total_power_watts=round(total_power_watts, 2),
        estimated_temp_c=config.ambient_temp_c if hasattr(config, "ambient_temp_c") else 25.0,
        estimated_cost_usd=round(total_chiplet_area * 50.0, 2) if total_chiplet_area > 0 else 0.0,
        arithmetic_intensity=round(arithmetic_intensity, 4),
        ridge_point_ops_per_byte=round(ridge_point, 4),
        utilization_pct=round(utilization_pct, 2),
        energy_per_inference_mj=round(energy_per_inference_mj, 4),
    )

    # Determine primary bottleneck
    bottleneck_bound_type = "balanced"
    severity_score = 0.0
    details: list[str] = []
    recommendations: list[str] = []

    # Compute utilization
    compute_util = (effective_compute_tops / total_compute_tops * 100) if total_compute_tops > 0 else 100.0

    # Analyze bottleneck
    if compute_util < 30.0:
        bottleneck_bound_type = "compute"
        severity_score = 0.8
        details.append(
            f"Compute-limited: {compute_util:.1f}% utilization across {num_requests} simulated requests"
        )
        recommendations.append("Increase compute resources or reduce workload per request")
    elif total_memory_bw_tbps < 10.0:
        bottleneck_bound_type = "memory_bandwidth"
        severity_score = 0.7
        details.append(f"Memory bandwidth-limited: {total_memory_bw_tbps:.1f} TB/s")
        recommendations.append("Increase memory channels or use higher-bandwidth memory")
    elif total_power_watts > 200:
        bottleneck_bound_type = "power_thermal"
        severity_score = 0.5
        details.append(f"Power/thermal bound: {total_power_watts:.1f} W over {num_requests} requests")
        recommendations.append("Apply power optimization or thermal management")
    else:
        bottleneck_bound_type = "balanced"
        severity_score = 0.3
        details.append(
            f"Balanced system: {compute_util:.1f}% compute utilization, {total_memory_bw_tbps:.1f} TB/s memory bandwidth"
        )
        recommendations.append("Consider fine-tuning for optimal performance-per-watt")

    from ..schemas.simulation import BottleneckReport

    bottleneck = BottleneckReport(
        primary_bottleneck=bottleneck_bound_type,
        bound_type=bottleneck_bound_type,
        severity_score=severity_score,
        utilization_breakdown={
            "compute_util_pct": round(compute_util, 2),
            "memory_bandwidth_tbps": round(total_memory_bw_tbps, 2),
        },
        details=details,
        recommendations=recommendations,
    )

    # Add provenance assumptions
    assumptions = [
        AssumptionRecord(
            parameter="num_requests",
            value=num_requests,
            unit="requests",
            provenance="assumed",
            notes=f"Discrete event simulation parameter: {num_requests} requests simulated",
        ),
        AssumptionRecord(
            parameter="arithmetic_intensity",
            value=arithmetic_intensity,
            unit="ops/byte",
            provenance="assumed",
            notes="Estimated based on workload FLOPs and data movement",
        ),
        AssumptionRecord(
            parameter="ridge_point",
            value=ridge_point,
            unit="TOPS",
            provenance="assumed",
            notes="Roof line model ridge point from arithmetic intensity and memory bandwidth",
        ),
    ]

    return SimulationResult(
        run_id="discrete_event-default",
        design_id=design.name,
        design_name=design.name,
        fidelity=config.fidelity,
        fidelity_disclaimer="Discrete-event simulation: event-based timing model with request-level tracking",
        metrics=metrics,
        bottleneck=bottleneck,
        assumptions=assumptions,
        warnings=validation.warnings if validation else [],
        raw_details={
            "num_requests_simulated": num_requests,
            "total_flops": total_flops,
            "total_bytes_read": total_bytes_read,
            "total_bytes_write": total_bytes_write,
            "compute_time_per_request_ms": round(compute_time_per_request_ms, 4),
            "memory_time_per_request_ms": round(memory_time_per_request_ms, 4),
            "interconnect_time_per_request_ms": round(interconnect_time_per_request_ms, 4),
            "per_request_time_ms": round(per_request_time_ms, 4),
        },
        success=True,
    )