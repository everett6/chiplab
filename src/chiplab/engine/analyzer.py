"""Analysis engine for ChipLab simulation results."""

from __future__ import annotations

from typing import Any

from ..schemas.simulation import (
    SimulationResult,
    BottleneckReport,
    ComparisonResult,
    SimulationMetrics,
    FidelityLevel,
)


def analyze_bottleneck(result: SimulationResult) -> BottleneckReport:
    """Deep-dive into result.metrics and result.bottleneck, returning an enriched BottleneckReport.

    Provides actionable architectural recommendations based on the simulation metrics.
    """
    metrics = result.metrics
    bottleneck = result.bottleneck

    # Extract key metric values for analysis
    def get_metric(name: str, default: float = 0.0) -> float:
        if hasattr(metrics, name):
            val = getattr(metrics, name)
            return float(val) if val is not None else default
        return default

    total_memory_capacity_gb = get_metric("total_memory_capacity_gb")
    aggregate_memory_bandwidth_tbps = get_metric("aggregate_memory_bandwidth_tbps")
    peak_compute_tops = get_metric("peak_compute_tops")
    effective_compute_tops = get_metric("effective_compute_tops")
    interconnect_bisection_bw_tbps = get_metric("interconnect_bisection_bw_tbps")
    estimated_latency_ms = get_metric("estimated_latency_ms")
    throughput_tokens_per_sec = get_metric("throughput_tokens_per_sec")
    throughput_ops_per_sec = get_metric("throughput_ops_per_sec")
    total_power_watts = get_metric("total_power_watts")
    total_area_mm2 = get_metric("total_area_mm2")
    energy_per_inference_mj = get_metric("energy_per_inference_mj")
    utilization_pct = get_metric("utilization_pct")
    arithmetic_intensity = get_metric("arithmetic_intensity")
    ridge_point_ops_per_byte = get_metric("ridge_point_ops_per_byte")

    # Make a copy of recommendations from the original bottleneck
    recommendations: list[str] = list(bottleneck.recommendations) if bottleneck.recommendations else []
    details: list[str] = list(bottleneck.details) if bottleneck.details else []

    # Analyze and add recommendations based on metrics
    # Check compute-bound vs memory-bound
    if peak_compute_tops > 0 and effective_compute_tops > 0:
        compute_util = effective_compute_tops / peak_compute_tops
    else:
        compute_util = 0.0

    if aggregate_memory_bandwidth_tbps > 0 and arithmetic_intensity > 0 and peak_compute_tops > 0:
        # Roofline attainable = min(peak, AI * BW); AI (ops/byte) * BW (TB/s) is in TOPS.
        roofline_util = min(1.0, arithmetic_intensity * aggregate_memory_bandwidth_tbps / peak_compute_tops)
    else:
        roofline_util = 0.0

    # Primary bottleneck analysis
    primary = bottleneck.primary_bottleneck

    new_recommendations: list[str] = []

    if primary == "memory_bandwidth" or (
        aggregate_memory_bandwidth_tbps < 50.0 and compute_util > 0.7
    ):
        new_recommendations.append(
            "Consider increasing memory bandwidth or reducing compute intensity to alleviate memory bottleneck"
        )
        if total_memory_capacity_gb < 32:
            new_recommendations.append(
                "Increase total memory capacity; current configuration has less than 32GB"
            )
        if interconnect_bisection_bw_tbps < aggregate_memory_bandwidth_tbps * 0.1:
            new_recommendations.append(
                "Interconnect bisection bandwidth is a limiting factor; consider a wider interconnect topology"
            )

    elif primary == "compute" or (
        compute_util > 0.9 and aggregate_memory_bandwidth_tbps > 50.0
    ):
        new_recommendations.append(
            "Compute units are heavily utilized; consider increasing peak compute throughput or adding more compute units"
        )
        if peak_compute_tops > 0 and effective_compute_tops / peak_compute_tops < 0.5:
            new_recommendations.append(
                "Consider pipelining or increasing clock frequency to improve compute utilization"
            )

    elif primary == "interconnect" or (
        interconnect_bisection_bw_tbps < 10.0
    ):
        new_recommendations.append(
            "Interconnect bandwidth is the primary bottleneck; consider mesh/topology redesign or higher link bandwidth"
        )
        new_recommendations.append(
            "Reduce communication volume through operator scheduling or data partitioning"
        )

    elif primary == "capacity" or total_memory_capacity_gb < 16:
        new_recommendations.append(
            "Memory capacity is limiting; consider adding more chiplets or higher-capacity memory modules"
        )

    elif primary == "power_thermal" or total_power_watts > 200:
        new_recommendations.append(
            "Power/thermal constraints are active; consider DVFS, power gating, or thermal optimization"
        )
        if energy_per_inference_mj > 1.0:
            new_recommendations.append(
                "High energy per inference; consider model quantization or architecture optimization"
            )

    else:
        new_recommendations.append(
            "System appears balanced; optimization opportunities lie in fine-tuning parameters"
        )

    # Add roofline analysis recommendation
    if roofline_util > 0.8:
        new_recommendations.append(
            "System is operating near the roofline; consider algorithm-level optimizations to increase arithmetic intensity"
        )
    elif roofline_util < 0.3:
        new_recommendations.append(
            "System is compute- or memory-bound below the roofline; investigate kernel optimization or data reuse"
        )

    # Update the bottleneck report with enriched data
    enriched_bottleneck = BottleneckReport(
        primary_bottleneck=primary,
        bound_type=primary,
        severity_score=bottleneck.severity_score,
        utilization_breakdown={
            "compute_util_pct": round(compute_util * 100, 2),
            "memory_bandwidth_tbps": round(aggregate_memory_bandwidth_tbps, 2),
            "roofline_util_pct": round(roofline_util * 100, 2),
        },
        details=details + [f"Analysis: compute_util={compute_util:.1%}, roofline_util={roofline_util:.1%}"],
        recommendations=list(set(recommendations + new_recommendations)),
    )

    return enriched_bottleneck


def compare_results(results: list[SimulationResult]) -> ComparisonResult:
    """Compare 2+ SimulationResult objects across key metrics.

    Computes metric_deltas, winner_by_metric, and a concise summary.
    """
    if len(results) < 2:
        raise ValueError("compare_results requires at least 2 SimulationResult objects")

    # Define the key metrics to compare
    metric_names = [
        "throughput_tokens_per_sec",
        "throughput_ops_per_sec",
        "estimated_latency_ms",
        "total_power_watts",
        "total_area_mm2",
        "estimated_cost_usd",
        "energy_per_inference_mj",
        "utilization_pct",
    ]

    # Collect metric values for each result
    metric_values: dict[str, list[float]] = {name: [] for name in metric_names}
    for result in results:
        for name in metric_names:
            val = 0.0
            if hasattr(result.metrics, name):
                val = float(getattr(result.metrics, name))
            metric_values[name].append(val)

    # Compute metric_deltas and winner_by_metric
    metric_deltas: dict[str, dict[str, float]] = {}
    winner_by_metric: dict[str, str] = {}

    # Use result[0] as baseline for all metrics
    if not results:
        return ComparisonResult(
            comparison_id="comparison-0-results",
            fidelity=FidelityLevel.ANALYTICAL,
            fidelity_disclaimer="",
            designs=[],
            metric_deltas={},
            winner_by_metric={},
            summary="",
        )

    baseline_results = results[0]

    for i, name in enumerate(metric_names):
        values = metric_values[name]
        if not values:
            continue

        # Baseline is always the first result's value
        baseline = values[0] if len(values) > 0 else 0.0

        # Store baseline delta info
        metric_deltas[name] = {
            "baseline": baseline,
            "baseline_delta": 0.0,
            "baseline_pct_change": 0.0,
        }

        # Add comparisons with other results
        for j in range(1, len(results)):
            result_val = values[j] if len(values) > j else baseline
            delta = result_val - baseline
            pct_change = (delta / baseline * 100) if baseline != 0 else 0.0
            metric_deltas[name][f"result_{j}"] = result_val
            metric_deltas[name][f"delta_{j}"] = delta
            metric_deltas[name][f"pct_change_{j}"] = pct_change

    # Determine winner for each metric
    for i, name in enumerate(metric_names):
        values = metric_values[name]
        if not values:
            continue

        is_minimize = name in {"estimated_latency_ms", "total_power_watts", "total_area_mm2", "estimated_cost_usd", "energy_per_inference_mj"}

        best_idx = 0
        best_val = values[0] if len(values) > 0 else 0.0
        for j in range(1, len(values)):
            val = values[j]
            if is_minimize:
                if val < best_val:
                    best_val = val
                    best_idx = j
            else:
                if val > best_val:
                    best_val = val
                    best_idx = j

        if best_idx < len(results):
            winner_by_metric[name] = results[best_idx].design_name
        else:
            winner_by_metric[name] = "unknown"

    # Compute concise summary
    design_names = [r.design_name for r in results]

    # Find the overall winner based on throughput_tokens_per_sec (primary objective)
    throughput_values = metric_values["throughput_tokens_per_sec"]
    best_throughput_idx = 0
    best_throughput = throughput_values[0] if len(throughput_values) > 0 else 0.0
    for i in range(1, len(throughput_values)):
        if throughput_values[i] > best_throughput:
            best_throughput = throughput_values[i]
            best_throughput_idx = i

    summary_parts = [
        f"Compared {len(results)} simulation results: {', '.join(design_names)}",
        f"Best throughput: {results[best_throughput_idx].design_name} "
        f"({best_throughput:.1f} tokens/sec)",
    ]

    # Add per-metric winners (top 4)
    for name in metric_names[:4]:  # Top 4 metrics
        winner = winner_by_metric.get(name, "unknown")
        summary_parts.append(f"{name}: winner is {winner}")

    summary = " | ".join(summary_parts)

    return ComparisonResult(
        comparison_id=f"comparison-{len(results)}-results",
        fidelity=baseline_results.fidelity,
        fidelity_disclaimer=baseline_results.fidelity_disclaimer,
        designs= [{"name": d} for d in design_names],
        metric_deltas=metric_deltas,
        winner_by_metric=winner_by_metric,
        summary=summary,
    )