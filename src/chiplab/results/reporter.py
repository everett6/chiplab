"""Export functions for ChipLab simulation results, comparisons, and sweeps.

Markdown reports include: Fidelity level & disclaimer banner, Key Metrics table,
Bottleneck Analysis & Recommendations, and Assumptions & Provenance table.
CSV/JSON use pandas cleanly. If output_path is provided, write to disk and return text/path.
"""

from __future__ import annotations

from typing import Any, Literal
import json
from pathlib import Path

import pandas as pd


def _to_dict(obj: Any) -> dict:
    """Convert a pydantic model or dict to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _result_to_dataframe(result: Any) -> pd.DataFrame:
    """Convert a SimulationResult to a flat dict suitable for a DataFrame row."""

    metrics = _to_dict(getattr(result, "metrics", {})) or {}
    bottleneck = _to_dict(getattr(result, "bottleneck", {})) or {}
    assumptions = getattr(result, "assumptions", []) or []

    rows = {
        "run_id": getattr(result, "run_id", ""),
        "design_id": getattr(result, "design_id", ""),
        "design_name": getattr(result, "design_name", ""),
        "design_version": getattr(result, "design_version", "0.1"),
        "workload_name": getattr(result, "workload_name", "default_workload"),
        "fidelity": getattr(result, "fidelity", "analytical"),
        "fidelity_disclaimer": getattr(result, "fidelity_disclaimer", ""),
        "timestamp": getattr(result, "timestamp", ""),
        "success": getattr(result, "success", True),
    }

    # Key metrics
    for key in [
        "total_memory_capacity_gb",
        "aggregate_memory_bandwidth_tbps",
        "peak_compute_tops",
        "effective_compute_tops",
        "interconnect_bisection_bw_tbps",
        "estimated_latency_ms",
        "throughput_tokens_per_sec",
        "throughput_ops_per_sec",
        "total_area_mm2",
        "total_power_watts",
        "estimated_temp_c",
        "estimated_cost_usd",
        "arithmetic_intensity",
        "ridge_point_ops_per_byte",
        "utilization_pct",
        "energy_per_inference_mj",
    ]:
        rows[key] = metrics.get(key, 0.0)

    # Bottleneck
    rows["primary_bottleneck"] = bottleneck.get("primary_bottleneck", "balanced")
    rows["severity_score"] = bottleneck.get("severity_score", 0.0)

    # Assumptions as concatenated string
    assumptions_str = "; ".join(
        f"{_to_dict(a).get('parameter', '')} ({_to_dict(a).get('provenance', 'assumed')})"
        for a in assumptions
    )
    rows["assumptions"] = assumptions_str

    return pd.DataFrame([rows])


def _comparison_to_dataframe(comp: Any) -> pd.DataFrame:
    """Convert a ComparisonResult to a DataFrame for export."""

    designs = getattr(comp, "designs", []) or []
    metric_deltas = _to_dict(getattr(comp, "metric_deltas", {})) or {}
    winner_by_metric = _to_dict(getattr(comp, "winner_by_metric", {})) or {}

    rows = []
    for i, d in enumerate(designs):
        row = {
            "design_index": i,
            "design_name": d.get("design_name", f"design_{i}"),
            "design_version": d.get("design_version", "0.1"),
            "fidelity": d.get("fidelity", "analytical"),
        }
        # Add metric values
        all_metrics = set()
        for deltas in metric_deltas.values():
            all_metrics.update(deltas.keys())
        for m in sorted(all_metrics):
            row[f"metric_{m}"] = metric_deltas.get(d.get("design_name", ""), {}).get(m, 0.0)
        row["winner_by_metric"] = winner_by_metric.get(d.get("design_name", ""), "")
        rows.append(row)

    return pd.DataFrame(rows)


def _sweep_to_dataframe(sweep: Any) -> pd.DataFrame:
    """Convert a SweepResult to a DataFrame for export."""

    points = getattr(sweep, "points", []) or []
    swept_params = getattr(sweep, "swept_parameters", []) or []

    rows = []
    for p in points:
        row = {"sweep_id": getattr(sweep, "sweep_id", ""),
               "design_name": getattr(sweep, "design_name", ""),
               "design_version": getattr(sweep, "design_version", "0.1"),
               "fidelity": getattr(sweep, "fidelity", "analytical")}
        row.update(_to_dict(p) if not isinstance(p, dict) else p)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        # Ensure swept params are columns
        for param in swept_params:
            if param not in df.columns:
                df[param] = None
        return df
    return pd.DataFrame()


def _format_markdown_banner(fidelity: str, disclaimer: str = "") -> str:
    """Format a fidelity level banner with disclaimer."""
    fidelity_labels = {
        "analytical": "Analytical Fidelity",
        "discrete_event": "Discrete-Event Fidelity",
        "specialized": "Specialized Fidelity",
        "rtl_circuit": "RTL Circuit Fidelity",
    }
    label = fidelity_labels.get(fidelity, fidelity.upper())
    banner = f"## Fidelity: {label}\n"
    if disclaimer:
        banner += f"> **Disclaimer:** {disclaimer}\n"
    return banner


def _format_metrics_table(result: Any) -> str:
    """Format key metrics as a markdown table."""
    metrics = _to_dict(getattr(result, "metrics", {})) or {}

    # Build a compact metrics table
    rows = []
    for key, label in [
        ("total_memory_capacity_gb", "Total Memory Capacity (GB)"),
        ("aggregate_memory_bandwidth_tbps", "Memory Bandwidth (TB/s)"),
        ("peak_compute_tops", "Peak Compute (TOPS)"),
        ("effective_compute_tops", "Effective Compute (TOPS)"),
        ("interconnect_bisection_bw_tbps", "Interconnect BW (TB/s)"),
        ("estimated_latency_ms", "Est. Latency (ms)"),
        ("throughput_tokens_per_sec", "Throughput (tokens/s)"),
        ("throughput_ops_per_sec", "Throughput (ops/s)"),
        ("total_area_mm2", "Area (mm²)"),
        ("total_power_watts", "Power (W)"),
        ("estimated_temp_c", "Est. Temp (°C)"),
        ("estimated_cost_usd", "Est. Cost (USD)"),
        ("arithmetic_intensity", "Arithmetic Intensity"),
        ("ridge_point_ops_per_byte", "Ridge Point (ops/byte)"),
        ("utilization_pct", "Utilization (%)"),
        ("energy_per_inference_mj", "Energy per Inference (mJ)"),
    ]:
        val = metrics.get(key, 0.0)
        if isinstance(val, float) and val == int(val):
            val = int(val)
        rows.append(f"| {label} | `{val}` |")

    metric_md = "| Metric | Value |\n|---|---|\n" + "\n".join(rows)
    return metric_md


def _format_bottleneck_section(result: Any) -> str:
    """Format bottleneck analysis and recommendations."""
    bottleneck = _to_dict(getattr(result, "bottleneck", {})) or {}

    primary = bottleneck.get("primary_bottleneck", "balanced")
    severity = bottleneck.get("severity_score", 0.0)
    details = bottleneck.get("details", []) or []
    recommendations = bottleneck.get("recommendations", []) or []

    md = f"**Primary Bottleneck:** {primary} (severity: {severity:.2f})\n\n"
    if details:
        md += "**Details:**\n"
        for d in details:
            md += f"- {d}\n"
    if recommendations:
        md += "**Recommendations:**\n"
        for r in recommendations:
            md += f"- {r}\n"
    return md


def _format_assumptions_table(result: Any) -> str:
    """Format assumptions and provenance table."""
    assumptions = getattr(result, "assumptions", []) or []
    # Convert pydantic models to dicts
    converted_assumptions = [_to_dict(a) for a in assumptions]

    if not converted_assumptions:
        return "*No assumptions recorded.*"

    rows = []
    for a in converted_assumptions:
        param = a.get("parameter", "")
        value = a.get("value", "")
        unit = a.get("unit", "")
        provenance = a.get("provenance", "assumed")
        source = a.get("source", "")
        notes = a.get("notes", "")
        rows.append(f"| {param} | {value} {unit} | {provenance} | {source} | {notes} |")

    md = "| Parameter | Value | Provenance | Source | Notes |\n"
    md += "|---|---|---|---|---|\n"
    md += "\n".join(rows)
    return md


def export_simulation_report(
    result: Any,
    fmt: Literal["markdown", "json", "csv"] = "markdown",
    output_path: str | Path | None = None,
) -> str:
    """Export a simulation result report in the specified format.

    Args:
        result: A SimulationResult instance.
        fmt: Output format ("markdown", "json", or "csv").
        output_path: If provided, write report to this path and return the path.

    Returns:
        The report text (markdown) or path (json/csv if output_path given).
    """
    fmt = fmt.lower()

    # Convert metrics to dict for consistent access
    metrics = _to_dict(getattr(result, "metrics", {})) or {}

    if fmt == "markdown":
        lines = []
        # Fidelity banner
        lines.append(_format_markdown_banner(result.fidelity, result.fidelity_disclaimer))
        # Key metrics table
        lines.append(_format_metrics_table(result))
        # Bottleneck analysis
        lines.append(_format_bottleneck_section(result))
        # Assumptions & Provenance table
        lines.append(_format_assumptions_table(result))
        text = "\n\n".join(lines)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"Report written to {out}"
        return text

    elif fmt == "json":
        # Convert assumptions to dicts
        assumptions_list = getattr(result, "assumptions", []) or []
        conv_assumptions = [_to_dict(a) for a in assumptions_list]

        data = {
            "run_id": getattr(result, "run_id", ""),
            "design_id": getattr(result, "design_id", ""),
            "design_name": getattr(result, "design_name", ""),
            "design_version": getattr(result, "design_version", "0.1"),
            "workload_name": getattr(result, "workload_name", "default_workload"),
            "fidelity": getattr(result, "fidelity", "analytical"),
            "fidelity_disclaimer": getattr(result, "fidelity_disclaimer", ""),
            "timestamp": getattr(result, "timestamp", ""),
            "success": getattr(result, "success", True),
            "metrics": {
                "total_memory_capacity_gb": metrics.get("total_memory_capacity_gb", 0.0),
                "aggregate_memory_bandwidth_tbps": metrics.get("aggregate_memory_bandwidth_tbps", 0.0),
                "peak_compute_tops": metrics.get("peak_compute_tops", 0.0),
                "effective_compute_tops": metrics.get("effective_compute_tops", 0.0),
                "interconnect_bisection_bw_tbps": metrics.get("interconnect_bisection_bw_tbps", 0.0),
                "estimated_latency_ms": metrics.get("estimated_latency_ms", 0.0),
                "throughput_tokens_per_sec": metrics.get("throughput_tokens_per_sec", None),
                "throughput_ops_per_sec": metrics.get("throughput_ops_per_sec", 0.0),
                "total_area_mm2": metrics.get("total_area_mm2", 0.0),
                "total_power_watts": metrics.get("total_power_watts", 0.0),
                "estimated_temp_c": metrics.get("estimated_temp_c", 25.0),
                "estimated_cost_usd": metrics.get("estimated_cost_usd", 0.0),
                "arithmetic_intensity": metrics.get("arithmetic_intensity", 0.0),
                "ridge_point_ops_per_byte": metrics.get("ridge_point_ops_per_byte", 0.0),
                "utilization_pct": metrics.get("utilization_pct", 0.0),
                "energy_per_inference_mj": metrics.get("energy_per_inference_mj", 0.0),
            },
            "bottleneck": {
                "primary_bottleneck": _to_dict(getattr(result, "bottleneck", {}))
                .get("primary_bottleneck", "balanced"),
                "severity_score": _to_dict(getattr(result, "bottleneck", {}))
                .get("severity_score", 0.0),
            },
            "assumptions": [
                {
                    "parameter": a.get("parameter", ""),
                    "value": a.get("value", ""),
                    "unit": a.get("unit", ""),
                    "provenance": a.get("provenance", "assumed"),
                    "source": a.get("source", ""),
                    "notes": a.get("notes", ""),
                }
                for a in conv_assumptions
            ],
        }
        text = json.dumps(data, indent=2)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"JSON report written to {out}"
        return text

    elif fmt == "csv":
        df = _result_to_dataframe(result)
        text = df.to_csv(index=False)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"CSV report written to {out}"
        return text

    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'markdown', 'json', or 'csv'.")


def export_comparison_report(
    comparison: Any,
    fmt: Literal["markdown", "json", "csv"] = "markdown",
    output_path: str | Path | None = None,
) -> str:
    """Export a comparison report in the specified format.

    Args:
        comparison: A ComparisonResult instance.
        fmt: Output format ("markdown", "json", or "csv").
        output_path: If provided, write report to this path and return the path.

    Returns:
        The report text (markdown) or path (json/csv if output_path given).
    """
    fmt = fmt.lower()

    if fmt == "markdown":
        # Build a simple markdown report for comparison
        lines = []
        lines.append("# Comparison Report")
        lines.append(f"**Fidelity:** {getattr(comparison, 'fidelity', 'analytical')}")
        lines.append(f"**Summary:** {getattr(comparison, 'summary', '')}")
        lines.append("")

        # Designs table
        designs = getattr(comparison, "designs", []) or []
        lines.append("| Design | Version | Fidelity |")
        lines.append("|---|---|---|")
        for d in designs:
            lines.append(f"| {d.get('design_name', '?')} | {d.get('design_version', '?')} | {d.get('fidelity', '?')} |")
        lines.append("")

        # Metric deltas
        metric_deltas = _to_dict(getattr(comparison, "metric_deltas", {})) or {}
        if metric_deltas:
            lines.append("| Metric | Delta | Winner |")
            lines.append("|---|---|---|")
            for metric, deltas in metric_deltas.items():
                for design_name, delta in deltas.items():
                    winner = metric_deltas.get("winner_by_metric", {}).get(metric, "")
                    lines.append(f"| {metric} | {delta} | {winner} |")
        lines.append("")

        # Winner by metric summary
        winner_by_metric = _to_dict(getattr(comparison, "winner_by_metric", {})) or {}
        if winner_by_metric:
            lines.append("**Winner by metric:**")
            for metric, winner in winner_by_metric.items():
                lines.append(f"- {metric}: {winner}")
            lines.append("")

        text = "\n".join(lines)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"Comparison report written to {out}"
        return text

    elif fmt == "json":
        data = {
            "comparison_id": getattr(comparison, "comparison_id", ""),
            "fidelity": getattr(comparison, "fidelity", "analytical"),
            "fidelity_disclaimer": getattr(comparison, "fidelity_disclaimer", ""),
            "designs": getattr(comparison, "designs", []),
            "metric_deltas": _to_dict(getattr(comparison, "metric_deltas", {})),
            "winner_by_metric": _to_dict(getattr(comparison, "winner_by_metric", {})),
            "summary": getattr(comparison, "summary", ""),
        }
        text = json.dumps(data, indent=2)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"Comparison JSON report written to {out}"
        return text

    elif fmt == "csv":
        df = _comparison_to_dataframe(comparison)
        text = df.to_csv(index=False)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"Comparison CSV report written to {out}"
        return text

    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'markdown', 'json', or 'csv'.")


def export_sweep_report(
    sweep: Any,
    fmt: Literal["markdown", "json", "csv"] = "markdown",
    output_path: str | Path | None = None,
) -> str:
    """Export a sweep report in the specified format.

    Args:
        sweep: A SweepResult instance.
        fmt: Output format ("markdown", "json", or "csv").
        output_path: If provided, write report to this path and return the path.

    Returns:
        The report text (markdown) or path (json/csv if output_path given).
    """
    fmt = fmt.lower()

    if fmt == "markdown":
        lines = []
        # Fidelity banner
        lines.append(_format_markdown_banner(sweep.fidelity, sweep.fidelity_disclaimer))
        # Summary info
        lines.append(f"**Sweep ID:** {getattr(sweep, 'sweep_id', '')}")
        lines.append(f"**Objective:** {getattr(sweep, 'objective', 'throughput_tokens_per_sec')}")
        lines.append(f"**Swept parameters:** {', '.join(getattr(sweep, 'swept_parameters', []))}")
        lines.append("")

        # Best point if available
        best_point = _to_dict(getattr(sweep, "best_point", None)) or {}
        if best_point:
            lines.append("**Best point:**")
            for key, val in best_point.items():
                lines.append(f"- {key}: {val}")
            lines.append("")

        # Points table
        points = getattr(sweep, "points", []) or []
        if points:
            lines.append("| Point | " + " | ".join(getattr(sweep, 'swept_parameters', [])) + " |")
            lines.append("---" + "|---|" * (len(getattr(sweep, 'swept_parameters', [])) + 1))
            for p in points:
                row_vals = [str(_to_dict(p).get(param, "")) for param in getattr(sweep, 'swept_parameters', [])]
                lines.append("| " + " | ".join(row_vals) + " |")
        lines.append("")

        text = "\n".join(lines)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"Sweep report written to {out}"
        return text

    elif fmt == "json":
        data = {
            "sweep_id": getattr(sweep, "sweep_id", ""),
            "design_name": getattr(sweep, "design_name", ""),
            "design_version": getattr(sweep, "design_version", "0.1"),
            "fidelity": getattr(sweep, "fidelity", "analytical"),
            "fidelity_disclaimer": getattr(sweep, "fidelity_disclaimer", ""),
            "swept_parameters": getattr(sweep, "swept_parameters", []),
            "objective": getattr(sweep, "objective", "throughput_tokens_per_sec"),
            "best_point": _to_dict(getattr(sweep, "best_point", None)),
            "point_count": len(getattr(sweep, "points", []) or []),
        }
        text = json.dumps(data, indent=2)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"Sweep JSON report written to {out}"
        return text

    elif fmt == "csv":
        df = _sweep_to_dataframe(sweep)
        text = df.to_csv(index=False)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            return f"Sweep CSV report written to {out}"
        return text

    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'markdown', 'json', or 'csv'.")