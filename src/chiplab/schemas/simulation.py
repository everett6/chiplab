from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import FidelityLevel, Provenance


class SimulationConfig(BaseModel):
    """Configuration for a ChipLab simulation run."""

    fidelity: FidelityLevel = FidelityLevel.ANALYTICAL
    seed: int = 42
    num_requests: int = 32
    allow_estimated_defaults: bool = True
    sim_time_ns: float | None = None


class AssumptionRecord(BaseModel):
    """Tracks provenance, source, and uncertainty for any assumed/estimated parameter."""

    parameter: str
    value: Any
    unit: str | None = None
    provenance: Provenance = Provenance.ASSUMED
    source: str | None = None
    uncertainty_pct: float | None = None
    notes: str | None = None


class SimulationMetrics(BaseModel):
    """Hardware architecture and performance metrics from a simulation run."""

    total_memory_capacity_gb: float = 0.0
    aggregate_memory_bandwidth_tbps: float = 0.0
    peak_compute_tops: float = 0.0
    effective_compute_tops: float = 0.0
    interconnect_bisection_bw_tbps: float = 0.0
    estimated_latency_ms: float = 0.0
    throughput_tokens_per_sec: float | None = None
    throughput_ops_per_sec: float = 0.0
    total_area_mm2: float = 0.0
    total_power_watts: float = 0.0
    estimated_temp_c: float = 25.0
    estimated_cost_usd: float = 0.0
    arithmetic_intensity: float = 0.0
    ridge_point_ops_per_byte: float = 0.0
    utilization_pct: float = 0.0
    energy_per_inference_mj: float = 0.0


class BottleneckReport(BaseModel):
    """Identifies what limits system performance and suggests improvements."""

    primary_bottleneck: str
    bound_type: Literal[
        "memory_bandwidth",
        "compute",
        "interconnect",
        "capacity",
        "power_thermal",
        "balanced",
    ] | str
    severity_score: float = 0.0
    utilization_breakdown: dict[str, float] = Field(default_factory=dict)
    details: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Complete result of a single simulation run, including fidelity and assumptions."""

    run_id: str
    design_id: str
    design_name: str
    design_version: str = "0.1"
    workload_name: str = "default_workload"
    fidelity: FidelityLevel
    fidelity_disclaimer: str
    timestamp: float = Field(default_factory=time.time)
    metrics: SimulationMetrics
    bottleneck: BottleneckReport
    assumptions: list[AssumptionRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw_details: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_message: str | None = None
    uncertainty_bounds: dict[str, dict[str, float]] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of validating a DesignSpec for contradictions and physical limits."""

    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_parameters: list[str] = Field(default_factory=list)
    provenance_summary: dict[str, int] = Field(default_factory=dict)


class SweepPoint(BaseModel):
    """Single point in a parameter sweep."""

    parameters: dict[str, Any]
    run_id: str
    metrics: SimulationMetrics
    primary_bottleneck: str
    bound_type: str


class SweepResult(BaseModel):
    """Result of sweeping one or more design/workload parameters."""

    sweep_id: str
    design_name: str
    design_version: str
    fidelity: FidelityLevel
    fidelity_disclaimer: str
    swept_parameters: list[str]
    points: list[SweepPoint] = Field(default_factory=list)
    best_point: SweepPoint | None = None
    objective: str = "throughput_tokens_per_sec"


class ComparisonResult(BaseModel):
    """Side-by-side comparison of two or more simulation runs/designs."""

    comparison_id: str
    fidelity: FidelityLevel
    fidelity_disclaimer: str
    designs: list[dict[str, Any]] = Field(default_factory=list)
    metric_deltas: dict[str, dict[str, float]] = Field(default_factory=dict)
    winner_by_metric: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
