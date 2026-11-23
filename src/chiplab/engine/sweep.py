"""Parameter sweep engine for ChipLab."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

# Cartesian sweeps grow fast; beyond this, use run_digital_twin_search (sampled search).
MAX_SWEEP_POINTS = 5000

from ..schemas.design import DesignSpec
from ..schemas.simulation import (
    SimulationConfig,
    SimulationResult,
    SweepResult,
    SweepPoint,
    FidelityLevel,
)
from ..schemas.workload import WorkloadSpec
from ..backends import list_backends, SimulatorBackend, RamulatorBackend, ScaleSimBackend, AccelSimBackend
from . import analytical, discrete_event


def _set_nested_value(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a value on an object using a dotted path like 'chiplets[0].count'.

    Supports Pydantic BaseModel attributes and list index notation.
    Also handles plain dicts from model_dump().
    """
    parts = dotted_path.split(".")

    # Walk to the parent of the final target
    if not parts:
        return

    # Navigate to the parent object
    parent = obj
    for i, part in enumerate(parts[:-1]):
        if "[" in part and part.endswith("]"):
            # Handle index notation like chiplets[0]
            name = part[: part.index("[")]
            idx = int(part[part.index("[") + 1 : -1])
            if isinstance(parent, dict):
                if name not in parent:
                    raise AttributeError(f"Key '{name}' not found in dict")
                parent = parent[name]
                if idx < 0 or idx >= len(parent):
                    raise IndexError(f"Index {idx} out of range for list of length {len(parent)}")
                parent = parent[idx]
            elif hasattr(parent, name):
                parent = getattr(parent, name)
                if idx < 0 or idx >= len(parent):
                    raise IndexError(f"Index {idx} out of range for list of length {len(parent)}")
                parent = parent[idx]
            else:
                raise AttributeError(f"Cannot navigate '{dotted_path}': '{name}' not found on object")
        else:
            if isinstance(parent, dict):
                if part not in parent:
                    raise AttributeError(f"Key '{part}' not found in dict")
                parent = parent[part]
            elif hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                raise AttributeError(f"Cannot navigate '{dotted_path}': '{part}' not found on object")

    # Set the final attribute
    final_part = parts[-1]
    if "[" in final_part and final_part.endswith("]"):
        # Handle index notation like chiplets[0].count -> set count[0]
        name = final_part[: final_part.index("[")]
        idx = int(final_part[final_part.index("[") + 1 : -1])
        if isinstance(parent, dict):
            if name not in parent:
                raise AttributeError(f"Key '{name}' not found in dict")
            target = parent[name]
            if idx < 0 or idx >= len(target):
                raise IndexError(f"Index {idx} out of range for list of length {len(target)}")
            target[idx] = value
        elif hasattr(parent, name):
            target = getattr(parent, name)
            if idx < 0 or idx >= len(target):
                raise IndexError(f"Index {idx} out of range for list of length {len(target)}")
            target[idx] = value
        else:
            raise AttributeError(f"Cannot set '{dotted_path}': '{name}' not found on object")
    else:
        if isinstance(parent, dict):
            parent[final_part] = value
        elif hasattr(parent, final_part):
            setattr(parent, final_part, value)
        else:
            raise AttributeError(f"Cannot set '{dotted_path}': '{final_part}' not found on object")


def _apply_parameters_to_design(design: DesignSpec, parameters: dict[str, Any]) -> DesignSpec:
    """Apply a dictionary of parameter values to a DesignSpec using dotted paths.

    Supports paths like: 'chiplets[0].count', 'interconnect.link_bandwidth_gbps',
    'design.chiplets[0].count' (with design. prefix).
    Paths starting with 'workload.' are ignored.
    """
    # Make a copy via model_dump and re-construct
    result = design.model_dump()
    for path, value in parameters.items():
        # Skip workload parameters
        if path.startswith("workload."):
            continue
        # Strip 'design.' prefix if present
        if path.startswith("design."):
            path = path[len("design."):]
        _set_nested_value(result, path, value)
    return DesignSpec(**result)


def _apply_parameters_to_workload(workload: WorkloadSpec, parameters: dict[str, Any]) -> WorkloadSpec:
    """Apply a dictionary of parameter values to a WorkloadSpec using dotted paths.

    Supports paths like: 'workload.batch_size', 'workload.hidden_dim'
    Only processes keys starting with 'workload.' (prefix stripped).
    """
    result = workload.model_dump()
    for path, value in parameters.items():
        # Only process workload parameters (keys starting with 'workload.')
        if not path.startswith("workload."):
            continue
        # Strip 'workload.' prefix
        path = path[len("workload."):]
        _set_nested_value(result, path, value)
    return WorkloadSpec(**result)


def _run_simulation(
    design: DesignSpec,
    workload: WorkloadSpec,
    config: SimulationConfig,
) -> SimulationResult:
    """Run the appropriate simulation based on config.fidelity."""
    if config.fidelity == FidelityLevel.ANALYTICAL:
        return analytical.run_analytical_simulation(design, workload, config)
    elif config.fidelity == FidelityLevel.DISCRETE_EVENT:
        return discrete_event.run_discrete_event_simulation(design, workload, config)
    else:
        # Fallback to analytical
        return analytical.run_analytical_simulation(design, workload, config)


def _evaluate_objective(
    metrics: Any,
    objective: str,
    minimize_objectives: set[str] | None = None,
) -> float:
    """Extract the objective value from SimulationMetrics.

    Returns a float for comparison purposes.
    """
    if minimize_objectives is None:
        minimize_objectives = {
            "estimated_latency_ms",
            "total_power_watts",
            "estimated_cost_usd",
            "total_area_mm2",
        }

    # Helper to get metric value
    def get_metric(name: str) -> float:
        if hasattr(metrics, name):
            val = getattr(metrics, name)
            return float(val) if val is not None else 0.0
        return 0.0

    if objective == "throughput_tokens_per_sec":
        return get_metric("throughput_tokens_per_sec") or 0.0
    elif objective == "throughput_ops_per_sec":
        return get_metric("throughput_ops_per_sec") or 0.0
    elif objective in minimize_objectives:
        return get_metric(objective)
    else:
        # Default: maximize, try common metrics
        return get_metric(objective) or 0.0


def run_parameter_sweep(
    design: DesignSpec,
    workload: WorkloadSpec,
    config: SimulationConfig,
    parameters: dict[str, list[Any]],
    objective: str = "throughput_tokens_per_sec",
    design_id: str = "unsaved",
    progress: Callable[[int, int], None] | None = None,
) -> SweepResult:
    """Run a parameter sweep over the Cartesian product of given parameters.

    ``progress(done, total)`` is called after each point (MCP progress notifications).
    Raises ValueError if the sweep exceeds MAX_SWEEP_POINTS points.

    Generates all combinations using itertools.product, runs simulations,
    and returns a SweepResult with the best point based on the objective.

    Supports dotted paths like:
      - "chiplets[0].count"
      - "chiplets[0].memory.channels"
      - "interconnect.link_bandwidth_gbps"
      - "workload.batch_size"

    Parameters dict should have keys like:
      - "chiplets[0].count"
      - "workload.batch_size"
    """
    # Generate the Cartesian product of parameter values
    param_keys = list(parameters.keys())
    param_values = list(parameters.values())

    if not param_keys:
        # No parameters to sweep, just run one simulation
        point_params: dict[str, Any] = {}
        adjusted_design = _apply_parameters_to_design(design, point_params)
        adjusted_workload = _apply_parameters_to_workload(workload, point_params)
        result = _run_simulation(adjusted_design, adjusted_workload, config)
        best_point = SweepPoint(
            parameters=point_params,
            run_id=f"sweep-{design_id}-0",
            metrics=result.metrics,
            primary_bottleneck=result.bottleneck.primary_bottleneck,
            bound_type=result.bottleneck.bound_type,
        )
        return SweepResult(
            sweep_id=f"sweep-{design_id}",
            design_name=design.name,
            design_version=design.version,
            fidelity=config.fidelity,
            fidelity_disclaimer=result.fidelity_disclaimer,
            swept_parameters=param_keys,
            points=[best_point],
            best_point=best_point,
            objective=objective,
        )

    total = math.prod(len(v) for v in param_values)
    if total > MAX_SWEEP_POINTS:
        raise ValueError(
            f"Sweep has {total} points (> {MAX_SWEEP_POINTS}). Narrow the value lists, or use "
            "run_digital_twin_search for sampled search over large spaces.")

    points = []
    # Track best based on objective
    # We need to know if we're minimizing or maximizing
    minimize_objectives = {
        "estimated_latency_ms",
        "total_power_watts",
        "estimated_cost_usd",
        "total_area_mm2",
    }
    minimize = objective in minimize_objectives

    best_value: float | None = None
    best_point: SweepPoint | None = None
    last_fidelity_disclaimer = ""

    # Iterate over all combinations
    for combination in itertools.product(*param_values):
        point_params = dict(zip(param_keys, combination))
        adjusted_design = _apply_parameters_to_design(design, point_params)
        adjusted_workload = _apply_parameters_to_workload(workload, point_params)
        result = _run_simulation(adjusted_design, adjusted_workload, config)

        metric_value = _evaluate_objective(result.metrics, objective, minimize)

        point = SweepPoint(
            parameters=point_params,
            run_id=f"sweep-{design_id}-{len(points)}",
            metrics=result.metrics,
            primary_bottleneck=result.bottleneck.primary_bottleneck,
            bound_type=result.bottleneck.bound_type,
        )
        points.append(point)
        if progress is not None:
            progress(len(points), total)

        # Update best and store fidelity_disclaimer
        if best_point is None:
            best_point = point
            best_value = metric_value
            last_fidelity_disclaimer = result.fidelity_disclaimer
        else:
            if minimize:
                if metric_value < best_value:  # type: ignore[comparison]
                    best_value = metric_value
                    best_point = point
                    last_fidelity_disclaimer = result.fidelity_disclaimer
            else:
                if metric_value > best_value:  # type: ignore[comparison]
                    best_value = metric_value
                    best_point = point
                    last_fidelity_disclaimer = result.fidelity_disclaimer

    return SweepResult(
        sweep_id=f"sweep-{design_id}",
        design_name=design.name,
        design_version=design.version,
        fidelity=config.fidelity,
        fidelity_disclaimer=last_fidelity_disclaimer if best_point else "",
        swept_parameters=param_keys,
        points=points,
        best_point=best_point,
        objective=objective,
    )