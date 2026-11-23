"""Chiplab MCP Server - FastMCP-based server for chip design, simulation, and analysis tools."""

from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

# --- Database path configuration ---
DB_PATH = os.environ.get("CHIPLAB_DB_PATH", "chiplab_experiments.db")

from chiplab.experiments.store import ExperimentStore
from chiplab.library.components import list_components, get_component_defaults, COMPONENT_LIBRARY
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.simulation import (
    SimulationResult,
    ValidationResult,
    SimulationConfig,
    BottleneckReport,
    AssumptionRecord,
    FidelityLevel,
)
from chiplab.engine.validator import validate_design
from chiplab.engine.sweep import run_parameter_sweep
from chiplab.engine.analyzer import analyze_bottleneck, compare_results
from chiplab.results.reporter import export_simulation_report

# Initialize the experiment store
store = ExperimentStore(db_path=DB_PATH)


# --- Helper functions ---

def _parse_spec(spec: str | dict) -> DesignSpec:
    """Parse a YAML string, JSON string, or dict into a DesignSpec."""
    import yaml

    if isinstance(spec, dict):
        data = spec
    else:
        data = yaml.safe_load(spec)
        if not isinstance(data, dict):
            raise ValueError("Design specification must parse to a mapping/dict.")

    if "design" in data and isinstance(data["design"], dict):
        data = data["design"]
    if "project" in data and isinstance(data["project"], dict):
        data = data["project"]
    return DesignSpec.model_validate(data)


# --- MCP Tools ---

def chip_create_design(
    spec: str | dict,
    name: str | None = None,
    version: str | None = None,
    parent_version: str | None = None,
) -> dict:
    """Create a new design specification.

    Accepts YAML string, JSON string, or dict.
    Validates DesignSpec and saves to ExperimentStore.
    Returns {design_id, name, version, success, validation}.
    """
    design_spec = _parse_spec(spec)

    # Set name and version
    if name is not None:
        design_spec.name = name
    if version is not None:
        design_spec.version = version

    # Set parent_version if provided
    if parent_version is not None:
        design_spec.parent_version = parent_version

    # Validate the design
    validation_result = validate_design(design_spec)

    # Save to experiment store
    design_id = store.save_design(design_spec)

    return {
        "design_id": design_id,
        "success": True,
        "name": design_spec.name,
        "version": design_spec.version,
        "parent_version": design_spec.parent_version,
        "validation": {
            "valid": validation_result.valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "missing_parameters": validation_result.missing_parameters,
            "provenance_summary": validation_result.provenance_summary,
        },
    }


def chip_get_design(design_id_or_name: str, version: str | None = None) -> dict:
    """Get a design specification by ID or name.

    Returns design spec + history.
    """
    design_id, design_spec = store.get_design(design_id_or_name)

    # Get history
    history = store.get_design_history(design_spec.name)

    # Find matching version if specified
    design = design_spec
    if version is not None:
        for entry in history:
            if entry.get("version") == version:
                design = DesignSpec.model_validate(json.loads(entry.get("spec_json", "{}")))
                break

    return {
        "design_id": design_id,
        "success": True,
        "name": design.name,
        "version": design.version,
        "parent_version": design.parent_version,
        "description": design.description,
        "architecture": design.architecture,
        "spec_yaml": design.to_yaml() if hasattr(design, "to_yaml") else "",
        "history": history,
    }


def chip_validate_design(
    design_id_or_name: str,
    version: str | None = None,
    allow_estimated_defaults: bool = True,
) -> dict:
    """Validate a design specification.

    Runs validate_design and returns ValidationResult dict.
    """
    # Get the design
    design_id, design_spec = store.get_design(design_id_or_name)

    # If version is specified, try to get the right version
    if version is not None:
        history = store.get_design_history(design_spec.name)
        for entry in history:
            if entry.get("version") == version:
                design_spec = DesignSpec.model_validate(json.loads(entry.get("spec_json", "{}")))
                break

    # Run validation
    validation_result = validate_design(design_spec, allow_estimated_defaults=allow_estimated_defaults)

    return {
        "design_id_or_name": design_id_or_name,
        "version": version,
        "allow_estimated_defaults": allow_estimated_defaults,
        "valid": validation_result.valid,
        "errors": validation_result.errors,
        "warnings": validation_result.warnings,
        "missing_parameters": validation_result.missing_parameters,
        "provenance_summary": validation_result.provenance_summary,
    }


def chip_run_simulation(
    design_id_or_name: str,
    workload: dict | None = None,
    fidelity: str = "analytical",
    seed: int = 42,
    num_requests: int = 32,
    allow_estimated_defaults: bool = True,
    version: str | None = None,
) -> dict:
    """Run a simulation for a design.

    Runs analytical or discrete-event simulation, persists in ExperimentStore.
    Returns SimulationResult dict.
    """
    # Get the design
    design_id, design_spec = store.get_design(design_id_or_name, version)

    # Determine fidelity level
    fidelity_level = FidelityLevel.ANALYTICAL
    if fidelity == "discrete_event":
        fidelity_level = FidelityLevel.DISCRETE_EVENT
    elif fidelity == "specialized":
        fidelity_level = FidelityLevel.SPECIALIZED
    elif fidelity == "rtl_circuit":
        fidelity_level = FidelityLevel.RTL_CIRCUIT

    # Build config
    config = SimulationConfig(
        fidelity=fidelity_level,
        seed=seed,
        num_requests=num_requests,
        allow_estimated_defaults=allow_estimated_defaults,
    )

    # Prepare workload
    workload_obj = None
    if workload is not None:
        from chiplab.schemas.workload import WorkloadSpec
        workload_obj = WorkloadSpec.model_validate(workload)

    # Run simulation based on fidelity
    from chiplab.engine import run_analytical_simulation, run_discrete_event_simulation

    if fidelity_level == FidelityLevel.DISCRETE_EVENT:
        result = run_discrete_event_simulation(design_spec, workload_obj, config, design_id)
    else:
        result = run_analytical_simulation(design_spec, workload_obj, config, design_id)

    # Save to experiment store
    simulation_result = SimulationResult(
        run_id=getattr(result, "run_id", f"{design_spec.name}_{design_spec.version}_{seed}"),
        design_id=design_id,
        design_name=design_spec.name,
        design_version=design_spec.version,
        workload_name=workload_obj.name if workload_obj else "default_workload",
        fidelity=fidelity_level,
        fidelity_disclaimer=getattr(result, "fidelity_disclaimer", ""),
        metrics=getattr(result, "metrics", {}),
        bottleneck=getattr(result, "bottleneck", {}),
        assumptions=getattr(result, "assumptions", []),
        warnings=getattr(result, "warnings", []),
        raw_details=getattr(result, "raw_details", {}),
        success=getattr(result, "success", True),
        error_message=getattr(result, "error_message", None),
    )

    run_id = store.save_simulation(simulation_result)

    return {
        "run_id": run_id,
        "design_id": design_id,
        "design_name": design_spec.name,
        "design_version": design_spec.version,
        "workload_name": simulation_result.workload_name,
        "fidelity": fidelity_level.value,
        "fidelity_disclaimer": simulation_result.fidelity_disclaimer,
        "metrics": simulation_result.metrics.model_dump(),
        "bottleneck": simulation_result.bottleneck.model_dump(),
        "assumptions": [a.model_dump() for a in simulation_result.assumptions],
        "success": simulation_result.success,
        "error_message": simulation_result.error_message,
    }


def chip_sweep_parameters(
    design_id_or_name: str,
    parameters: dict[str, list],
    workload: dict | None = None,
    fidelity: str = "analytical",
    objective: str = "throughput_tokens_per_sec",
    version: str | None = None,
) -> dict:
    """Run parameter sweep for a design.

    Runs run_parameter_sweep, persists sweep, returns summary.
    """
    # Get the design
    design_id, design_spec = store.get_design(design_id_or_name)

    # If version is specified, get the right version
    if version is not None:
        history = store.get_design_history(design_spec.name)
        for entry in history:
            if entry.get("version") == version:
                design_spec = DesignSpec.model_validate(json.loads(entry.get("spec_json", "{}")))
                break

    # Build workload if provided
    from chiplab.schemas.workload import WorkloadSpec
    workload_obj = WorkloadSpec.model_validate(workload) if workload is not None else WorkloadSpec()

    # Run parameter sweep
    config = SimulationConfig(fidelity=FidelityLevel(fidelity))
    result = run_parameter_sweep(
        design=design_spec,
        workload=workload_obj,
        config=config,
        parameters=parameters,
        objective=objective,
        design_id=design_id,
    )

    store.save_sweep(result, design_id=design_id)

    return {
        "sweep_id": result.sweep_id,
        "design_id": design_id,
        "design_name": design_spec.name,
        "design_version": design_spec.version,
        "fidelity": result.fidelity.value,
        "fidelity_disclaimer": result.fidelity_disclaimer,
        "swept_parameters": result.swept_parameters,
        "points": [p.model_dump() for p in result.points],
        "best_point": result.best_point.model_dump() if result.best_point else None,
        "objective": result.objective,
    }


def chip_compare_designs(run_ids: list[str]) -> dict:
    """Compare multiple simulation runs.

    Loads runs from ExperimentStore, runs compare_results, saves comparison.
    Returns ComparisonResult dict.
    """
    # Load all simulation results
    results = []
    for run_id in run_ids:
        try:
            sim_result = store.get_simulation(run_id)
            results.append(sim_result)
        except ValueError:
            continue

    if not results:
        return {
            "comparison_id": "",
            "error": "No valid simulation runs found",
            "run_ids": run_ids,
        }

    # Run engine comparison
    comparison = compare_results(results)

    # Save to experiment store
    comparison_id = f"comp_{int(__import__('time').time())}"
    comp_result_type = type(
        "ComparisonResult",
        (),
        {
            "comparison_id": comparison_id,
            "fidelity": comparison.fidelity if hasattr(comparison, "fidelity") else "analytical",
            "fidelity_disclaimer": getattr(comparison, "fidelity_disclaimer", ""),
            "designs": comparison.designs if hasattr(comparison, "designs") else [],
            "metric_deltas": comparison.metric_deltas if hasattr(comparison, "metric_deltas") else {},
            "winner_by_metric": comparison.winner_by_metric if hasattr(comparison, "winner_by_metric") else {},
            "summary": comparison.summary if hasattr(comparison, "summary") else "",
        },
    )()

    store.save_comparison(comp_result_type)

    return {
        "comparison_id": comp_result_type.comparison_id,
        "fidelity": comp_result_type.fidelity,
        "fidelity_disclaimer": comp_result_type.fidelity_disclaimer,
        "designs": comp_result_type.designs,
        "metric_deltas": comp_result_type.metric_deltas,
        "winner_by_metric": comp_result_type.winner_by_metric,
        "summary": comp_result_type.summary,
        "run_ids": run_ids,
    }


def chip_analyze_bottleneck(run_id: str) -> dict:
    """Analyze bottleneck for a simulation run.

    Loads run, runs analyze_bottleneck, returns BottleneckReport dict.
    """
    # Get the simulation result
    sim_result = store.get_simulation(run_id)

    # Run bottleneck analysis
    bottleneck = analyze_bottleneck(sim_result)

    return {
        "run_id": run_id,
        "primary_bottleneck": bottleneck.primary_bottleneck if hasattr(bottleneck, "primary_bottleneck") else "balanced",
        "severity_score": bottleneck.severity_score if hasattr(bottleneck, "severity_score") else 0.0,
        "utilization_breakdown": bottleneck.utilization_breakdown if hasattr(bottleneck, "utilization_breakdown") else {},
        "details": bottleneck.details if hasattr(bottleneck, "details") else [],
        "recommendations": bottleneck.recommendations if hasattr(bottleneck, "recommendations") else [],
    }


def chip_get_results(
    run_id: str | None = None,
    design_id: str | None = None,
    limit: int = 20,
) -> dict:
    """Retrieve simulation results.

    Gets specific run or lists recent runs.
    """
    # List simulations
    if run_id is not None:
        try:
            sim_result = store.get_simulation(run_id)
            return {
                "run_id": sim_result.run_id,
                "design_id": sim_result.design_id,
                "design_name": sim_result.design_name,
                "design_version": sim_result.design_version,
                "workload_name": sim_result.workload_name,
                "fidelity": sim_result.fidelity,
                "metrics": sim_result.metrics,
                "success": sim_result.success,
                "error_message": sim_result.error_message,
            }
        except ValueError:
            return {"error": f"Simulation not found: {run_id}"}

    # List recent results
    simulations = store.list_simulations(design_id=design_id, limit=limit)

    return {
        "results": simulations,
        "limit": limit,
        "design_id_filter": design_id,
    }


def chip_export_report(run_id: str, fmt: str = "markdown", output_path: str | None = None) -> dict:
    """Export a simulation report.

    Exports report via export_simulation_report.
    """
    # Get the simulation result
    sim_result = store.get_simulation(run_id)

    # Export report
    report_text = export_simulation_report(sim_result, fmt=fmt, output_path=output_path)

    return {
        "run_id": run_id,
        "format": fmt,
        "report": report_text,
        "output_path": output_path,
    }


# --- MCP Resources ---

def get_library_components() -> dict:
    """Get JSON of list_components()."""
    names = list_components()
    components = {}
    for name in names:
        comp = get_component_defaults(
            "memory" if name in ["SRAM", "DRAM", "HBM", "3D_MEMORY"] else "compute", name
        )
        if comp:
            components[name] = comp
        else:
            components[name] = COMPONENT_LIBRARY.get(name, {})

    return {"components": components, "count": len(components)}


def get_design_resource(design_id: str) -> dict:
    """Get JSON/YAML of design by ID."""
    try:
        design_id_num, design_spec = store.get_design(design_id)
        yaml_str = design_spec.to_yaml() if hasattr(design_spec, "to_yaml") else ""
        return {
            "design_id": design_id_num,
            "name": design_spec.name,
            "version": design_spec.version,
            "spec_yaml": yaml_str,
            "spec_json": json.dumps(design_spec.model_dump(), indent=2),
        }
    except ValueError:
        return {"error": f"Design not found: {design_id}"}


def get_result_resource(run_id: str) -> dict:
    """Get JSON of simulation result by run_id."""
    try:
        sim_result = store.get_simulation(run_id)
        return {
            "run_id": sim_result.run_id,
            "design_id": sim_result.design_id,
            "design_name": sim_result.design_name,
            "design_version": sim_result.design_version,
            "workload_name": sim_result.workload_name,
            "fidelity": sim_result.fidelity,
            "metrics": sim_result.metrics,
            "bottleneck": sim_result.bottleneck,
            "assumptions": sim_result.assumptions,
            "success": sim_result.success,
            "error_message": sim_result.error_message,
        }
    except ValueError:
        return {"error": f"Simulation not found: {run_id}"}


def get_models_assumptions() -> dict:
    """Get JSON describing fidelity levels, provenance categories, and default model assumptions."""
    from chiplab.schemas.common import Provenance

    fidelity_levels = {
        "analytical": {
            "name": "Analytical Fidelity",
            "description": "Level 1 analytical model: theoretical roofline and first-order estimates; not cycle-accurate.",
        },
        "discrete_event": {
            "name": "Discrete-Event Fidelity",
            "description": "Level 2 discrete-event simulation (SimPy): models request queueing, bank/channel contention, compute scheduling, and interconnect contention; not RTL/cycle-accurate.",
        },
        "specialized": {
            "name": "Specialized Fidelity",
            "description": "Level 3 specialized model: optimized cycle-accurate simulation for specific architectures.",
        },
        "rtl_circuit": {
            "name": "RTL Circuit Fidelity",
            "description": "Level 4 RTL circuit simulation: cycle-accurate verification at the RTL level.",
        },
    }

    provenance_categories = {
        "measured": {
            "name": "Measured",
            "description": "Values measured from physical implementations or validated benchmarks.",
        },
        "published": {
            "name": "Published",
            "description": "Values from published papers, datasheets, and industry specifications.",
        },
        "estimated": {
            "name": "Estimated",
            "description": "Values estimated based on similar components or architectural constraints.",
        },
        "assumed": {
            "name": "Assumed",
            "description": "Default values used when no other information is available.",
        },
    }

    default_assumptions = {
        "default_fidelity": "analytical",
        "default_seed": 42,
        "default_num_requests": 32,
        "allow_estimated_defaults": True,
    }

    return {
        "fidelity_levels": fidelity_levels,
        "provenance_categories": provenance_categories,
        "default_assumptions": default_assumptions,
    }


get_fidelity_levels = get_models_assumptions


# --- MCP Prompts ---

def explore_architecture(goal: str) -> dict:
    """Engineering workflow prompt guiding an agent through create -> validate -> simulate -> sweep -> compare -> iterate.

    Args:
        goal: The engineering goal or question to address.

    Returns:
        A structured prompt dict guiding the design workflow.
    """
    return {
        "goal": goal,
        "workflow": [
            "1. CREATE: Design the architecture using chip_create_design",
            "   - Provide YAML spec, name, and version",
            "   - Validate with chip_validate_design before proceeding",
            "",
            "2. VALIDATE: Check design integrity using chip_validate_design",
            "   - Review errors, warnings, and missing parameters",
            "   - Fix any issues before simulation",
            "",
            "3. SIMULATE: Run initial simulation using chip_run_simulation",
            "   - Choose fidelity level (analytical recommended for start)",
            "   - Review metrics, bottleneck, and assumptions",
            "",
            "4. SWEEP: Optimize parameters using chip_sweep_parameters",
            "   - Sweep key parameters (e.g., chiplet counts, frequencies)",
            "   - Use objective: throughput_tokens_per_sec or custom",
            "",
            "5. COMPARE: Compare design variants using chip_compare_designs",
            "   - Compare runs side-by-side",
            "   - Review metric_deltas and winner_by_metric",
            "",
            "6. ITERATE: Refine and repeat as needed",
            "   - Use comparison results to guide next design iteration",
            "   - Loop back to validation or simulation as needed",
        ],
        "steps_completed": [],
        "next_step": "create",
        "goal": goal,
    }



# --- Easy design, cost and price tools (work the same from any MCP client) ---

from chiplab import templates as _templates
from chiplab.cost import model as _cost_model
from chiplab.cost import prices as _prices
from chiplab.cost.reference import MEMORY_TECHS, NODES, PACKAGES

SERVER_INSTRUCTIONS = """ChipLab designs and tests chips: performance (roofline or discrete-event
simulation), power/thermal, and unit cost from TSMC wafer prices, yield, SRAM area, DRAM/HBM
$/GB, packaging (FCBGA, CoWoS-S/L, EMIB, SoIC), test, PCB and mask NRE.
Fastest path: chip_quickstart -> chip_design_from_template (or chip_create_design) ->
chip_evaluate. For broad design-space exploration without spending tokens per design, use
run_digital_twin_search (thousands of designs + Monte Carlo locally, compact summary back),
calibrate it with ingest_byo_pdk / calibrate_from_eda_report, and hand off the winner with
export_pareto_handoff_bundle. Every cost carries a source and date; built-in prices are estimates. If
chip_evaluate reports stale prices and you can search the web, run chip_price_lookup_plan,
search each query, then chip_set_price with the number, unit-matched, and the source URL."""


def _resolve_design(design_id_or_name: str | None, spec: str | dict | None,
                    version: str | None = None) -> tuple[str | None, DesignSpec]:
    if spec is not None:
        return None, _parse_spec(spec)
    if not design_id_or_name:
        raise ValueError("Pass design_id_or_name (a saved design) or spec (YAML/JSON/dict).")
    return store.get_design(design_id_or_name, version)


def _volume(design: DesignSpec, volume: int | None) -> int:
    return int(volume or design.metadata.get("suggested_volume") or 100_000)


def chip_quickstart() -> dict:
    """Start here. Explains the workflow, lists templates, TSMC nodes, memory types and
    package types, and shows the minimal spec format."""
    return {
        "workflow": [
            "1. chip_list_templates or chip_design_from_template(template, overrides) "
            "to get a saved design (or chip_create_design with your own spec).",
            "2. chip_evaluate(design_id_or_name, workload, volume) -> performance, power, "
            "unit cost range, $/TOPS, warnings (reticle limit, SRAM fit, HBM packaging).",
            "3. Change one thing (node, die area, SRAM, HBM stacks, package) with "
            "chip_design_from_template overrides or a new version and evaluate again; "
            "chip_compare_designs compares runs.",
            "4. Prices: chip_cost_references shows every number with source/date; "
            "chip_price_lookup_plan + web search + chip_set_price refreshes them.",
        ],
        "templates": _templates.list_templates(),
        "tsmc_nodes": list(NODES),
        "memory_technologies": list(MEMORY_TECHS),
        "package_types": list(PACKAGES) + ["(aliases: 2D, 2.5D, 3D, cowos, emib, organic)"],
        "minimal_spec_example": {
            "name": "my_chip",
            "package": {"type": "2.5D", "interposer": "cowos_s"},
            "chiplets": [
                {"id": "core", "type": "compute", "count": 1, "area_mm2": 400,
                 "process_nm": 3, "tdp_watts": 300,
                 "compute": {"architecture": "systolic", "precision": "FP8",
                             "peak_tops": 1000, "clock_ghz": 1.6, "sram_mb": 64}},
                {"id": "hbm", "type": "HBM3E", "count": 4,
                 "memory": {"technology": "HBM3E", "capacity_per_chiplet_gb": 24,
                            "channels": 16, "bandwidth_per_channel_gbps": 75}},
            ],
        },
        "workload_example": {"name": "70B dense decode", "type": "llm_decode",
                             "model_params_b": 70, "hidden_dim": 8192, "num_layers": 80,
                             "precision": "FP8", "batch_size": 8, "sequence_length": 4096,
                             "kv_cache_fraction": 0.125},
        "workload_types": "llm_decode, llm_prefill, moe_llm_decode (set num_experts, "
                          "active_experts), plus legacy transformer/moe_decode/ssm/cnn",
    }


def chip_list_templates() -> dict:
    """List starter chip designs with a one-line summary each."""
    return {"templates": _templates.list_templates()}


def chip_design_from_template(template: str, name: str | None = None,
                              overrides: dict | None = None) -> dict:
    """Create and save a design from a template, optionally changed by overrides.

    overrides deep-merges into the template. Chiplets merge by id: same id edits it, a new
    id adds it, {"id": "hbm", "remove": true} drops it. Example:
    {"chiplets": [{"id": "compute", "process_nm": 2, "compute": {"sram_mb": 128}},
                  {"id": "hbm", "count": 6}], "package": {"interposer": "cowos_s"}}
    """
    spec = _templates.build_from_template(template, overrides)
    if name:
        spec["name"] = name
    created = chip_create_design(spec)
    created["spec"] = spec
    return created


def chip_estimate_cost(design_id_or_name: str | None = None, spec: str | dict | None = None,
                       volume: int | None = None, board: dict | None = None,
                       version: str | None = None) -> dict:
    """Estimate unit cost (typical, optimistic, pessimistic) with a full breakdown:
    per-die TSMC wafer cost, dies/wafer, yield, SRAM area, DRAM/HBM $/GB, packaging,
    assembly yield, test, optional board (PCB/VRM/cooling) and mask NRE spread over volume
    (default: the design's metadata.suggested_volume, else 100,000 units).

    board example: {"area_cm2": 400, "layers": 14, "hdi": false, "high_speed": true}
    """
    _, design = _resolve_design(design_id_or_name, spec, version)
    return _cost_model.estimate_cost_range(design, volume=_volume(design, volume), board=board)


def chip_evaluate(design_id_or_name: str | None = None, spec: str | dict | None = None,
                  workload: dict | None = None, volume: int | None = None,
                  board: dict | None = None, fidelity: str = "analytical") -> dict:
    """One-call realistic check: validate, simulate performance/power/thermal, estimate cost,
    and report $/TOPS, TOPS/W, tokens/s per $1k, constraint checks and warnings.
    Pass a saved design name/id, or a spec to save and evaluate in one go."""
    if spec is not None:
        created = chip_create_design(spec)
        design_id_or_name = created["design_id"]
    _, design = _resolve_design(design_id_or_name, None)
    validation = validate_design(design)
    sim = chip_run_simulation(design_id_or_name, workload=workload, fidelity=fidelity)
    cost = _cost_model.estimate_cost_range(design, volume=_volume(design, volume), board=board)
    m = sim["metrics"]
    unit = cost["unit_cost_usd"] or 0.0
    peak = m.get("peak_compute_tops") or 0.0
    power = m.get("total_power_watts") or 0.0
    tps = (m.get("throughput_tokens_per_sec") or 0.0) if workload else 0.0
    if not workload:
        m = {**m, "throughput_tokens_per_sec": None, "estimated_latency_ms": None}
    checks = []
    c = design.constraints
    for label, limit, actual in (
        ("power_w", c.max_power_watts, power),
        ("temp_c", c.max_temp_c, m.get("estimated_temp_c")),
        ("unit_cost_usd", c.max_cost_usd, unit),
    ):
        if limit is not None and actual is not None:
            lim = float(getattr(limit, "value", limit))
            checks.append({"constraint": label, "limit": lim, "actual": round(actual, 2),
                           "pass": actual <= lim})
    return {
        "design_id": sim["design_id"], "design": design.name, "version": design.version,
        "run_id": sim["run_id"],
        "valid": validation.valid, "validation_errors": validation.errors,
        "performance": {k: m.get(k) for k in (
            "peak_compute_tops", "effective_compute_tops", "aggregate_memory_bandwidth_tbps",
            "total_memory_capacity_gb", "estimated_latency_ms", "throughput_tokens_per_sec",
            "utilization_pct", "arithmetic_intensity", "ridge_point_ops_per_byte")},
        "power_thermal": {"total_power_watts": power,
                          "estimated_temp_c": m.get("estimated_temp_c"),
                          "energy_per_inference_mj": m.get("energy_per_inference_mj")},
        "bottleneck": sim["bottleneck"],
        "cost": {"unit_cost_usd": unit, "volume": cost["volume"],
                 "range_usd": cost["range_usd"],
                 "breakdown_usd": cost["breakdown_usd"], "package_type": cost["package_type"],
                 "chiplets": cost["chiplets"]},
        "efficiency": {
            "usd_per_peak_tops": round(unit / peak, 3) if peak else None,
            "tops_per_watt": round(peak / power, 3) if power else None,
            "tokens_per_sec_per_1k_usd": round(tps / unit * 1000, 2) if unit and tps else None,
        },
        "constraint_checks": checks,
        "warnings": validation.warnings + cost["warnings"] + sim.get("warnings", []),
        "notes": cost["notes"] + ([] if workload else [
            "No workload given: latency and tokens/s omitted. Pass e.g. "
            "{'type': 'llm_decode', 'model_params_b': 70, 'batch_size': 8, ...}."]),
        "price_data": cost["price_data"],
        "fidelity_disclaimer": sim["fidelity_disclaimer"],
    }


def chip_cost_references(prefix: str = "") -> dict:
    """Show the price/process table (value, low-high range, unit, source, date, stale flag).
    prefix filters keys, e.g. 'wafer.', 'memory.', 'package.cowos', 'pcb.', 'sram_bitcell.'."""
    entries = _prices.list_entries(prefix)
    return {"count": len(entries), "price_file": str(_prices.price_file()),
            "stale_after_days": _prices.STALE_DAYS, "entries": entries}


def chip_set_price(key: str, value: float, source: str, as_of: str | None = None,
                   low: float | None = None, high: float | None = None,
                   unit: str | None = None) -> dict:
    """Store a live/looked-up price that overrides the built-in estimate. Use the key's
    unit from chip_cost_references (e.g. memory.HBM3E is USD per GB, wafer.tsmc.N3 is USD per
    300mm wafer). source must be a URL or citation; as_of is YYYY-MM-DD (default today)."""
    return {"key": key, "entry": _prices.set_override(key, value, source, as_of, low, high,
                                                       unit)}


def chip_clear_price(key: str) -> dict:
    """Remove an override so the built-in estimate is used again."""
    return {"key": key, "removed": _prices.clear_override(key)}


def chip_price_lookup_plan(prefix: str = "", only_stale: bool = True) -> dict:
    """List web searches to refresh prices (wafer per node, DRAM/HBM $/GB, packaging, PCB).
    Run each search, then call chip_set_price(key, value, source_url)."""
    plan = _prices.lookup_plan(prefix, only_stale)
    return {"count": len(plan), "items": plan,
            "how": "Search each query; convert to the listed unit; call chip_set_price with "
                   "the source URL and the date the source gives."}


def chip_refresh_prices(feed_url: str | None = None) -> dict:
    """Pull prices from a JSON price feed (https) set in CHIPLAB_PRICE_FEED_URL or passed in.
    Feed format: {"memory.HBM3E": {"value": 22, "source": "...", "as_of": "2026-10-01"}}."""
    return _prices.refresh_from_feed(feed_url)


def search(query: str) -> dict:
    """Search ChipLab designs, templates and price keys (ChatGPT connector compatible)."""
    q = query.lower().strip()
    results = []
    for d in store.list_designs():
        text = f"{d['name']} {d.get('description') or ''}".lower()
        if not q or any(w in text for w in q.split()):
            results.append({"id": f"design:{d['design_id']}",
                            "title": f"Design {d['name']} v{d['version']}",
                            "url": f"chiplab://designs/{d['design_id']}"})
    for name, summary in _templates.list_templates().items():
        if not q or any(w in f"{name} {summary}".lower() for w in q.split()):
            results.append({"id": f"template:{name}", "title": f"Template {name}",
                            "url": f"chiplab://templates/{name}"})
    for key in _prices.list_entries():
        if q and any(w in key.lower() for w in q.split()):
            results.append({"id": f"price:{key}", "title": f"Price {key}",
                            "url": f"chiplab://prices/{key}"})
    return {"results": results[:50]}


def fetch(id: str) -> dict:
    """Fetch a search result by id: design:<id>, template:<name> or price:<key>."""
    kind, _, ref = id.partition(":")
    if kind == "design":
        data = chip_get_design(ref)
        title, text = f"Design {data['name']} v{data['version']}", data["spec_yaml"]
    elif kind == "template":
        data = _templates.TEMPLATES[ref]
        title, text = f"Template {ref}", json.dumps(data, indent=2)
    elif kind == "price":
        data = _prices.get(ref)
        title, text = f"Price {ref}", json.dumps(data, indent=2)
    else:
        raise ValueError("id must start with design:, template: or price:")
    return {"id": id, "title": title, "text": text, "url": f"chiplab://{kind}s/{ref}",
            "metadata": {"kind": kind}}



# --- Zero-token digital twin, BYO-PDK calibration, IP catalog, handoff ---------------------

_TWIN_KEY_METRICS = ("slack_ps", "fmax_ghz", "temp_c", "perf_per_watt", "area_mm2", "yield",
                     "refresh_overhead", "mem_conflict_rate", "pim_gmacs", "mem_power_w")


def _twin_compact(summary: dict) -> dict:
    """Token-lean view of a twin summary (full data stays in the ExperimentStore)."""
    out = dict(summary)
    out["top"] = [{
        "rank": p["rank"], "params": p["params"], "objectives": p["objectives"],
        "key_metrics": {k: p["metrics"][k] for k in _TWIN_KEY_METRICS if k in p["metrics"]},
        "feasible": p["feasible"], "robustness": p.get("robustness")} for p in summary["top"]]
    out["sensitivity"] = [{"parameter": e["parameter"], "provenance": e["provenance"],
                           "score": e["score"],
                           "most_affects": max(e["impact"], key=e["impact"].get)}
                          for e in summary["sensitivity"]]
    return out


def run_digital_twin_search(config: dict | None = None, calibration_id: str | None = None,
                            blackboxes: list | None = None, strategy: str | None = None,
                            population: int | None = None, generations: int | None = None,
                            mc_trials: int | None = None, top_k: int | None = None,
                            seed: int | None = None, fixed: dict | None = None,
                            node_set: str | None = None, memory: dict | None = None) -> dict:
    """Run the zero-token digital twin: NSGA-II / Latin hypercube / GP-guided search over a
    RISC-V SoC space (cores, VLEN/DLEN, L1/L2/LLC, Vdd/freq, node, UCIe lanes, utilization),
    then Monte Carlo over every 'estimated'/'assumed' parameter (+/-15-25%), with technology
    factors widened to each node's PDK accuracy margin (5% sky130/GF180/IHP ... 25% 2nm).
    Returns top-K Pareto designs with robustness (slack yield, thermal- and refresh-violation
    probability, robustness score) and a sensitivity ranking; the full trace is saved.

    node_set: 'tsmc' (N16/N7/N5/N3, default) or 'ladder' (GF180 ... 2nm PDK ladder, Vdd as a
    ratio of each node's nominal, 50 MHz-3.5 GHz). memory: enables the on-chip memory model
    and its knobs (SRAM vs 2T/3T gain-cell eDRAM, banks, interleaving, PIM lanes), e.g.
    {"capacity_kib": 2048, "requests_per_cycle": 2.0}; objectives may then use pim_gmacs.

    config: optional DigitalTwinConfig dict (search_space, objectives, constraints, workload,
    priors). Shortcut args override it. blackboxes: preset names (see list_ip_catalog) or
    BlackBoxMacro dicts. calibration_id: from ingest_byo_pdk / calibrate_from_eda_report;
    calibrated factors become 'measured' and are no longer perturbed."""
    from chiplab.engine.digital_twin import run_digital_twin
    from chiplab.library.ip_catalog import get_blackbox
    from chiplab.schemas.digital_twin import DigitalTwinConfig, default_search_space

    data = dict(config or {})
    if memory is not None:
        data["memory"] = memory
    if node_set is not None or (memory is not None and "search_space" not in data):
        if node_set not in (None, "tsmc", "ladder"):
            raise ValueError("node_set must be 'tsmc' or 'ladder'")
        data.setdefault("search_space", [sp.model_dump() for sp in default_search_space(
            node_set or "tsmc", memory=data.get("memory") is not None)])
    for k, v in (("strategy", strategy), ("population", population),
                 ("generations", generations), ("mc_trials", mc_trials), ("top_k", top_k),
                 ("seed", seed), ("fixed", fixed)):
        if v is not None:
            data[k] = v
    if blackboxes:
        data["blackboxes"] = [get_blackbox(b).model_dump() for b in blackboxes]
    if calibration_id:
        rep = store.get_calibration(calibration_id)
        data["calibration"] = {**{f["name"]: f["value"] for f in rep["factors"]},
                               **(data.get("calibration") or {})}
    cfg = DigitalTwinConfig.model_validate(data)
    summary, trace = run_digital_twin(cfg)
    dumped = summary.model_dump(mode="json")
    store.save_twin_run(dumped, trace)
    return _twin_compact(dumped)


def get_digital_twin_run(run_id: str, include_rows: int = 0) -> dict:
    """Fetch a saved twin run summary; include_rows > 0 adds that many trace rows."""
    data = store.get_twin_run(run_id, with_trace=include_rows > 0)
    out = {"summary": _twin_compact(data["summary"]), "created_at": data["created_at"]}
    if include_rows > 0:
        out["trace_rows"] = data["trace"]["rows"][: min(include_rows, 500)]
        out["trace_total"] = len(data["trace"]["rows"])
    return out


def _eda_text(content: str | None, path: str | None,
              private: bool = False) -> tuple[str | None, dict | None]:
    from chiplab.calibration.pdk_ingestor import read_input

    if content is None and not path:
        return None, None
    try:
        return read_input(content, path)
    except (PermissionError, FileNotFoundError, ValueError) as exc:
        if not private:
            raise
        # NDA mode: never echo a foundry path or filename back to the client or the log.
        msg = {PermissionError: "input path is outside CHIPLAB_DATA_DIRS",
               FileNotFoundError: "input file not found"}.get(type(exc), "input rejected "
                                                                         "(type or size)")
        raise type(exc)(f"{msg} (path withheld: private/NDA mode)") from None


def _nda_parse(fn: Any, text: str, kind: str, private: bool) -> Any:
    try:
        return fn(text)
    except Exception as exc:
        if not private:
            raise
        raise ValueError(f"{kind} parse failed ({type(exc).__name__}); details withheld in "
                         "private/NDA mode") from None


def calibrate_from_eda_report(node: str, timing_report: str | None = None,
                              timing_report_path: str | None = None, spef: str | None = None,
                              spef_path: str | None = None, liberty: str | None = None,
                              liberty_path: str | None = None,
                              design_point: dict | None = None, private: bool = True) -> dict:
    """Back-annotate the twin from real EDA results. Parses an OpenSTA/PrimeTime timing
    report (critical path cell vs wire delay, slack, clock period), optional SPEF and Liberty,
    compares with the model at the same design point, and stores calibration factors
    (gate_delay, wire_rc, leakage) as 'measured'. Pass text or a path under
    CHIPLAB_DATA_DIRS. Use the returned calibration_id in run_digital_twin_search.
    private (default True, NDA mode): file paths, library/cell names and timing-path
    instance names are redacted from the response, the stored record and the activity log."""
    from chiplab.calibration import pdk_ingestor as pi
    from chiplab.library import private_pdk as pp

    sources, timing = [], None
    text, src = _eda_text(timing_report, timing_report_path, private)
    if text is not None:
        timing = _nda_parse(pi.parse_timing_report, text, "timing report", private)
        sources.append({**src, "type": "timing_report"})
    spef_d = lib_d = None
    text, src = _eda_text(spef, spef_path, private)
    if text is not None:
        spef_d = _nda_parse(pi.parse_spef, text, "SPEF", private)
        sources.append({**src, "type": "spef"})
    text, src = _eda_text(liberty, liberty_path, private)
    if text is not None:
        lib_d = _nda_parse(pi.parse_liberty, text, "Liberty", private)
        sources.append({**src, "type": "liberty"})
    if not sources:
        raise ValueError("Provide at least a timing report, SPEF or Liberty input.")
    rep = pi.calibrate(node, design_point, lib_d, None, timing, spef_d, sources)
    out = rep.model_dump(mode="json")
    paths = [{k: p[k] for k in ("startpoint", "endpoint", "slack_ps", "status")}
             for p in (timing or {}).get("paths", [])[:10]]
    if private:
        secrets = pp.collect_secrets(lib_d, None, sources)
        for p in (timing or {}).get("paths", []):
            secrets |= {str(p.get("startpoint", "")), str(p.get("endpoint", ""))}
        secrets = {x for x in secrets if len(x) >= 3}
        out["sources"] = pp.redact_sources(sources)
        out = pp.scrub(out, secrets)
        paths = [{**p, "startpoint": f"path_{i + 1}_start", "endpoint": f"path_{i + 1}_end"}
                 for i, p in enumerate(paths)]
        out["nda_redacted"] = True
    store.save_calibration(out)
    if timing:
        out["timing_paths"] = paths
    return out


def ingest_byo_pdk(node: str, liberty: str | None = None, liberty_path: str | None = None,
                   lef: str | None = None, lef_path: str | None = None,
                   keepout_um: float = 10.0, private: bool = True,
                   label: str | None = None) -> dict:
    """Bring your own PDK: parse Liberty (.lib: units, FO4 from the smallest inverter's delay
    table, leakage density, pin caps) and LEF (routing layers, pitches, site height, macros).
    Derives gate_delay and leakage calibration factors vs the twin's node model, stores them,
    and turns LEF BLOCK macros into black-box candidates. Nothing leaves your machine.

    private (default True, NDA mode): a compiled numeric profile is saved under the
    git-ignored .chiplab_private_pdks/ with an opaque pdk_<hash> id; library, cell, macro,
    pin and layer names and file paths are redacted from the response, the stored
    calibration and the activity log (macros become macro_1, macro_2, ...). Use
    private=False for open PDKs (sky130, GF180, IHP, ASAP7) to keep names."""
    from chiplab.calibration import pdk_ingestor as pi
    from chiplab.library import private_pdk as pp

    sources, lib_d, lef_d = [], None, None
    text, src = _eda_text(liberty, liberty_path, private)
    if text is not None:
        lib_d = _nda_parse(pi.parse_liberty, text, "Liberty", private)
        sources.append({**src, "type": "liberty"})
    text, src = _eda_text(lef, lef_path, private)
    if text is not None:
        lef_d = _nda_parse(pi.parse_lef, text, "LEF", private)
        sources.append({**src, "type": "lef"})
    if not sources:
        raise ValueError("Provide Liberty and/or LEF input.")
    rep = pi.calibrate(node, None, lib_d, lef_d, None, None, sources)
    report = rep.model_dump(mode="json")
    blackbox_candidates = []
    if lef_d:
        for m in lef_d["macros"]:
            if str(m.get("class", "")).startswith("BLOCK") and m.get("width_um"):
                bb = pi.lef_macro_to_blackbox(m, keepout_um)
                blackbox_candidates.append(bb.model_dump(mode="json"))
    if private:
        secrets = pp.collect_secrets(lib_d, lef_d, sources)
        for i, bb in enumerate(blackbox_candidates):
            bb["name"] = f"macro_{i + 1}"
            bb["datasheet"] = "private PDK (redacted)"
            for j, pin in enumerate(bb.get("pins") or []):
                pin["name"] = f"pin_{j + 1}"
                pin["clock"] = "clk" if pin.get("clock") else pin.get("clock")
            for j, rail in enumerate(bb.get("rails") or []):
                rail["name"] = f"rail_{j + 1}"
        report["sources"] = pp.redact_sources(sources)
        report = pp.scrub(report, secrets)
        report["summary"]["blackbox_candidates"] = [b["name"] for b in blackbox_candidates]
        report["nda_redacted"] = True
        store.save_calibration(report)
        prof = pp.save_profile(node, lib_d, lef_d, report, sources, label)
        return pp.scrub({**report, "pdk_id": prof["pdk_id"],
                         "private_store": "git-ignored .chiplab_private_pdks/ (local only)",
                         "blackbox_candidates": blackbox_candidates[:20],
                         "liberty": prof["liberty"], "lef_summary": prof["lef"]}, secrets)
    report["summary"]["blackbox_candidates"] = [b["name"] for b in blackbox_candidates]
    store.save_calibration(report)
    return {**report, "blackbox_candidates": blackbox_candidates[:20],
            "liberty": {k: v for k, v in (lib_d or {}).items() if k != "cell_names"},
            "lef_summary": (lef_d or {}).get("summary")}


def design_onchip_memory_subsystem(config: dict | None = None, workload: dict | None = None,
                                   simulate_cycles: int = 20_000, generate_rtl: bool = False,
                                   output_name: str | None = None,
                                   intended_use: str = "research", ip: list | None = None,
                                   seed: int = 1, trace_cycles: int = 0) -> dict:
    """Design an on-chip memory subsystem: multi-bank 6T SRAM or 2T/3T gain-cell eDRAM on any
    PDK-ladder node (gf180, sky130, ihp_sg13g2, n65, n28, freepdk15, asap7, n3, n2).

    Returns the analytical model (area, fmax/slack, retention, refresh overhead and stall %,
    bank-conflict rate, port contention, SECDED overhead, pJ/bit, near-bank PIM GMAC/s) and a
    deterministic cycle-accurate cross-check (simulate_cycles, max 200k; 0 skips it).
    generate_rtl=True writes SystemVerilog-2012 (onchip_mem_subsystem_top.sv with crossbar,
    interleaver, RR/priority arbiter, auto-refresh, SECDED, PIM MAC; behavioral bitcell macro
    with retention assertions; self-checking Verilator testbench; SDC) under
    CHIPLAB_EXPORT_DIR. Licence / tape-out readiness warnings follow intended_use
    ('research' or 'commercial') and ip (catalog names).

    config: OnChipMemoryConfig fields, e.g. {"technology": "gc2t", "pdk": "sky130",
    "capacity_kib": 64, "banks": 8, "word_bits": 64, "ecc": "secded", "interleave": "xor",
    "num_requesters": 8, "pim": {"enabled": true, "lanes": 8}}. workload: MemWorkload
    fields (requests_per_cycle, read_fraction, pattern random/sequential/strided, stride,
    pim_fraction)."""
    from chiplab.library.pdk_library import commercial_readiness
    from chiplab.memory_gen.rtl_generator import write_bundle
    from chiplab.models.onchip_memory import analyze, simulate
    from chiplab.schemas.digital_twin import MemWorkload, OnChipMemoryConfig

    cfg = OnChipMemoryConfig.model_validate(config or {})
    wl = MemWorkload.model_validate(workload or {})
    report = analyze(cfg, wl)
    out: dict = {"design": cfg.name, **report}
    if simulate_cycles > 0:
        sim = simulate(cfg, wl, cycles=min(int(simulate_cycles), 200_000), seed=seed,
                       freq_mhz=report["freq_mhz"], trace_cycles=min(int(trace_cycles), 400))
        m = report["metrics"]
        out["simulation"] = {k: v for k, v in sim.items() if k != "timeline"}
        out["cross_check"] = {
            "conflict_rate": {"analytical": m["conflict_rate"], "simulated":
                              sim["conflict_rate"]},
            "throughput_per_cycle": {"analytical": m["served_per_cycle"],
                                     "simulated": sim["throughput_per_cycle"]},
            "refresh_overhead": {"analytical": m["refresh_overhead"],
                                 "simulated": sim["refresh_overhead"]},
            "note": "Closed-loop binomial model vs cycle simulation; expect agreement within "
                    "~0.06 absolute on conflict rate and ~12% on throughput up to saturation."}
    if generate_rtl:
        b = write_bundle(cfg, wl, output_name, seed)
        out["rtl"] = {"directory": b["directory"], "files": b["files"], "lint": b["lint"],
                      "params": {k: v for k, v in b["params"].items()
                                 if k in ("NUM_BANKS", "DATA_W", "CODE_W", "ECC_EN",
                                          "REFRESH_INTERVAL", "RETENTION_CYCLES", "PIM_EN",
                                          "PIM_LANES", "ACC_W")},
                      "verify": "bash run_verilator.sh (Verilator 5); not executed by ChipLab"}
    extra = commercial_readiness(cfg.pdk, intended_use, list(ip or []) + ["openram"])
    out["warnings"] = list(dict.fromkeys(
        [w for w in report["warnings"] if not w.startswith(f"PDK '{cfg.pdk}'")] + extra))
    return out


def list_pdk_ladder() -> dict:
    """The 180nm-to-2nm PDK ladder: per node Vdd, FO4, gate density, 6T bitcell, energy and
    leakage per gate, M1 pitch, wire cap, gain-cell retention, tape-out readiness, accuracy
    margin (used by Monte Carlo), licence and commercial-use flag, each with provenance."""
    from chiplab.library.pdk_library import ladder_table

    return {"nodes": ladder_table(),
            "doc": "docs/sources_and_accuracy.md (python -m chiplab.library.pdk_library)"}


def list_private_pdks() -> dict:
    """List compiled private (NDA) PDK profiles saved by ingest_byo_pdk: opaque pdk_id, node,
    optional label and calibration factors only (no names, paths or raw tables)."""
    from chiplab.library import private_pdk as pp

    return {"profiles": pp.list_profiles()}


def export_pareto_handoff_bundle(run_id: str, rank: int = 1, output_name: str | None = None,
                                 blackboxes: list | None = None,
                                 lef_paths: list[str] | None = None,
                                 lib_paths: list[str] | None = None) -> dict:
    """Write a pre-signoff bundle for a Pareto winner of a twin run: SystemVerilog wrapper
    (stub tiles + black-box macros), SDC, UPF, OpenROAD and Synopsys DC TCL, Verilator lint
    and Yosys equivalence scripts, README and a sha256 manifest. Files go under
    CHIPLAB_EXPORT_DIR (default ./chiplab_exports). Scripts are generated, never executed."""
    from chiplab.backends.handoff import build_bundle
    from chiplab.library.ip_catalog import get_blackbox
    from chiplab.schemas.digital_twin import BlackBoxMacro

    data = store.get_twin_run(run_id, with_trace=True)
    top = data["summary"]["top"]
    match = [p for p in top if p["rank"] == rank]
    if not match:
        raise ValueError(f"Run {run_id} has ranks 1..{len(top)}")
    if blackboxes is None:
        bbs = [BlackBoxMacro.model_validate(b)
               for b in data["trace"]["config"].get("blackboxes", [])]
    else:
        bbs = [get_blackbox(b) for b in blackboxes]
    res = build_bundle(match[0], run_id, rank, bbs, output_name, lef_paths, lib_paths)
    return {k: res[k] for k in ("directory", "files", "structural_lint")}


def list_ip_catalog(kind: str | None = None) -> dict:
    """Open-source silicon-proven IP presets (CVA6, Ara, Rocket, XiangShan, Ibex, OpenRAM,
    BaseJump STL, PULP AXI, TileLink, UCIe adapter status) and black-box macro examples.
    kind filters: core, vector_unit, sram_compiler, library, interconnect."""
    from chiplab.library.ip_catalog import BLACKBOX_EXAMPLES, list_ip

    return {"ip": list_ip(kind),
            "blackbox_examples": {k: v.model_dump(mode="json")
                                  for k, v in BLACKBOX_EXAMPLES.items()}}


# --- Main ---

def create_mcp_server() -> MCPServer:
    """Create and configure the ChipLab MCPServer instance."""
    from mcp.types import ToolAnnotations

    from chiplab.activity import ActivityLog, ActivityMiddleware

    mcp = MCPServer("chiplab", instructions=SERVER_INSTRUCTIONS,
                    middleware=[ActivityMiddleware(ActivityLog(DB_PATH))])
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    writes = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    for fn in (chip_quickstart, chip_list_templates, chip_get_design, chip_validate_design,
               chip_estimate_cost, chip_cost_references, chip_price_lookup_plan,
               chip_analyze_bottleneck, chip_get_results, search, fetch,
               get_digital_twin_run, list_ip_catalog, list_pdk_ladder, list_private_pdks):
        mcp.tool(annotations=read_only)(fn)
    for fn in (chip_design_from_template, chip_create_design, chip_evaluate,
               chip_run_simulation, chip_sweep_parameters, chip_compare_designs,
               chip_export_report, chip_set_price, chip_clear_price,
               run_digital_twin_search, calibrate_from_eda_report, ingest_byo_pdk,
               export_pareto_handoff_bundle, design_onchip_memory_subsystem):
        mcp.tool(annotations=writes)(fn)
    mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))(
        chip_refresh_prices)

    # Register resources
    mcp.resource("chiplab://library/components")(get_library_components)
    mcp.resource("chiplab://designs/{design_id}")(get_design_resource)
    mcp.resource("chiplab://results/{run_id}")(get_result_resource)
    mcp.resource("chiplab://models/assumptions")(get_models_assumptions)

    # Register prompts
    mcp.prompt("explore_architecture")(explore_architecture)

    return mcp


def main(argv: list[str] | None = None) -> None:
    """Entry point. stdio by default (Claude Code, Codex CLI, Gemini CLI); HTTP for web
    clients such as ChatGPT: chiplab-mcp --transport streamable-http --port 8000."""
    import argparse

    ap = argparse.ArgumentParser(prog="chiplab-mcp", description=main.__doc__)
    ap.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--allowed-host", action="append", default=[],
                    help="Public hostname allowed to reach the HTTP server (e.g. your tunnel "
                         "domain). Repeatable. Without it only localhost Host headers pass.")
    args = ap.parse_args(argv)

    mcp = create_mcp_server()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"] + args.allowed_host
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts,
        allowed_origins=["https://chatgpt.com", "https://chat.openai.com"]
        + [f"https://{h}" for h in args.allowed_host])
    mcp.run(transport=args.transport, host=args.host, port=args.port,
            transport_security=security)


if __name__ == "__main__":
    main()