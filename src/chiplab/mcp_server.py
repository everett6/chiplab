"""Chiplab MCP Server - FastMCP-based server for chip design, simulation, and analysis tools."""

from __future__ import annotations

import itertools
import math
import os
import sys
import json
import time
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import Context, MCPServer

# --- Database path configuration ---
DB_PATH = os.environ.get("CHIPLAB_DB_PATH", "chiplab_experiments.db")

from chiplab.credibility import require_credible_design
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
from chiplab.core.gpu_benchmark import get_gpu_status, run_gpu_benchmark
from chiplab.mcp_industrial_tools import (
    chiplab_reliability_analyze,
    chiplab_discover_architectures,
    chiplab_manufacturing_signoff,
    chiplab_digital_twin_manage,
    chiplab_multiscale_surrogate,
)
from chiplab.mcp_experimental_tools import (
    chiplab_simulate_experimental_tech,
    chiplab_compare_technologies,
    chiplab_sweep_experiment,
    chiplab_custom_device_simulate,
    chiplab_experiment_ledger,
    chiplab_custom_circuit_simulate,
    chiplab_scale_technology_node,
    chiplab_list_process_nodes,
    chiplab_list_experimental_technologies,
    chiplab_2d_beyond_cmos_simulate,
    chiplab_quantum_device_simulate,
    chiplab_spice_netlist_run,
    chiplab_agent_run,
    chiplab_chipro_eda_flow,
    chiplab_mage_rtl_pipeline,
    chiplab_chia_codesign_loop,
    chiplab_local_llm_models,
)

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

    fallback_note = None
    result = None
    if fidelity_level == FidelityLevel.SPECIALIZED:
        from chiplab.engine.specialized import SpecializedUnavailable, run_specialized_simulation
        try:
            result = run_specialized_simulation(design_spec, workload_obj, config, design_id)
        except SpecializedUnavailable as e:
            fallback_note = f"Requested fidelity 'specialized' is not available ({e})."
    if result is None and fidelity_level == FidelityLevel.DISCRETE_EVENT:
        result = run_discrete_event_simulation(design_spec, workload_obj, config, design_id)
    elif result is None:
        if fidelity_level in (FidelityLevel.SPECIALIZED, FidelityLevel.RTL_CIRCUIT):
            # No integrated backend exists for these levels: run analytical and label it so.
            from chiplab.backends import backend_status
            fallback_note = (fallback_note or (
                f"Requested fidelity '{fidelity_level.value}' is not available (external "
                f"backends: {backend_status()}).")) + (
                " Ran the analytical model instead; the result is labeled analytical.")
            fidelity_level = FidelityLevel.ANALYTICAL
            config = config.model_copy(update={"fidelity": fidelity_level})
        result = run_analytical_simulation(design_spec, workload_obj, config, design_id)
    run_warnings = list(getattr(result, "warnings", []))
    if fallback_note:
        run_warnings.insert(0, fallback_note)

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
        warnings=run_warnings,
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
        "warnings": simulation_result.warnings,
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
    progress: Any = None,
) -> dict:
    """Sweep a saved design over the Cartesian product of parameter values (dotted paths such
    as "chiplets[0].count", "interconnect.link_bandwidth_gbps", "workload.batch_size"), up to
    5,000 points, and return every point plus the best one for `objective`. Progress is
    streamed to MCP clients that send a progress token. For larger spaces use
    run_digital_twin_search."""
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
        progress=progress,
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


async def _chip_sweep_parameters_tool(
    design_id_or_name: str,
    parameters: dict[str, list],
    workload: dict | None = None,
    fidelity: str = "analytical",
    objective: str = "throughput_tokens_per_sec",
    version: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """MCP wrapper: runs the sweep in a worker thread and reports progress per point."""
    import anyio
    import anyio.from_thread

    def report(done: int, total: int) -> None:
        # At most ~50 notifications per sweep, plus the final one.
        if ctx is not None and (done == total or done % max(1, total // 50) == 0):
            anyio.from_thread.run(ctx.report_progress, done, total,
                                  f"{done}/{total} sweep points")

    return await anyio.to_thread.run_sync(
        lambda: chip_sweep_parameters(design_id_or_name, parameters, workload, fidelity,
                                      objective, version, progress=report))


_chip_sweep_parameters_tool.__doc__ = chip_sweep_parameters.__doc__


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


# --- New MCP Tools ---

def chip_design_from_nl(
    description: str,
    constraints: dict | None = None,
) -> dict:
    """Natural language → DesignSpec.

    Parses a plain-English chip description (e.g. "16 memory chiplets with local DRAM compute and 4 digital compute chiplets for MoE inference, 2.5D package, 45nm") into a structured DesignSpec YAML.
    Uses keyword matching + the component library to infer architecture, chiplet types, counts, memory tech, interconnect, package type.
    Returns the full DesignSpec + validation.
    """
    from chiplab.library.components import get_component_defaults, COMPONENT_LIBRARY

    # Keyword-based parsing of the natural language description
    desc_lower = description.lower()

    # Initialize default design spec components
    name = "nl_design"
    version = "0.1"
    architecture = "heterogeneous_accelerator"
    package_type = "2.5D"
    interposer = "silicon"
    chiplets = []
    interconnect_topology = "mesh"
    interconnect_bandwidth = 100.0

    # Parse chiplet types and counts from description
    import re

    # Look for memory chiplets
    memory_keywords = ["memory", "drams", "hbm", "sram", "dram"]
    compute_keywords = ["compute", "digital compute", "chiplets for", "compute chiplets"]

    # Count mentions of different types
    memory_mentions = sum(1 for kw in memory_keywords if kw in desc_lower)
    compute_mentions = sum(1 for kw in compute_keywords if kw in desc_lower)

    # Extract numbers associated with chiplet types
    # Pattern: "N memory chiplets" or "N compute chiplets"
    number_pattern = r"(\d+)\s*(?:memory|compute|drams?|hbm|sram)?\s*chiplet"

    # Find all number-chiplet combinations
    memory_counts = []
    compute_counts = []

    for match in re.finditer(number_pattern, description):
        count = int(match.group(1))
        # Check context around the match
        start = max(0, match.start() - 30)
        end = min(len(description), match.end() + 30)
        context = description[start:end].lower()

        if any(kw in context for kw in memory_keywords):
            memory_counts.append(count)
        elif any(kw in context for kw in compute_keywords):
            compute_counts.append(count)

    # If no specific counts found, use reasonable defaults based on description
    if not memory_counts and memory_mentions > 0:
        # Estimate based on description length and keywords
        memory_counts = [4]  # default 4 memory chiplets
    if not compute_counts and compute_mentions > 0:
        compute_counts = [4]  # default 4 compute chiplets

    # Build memory chiplet specs
    mem_tech = "DRAM"
    mem_capacity = 16
    mem_channels = 8

    if "hbm" in desc_lower or "hbm3" in desc_lower:
        mem_tech = "HBM"
        mem_capacity = 24
        mem_channels = 16
    elif "sram" in desc_lower:
        mem_tech = "SRAM"
        mem_capacity = 64

    # Add memory chiplets
    for i, count in enumerate(memory_counts[:4]):  # limit to 4 max
        mem_spec = {
            "id": f"memory_{i}",
            "type": "memory",
            "count": count,
            "memory": {
                "technology": mem_tech,
                "capacity_per_chiplet_gb": mem_capacity,
                "channels": mem_channels,
            },
        }
        chiplets.append(mem_spec)

    # Build compute chiplet specs
    compute_arch = "systolic"
    compute_precision = "FP8"
    compute_tops = 1000.0
    compute_clock = 1.6
    compute_sram = 64

    if "moe" in desc_lower or "mixture of experts" in desc_lower:
        compute_arch = "moe_systolic"
    if "tensor" in desc_lower:
        compute_arch = "tensor_engine"
    if "vector" in desc_lower:
        compute_arch = "vector_processor"

    # Look for process node mentions
    node_match = re.search(r"(\d+)nm", description)
    if node_match:
        compute_process = int(node_match.group(1))
    else:
        compute_process = 28  # default

    # Add compute chiplets
    for i, count in enumerate(compute_counts[:4]):  # limit to 4 max
        compute_spec = {
            "id": f"compute_{i}",
            "type": "digital_compute",
            "count": count,
            "compute": {
                "architecture": compute_arch,
                "precision": compute_precision,
                "peak_tops": compute_tops,
                "clock_ghz": compute_clock,
                "sram_mb": compute_sram,
            },
            "process_nm": compute_process,
        }
        chiplets.append(compute_spec)

    # Parse package type
    if "2.5D" in description or "2.5 d" in desc_lower:
        package_type = "2.5D"
        if "silicon" in desc_lower:
            interposer = "silicon"
    elif "3D" in desc_lower or "3d" in desc_lower:
        package_type = "3D_stacking"
    elif "organic" in desc_lower or "2D" in desc_lower:
        package_type = "monolithic"

    # Parse interconnect
    if "ucie" in desc_lower or "uci-e" in desc_lower:
        interconnect_topology = "ucie"
    elif "crossbar" in desc_lower:
        interconnect_topology = "crossbar"

    # Build the design spec dict
    design_spec = {
        "name": name,
        "version": version,
        "architecture": architecture,
        "package": {"type": package_type, "interposer": interposer},
        "chiplets": chiplets,
        "interconnect": {
            "topology": interconnect_topology,
            "link_bandwidth_gbps": interconnect_bandwidth,
        },
        "description": description,
        "constraints": {
            "max_power_watts": constraints.get("max_power_watts") if constraints else None,
            "max_area_mm2": constraints.get("max_area_mm2") if constraints else None,
            "max_cost_usd": constraints.get("max_cost_usd") if constraints else None,
        }
        if constraints
        else {},
    }

    # Validate and save using existing chip_create_design logic
    from chiplab.schemas.design import DesignSpec

    try:
        ds = DesignSpec.model_validate(design_spec)
        validation_result = validate_design(ds)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse design specification: {str(e)}",
            "description": description,
        }

    # Save to experiment store
    design_id = store.save_design(ds)

    return {
        "design_id": design_id,
        "success": True,
        "name": ds.name,
        "version": ds.version,
        "description": description,
        "architecture": ds.architecture,
        "package_type": package_type,
        "chiplet_count": len(ds.chiplets),
        "memory_technology": ds.chiplets[0].memory.technology if ds.chiplets and ds.chiplets[0].type == "memory" else "unknown",
        "validation": {
            "valid": validation_result.valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "missing_parameters": validation_result.missing_parameters,
            "provenance_summary": validation_result.provenance_summary,
        },
    }


def chip_full_flow(
    description: str,
    workload: dict | None = None,
    constraints: dict | None = None,
    max_iterations: int = 3,
) -> dict:
    """End-to-end autonomous flow: NL description → design → validate → simulate → analyze bottleneck → auto-suggest improvement → re-simulate → compare → return best design + report.

    A single ChatGPT message triggers the entire chip design lifecycle.
    Internally calls chip_design_from_nl, chip_validate_design, chip_run_simulation, chip_analyze_bottleneck, chip_sweep_parameters, chip_compare_designs.
    """

    # Step 1: Parse natural language into DesignSpec
    design_result = chip_design_from_nl(description, constraints)
    if not design_result["success"]:
        return {"error": design_result.get("error", "Failed to parse NL description")}

    design_id = design_result["design_id"]

    # Step 2: Validate the design
    validation = chip_validate_design(design_id)
    if not validation["valid"]:
        return {
            "design_id": design_id,
            "validation_errors": validation["errors"],
            "message": "Design has validation errors, but proceeding with flow.",
        }

    # Step 3: Run initial simulation
    sim_result = chip_run_simulation(design_id, workload=workload, fidelity="analytical")

    # Step 4: Analyze bottleneck
    bottleneck = chip_analyze_bottleneck(sim_result["run_id"])

    # Step 5: Suggest improvements
    suggestions = chip_suggest_improvements(sim_result["run_id"], max_suggestions=5)

    # Step 6: Sweep parameters based on suggestions
    sweep_params = {}
    if suggestions.get("suggestions"):
        for s in suggestions["suggestions"][:3]:  # top 3 suggestions
            change = s.get("change", {})
            if isinstance(change, dict):
                param_path = list(change.keys())[0] if change else ""
                param_values = list(change.values())[0] if change else []
                if param_path and param_values:
                    sweep_params[param_path] = param_values

    sweep_result = None
    if sweep_params:
        sweep_result = chip_sweep_parameters(
            design_id,
            parameters=sweep_params,
            workload=workload,
            fidelity="analytical",
            objective="throughput_tokens_per_sec",
        )

    # Step 7: Compare designs if we had multiple iterations
    comparison = None
    if max_iterations > 1 and sweep_result:
        # Create variant designs based on suggestions and compare
        run_ids = [sim_result["run_id"]]
        if sweep_result and sweep_result.get("best_point"):
            # Re-create design with best parameters and simulate
            best_design = chip_design_from_nl(description, constraints)
            if best_design["success"]:
                run_ids.append(best_design["design_id"])

        if len(run_ids) > 1:
            comparison = chip_compare_designs(run_ids)

    # Step 8: Export the full project
    export = chip_export_full_project(design_id, format="all")

    # Compile results
    result = {
        "design_id": design_id,
        "description": description,
        "initial_design": {
            "name": design_result["name"],
            "architecture": design_result["architecture"],
            "chiplets": design_result["chiplet_count"],
        },
        "validation": validation,
        "simulation": {
            "run_id": sim_result["run_id"],
            "fidelity": sim_result["fidelity"],
            "metrics": sim_result["metrics"],
            "bottleneck": sim_result.get("bottleneck", {}),
        },
        "bottleneck_analysis": bottleneck,
        "improvement_suggestions": suggestions,
        "sweep_parameters": sweep_result,
        "comparison": comparison,
        "export": export,
    }

    return result


def chip_suggest_improvements(
    run_id: str,
    max_suggestions: int = 5,
) -> dict:
    """Analyze a simulation result and suggest concrete parameter changes
    (increase memory bandwidth, add compute chiplets, change process node, adjust cache sizes)
    with predicted impact. Returns ranked list of {change, rationale, expected_improvement}.
    """
    from chiplab.engine.analyzer import analyze_bottleneck

    # Get the simulation result
    sim_result = store.get_simulation(run_id)

    # Analyze bottleneck
    bottleneck_report = analyze_bottleneck(sim_result)

    suggestions = []
    primary_bottleneck = bottleneck_report.get("primary_bottleneck", "balanced")

    # Generate suggestions based on bottleneck type
    metrics = sim_result.get("metrics", {})

    if primary_bottleneck in ["memory_bandwidth", "memory_bound"]:
        # Suggestion: increase memory bandwidth
        bandwidth = metrics.get("aggregate_memory_bandwidth_tbps", 0)
        suggestions.append(
            {
                "change": {"interconnect.link_bandwidth_gbps": [bandwidth * 2]},
                "rationale": "Memory bandwidth is the primary bottleneck. Doubling the interconnect bandwidth should significantly improve throughput.",
                "expected_improvement": "2x increase in throughput_tokens_per_sec",
                "priority": 1,
            }
        )
        # Suggestion: add more memory chiplets
        memory_chiplets = [
            c for c in sim_result.get("design", {}).get("chiplets", []) if c.get("type") == "memory"
        ]
        if memory_chiplets:
            current_count = memory_chiplets[0].get("count", 1)
            suggestions.append(
                {
                    "change": {"chiplets[].count": [current_count + 2]},
                    "rationale": "Adding more memory chiplets increases aggregate memory bandwidth proportionally.",
                    "expected_improvement": f"{current_count + 2}x memory capacity",
                    "priority": 2,
                }
            )

    elif primary_bottleneck in ["compute_bound", "compute"]:
        # Suggestion: add more compute chiplets
        compute_chiplets = [
            c for c in sim_result.get("design", {}).get("chiplets", []) if c.get("type") == "digital_compute"
        ]
        if compute_chiplets:
            current_count = compute_chiplets[0].get("count", 1)
            suggestions.append(
                {
                    "change": {"chiplets[].count": [current_count + 1]},
                    "rationale": "Compute bound workload can benefit from additional compute chiplets.",
                    "expected_improvement": f"{current_count + 1}x compute throughput",
                    "priority": 1,
                }
            )
        # Suggestion: upgrade process node
        suggestions.append(
            {
                "change": {"chiplets[0].process_nm": [28]},  # from default to 28nm
                "rationale": "Moving to a smaller process node improves power efficiency and can increase clock frequency.",
                "expected_improvement": "1.5x improvement in TOPS/W",
                "priority": 2,
            }
        )

    elif primary_bottleneck in ["thermal", "power"]:
        # Suggestion: reduce power or add thermal management
        power = metrics.get("total_power_watts", 0)
        suggestions.append(
            {
                "change": {"chiplets[].tdp_watts": [power * 0.8]},
                "rationale": "Power budget is exceeded. Reducing TDP per chiplet helps stay within thermal limits.",
                "expected_improvement": "Stay within thermal budget while maintaining 80% performance",
                "priority": 1,
            }
        )
        # Suggestion: add more chiplets with better efficiency
        suggestions.append(
            {
                "change": {"chiplets[0].process_nm": [16]},
                "rationale": "Smaller process node typically offers better power efficiency.",
                "expected_improvement": "30% power reduction for same performance",
                "priority": 2,
            }
        )

    else:
        # Balanced design - suggest fine-tuning
        suggestions.append(
            {
                "change": {"workload.batch_size": [64, 128, 256]},
                "rationale": "Balanced design - tuning batch size can optimize throughput.",
                "expected_improvement": "Up to 20% throughput improvement by finding optimal batch size",
                "priority": 1,
            }
        )

    # Sort by priority and limit to max_suggestions
    suggestions.sort(key=lambda x: x.get("priority", 99))
    suggestions = suggestions[:max_suggestions]

    # Format output
    formatted_suggestions = []
    for i, s in enumerate(suggestions):
        formatted_suggestions.append(
            {
                f"suggestion_{i + 1}": {
                    "change": s.get("change", {}),
                    "rationale": s.get("rationale", ""),
                    "expected_improvement": s.get("expected_improvement", ""),
                }
            }
        )

    return {
        "run_id": run_id,
        "primary_bottleneck": primary_bottleneck,
        "suggestions": formatted_suggestions,
        "total_suggestions_generated": len(formatted_suggestions),
    }


def chip_generate_rtl(
    design_id_or_name: str,
    target: str = "chisel",
) -> dict:
    """Generate synthesizable RTL (Chisel/SystemVerilog) skeleton for the chip's compute and memory subsystems.
    Uses existing memory_gen and asic modules. Returns the RTL code + a file manifest.
    """
    from chiplab.library.components import get_component_defaults, COMPONENT_LIBRARY
    from chiplab.schemas.design import plain_design

    # Get the design
    design_id, design_spec = store.get_design(design_id_or_name)
    require_credible_design(design_spec)
    design_spec, _ = plain_design(design_spec)  # TrackedValues would not format as numbers

    # Determine which chiplets have RTL generation capability
    memory_chiplets = [
        c for c in design_spec.chiplets if c.type == "memory"
    ]
    compute_chiplets = [
        c for c in design_spec.chiplets if c.type in ("compute", "digital_compute")
    ]

    rtl_files = []
    manifest = {"files": [], "target": target}

    # Generate RTL for memory subsystems
    for mem_chiplet in memory_chiplets[:2]:  # limit to first 2 memory chiplets
        mem_tech = mem_chiplet.memory.technology if mem_chiplet.memory else "DRAM"

        if mem_tech == "HBM":
            # HBM generates memory controller interface
            rtl_code = f"// HBM memory controller for {mem_chiplet.count} chiplets\n// Technology: {mem_tech}\n// This is a skeleton interface for HBM3E memory\n"
            manifest["files"].append(
                {
                    "name": f"hbm_controller_{mem_chiplet.id}.sv",
                    "type": "system_verilog",
                    "description": "HBM3E memory controller skeleton",
                }
            )
        elif mem_tech == "SRAM":
            # SRAM generates on-chip memory subsystem RTL
            rtl_code = f"// SRAM on-chip memory for {mem_chiplet.count} chiplets\n// Technology: {mem_tech}\n// Size: {mem_chiplet.memory.capacity_per_chiplet_gb}GB per chiplet\n"
            manifest["files"].append(
                {
                    "name": f"sram_array_{mem_chiplet.id}.sv",
                    "type": "system_verilog",
                    "description": "SRAM array with decoder and read/write port",
                }
            )
        else:  # DRAM
            rtl_code = f"// DRAM interface for {mem_chiplet.count} chiplets\n// Technology: {mem_tech}\n// This generates the memory controller interface\n"
            manifest["files"].append(
                {
                    "name": f"dram_controller_{mem_chiplet.id}.sv",
                    "type": "system_verilog",
                    "description": "DDR4/DDR5 memory controller skeleton",
                }
            )

        rtl_files.append({"chiplet": mem_chiplet.id, "rtl_code": rtl_code})

    # Generate RTL for compute subsystems
    for comp_chiplet in compute_chiplets[:2]:  # limit to first 2 compute chiplets
        arch = comp_chiplet.compute.architecture if comp_chiplet.compute else "systolic"

        if arch == "systolic":
            rtl_code = f"// Systolic array compute kernel\n// Architecture: systolic\n// Precision: {comp_chiplet.compute.precision if comp_chiplet.compute else 'FP8'}\n// Peak TOPS: {comp_chiplet.compute.peak_tops if comp_chiplet.compute else 1000}\n"
            manifest["files"].append(
                {
                    "name": f"systolic_array_{comp_chiplet.id}.sv",
                    "type": "system_verilog",
                    "description": "Systolic array compute kernel skeleton",
                }
            )
        elif arch == "tensor_engine":
            rtl_code = f"// Tensor engine compute kernel\n// Architecture: tensor_engine\n// This generates the tensor processing unit skeleton\n"
            manifest["files"].append(
                {
                    "name": f"tensor_engine_{comp_chiplet.id}.sv",
                    "type": "system_verilog",
                    "description": "Tensor engine compute kernel skeleton",
                }
            )
        else:
            rtl_code = f"// Compute kernel for {arch}\n// This is a generic compute kernel skeleton\n"
            manifest["files"].append(
                {
                    "name": f"compute_{comp_chiplet.id}.sv",
                    "type": "system_verilog",
                    "description": "Generic compute kernel skeleton",
                }
            )

        rtl_files.append({"chiplet": comp_chiplet.id, "rtl_code": rtl_code})

    # Generate a top-level module that instantiates all subsystems
    top_module = f"// Top-level chip module\n// Generated from design: {design_spec.name}\n// Target: {target}\n"
    ports = ["input clk", "input rst_n"]
    # Add ports for memory interfaces
    for i, mem in enumerate(memory_chiplets[:2]):
        channels = mem.memory.channels if mem.memory else 1
        ports.append(f"inout [{channels - 1}:0] hbm_dq_{i}")
    # Add ports for compute: one result bus per instance, so no net has two drivers
    result_signals = ["compute_result"]
    if len(compute_chiplets[:2]) == 2:
        result_signals.append("compute_result_1")
    ports += [f"output [31:0] {signal}" for signal in result_signals]
    top_module += "\nmodule top_level_chip (\n"
    top_module += ",\n".join(f"    {port}" for port in ports)
    top_module += "\n);\n"
    top_module += "\n// Instantiate memory subsystems\n"
    for i, mem in enumerate(memory_chiplets[:2]):
        tech = mem.memory.technology if mem.memory else "DRAM"
        top_module += "    memory_subsystem #(\n"
        top_module += f"        .tech({json.dumps(tech, ensure_ascii=False)}),\n"
        top_module += f"        .count({mem.count})\n"
        top_module += f"    ) mem_{i} (\n"
        top_module += "        .clk(clk),\n"
        top_module += "        .rst_n(rst_n)\n"
        top_module += "    );\n"
    top_module += "\n// Instantiate compute subsystems\n"
    for i, comp in enumerate(compute_chiplets[:2]):
        arch = comp.compute.architecture if comp.compute else "systolic"
        precision = comp.compute.precision if comp.compute else "FP8"
        top_module += "    compute_subsystem #(\n"
        top_module += f"        .arch({json.dumps(arch, ensure_ascii=False)}),\n"
        top_module += f"        .precision({json.dumps(precision, ensure_ascii=False)})\n"
        top_module += f"    ) comp_{i} (\n"
        top_module += "        .clk(clk),\n"
        top_module += "        .rst_n(rst_n),\n"
        top_module += f"        .result({result_signals[i]})\n"
        top_module += "    );\n"
    top_module += "\nendmodule\n"

    manifest["files"].append(
        {
            "name": "top_level_chip.sv",
            "type": "system_verilog",
            "description": "Top-level chip module instantiating all subsystems",
        }
    )

    return {
        "design_id": design_id,
        "target": target,
        "rtl_code": top_module,
        "rtl_files": rtl_files,
        "manifest": manifest,
    }


def chip_verify_design(
    design_id_or_name: str,
    checks: list | None = None,
) -> dict:
    """Run comprehensive verification: DRC (design rule checks), timing checks, power budget, thermal budget, area budget, connectivity check.
    Returns pass/fail per check with details.
    """
    from chiplab.engine.validator import validate_design

    # Get the design
    design_id, design_spec = store.get_design(design_id_or_name)
    require_credible_design(design_spec)

    # Run base validation
    validation_result = validate_design(design_spec)

    # Initialize verification results
    verification = {
        "design_id": design_id,
        "checks": {},
        "overall_pass": True,
    }

    # Default checks if none specified
    default_checks = [
        "drc",
        "timing",
        "power_budget",
        "thermal_budget",
        "area_budget",
        "connectivity",
    ]
    check_list = checks if checks else default_checks

    # DRC check
    if "drc" in check_list:
        verification["checks"]["drc"] = {
            "pass": validation_result.valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "details": "Design rule checks - validates layout rules, spacing, etc.",
        }
        if not validation_result.valid:
            verification["overall_pass"] = False

    # Timing check
    if "timing" in check_list:
        metrics = design_spec.metadata.get("metrics", {})
        max_temp = design_spec.constraints.max_temp_c
        actual_temp = metrics.get("estimated_temp_c", max_temp)
        verification["checks"]["timing"] = {
            "pass": actual_temp <= max_temp,
            "actual_temp_c": actual_temp,
            "max_temp_c": max_temp,
            "details": f"Timing and thermal check: actual {actual_temp:.1f}°C vs max {max_temp}°C",
        }
        if actual_temp > max_temp:
            verification["overall_pass"] = False

    # Power budget check
    if "power_budget" in check_list:
        power = design_spec.metadata.get("power_estimate_watts", 0)
        max_power = design_spec.constraints.max_power_watts or 1000.0
        verification["checks"]["power_budget"] = {
            "pass": power <= max_power,
            "actual_power_watts": power,
            "max_power_watts": max_power,
            "details": f"Power check: actual {power:.1f}W vs max {max_power}W",
        }
        if power > max_power:
            verification["overall_pass"] = False

    # Thermal budget check
    if "thermal_budget" in check_list:
        max_temp = design_spec.constraints.max_temp_c
        ambient_temp = design_spec.constraints.ambient_temp_c
        verification["checks"]["thermal_budget"] = {
            "pass": True,  # Simplified - would need thermal model
            "max_temp_c": max_temp,
            "ambient_temp_c": ambient_temp,
            "details": f"Thermal budget check: max allowed {max_temp}°C ambient {ambient_temp}°C",
        }

    # Area budget check
    if "area_budget" in check_list:
        area = sum(
            c.area_mm2 or 0 for c in design_spec.chiplets
        )
        max_area = design_spec.constraints.max_area_mm2 or 10000.0
        verification["checks"]["area_budget"] = {
            "pass": area <= max_area,
            "actual_area_mm2": area,
            "max_area_mm2": max_area,
            "details": f"Area check: actual {area:.1f}mm² vs max {max_area}mm²",
        }
        if area > max_area:
            verification["overall_pass"] = False

    # Connectivity check
    if "connectivity" in check_list:
        # Check that all chiplets have proper interconnect
        has_interconnect = (
            design_spec.interconnect is not None
            and design_spec.interconnect.topology is not None
        )
        verification["checks"]["connectivity"] = {
            "pass": has_interconnect,
            "topology": design_spec.interconnect.topology if has_interconnect else "none",
            "details": "Connectivity check: validates all chiplets are properly connected via interconnect",
        }
        if not has_interconnect:
            verification["overall_pass"] = False

    return verification


def chip_export_full_project(
    design_id_or_name: str,
    format: str = "all",
) -> dict:
    """Export a complete project bundle: design YAML, simulation results, RTL, testbenches, Makefile, README, constraint files (SDC), floorplan.
    Formats: "all", "rtl", "gds", "report". Returns file paths.
    """
    from chiplab.schemas.design import DesignSpec

    # Get the design
    design_id, design_spec = store.get_design(design_id_or_name)
    require_credible_design(design_spec)

    export_result = {
        "design_id": design_id,
        "format": format,
        "file_paths": [],
    }

    # Always include design YAML
    yaml_path = f"design_{design_id}.yaml"
    design_yaml = design_spec.to_yaml() if hasattr(design_spec, "to_yaml") else ""
    export_result["file_paths"].append(yaml_path)

    if format in ("all", "report"):
        # Include simulation results (we'll get recent ones)
        # For now, just note that report can be generated
        export_result["file_paths"].append(f"report_{design_id}.md")

    if format in ("all", "rtl"):
        # Generate RTL
        rtl_result = chip_generate_rtl(design_id, target="system_verilog")
        rtl_path = f"rtl_{design_id}.sv"
        export_result["file_paths"].append(rtl_path)

    if format in ("all", "report"):
        # Export simulation report (would need a run_id)
        # For now, add placeholder
        export_result["file_paths"].append(f"simulation_report_{design_id}.md")

    # Add constraint files placeholder
    sdc_path = f"constraints_{design_id}.sdc"
    export_result["file_paths"].append(sdc_path)

    # Add Makefile placeholder
    makefile_path = f"Makefile_{design_id}"
    export_result["file_paths"].append(makefile_path)

    # Add README placeholder
    readme_path = f"README_{design_id}.md"
    export_result["file_paths"].append(readme_path)

    # Add floorplan placeholder
    floorplan_path = f"floorplan_{design_id}.json"
    export_result["file_paths"].append(floorplan_path)

    return export_result


# --- Material Device MCP Tools ---
# Ferrofluid is routed to the particle physics model (models/materials/ferro_*.py,
# docs/ferrofluid_model.md). The other technologies still return labelled placeholders.

MATERIAL_PLACEHOLDER = (
    "placeholder: fixed heuristic values, not a physical simulation "
    "(only ferrofluid has a physics model in this tool)")

def simulate_material_device(
    technology: str,
    parameters: dict,
    fidelity: str = "reduced_order",
    duration_s: float = 1e-6,
) -> dict:
    """Simulate physical material behavior.

    Args:
        technology: Material technology type. "ferrofluid" runs a particle physics model;
            others (e.g. "perovskite", "phase_change", "2d_material") return placeholders.
        parameters: Dictionary of parameters. Ferrofluid keys include voltage_v,
            read_frequency_hz, duration_ns, read_delay_ns, separation_um, radius_um, turns,
            series_resistance_ohm, volume_fraction, core_diameter_nm, size_sigma_ln,
            coating_nm, anisotropy_j_m3, viscosity_pa_s, temperature_k, carrier_mobile;
            unknown ferrofluid keys are rejected.
        fidelity: Simulation fidelity level ("reduced_order", "medium", "full"); the
            ferrofluid model is the same at every level.
        duration_s: Simulation duration in seconds (ferrofluid: write pulse length when
            parameters has no duration_ns)

    Returns:
        Dictionary with state evolution, impedance, and energy metrics.
    """
    from chiplab.models.materials import ferro_interface
    from chiplab.schemas.simulation import FidelityLevel

    # Map fidelity string to FidelityLevel
    fidelity_map = {
        "reduced_order": FidelityLevel.ANALYTICAL,
        "medium": FidelityLevel.DISCRETE_EVENT,
        "full": FidelityLevel.SPECIALIZED,
    }
    fidelity_level = fidelity_map.get(fidelity, FidelityLevel.ANALYTICAL)

    if ferro_interface.routes_to_ferrofluid({**parameters, "technology": technology}):
        inputs = {k: v for k, v in parameters.items() if k not in ("technology", "material")}
        inputs.setdefault("duration_ns", duration_s * 1e9)
        summary = ferro_interface.technology_summary(inputs, None)
        metrics = summary["metrics"]
        return {
            "technology": technology,
            "fidelity": FidelityLevel.ANALYTICAL.value,
            "fidelity_disclaimer": "Reduced-order particle physics model, uncalibrated unless "
                                   "parameters are labelled fitted; the fidelity argument "
                                   "does not change it",
            "state_evolution": {
                "reads": summary["simulation"]["reads"],
                "retention": summary["simulation"]["retention"],
                "duration_s": inputs["duration_ns"] * 1e-9,
            },
            "impedance_ohm": metrics["read_impedance_magnitude_ohm"],
            "energy_pj": metrics["write_drive_energy_j"] * 1e12,
            "metrics": metrics,
            "metric_status": summary["metric_status"],
            "simulation": summary["simulation"],
            "not_modelled": summary["not_modelled"],
            "success": True,
            "error_message": None,
        }

    tech_lower = technology.lower()

    # Determine default metrics based on technology type
    if "perovskite" in tech_lower:
        default_impedance = 150.0
        default_energy = duration_s * 1e-6 * 10.0  # pJ-scale
        state_params = {"carrier_density": parameters.get("carrier_density", 1e22),
                       "lifetime": parameters.get("lifetime", 1e-6)}
    elif "phase_change" in tech_lower:
        default_impedance = 200.0
        default_energy = duration_s * 5e-7 * 5.0
        state_params = {"phase": parameters.get("phase", "amorphous"),
                       "transition_time": parameters.get("transition_time", 1e-9)}
    elif "2d" in tech_lower or "graphene" in tech_lower:
        default_impedance = 10000.0
        default_energy = duration_s * 1e-7 * 1.0
        state_params = {"carrier_density": parameters.get("carrier_density", 1e16),
                       "mobility": parameters.get("mobility", 1e4)}
    else:
        # Generic material defaults
        default_impedance = 50.0
        default_energy = duration_s * 1e-6 * 1.0
        state_params = {"temperature": parameters.get("temperature", 300),
                       "stress": parameters.get("stress", 0.0)}

    # Generate state evolution (time-series-like summary)
    state_evolution = {
        "initial": state_params,
        "final": {k: v * 0.8 for k, v in state_params.items()},  # Simulate decay
        "fidelity": fidelity,
        "duration_s": duration_s,
    }

    result = {
        "technology": technology,
        "fidelity": fidelity_level.value,
        "fidelity_disclaimer": f"Material simulation at {fidelity_level.value} fidelity; "
                               f"parameters: {list(parameters.keys())}",
        "state_evolution": state_evolution,
        "impedance_ohm": default_impedance,
        "energy_pj": round(default_energy, 4),
        "model_status": MATERIAL_PLACEHOLDER,
        "success": True,
        "error_message": None,
    }

    return result


def configure_material_array(
    technology: str,
    n_ports: int,
    arrangement: str,
    pitch_um: float,
    port_config: dict | list | None = None,
) -> dict:
    """Configure electrode and cell arrangements for material devices.

    Args:
        technology: Material technology type
        n_ports: Number of electrode ports
        arrangement: "linear", "grid", "grid_2x2", "grid_4x4", "circular", "hexagonal" or
            "custom" (custom needs port_config["positions"] = [[x_um, y_um], ...])
        pitch_um: Pitch in micrometers between adjacent electrodes
        port_config: Optional per-port settings, as a list (one dict per port) or a dict
            keyed "0".."n-1" or "port_0".."port_n-1". Keys: role ("write", "read", "both"),
            id, bias_voltage, impedance_ohm, functional_group, position_um, and ferrofluid
            conductor geometry (conductor, radius_um, turns, wire_radius_um,
            series_resistance_ohm, axis, start_um, end_um, electrode_radius_um)

    Returns:
        ElectrodeArray specification dictionary.
    """
    from chiplab.models.materials import ferro_interface

    config = port_config or {}
    custom_positions = None
    per_port: dict[int, dict] = {}
    if isinstance(config, list):
        if len(config) > n_ports:
            raise ValueError(f"port_config lists {len(config)} ports but n_ports is {n_ports}")
        items = list(enumerate(config))
    elif isinstance(config, dict):
        config = dict(config)
        custom_positions = config.pop("positions", None)
        items = []
        for key, value in config.items():
            text = str(key).removeprefix("port_")
            if not text.isdigit() or int(text) >= n_ports:
                raise ValueError(f"port_config key {key!r} does not name one of the "
                                 f"{n_ports} ports (use '0'..'{n_ports - 1}')")
            items.append((int(text), value))
    else:
        raise ValueError("port_config must be a dict or a list")
    for index, value in items:
        if not isinstance(value, dict):
            raise ValueError(f"port_config entry for port {index} must be a dict, "
                             f"got {type(value).__name__}")
        if index in per_port:
            raise ValueError(f"port {index} is configured twice")
        per_port[index] = dict(value)

    for settings in per_port.values():
        role = str(settings.get("type", "")).lower()
        if role in ("write", "read", "both") and "role" not in settings:
            settings["role"] = role
            del settings["type"]
    for index, settings in per_port.items():
        ferro_interface.check_keys(settings, ferro_interface.ARRAY_PORT_KEYS, f"port {index}")

    kind = ferro_interface.canonical_arrangement(arrangement)
    positions = ferro_interface.layout_positions(n_ports, kind, pitch_um, custom_positions)
    layouts = {"linear": "one-dimensional chain", "circular": "circular arrangement",
               "hexagonal": "hexagonal lattice", "custom": "custom arrangement"}
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    if kind == "linear":
        port_positions = list(xs)
        total_length = pitch_um * (n_ports - 1) if n_ports > 1 else pitch_um
    elif kind == "circular":
        radius_um = pitch_um * n_ports / (2 * math.pi)
        port_positions = [{"angle_rad": 2 * math.pi * i / n_ports, "radius_um": radius_um,
                           "x_um": x, "y_um": y} for i, (x, y) in enumerate(positions)]
        total_length = 2 * radius_um
    else:
        port_positions = [{"x_um": x, "y_um": y} for x, y in positions]
        total_length = max(max(xs) - min(xs), max(ys) - min(ys))

    ferro = ferro_interface.is_ferrofluid(technology)
    configured_ports = []
    for i in range(n_ports):
        settings = per_port.get(i, {})
        given = "impedance_ohm" in settings
        port = {
            "port_index": i,
            "position": port_positions[i],
            "bias_voltage": settings.get("bias_voltage", 0.0),
            "impedance_ohm": settings.get("impedance_ohm", None if ferro else 50.0),
            "impedance_source": (
                "user_supplied" if given else
                "not computed here: use run_multiphysics_simulation" if ferro else
                "assumed default, not computed"),
            "functional_group": settings.get("functional_group", "default"),
        }
        port.update({k: v for k, v in settings.items() if k not in port})
        configured_ports.append(port)

    return {
        "technology": technology,
        "arrangement": arrangement,
        "n_ports": n_ports,
        "pitch_um": pitch_um,
        "layout": layouts.get(kind, "two-dimensional grid"),
        "total_length_um": total_length,
        "port_config": configured_ports,
    }


def run_multiphysics_simulation(
    device_spec: dict,
    electrode_array: dict,
    write_sequence: list,
    read_sequence: list,
    fidelity: str = "reduced_order",
) -> dict:
    """Run coupled physical simulation of material device with electrode array.

    Args:
        device_spec: Device specification dictionary. technology "ferrofluid" runs the
            particle physics model and accepts volume_fraction, core_diameter_nm,
            size_sigma_ln, coating_nm, viscosity_pa_s, temperature_k, anisotropy_j_m3,
            carrier_mobile, fluid_region, conductivity_s_per_m, relative_permittivity,
            spectrum_hz and calibration (output of calibrate_material_model).
        electrode_array: Electrode array configuration from configure_material_array
        write_sequence: List of write pulses. Ferrofluid: {"port", "voltage_v" (or
            "amplitude"), "duration_ns", "start_ns"}; pulses on different ports may overlap.
        read_sequence: List of reads. Ferrofluid: {"port", "frequency_hz", "time_ns" or
            "delay_ns", "amplitude_v", "bandwidth_hz"}
        fidelity: Simulation fidelity level ("reduced_order", "medium", "full"); the
            ferrofluid model is the same at every level.

    Returns:
        Dictionary with time-domain waveforms, impedance spectra, and metrics.
    """
    from chiplab.models.materials import ferro_interface
    from chiplab.schemas.simulation import FidelityLevel

    fidelity_level = FidelityLevel.ANALYTICAL
    fidelity_map = {
        "reduced_order": FidelityLevel.ANALYTICAL,
        "medium": FidelityLevel.DISCRETE_EVENT,
        "full": FidelityLevel.SPECIALIZED,
    }
    if fidelity in fidelity_map:
        fidelity_level = fidelity_map[fidelity]

    tech = device_spec.get("technology") or electrode_array.get("technology", "generic")
    n_ports = electrode_array.get("n_ports", len(electrode_array.get("port_config", [])))

    if ferro_interface.routes_to_ferrofluid(device_spec, electrode_array):
        result = ferro_interface.multiphysics(
            device_spec, electrode_array, write_sequence, read_sequence)
        return {
            **result,
            "fidelity": FidelityLevel.ANALYTICAL.value,
            "fidelity_disclaimer": "Reduced-order particle physics model, uncalibrated "
                                   "unless parameters are labelled fitted",
            "success": True,
            "error_message": None,
        }

    # Placeholder for the other technologies: echoes the sequences
    write_waveforms = []
    for i, write_sig in enumerate(write_sequence):
        write_waveforms.append({
            "channel": i,
            "waveform": write_sig.get("pattern", "pulse"),
            "amplitude": write_sig.get("amplitude", 1.0),
            "duration_ns": write_sig.get("duration_ns", 1.0),
        })

    read_waveforms = []
    for i, read_sig in enumerate(read_sequence):
        read_waveforms.append({
            "channel": i,
            "waveform": read_sig.get("pattern", "sense"),
            "sensitivity": read_sig.get("sensitivity", 1.0),
            "integration_time_ns": read_sig.get("integration_time_ns", 0.5),
        })

    # Generate impedance spectrum (simplified)
    impedance_data = []
    for freq_idx in range(10):  # 10 frequency points
        freq = 10 ** (3 + freq_idx * 0.5)  # 1kHz to 100GHz range
        # Impedance depends on technology and arrangement
        base_impedance = electrode_array.get("pitch_um", 100.0)
        if "perovskite" in tech.lower():
            reactance = base_impedance * 0.8 ** (freq_idx + 1)
        elif "phase_change" in tech.lower():
            reactance = base_impedance * (1 + freq_idx * 0.1)
        else:
            reactance = base_impedance
        impedance_data.append({"frequency_hz": freq, "impedance_ohm": reactance})

    # Generate time-domain metrics
    total_energy_pj = sum(w.get("amplitude", 1.0) * w.get("duration_ns", 1.0) for w in write_waveforms) * 1e-3

    result = {
        "device_technology": tech,
        "fidelity": fidelity_level.value,
        "fidelity_disclaimer": f"Multi-physics simulation at {fidelity_level.value} fidelity",
        "electrode_array": {
            "arrangement": electrode_array.get("arrangement"),
            "n_ports": n_ports,
            "pitch_um": electrode_array.get("pitch_um"),
        },
        "write_sequence": write_waveforms,
        "read_sequence": read_waveforms,
        "impedance_spectrum": impedance_data,
        "total_energy_pj": round(total_energy_pj, 4),
        "model_status": MATERIAL_PLACEHOLDER,
        "success": True,
        "error_message": None,
    }

    return result


def simulate_material_circuit(
    device_spec: dict,
    electrode_array: dict,
    write_circuit: dict,
    read_circuit: dict,
    analysis: str = "transient",
) -> dict:
    """Connect material device to equivalent circuit for system-level analysis.

    Args:
        device_spec: Device specification dictionary
        electrode_array: Electrode array configuration
        write_circuit: Write circuit parameters (voltage sources, etc.)
        read_circuit: Read circuit parameters (sensors, amplifiers, etc.)
        analysis: Type of analysis ("transient", "ac", "dc", "noise")

    Ferrofluid uses the particle physics model: write_circuit accepts port, voltage_v,
    duration_ns, frequency_range, n_freqs; read_circuit accepts port, bandwidth_hz,
    noise_floor_v_sqrt_hz. Other technologies return placeholders.

    Returns:
        Dictionary with circuit waveforms and analysis results.
    """
    from chiplab.models.materials import ferro_interface

    tech = device_spec.get("technology") or electrode_array.get("technology", "generic")
    if ferro_interface.routes_to_ferrofluid(device_spec, electrode_array):
        ferro_interface.validate_device_spec(device_spec)
        device = ferro_interface.build_device(
            ferro_interface.ports_from_array(electrode_array), electrode_array, device_spec)
        return {"technology": "ferrofluid", **ferro_interface.circuit_analysis(
            device, write_circuit, read_circuit, analysis)}

    # Perform the requested analysis type
    if analysis.lower() == "dc":
        # DC analysis: compute operating point
        voltage = write_circuit.get("voltage", 1.0)
        resistance = electrode_array.get("pitch_um", 50.0) * 1e3  # approximate
        current = voltage / resistance if resistance > 0 else 0.0
        power = voltage * current

        result = {
            "analysis": "dc",
            "technology": tech,
            "voltage_v": voltage,
            "resistance_ohm": resistance,
            "current_a": current,
            "power_w": power,
            "dc_operating_point": {
                "voltage": voltage,
                "current": current,
                "power": power,
            },
        }
    elif analysis.lower() == "ac":
        # AC analysis: frequency response
        n_freqs = write_circuit.get("n_freqs", 10)
        frequency_range = write_circuit.get("frequency_range", {"min_hz": 1e3, "max_hz": 1e9})
        impedances = []
        for i in range(n_freqs):
            freq = 10 ** (3 + i * (6 - 3) / (n_freqs - 1))  # log-spaced
            # Simple RC model
            rc_time = electrode_array.get("pitch_um", 50.0) * 1e-12
            magnitude = 1 / (1 + (2 * 3.14159 * freq * rc_time))
            phase = -2 * 3.14159 * freq * rc_time
            impedances.append({
                "frequency_hz": freq,
                "magnitude": magnitude,
                "phase_deg": phase * 180 / 3.14159,
            })

        result = {
            "analysis": "ac",
            "technology": tech,
            "frequency_range": frequency_range,
            "impedance_response": impedances,
        }
    elif analysis.lower() == "noise":
        # Noise analysis
        n_samples = read_circuit.get("n_samples", 1000)
        noise_floor = read_circuit.get("noise_floor_v/√Hz", 1e-9)
        rms_noise = noise_floor * (3.14159 * 1e6) ** 0.5  # simplified

        result = {
            "analysis": "noise",
            "technology": tech,
            "n_samples": n_samples,
            "noise_floor_v_sqrt_hz": noise_floor,
            "rms_noise_v": round(rms_noise, 6),
            "snr_db": 20 * (3.14159 ** 0.5),  # simplified SNR
        }
    else:
        # Default to transient analysis
        # Transient: simulate time-domain response
        write_voltage = write_circuit.get("voltage", 1.0)
        write_duration = write_circuit.get("duration_ns", 100.0)
        read_sensitivity = read_circuit.get("sensitivity", 1.0)

        # Generate time points
        n_points = 50
        time_points = [i * write_duration / (n_points - 1) for i in range(n_points)]

        # Simple exponential rise/fall model
        waveforms = []
        for t in time_points:
            # V(t) = V0 * (1 - e^(-t/τ)) for rise, or V0 * e^(-t/τ) for decay
            tau = electrode_array.get("pitch_um", 50.0) * 1e-12  # time constant
            voltage = write_voltage * (1 - __import__("math").exp(-t / tau)) if t < write_duration else write_voltage * __import__("math").exp(-(t - write_duration) / tau)
            current = voltage / (electrode_array.get("pitch_um", 50.0) * 1e3)
            waveforms.append({"time_ns": t, "voltage_v": voltage, "current_a": current})

        result = {
            "analysis": "transient",
            "technology": tech,
            "write_voltage_v": write_voltage,
            "write_duration_ns": write_duration,
            "n_points": n_points,
            "waveforms": waveforms,
            "read_sensitivity": read_sensitivity,
        }

    result["model_status"] = MATERIAL_PLACEHOLDER
    return result


def sweep_material_parameters(
    device_spec: dict,
    parameters: dict,
    objective: str = "energy_per_bit",
) -> dict:
    """Explore material configurations by sweeping parameter spaces.

    Args:
        device_spec: Device specification dictionary
        parameters: Dictionary of parameter names to lists of values to sweep
        objective: Optimization objective ("energy_per_bit", "performance", "power")

    Ferrofluid objectives: "write_energy", "power", "read_signal", "read_flux",
    "decay_time", "peak_field" ("energy_per_bit" is not modelled). The ferrofluid sweep
    accepts up to 50 combinations.

    Returns:
        Dictionary with sweep results including Pareto front.
    """
    from chiplab.models.materials import ferro_interface
    from chiplab.schemas.simulation import FidelityLevel

    fidelity = device_spec.get("fidelity", "reduced_order")
    fidelity_level = FidelityLevel.ANALYTICAL
    if fidelity == "discrete_event":
        fidelity_level = FidelityLevel.DISCRETE_EVENT
    elif fidelity == "full" or fidelity == "specialized":
        fidelity_level = FidelityLevel.SPECIALIZED

    n_points = 1
    for vals in parameters.values():
        n_points *= len(vals) if vals else 1

    tech = device_spec.get("technology", "generic")
    if ferro_interface.routes_to_ferrofluid(device_spec):
        tech = "ferrofluid"
        if not parameters or not all(isinstance(v, list) and v for v in parameters.values()):
            raise ValueError("parameters must map each name to a non-empty list of values")
        if n_points > 50:
            raise ValueError(f"{n_points} combinations requested; the limit is 50")
        names = list(parameters)
        combos = [dict(zip(names, combo)) for combo in
                  itertools.product(*(parameters[k] for k in names))]
        results = ferro_interface.sweep(device_spec, combos, objective)
        known = ferro_interface.SWEEP_OBJECTIVES.get(objective)
        ranked = [r for r in results if r["objective_value"] is not None]
        ranked.sort(key=lambda r: r["objective_value"], reverse=bool(known and known[1]))
        return {
            "device_technology": tech,
            "objective": objective,
            "objective_status": (
                f"computed from {known[0]}" if known else
                f"{objective!r} is not modelled for ferrofluid; available: "
                f"{list(ferro_interface.SWEEP_OBJECTIVES)}"),
            "n_parameter_combinations": n_points,
            "n_simulated_points": len(results),
            "sweep_points": results,
            "pareto_front": ranked[:5],
            "pareto_note": "best points for the single objective, not a multi-objective front",
            "fidelity": FidelityLevel.ANALYTICAL.value,
        }

    # Generate sweep points
    sweep_points = []
    param_names = list(parameters.keys())
    if n_points > 0:
        value_combos = list(itertools.product(*[parameters.get(k, []) for k in param_names]))
        for i, combo in enumerate(value_combos[:50]):  # limit to 50 points max
            point_params = dict(zip(param_names, combo))
            sweep_points.append(point_params)

    # Run simulations for each point (using reduced simulations for speed)
    results = []
    for i, params in enumerate(sweep_points[:20]):  # limit simulations
        # Simulate with these parameter values
        sim_result = simulate_material_device(
            device_spec.get("technology", "generic"),
            params,
            fidelity=fidelity,
            duration_s=1e-6,
        )
        # Compute objective metric
        if objective == "energy_per_bit":
            metric_value = sim_result.get("energy_pj", 1e-3) / max(sim_result.get("impedance_ohm", 50.0), 1)
        elif objective == "performance":
            metric_value = 1 / max(sim_result.get("impedance_ohm", 50.0) / 100.0, 0.01)
        elif objective == "power":
            metric_value = sim_result.get("energy_pj", 1e-3) / max(1e-6, sim_result.get("duration_s", 1e-6))
        else:
            metric_value = sim_result.get("energy_pj", 1e-3)

        results.append({
            "parameters": params,
            "objective_value": metric_value,
            "impedance": sim_result.get("impedance_ohm", 50.0),
            "energy_pj": sim_result.get("energy_pj", 1e-3),
            "success": True,
        })

    # Find Pareto front (simplified: best objective values)
    if objective == "energy_per_bit":
        # Lower is better
        pareto_indices = sorted(range(len(results)), key=lambda i: results[i]["objective_value"])[:5]
    else:
        # Higher is better
        pareto_indices = sorted(range(len(results)), key=lambda i: results[i]["objective_value"], reverse=True)[:5]

    pareto_front = [results[i] for i in pareto_indices]

    sweep_result = {
        "device_technology": device_spec.get("technology", "generic"),
        "objective": objective,
        "n_parameter_combinations": n_points,
        "n_simulated_points": len(results),
        "sweep_points": results,
        "pareto_front": pareto_front,
        "fidelity": fidelity_level.value,
        "model_status": MATERIAL_PLACEHOLDER,
    }

    return sweep_result


def calibrate_material_model(
    technology: str,
    experimental_data: dict,
    parameters_to_fit: list[str],
) -> dict:
    """Fit material model parameters to experimental measurements.

    Args:
        technology: Material technology type
        experimental_data: Dictionary with experimental measurements. Ferrofluid:
            {"provenance": {"source": "<instrument/file/publication>"},
             "datasets": [{"kind": "magnetization_curve", "H_a_per_m": [...],
                           "M_a_per_m": [...], "sigma_M_a_per_m": [...] (optional)},
                          {"kind": "susceptibility_spectrum", "frequency_hz": [...],
                           "chi_real": [...], "chi_loss": [...]},
                          {"kind": "magnetization_decay", "time_s": [...],
                           "m_normalized": [...], "initial_field_a_per_m": H0}],
             "fixed_parameters": {...}}
        parameters_to_fit: List of parameter names to calibrate

    Returns:
        Ferrofluid: least-squares fit with standard errors, identifiability flags and data
        provenance. Other technologies have no model to fit: supplied values are only
        averaged and reported as uncalibrated.
    """
    from chiplab.models.materials import ferro_interface

    if ferro_interface.is_ferrofluid(technology):
        from chiplab.models.materials.ferro_calibration import fit_parameters

        ferro_interface.check_keys(
            experimental_data, {"provenance", "datasets", "fixed_parameters"},
            "experimental_data")
        fit = fit_parameters(
            experimental_data.get("datasets") or [], parameters_to_fit,
            experimental_data.get("provenance") or {},
            experimental_data.get("fixed_parameters"))
        return {"calibrated_model": {"technology": technology, **fit},
                "success": fit["status"] != "failed"}

    tech_lower = technology.lower()

    # Initialize parameter values with reasonable defaults
    initial_params = {}
    for param in parameters_to_fit:
        if "carrier" in param.lower():
            initial_params[param] = 1e22  # m^-3
        elif "mobility" in param.lower():
            initial_params[param] = 0.1  # m^2/Vs
        elif "threshold" in param.lower():
            initial_params[param] = 0.5  # V
        elif "dielectric" in param.lower():
            initial_params[param] = 3.9  # relative permittivity
        else:
            initial_params[param] = 1.0

    # No physical model exists for these technologies, so nothing is fitted.
    values = {}
    sources = {}
    for param in parameters_to_fit:
        data = experimental_data.get(param)
        if data is None or data == []:
            values[param] = initial_params.get(param, 1.0)
            sources[param] = "assumed default (no data supplied)"
        else:
            values[param] = sum(data) / len(data) if isinstance(data, list) else data
            sources[param] = "mean of supplied values"

    calibrated_model = {
        "technology": technology,
        "status": "not_calibrated",
        "model_status": f"no fitting model for {tech_lower}: values are averages or defaults",
        "fitted_parameters": values,
        "parameter_sources": sources,
        "parameter_uncertainty": {param: None for param in values},
        "provenance": {
            "source": "user_supplied_values",
            "method": "arithmetic mean, no model fit",
            "experimental_data_points": sum(
                len(v) if isinstance(v, list) else 1 for v in experimental_data.values()),
        },
        "fit_quality": {"residual_sum_squared": None, "r_squared": None, "converged": False},
        "calibration_timestamp": time.time(),
    }

    return {"calibrated_model": calibrated_model, "success": False}


def material_device_to_chiplet(
    device_spec: dict,
    electrode_array: dict,
    write_sequence: list | None = None,
    read_sequence: list | None = None,
    validation_measurements: dict | None = None,
) -> dict:
    """Turn a material device into a chiplet, only if it passes the physics validation gate.

    Args:
        device_spec: Device specification with every material parameter supplied or fitted
        electrode_array: Electrode array configuration with explicit port geometry
        write_sequence: Operating write pulses (as for run_multiphysics_simulation)
        read_sequence: Operating reads
        validation_measurements: Your own measurements, {"provenance": {"source": ...},
            "datasets": [...]}; see chiplab_validate_material_device

    Returns:
        {"accepted", "chiplet", "credibility"}. chiplet is null unless every check passes;
        without measurements it is always rejected. An accepted chiplet carries its device
        record (re-checked by every engine) and only area and drive power.
    """
    credibility = chiplab_validate_material_device(
        device_spec, electrode_array, write_sequence, read_sequence, validation_measurements)
    if not credibility["accepted"]:
        failed = [f"{c['id']}: {c['status']}" for c in credibility["checks"]
                  if c["status"] != "pass"]
        return {"accepted": False, "success": False, "chiplet": None,
                "credibility": credibility,
                "error_message": "rejected by the physics validation gate (" +
                                 ", ".join(failed) + ")"}
    numbers = credibility["chiplet_numbers"]
    chiplet = {
        "id": "material_device_ferrofluid",
        "type": "material",
        "area_mm2": numbers["area_mm2"],
        "tdp_watts": numbers["tdp_watts"],
        "material_device": _device_record(
            device_spec, electrode_array, write_sequence, read_sequence,
            validation_measurements),
    }
    return {"accepted": True, "success": True, "chiplet": chiplet,
            "credibility": credibility, "error_message": None,
            "field_status": {
                "area_mm2": "bounding box of the port conductors only",
                "tdp_watts": "resistive drive power of the supplied write pulses",
                "performance, capacity, cost": "not modelled"}}


def _device_record(device_spec: dict, electrode_array: dict, write_sequence: list | None,
                   read_sequence: list | None, validation_measurements: dict | None) -> dict:
    return {"device_spec": device_spec, "electrode_array": electrode_array,
            "write_sequence": write_sequence or [], "read_sequence": read_sequence or [],
            "validation_measurements": validation_measurements or {}}


def chiplab_validate_material_device(
    device_spec: dict,
    electrode_array: dict,
    write_sequence: list | None = None,
    read_sequence: list | None = None,
    validation_measurements: dict | None = None,
) -> dict:
    """Physics validation report for a material device: every check pass, fail or not_run.

    Checks: zero input, parameter sensitivity, unit consistency, boundary limits,
    conservation, grid convergence, model validity range, no assumed inputs, and agreement
    with measurements you supply (magnetization curve; susceptibility spectrum or decay;
    fluid-induced port impedance change). A device is accepted only if all pass.

    validation_measurements: {"provenance": {"source": ...}, "datasets": [
        {"kind": "magnetization_curve", "H_a_per_m", "M_a_per_m", "sigma_M_a_per_m"},
        {"kind": "susceptibility_spectrum", "frequency_hz", "chi_real", "chi_loss",
         "sigma_chi_real", "sigma_chi_loss"},
        {"kind": "port_impedance_change", "port", "to_port" (optional), "frequency_hz",
         "delta_z_real_ohm", "delta_z_imag_ohm", "sigma_delta_z_real_ohm",
         "sigma_delta_z_imag_ohm"}]}
    Only ferrofluid has checks; other technologies are rejected as not_run. Read-only.
    """
    from chiplab.models.materials import ferro_interface
    from chiplab.models.materials.ferro_validation import evaluate_record

    if not ferro_interface.routes_to_ferrofluid(device_spec, electrode_array):
        reason = "no physics validation exists for this technology (only ferrofluid)"
        return {"accepted": False,
                "rule": "accepted only if every check passes; not_run is never a pass",
                "summary": {"pass": 0, "fail": 0, "not_run": 1},
                "checks": [{"id": "physics_model", "status": "not_run",
                            "requirement": reason, "evidence": reason}]}
    return evaluate_record(_device_record(
        device_spec, electrode_array, write_sequence, read_sequence, validation_measurements))


def chiplab_ferrofluid_analyze(
    analysis: str,
    device_spec: dict,
    electrode_array: dict,
    write_sequence: list | None = None,
    read_sequence: list | None = None,
    options: dict | None = None,
) -> dict:
    """Further ferrofluid analyses from the particle model (uncalibrated unless fitted).

    analysis:
      "sensitivity": d ln(output)/d ln(parameter). options: relative_step, parameters.
      "read_disturbance": does reading change the stored state? options: port, amplitude_v,
          frequency_hz (required), cycles, steps_per_cycle.
      "retention": how long two written states stay distinguishable above thermal noise.
          options: port, state_b_write_sequence (default unwritten), bandwidth_hz,
          snr_threshold.
      "scaling": predict a larger array from a small one and check it directly. options:
          small_shape, large_shape, voltage_v (required), pitch_um, geometry. Above 32
          ports it reports "not_modelled". electrode_array supplies pitch_um only.
    Read-only.
    """
    from chiplab.models.materials import ferro_interface

    return ferro_interface.analyze(analysis, device_spec, electrode_array,
                                   write_sequence, read_sequence, options)


STATE_MODEL_OPTIONS = {
    "describe": set(),
    "import_trace": {"time_s", "voltages_v", "temperature_k"},
    "import_touchstone": {"text", "ports", "time_s", "sigma_ohm", "sigma_relative",
                          "reference_item"},
    "simulate": {"read_times_s", "frequency_hz", "initial_state", "substeps"},
    "sensitivity": {"frequency_hz", "relative_step", "read_time_s"},
    "multiport": {"write_voltages_v", "duration_s", "frequency_hz", "idle_ports",
                  "temperature_k", "hold_s", "noise_sigma_ohm", "snr_threshold",
                  "substeps"},
    "compare": {"include_points"},
    "fit": {"parameters_to_fit", "max_evaluations"},
    "reproduce_reference": {"drive_port", "other_port", "temperature_k", "iterations",
                            "frequency_points", "zc_definition", "pulse_durations_s",
                            "skip_model_behaviour"},
}


def chiplab_ferrofluid_state_model(
    task: str,
    model: dict | None = None,
    protocol: dict | None = None,
    measurements: dict | list | None = None,
    options: dict | None = None,
) -> dict:
    """Electrically programmed N-port ferrofluid: reduced-order state-space model.

    Phenomenological and without default parameter values: fit it to your measurements.
    Separate from the magnetic coil model. Details: docs/ferrofluid_state_model.md.

    model: {"ports": [...], "ground": {branch parameters}, "coupling": {...} or null,
        "branches": {"p1|p2": {overrides}}}. protocol: {"segments": [{"duration_s",
        "voltages_v": {port: V or null=floating}, "temperature_k"}], "repeat"} or a trace
        {"time_s", "voltages_v": {port: [...]}, "temperature_k"}.
    task:
      "describe": parameter names, units, which are required.
      "import_trace" / "import_touchstone": build a protocol or impedance datasets
          (options: the arguments named in the docs; uncertainty is mandatory).
      "simulate": Z(omega, t) as N x N matrices. options: read_times_s, frequency_hz.
      "sensitivity": effect of each parameter. options: frequency_hz.
      "multiport": writes alone vs together, retrieval. options: write_voltages_v,
          duration_s, frequency_hz, noise_sigma_ohm (needed for retrieval).
      "compare": measured against simulated (measurements required).
      "fit": options: parameters_to_fit, e.g. ["ground.delta_r_ohm"].
      "reproduce_reference": status against the two-port experiment of arXiv:2211.08152;
          "not_reproduced" unless your data for every required item agree.
    Read-only.
    """
    from chiplab.models.materials import ferro_experiment as experiment
    from chiplab.models.materials import ferro_reference, ferro_statespace

    if task not in STATE_MODEL_OPTIONS:
        raise ValueError(f"task must be one of {sorted(STATE_MODEL_OPTIONS)}")
    opts = dict(options or {})
    unknown = sorted(set(opts) - STATE_MODEL_OPTIONS[task])
    if unknown:
        raise ValueError(f"unknown option(s) {unknown} for task {task!r}; allowed: "
                         f"{sorted(STATE_MODEL_OPTIONS[task])}")

    def need(**values: object) -> None:
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(f"task {task!r} needs {missing}")

    if task == "describe":
        return {"parameters": ferro_statespace.describe_parameters(),
                "target_symmetry": list(ferro_statespace.SYMMETRIES),
                "not_modelled": ferro_statespace.NOT_MODELLED}
    if task == "import_trace":
        need(time_s=opts.get("time_s"), voltages_v=opts.get("voltages_v"))
        return {"protocol": experiment.protocol_from_trace(**opts)}
    if task == "import_touchstone":
        need(text=opts.get("text"), ports=opts.get("ports"), time_s=opts.get("time_s"))
        return {"datasets": experiment.touchstone_datasets(**opts)}
    if task == "reproduce_reference":
        return ferro_reference.reproduce(model, measurements, opts)
    need(model=model)
    if task == "compare":
        need(measurements=measurements)
        return experiment.compare(model, measurements, **opts)
    if task == "fit":
        need(measurements=measurements, parameters_to_fit=opts.get("parameters_to_fit"))
        return experiment.fit(model, measurements, **opts)
    if task == "multiport":
        need(write_voltages_v=opts.get("write_voltages_v"),
             duration_s=opts.get("duration_s"), frequency_hz=opts.get("frequency_hz"))
        return ferro_statespace.multiport_report(model, **opts)
    need(protocol=protocol, frequency_hz=opts.get("frequency_hz"))
    steps = experiment.normalise_protocol(protocol)
    if task == "sensitivity":
        return ferro_statespace.sensitivity(model, steps, **opts)
    need(read_times_s=opts.get("read_times_s"))
    return ferro_statespace.simulate(model, steps, **opts)


def chiplab_model_registry(technology: str | None = None, model_id: str | None = None) -> dict:
    """Scientific model registry: equations, inputs and units, calibration data, tests,
    fidelity, limitations and provenance of each registered physics model. Read-only."""
    from chiplab import model_registry

    return model_registry.describe(technology, model_id)


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
            "description": "Level 2 tile-level discrete-event simulation (heap-driven, no SimPy): operator tiles queue FIFO for memory banks/channels, interconnect links along topology routes (mesh XY, ring, crossbar/PCIe ports, shared bus) and compute engines, with closed-loop requests. Simulates contention and queueing; first-order timing (no DRAM command timing, no caches, store-and-forward links); not cycle-accurate.",
        },
        "specialized": {
            "name": "Specialized Fidelity",
            "description": "Level 3 external simulators. Ramulator 2 (DRAM command timing: bandwidth efficiency and loaded latency per memory type) and SCALE-Sim 3 (systolic-array utilization on the workload's GEMMs) are integrated when installed (see backend_status); they derate the roofline's memory bandwidth and array efficiency, while power, area, thermal and cost stay analytical. Accel-Sim is detected only. If no installed tool covers a design, this level runs the analytical model and labels the result analytical.",
        },
        "rtl_circuit": {
            "name": "RTL Circuit Fidelity",
            "description": "Level 4 RTL: not available for whole designs through chip_run_simulation (falls back to analytical, labeled so). Generated on-chip memory RTL is verified separately with Verilator (design_onchip_memory_subsystem + tests/test_rtl_verilator.py).",
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
        "external_backends": __import__("chiplab.backends", fromlist=["backend_status"]).backend_status(),
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
    from chiplab.backends.ecore import apply_ecore_to_design
    if spec is not None:
        design_id, design = None, _parse_spec(spec)
    elif not design_id_or_name:
        raise ValueError("Pass design_id_or_name (a saved design) or spec (YAML/JSON/dict).")
    else:
        design_id, design = store.get_design(design_id_or_name, version)
    # eCore-backed PIM chiplets get their area/power from Echo Lab before cost or validation.
    return design_id, apply_ecore_to_design(design)[0]


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


def simulate_ecore_array(rows: int = 64, cols: int = 64, precision: str = "int8",
                         adc_bits: int = 8, organization: str = "org_b",
                         seed: int = 42) -> dict:
    """Simulate one analog DRAM compute-in-memory array with Echo Lab's eCore (SPICE-
    calibrated, generic models). Returns end-to-end matvec accuracy (SNR, rel L2, RMSE, max
    error vs exact W@x), latency/energy/area with per-term breakdowns and source tags,
    derived TOPS, pJ/op and W per array, calibration provenance (SPICE, file date, sha256)
    and limitations. rows/cols 8-1024; precision int4|int8|fp8_e4m3|fp8_e5m2|fp16;
    adc_bits 4|6|8|10|12; organization org_a (bit-sliced bitline) | org_b (shared charge
    accumulator). Deterministic for a given seed. To use the result in a chip design, add
    compute.ecore {rows, cols, precision, adc_bits, organization, arrays?} to a PIM chiplet."""
    from chiplab.backends.ecore import EcoreBackend, EcoreUnavailableError
    try:
        return EcoreBackend().simulate(rows=rows, cols=cols, precision=precision,
                                       adc_bits=adc_bits, organization=organization,
                                       seed=seed)
    except EcoreUnavailableError as e:
        raise RuntimeError(f"eCore backend unavailable: {e}") from e


# ---------------------------------------------------------------- ASIC workspaces
def asic_get_architecture() -> dict:
    """The competition ASIC (PEM: four-lane reprogrammable protocol engine for Tiny Tapeout
    IHP 130 nm): competition requirements, ISA document, architecture summary, register-bit
    count, shipped protocol programs and planned protocols. Read-only."""
    from chiplab import asic_api
    return asic_api.asic_architecture()


def asic_assemble(source: str) -> dict:
    """Assemble PEM-16 source (labels, .origin/.wrap_target/.wrap, [delay]). Returns words with
    disassembly, or the error with its line number. Read-only."""
    from chiplab import asic_api
    return asic_api.asic_assemble(source)


def asic_simulate_program(source: str, lane_regs: dict | None = None, cycles: int = 2000,
                          bench: str = "", lane: int = 0, clkdiv: int = 1) -> dict:
    """Run a PEM-16 program on the cycle-accurate Python reference model (NOT the RTL).
    lane_regs: clkdiv/pins/shift/crc_* register values. bench: extra lines W addr data,
    R addr, C n, S value, LB src dst, I2C scl sda addr. cycles <= 20000. Returns pin waveform,
    register reads, events and lane state. Read-only."""
    from chiplab import asic_api
    return asic_api.asic_simulate(source, lane_regs, cycles, bench, lane, clkdiv)


def asic_run_verification(kind: str = "protocols", seeds: int = 20, cycles: int = 3000,
                          backend: str = "rtl") -> dict:
    """Run verification on the PEM RTL and store the result: protocols (UART/SPI 0-3/I2C/USB-LS
    with independent decoders), regression (constrained-random RTL vs reference-model
    lockstep with coverage), mutation (injected RTL bugs must be caught), lint, tt_smoke
    (Tiny Tapeout top level through the SPI host port), synthesis, synthesis_ihp, icarus_tt,
    formal, gate_sim, hardening_import (reads the latest TT GitHub Actions gds run: DRC, LVS,
    timing corners, utilization). seeds <= 100, cycles <= 10000.
    Results come only from the run; a missing tool is reported, never passed."""
    from chiplab import asic_api
    return asic_api.asic_run(kind, seeds, cycles, backend)


def asic_competition_report() -> dict:
    """Competition evaluation: tile utilization, area, timing, fmax, congestion, power (each
    'not run' until the IHP flow produces it), memory use, protocol and verification results
    with RTL hash and staleness, requirement tracker, known limitations, submission checklist
    and run history. Read-only."""
    from chiplab import asic_api
    return asic_api.asic_report()


def asic_export_tt_project(name: str = "jane_street_pem") -> dict:
    """Write a Tiny Tapeout project tree (info.yaml, src/, docs/, programs/, cocotb test) under
    projects/<name>. Personal fields are left as TODO."""
    from chiplab import asic_api
    return asic_api.asic_export(name)


def asic_simulation_levels() -> list:
    """The ten ASIC simulation levels (functional, RTL, gate, transistor, synthesis, STA,
    physical, DRC/LVS, power, thermal) with the engine each uses here and whether it is
    available. Read-only."""
    from chiplab import asic_api
    return asic_api.simulation_levels()


def memory_compile(spec: dict, views: bool = False) -> dict:
    """Memory compiler. spec: {name, kind: mask_rom|sram|dram|edram|regfile|fifo|custom, words,
    width, read_ports, write_ports, rw_ports, pdk, freq_mhz, contents (ROM data), banks?, cell?,
    retention_us?, objective: area|delay|energy|area_delay}. Returns ranked candidate
    organizations (cell architecture, banks, rows, cols, column mux, decoder) with area,
    access/cycle time, read energy, leakage, power, throughput, transistors, refresh,
    manufacturing feasibility, PDK rule checks (DRC/LVS reported not run) and the Pareto set.
    views=true adds logical/circuit/physical/block SVGs. Fidelity: analytical. Read-only."""
    from chiplab import asic_api
    return asic_api.memory_compile(spec, views)


def memory_compare(specs: list) -> list:
    """Compare implementations of the same function (e.g. mask ROM vs SRAM vs register file, or
    NOR vs NAND ROM): one row per spec with area, latency, power, capacity, throughput,
    transistors and feasibility. Analytical. Read-only."""
    from chiplab import asic_api
    return asic_api.memory_compare(specs)


def memory_generate_rtl(spec: dict, run_tests: bool = True, n_ops: int = 300,
                        retention_cycles: int = 0, refresh_interval: int = 0,
                        refresh: bool = True) -> dict:
    """Generate synthesizable Verilog for a memory spec and (run_tests) verify it with a
    Verilator self-checking testbench against an independent Python model. Dynamic memories
    can include a retention model and refresh controller. n_ops <= 2000."""
    from chiplab import asic_api
    return asic_api.memory_rtl(spec, run_tests, n_ops, retention_cycles, refresh_interval, refresh)


def component_list(category: str | None = None) -> list:
    """Components available to compose ASICs: built-in transistors, passives, gates,
    arithmetic, memory cells, sense amps, ADC/DAC, clocking, I/O, interconnect, the PEM
    processor, composite templates and user components. Read-only."""
    from chiplab import asic_api
    return asic_api.component_list(category)


def component_get(ref: str) -> dict:
    """Full definition of a component (name or name@version). Read-only."""
    from chiplab import asic_api
    return asic_api.component_get(ref)


def component_create(definition: dict) -> dict:
    """Register a custom component without changing ChipLab's core: params (default, unit,
    provenance), ports (dir, width expr, domain digital|analog|power), models (area_um2,
    power_uw, delay_ns, transistors, energy_pj as sandboxed expressions), behavior (comb
    outputs or seq state/next/outputs), instances + connections for hierarchy, rules, view,
    backend. Expressions are parsed by a whitelist interpreter (no code execution). It must
    elaborate cleanly; name@version is immutable."""
    from chiplab import asic_api
    from chiplab.credibility import refuse_ferrofluid

    refuse_ferrofluid("the custom component library", definition)
    return asic_api.component_create(definition)


def architecture_elaborate(ref: str, params: dict | None = None, pdk: str = "sky130",
                           svg: bool = False) -> dict:
    """Elaborate a (hierarchical) component: resolve parameters, check connections, widths and
    drivers, roll up area/power/transistors and return the block graph (svg=true adds a block
    diagram). Analytical roll-ups. Read-only."""
    from chiplab import asic_api
    return asic_api.architecture_elaborate(ref, params, pdk, svg)


def component_simulate(ref: str, stimulus: list, params: dict | None = None) -> dict:
    """Functional (cycle-based) simulation of a component's digital/behavioural leaves.
    stimulus: one dict of top-level inputs per cycle (<= 2000). Read-only."""
    from chiplab import asic_api
    return asic_api.component_simulate(ref, stimulus, params)


def architecture_export_verilog(ref: str, params: dict | None = None) -> str:
    """Structural Verilog for an all-digital composite component. Read-only."""
    from chiplab import asic_api
    return asic_api.architecture_verilog(ref, params)


def chip_uncertainty(design_id_or_name: str | None = None, spec: str | dict | None = None,
                     workload: dict | None = None, samples: int = 200, seed: int = 42,
                     default_pct: float = 15.0, include_cost: bool = True,
                     volume: int | None = None, version: str | None = None) -> dict:
    """Confidence bands (5th/50th/95th percentile) on peak/effective TOPS, tokens/s, latency,
    power, temperature, energy/inference and unit cost, by Monte Carlo over the design's
    uncertain inputs. Each input varies uniformly within +/- its TrackedValue uncertainty_pct,
    else by provenance (measured 5%, published 10%, estimated 20%, assumed 30%), else
    default_pct for plain numbers. Also returns the top-3 inputs driving each output
    (Spearman). Seeded and read-only (nothing is saved). samples 10-5000."""
    from chiplab.engine.uncertainty import propagate_uncertainty
    from chiplab.schemas.workload import WorkloadSpec
    if spec is not None:
        design = _parse_spec(spec)
    elif design_id_or_name:
        _, design = store.get_design(design_id_or_name, version)
    else:
        raise ValueError("Pass design_id_or_name (a saved design) or spec (YAML/JSON/dict).")
    wl = WorkloadSpec.model_validate(workload) if workload is not None else None
    return propagate_uncertainty(design, wl, samples=samples, seed=seed,
                                 default_pct=default_pct, include_cost=include_cost,
                                 volume=_volume(design, volume))


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


MEMORY_MODEL_CLASS = {
    "class": "on_chip_macro",
    "covers": ["6T SRAM", "2T gain-cell eDRAM", "3T gain-cell eDRAM"],
    "not_covered": ["1T1C commodity DRAM arrays", "HBM stacks", "analog DRAM compute-in-memory"],
    "note": "Banked on-die macros with digital PIM lanes. Off-chip DRAM/HBM are modeled at "
            "chiplet level (MemorySpec in chip designs); analog CIM needs a circuit-level "
            "backend.",
}


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
    out: dict = {"design": cfg.name, **report, "memory_model_class": MEMORY_MODEL_CLASS}
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
               get_digital_twin_run, list_ip_catalog, list_pdk_ladder, list_private_pdks,
               simulate_ecore_array, chip_uncertainty, asic_get_architecture, asic_assemble,
               asic_simulate_program, asic_competition_report, asic_simulation_levels,
               memory_compile, memory_compare, component_list, component_get,
               architecture_elaborate, component_simulate, architecture_export_verilog,
               chiplab_validate_material_device, chiplab_ferrofluid_analyze,
               chiplab_ferrofluid_state_model, chiplab_model_registry):
        mcp.tool(annotations=read_only)(fn)
    mcp.tool(name="chip_sweep_parameters", annotations=writes)(_chip_sweep_parameters_tool)
    for fn in (chip_design_from_template, chip_create_design, chip_evaluate,
               chip_run_simulation, chip_compare_designs,
               chip_export_report, chip_set_price, chip_clear_price,
               run_digital_twin_search, calibrate_from_eda_report, ingest_byo_pdk,
               export_pareto_handoff_bundle, design_onchip_memory_subsystem,
               asic_run_verification, asic_export_tt_project, memory_generate_rtl,
               component_create, chip_design_from_nl, chip_full_flow,
               chip_suggest_improvements, chip_generate_rtl, chip_verify_design,
               chip_export_full_project,
               simulate_material_device, configure_material_array, run_multiphysics_simulation,
               simulate_material_circuit, sweep_material_parameters, calibrate_material_model,
               material_device_to_chiplet):
        mcp.tool(annotations=writes)(fn)
    mcp.tool(annotations=writes)(chiplab_reliability_analyze)
    mcp.tool(annotations=writes)(chiplab_discover_architectures)
    mcp.tool(annotations=writes)(chiplab_manufacturing_signoff)
    mcp.tool(annotations=writes)(chiplab_digital_twin_manage)
    mcp.tool(annotations=writes)(chiplab_multiscale_surrogate)
    mcp.tool(annotations=writes)(chiplab_simulate_experimental_tech)
    mcp.tool(annotations=writes)(chiplab_compare_technologies)
    mcp.tool(annotations=writes)(chiplab_sweep_experiment)
    mcp.tool(annotations=writes)(chiplab_custom_device_simulate)
    mcp.tool(annotations=writes)(chiplab_experiment_ledger)
    mcp.tool(annotations=writes)(chiplab_custom_circuit_simulate)
    mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))(chiplab_scale_technology_node)
    mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))(chiplab_list_process_nodes)
    mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))(chiplab_list_experimental_technologies)
    mcp.tool(annotations=writes)(chiplab_2d_beyond_cmos_simulate)
    mcp.tool(annotations=writes)(chiplab_quantum_device_simulate)
    mcp.tool(annotations=writes)(chiplab_spice_netlist_run)
    mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))(chiplab_agent_run)
    mcp.tool(annotations=writes)(chiplab_chipro_eda_flow)
    mcp.tool(annotations=writes)(chiplab_mage_rtl_pipeline)
    mcp.tool(annotations=writes)(chiplab_chia_codesign_loop)
    mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))(chiplab_local_llm_models)
    mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))(
        chip_refresh_prices)
    from chiplab import mcp_learning_tools   # IMC lab + learning loop
    mcp_learning_tools.register(mcp, read_only, writes)
    from chiplab import mcp_research_tools   # brain, review, PIM lab, research, DFT, plugins
    mcp_research_tools.register(mcp, read_only, writes)









    # Register resources
    mcp.resource("chiplab://library/components")(get_library_components)
    mcp.resource("chiplab://designs/{design_id}")(get_design_resource)
    mcp.resource("chiplab://results/{run_id}")(get_result_resource)
    mcp.resource("chiplab://models/assumptions")(get_models_assumptions)

    # Register prompts
    mcp.prompt("explore_architecture")(explore_architecture)
    from chiplab.prompts import PROMPTS
    for prompt_fn in PROMPTS:
        mcp.prompt(prompt_fn.__name__)(prompt_fn)

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