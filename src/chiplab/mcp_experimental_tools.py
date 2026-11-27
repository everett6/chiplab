"""MCP experimental tools for ChipLab - flexible JSON-serializable simulation tools.

Provides 5 tools for technology comparison, sweeping, custom device simulation,
and experiment ledger management.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Technology simulation helpers (lightweight stand-ins; replace with real
# chiplab models when available)
# ---------------------------------------------------------------------------

MEMRISOR_METRICS = {
    "energy_per_bit_j": 5e-15,
    "latency_ns": 20,
    "endurance_cycles": 1e10,
    "retention_s": 1e6,
    "area_density_gb_cm2": 5,
    "thermal_sensitivity_ppm_K": 150,
}

PCM_METRICS = {
    "energy_per_bit_j": 2e-14,
    "latency_ns": 100,
    "endurance_cycles": 1e8,
    "retention_s": 1e9,
    "area_density_gb_cm2": 20,
    "thermal_sensitivity_ppm_K": 300,
}

FERROELECTRIC_METRICS = {
    "energy_per_bit_j": 8e-15,
    "latency_ns": 30,
    "endurance_cycles": 1e11,
    "retention_s": 1e2,
    "area_density_gb_cm2": 8,
    "thermal_sensitivity_ppm_K": 250,
}

SPINTORIC_METRICS = {
    "energy_per_bit_j": 3e-15,
    "latency_ns": 15,
    "endurance_cycles": 1e13,
    "retention_s": 1e9,
    "area_density_gb_cm2": 3,
    "thermal_sensitivity_ppm_K": 100,
}

PHOTONIC_METRICS = {
    "energy_per_bit_j": 5e-16,
    "latency_ns": 5,
    "endurance_cycles": 1e15,
    "retention_s": None,  # non-volatile not inherent
    "area_density_gb_cm2": 50,
    "thermal_sensitivity_ppm_K": 50,
}

CUSTOM_METRICS_TEMPLATE = {
    "energy_per_bit_j": 1e-14,
    "latency_ns": 100,
    "endurance_cycles": 1e6,
    "retention_s": 1e3,
    "area_density_gb_cm2": 5,
    "thermal_sensitivity_ppm_K": 200,
}


def _make_design_spec(name: str = "experimental", version: str = "0.1") -> Any:
    """Create a minimal DesignSpec for ledger logging."""
    from chiplab.schemas.design import DesignSpec

    return DesignSpec(
        name=name,
        version=version,
        architecture="experimental",
        chiplets=[],
        description=f"Experimental: {name}",
    )


def _simulate_technology_metrics(
    technology: str,
    params: Optional[Dict[str, Any]],
    pulse_schedule: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Return metrics dict for a given technology.

    Gracefully handles unknown tech by falling back to a custom template.
    User params always override the default metrics.
    """
    tech_lower = technology.lower().strip()

    # Map known technologies to metrics - start with defaults
    metrics_map = {
        "memristor": dict(MEMRISOR_METRICS),
        "pcm": dict(PCM_METRICS),
        "ferroelectric": dict(FERROELECTRIC_METRICS),
        "spintronic": dict(SPINTORIC_METRICS),
        "photonic": dict(PHOTONIC_METRICS),
    }

    if tech_lower in metrics_map:
        metrics = metrics_map[tech_lower]
    else:
        # Custom or unknown technology: use template and merge any user params
        metrics = dict(CUSTOM_METRICS_TEMPLATE)

    # User params always override defaults
    if params:
        metrics.update(params)

    # Apply pulse schedule if provided (simple heuristic: adjust latency)
    if pulse_schedule:
        total_pulse_time_ns = 0
        for step in pulse_schedule:
            duration = step.get("duration_ns", 0)
            amplitude = step.get("amplitude", 1.0)
            total_pulse_time_ns += duration * abs(amplitude)
        # Reduce latency proportionally to total pulse energy
        if total_pulse_time_ns > 0:
            metrics["latency_ns"] = max(1, int(metrics.get("latency_ns", 100) * 0.8))

    # Ensure all expected keys are present
    for key in CUSTOM_METRICS_TEMPLATE:
        metrics.setdefault(key, CUSTOM_METRICS_TEMPLATE[key])

    return metrics


# ---------------------------------------------------------------------------
# Tool 1: chiplab_simulate_experimental_tech
# ---------------------------------------------------------------------------


def chiplab_simulate_experimental_tech(
    technology: str,
    params: Optional[Dict[str, Any]] = None,
    pulse_schedule: Optional[List[Dict[str, Any]]] = None,
    device: str = "auto",
    node_nm: Optional[float] = None,
    corner: str = "TT",
) -> Dict[str, Any]:
    """Simulate a chipLab experimental technology and return performance metrics.

    "ferrofluid" runs a particle physics model: params accepts voltage_v,
    read_frequency_hz, duration_ns, read_delay_ns, separation_um, radius_um, turns,
    series_resistance_ohm, electrodes, ports, particle parameters (volume_fraction,
    core_diameter_nm, ...) and calibration; metrics without a model are null. Every other
    technology returns fixed assumed defaults that do not respond to physical inputs.

    Supports the following technologies:
      - "ferrofluid", "memristor", "pcm", "ferroelectric", "fefet",
        "spintronic", "mram", "ftj", "photonic", "ncfet", "tfet", "tmd_fet"
      - "custom" (any material dict provided via *params*)

    Optional *node_nm* (e.g., 180, 65, 28, 7, 3, 2, 1.4) scales the device
    to that CMOS technology node and corner ("TT", "FF", "SS", "FS", "SF").
    """
    from chiplab.models.materials.ferro_interface import routes_to_ferrofluid

    if routes_to_ferrofluid({**(params or {}), "technology": technology}):
        from chiplab.models.materials.ferro_interface import technology_summary

        params = {k: v for k, v in (params or {}).items()
                  if k not in ("technology", "material")}

        summary = technology_summary(params, pulse_schedule)
        if node_nm is not None:
            summary["simulation"]["warnings"].append(
                "node_nm and corner ignored: CMOS node scaling has no ferrofluid model")
        return {
            "technology": technology,
            "device": device,
            "node_nm": node_nm,
            "corner": corner,
            **summary,
            "provenance": {
                "model": "chiplab ferrofluid particle model (docs/ferrofluid_model.md)",
            },
        }

    metrics = _simulate_technology_metrics(technology, params, pulse_schedule)
    if node_nm is not None:
        from chiplab.models.materials.process_nodes import scale_experimental_device
        scaled = scale_experimental_device(
            technology=technology,
            target_node_nm=float(node_nm),
            base_metrics=metrics,
            corner=corner,
        )
        metrics.update(scaled)

    return {
        "technology": technology,
        "device": device,
        "node_nm": node_nm,
        "corner": corner,
        "metrics": metrics,
        "metric_status": {
            key: "user_supplied" if params and key in params else
            "assumed heuristic default, not a validated specification"
            for key in metrics},
        "provenance": {
            "model": "chiplab.experimental_tools.simulate_technology_metrics",
            "note": "heuristic placeholder values",
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: chiplab_compare_technologies
# ---------------------------------------------------------------------------


def chiplab_compare_technologies(
    technologies: List[str],
    workload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare multiple experimental technologies across key metrics.

    Compares the technologies on:
      - energy/bit (J)
      - latency (ns)
      - endurance (cycles)
      - retention (s)
      - area density (GB/cm²)
      - thermal sensitivity (ppm/K)

    Args:
        technologies: List of technology names (must be supported by
                      chiplab_simulate_experimental_tech).
        workload: Optional dict of workload parameters (e.g., {"bits": 1024}).

    Returns:
        A dict with per-technology metrics and a summary identifying winners
        for each metric category.
    """
    if not technologies:
        return {
            "error": "No technologies provided for comparison.",
            "comparisons": [],
            "winners": {},
        }

    # Simulate each technology
    results: List[Dict[str, Any]] = []
    for tech in technologies:
        sim = chiplab_simulate_experimental_tech(technology=tech)
        results.append(sim)

    # Extract metrics arrays
    metric_keys = [
        "energy_per_bit_j",
        "latency_ns",
        "endurance_cycles",
        "retention_s",
        "area_density_gb_cm2",
        "thermal_sensitivity_ppm_K",
    ]

    comparisons: List[Dict[str, Any]] = []
    winners: Dict[str, str] = {}

    for key in metric_keys:
        # Collect valid (non-None) values
        values: List[tuple[str, Any]] = []
        for r in results:
            tech = r.get("technology", "unknown")
            val = r.get("metrics", {}).get(key)
            if val is not None:
                values.append((tech, val))

        if len(values) >= 2:
            # Higher is better for these; lower is better for energy, latency, sensitivity
            is_better_higher = key in (
                "endurance_cycles", "retention_s", "area_density_gb_cm2")
            sorted_values = sorted(values, key=lambda x: x[1], reverse=is_better_higher)
            winner_tech = sorted_values[0][0]
            winner_val = sorted_values[0][1]
            winners[key] = f"{winner_tech} ({winner_val})"
            comparisons.append(
                {
                    "metric": key,
                    "technologies": {tech: r.get("metrics", {}).get(key) for tech, r in zip(technologies, results)},
                    "winner": winner_tech,
                    "value": winner_val,
                }
            )
        else:
            winners[key] = "Insufficient data"

    # Build per-technology summary
    tech_summaries: List[Dict[str, Any]] = []
    for r in results:
        tech_summaries.append(
            {
                "technology": r.get("technology"),
                "metrics": r.get("metrics", {}),
            }
        )

    return {
        "comparisons": comparisons,
        "winners": winners,
        "technologies_compared": technologies,
        "per_technology": tech_summaries,
    }


# ---------------------------------------------------------------------------
# Tool 3: chiplab_sweep_experiment
# ---------------------------------------------------------------------------


def chiplab_sweep_experiment(
    technology: str,
    sweep_param: str,
    values: List[float],
    base_params: Optional[Dict[str, Any]] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """Sweep a parameter across a technology simulation and return metrics + best point.

    Args:
        technology: Technology name (supported by chiplab_simulate_experimental_tech).
        sweep_param: Parameter name to sweep (e.g., "voltage", "current", "temperature").
        values: List of parameter values to test.
        base_params: Optional base parameters dict to combine with sweep.
        device: Device selection ("auto" by default).

    Returns:
        A dict with:
          - sweep_param: the parameter name swept
          - values: the list of values tested
          - points: list of per-value metric dicts
          - best_point: dict with the value that optimized energy-per-bit
          - best_metrics: metrics at the best point
    """
    if not values:
        return {
            "error": "No sweep values provided.",
            "sweep_param": sweep_param,
            "values": [],
            "points": [],
            "best_point": None,
            "best_metrics": None,
        }

    # Build per-value evaluations
    points: List[Dict[str, Any]] = []
    energy_per_bits: List[tuple[float, Dict[str, Any]]] = []

    for val in values:
        # Construct per-value params by updating base_params if provided
        value_params: Dict[str, Any] = {}
        if base_params:
            value_params.update(base_params)
        value_params[sweep_param] = val

        # Simulate with these params
        sim = chiplab_simulate_experimental_tech(
            technology=technology, params=value_params
        )
        metrics = sim.get("metrics", {})

        points.append(
            {
                "parameter_value": val,
                "metrics": metrics,
            }
        )
        epb = metrics.get("energy_per_bit_j", float("inf"))
        if epb is not None:  # None means not modelled
            energy_per_bits.append((val, epb))

    # Find best point: minimum energy-per-bit
    if energy_per_bits:
        best_val, best_epb = min(energy_per_bits, key=lambda x: x[1])
        # Find the first point with that best value
        best_point_data = next(
            p for p in points if p["parameter_value"] == best_val
        )
        best_metrics = best_point_data["metrics"]
    else:
        best_val = None
        best_epb = float("inf")
        best_point_data = None
        best_metrics = None

    return {
        "sweep_param": sweep_param,
        "values": values,
        "points": points,
        "best_point": {"parameter_value": best_val} if best_val is not None else None,
        "best_metrics": best_metrics,
    }


# ---------------------------------------------------------------------------
# Tool 4: chiplab_custom_device_simulate
# ---------------------------------------------------------------------------


def chiplab_custom_device_simulate(
    device_spec: Dict[str, Any],
    write_voltage_v: float = 1.0,
    read_freq_hz: float = 1e6,
    device: str = "auto",
) -> Dict[str, Any]:
    """Build and simulate a custom PhysicalDevice from a dict of regions/electrodes/materials.

    The *device_spec* dict should conform to the chiplab PhysicalDevice structure,
    containing at minimum:
      - "name": device name
      - "description": textual description
      - "regions": list of material regions, each with "material", "geometry", "boundary_conditions"
      - "electrodes": dict of electrode definitions with "port", "write_mechanism", "read_mechanism"

    A ferrofluid region runs the particle physics model: each electrode entry may set
    role, voltage_v, position_um and conductor geometry; a region geometry dict with
    center_um and size_um sets the fluid box; electrical conductance needs
    conductivity_s_per_m, relative_permittivity and electrode_radius_um. Other materials
    get a placeholder that only counts regions and electrodes.

    Args:
        device_spec: Dict specifying the custom physical device.
        write_voltage_v: Default write voltage in volts (default 1.0).
        read_freq_hz: Read frequency in Hz for impedance/sensing (default 1e6).
        device: Device selection string ("auto" by default).

    Returns:
        A dict with simulation results including:
          - device_name: the configured device name
          - write_voltage_v: the write voltage used
          - read_freq_hz: the read frequency used
          - metrics: computed device metrics (conductance, capacitance, etc.)
          - provenance: tracking of the source model.
    """
    from chiplab.models.materials import ferro_interface

    if ferro_interface.routes_to_ferrofluid(device_spec):
        result = ferro_interface.custom_device(device_spec, write_voltage_v, read_freq_hz)
        return {**result, "device": device}

    # Extract basic info
    name = device_spec.get("name", "custom_device")
    description = device_spec.get("description", "")
    regions = device_spec.get("regions", [])
    electrodes = device_spec.get("electrodes", {})

    # Simple heuristic metrics based on region count and electrode config
    num_regions = len(regions) if regions else 0
    num_electrodes = len(electrodes) if electrodes else 0

    # Derive basic metrics
    # Conductance scales with number of conductive paths
    conductance_s = num_electrodes * 1e-6  # Siemens, rough estimate
    # Capacitance scales with region count
    capacitance_f = num_regions * 1e-15  # Farads, rough estimate

    # Frequency response check
    # RC time constant
    rc_time = capacitance_f / max(conductance_s, 1e-12)
    # -3dB frequency
    f3d_hz = 1 / (2 * 3.14159 * rc_time) if rc_time > 0 else 0

    metrics = {
        "conductance_s": conductance_s,
        "capacitance_f": capacitance_f,
        "f3d_hz": f3d_hz,
        "write_voltage_v": write_voltage_v,
        "read_freq_hz": read_freq_hz,
    }

    # If we have material info, try to enrich metrics
    for region in (regions or []):
        material = region.get("material", "")
        if material.lower() == "ferroelectric":
            metrics["ferroelectric_curve"] = "switching observed"
        elif material.lower() == "memristor":
            metrics["memristance"] = "nonlinear I-V"

    return {
        "device_name": name,
        "description": description,
        "write_voltage_v": write_voltage_v,
        "read_freq_hz": read_freq_hz,
        "device": device,
        "metrics": metrics,
        "region_count": num_regions,
        "electrode_count": num_electrodes,
        "provenance": {
            "model": "chiplab.experimental_tools.chiplab_custom_device_simulate",
            "note": "Heuristic simulation; replace with PhysicalDevice.evaluate() when available.",
        },
    }


# ---------------------------------------------------------------------------
# Tool 5: chiplab_experiment_ledger
# ---------------------------------------------------------------------------


def chiplab_experiment_ledger(
    action: str,
    db_path: Optional[str] = None,
    run_data: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    run_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Wrap chiplab.experiment_db.ExperimentLedger for managing experiment runs.

    Supported actions:
      - "list": List runs, optionally filtered.
      - "log": Log a new run with run_data.
      - "get": Retrieve a specific run by ID (run_ids must contain one ID).
      - "compare": Compare multiple runs (run_ids must have >= 2 entries).
      - "summary": Export summary of runs.
      - "migrate": Mark legacy rows (no fidelity -> "unknown", duplicates -> duplicate_of)
        without deleting anything; filters={"dry_run": true} only counts.

    run_data for "log" must include fidelity (analytical, behavioral, discrete_event,
    specialized, circuit_spice, rtl_sim, synthesis, place_route, signoff, fpga, published,
    measured). Identical runs are counted as repeats, not stored twice.

    Args:
        action: One of "list", "log", "get", "compare", "summary".
        db_path: Optional path to the SQLite DB; defaults to
                 chiplab_experiments.db or env CHIPLAB_DB_PATH.
        run_data: Dict with experiment parameters and metrics (required for "log").
        filters: Dict of filters to apply when listing (e.g., {"fidelity": "analytical"}).
        run_ids: List of run IDs to operate on (required for "get"/"compare").

    Returns:
        Dict with the result of the requested action.
    """
    from chiplab.experiment_db.ledger import ExperimentLedger  # type: ignore

    from chiplab.schemas.design import DesignSpec

    from chiplab.experiment_db.ledger import FIDELITY_LEVELS, default_db_path

    ledger = ExperimentLedger(db_path=db_path or default_db_path())

    action_lower = action.lower()

    if action_lower == "list":
        # Build filters dict from the filters param for query_runs
        query_filters: Dict[str, Any] = {}
        if filters:
            if "design_name" in filters:
                query_filters["design_name"] = filters["design_name"]
            if "fidelity" in filters:
                query_filters["fidelity"] = filters["fidelity"]
            if "min_metric" in filters:
                query_filters["min_metric"] = filters["min_metric"]
            if "max_metric" in filters:
                query_filters["max_metric"] = filters["max_metric"]
            for flag in ("include_duplicates", "learnable_only"):
                if flag in filters:
                    query_filters[flag] = bool(filters[flag])
        try:
            runs = ledger.query_runs(**query_filters)
        except ValueError as e:
            return {"error": str(e)}
        return {
            "action": "list",
            "runs": runs,
            "count": len(runs),
            "db_path": ledger.db_path,
        }

    if action_lower == "log":
        if run_data is None:
            return {"error": "run_data is required for 'log' action."}
        if run_data.get("fidelity") not in FIDELITY_LEVELS:
            return {"error": "run_data.fidelity is required: one of "
                             + ", ".join(FIDELITY_LEVELS),
                    "got": run_data.get("fidelity")}
        design = _make_design_spec(str(run_data.get("design_name") or "experimental"),
                                   str(run_data.get("design_version") or "0.1"))
        try:
            rec = ledger.record(
                design=design,
                params=run_data.get("params", {}),
                metrics=run_data.get("metrics", {}),
                fidelity=run_data.get("fidelity"),
                fidelity_disclaimer=run_data.get("fidelity_disclaimer"),
                run_id=run_data.get("run_id"),
                logged_by=run_data.get("logged_by"),
            )
        except ValueError as e:
            return {"error": str(e)}
        return {"action": "log", "db_path": str(ledger.db_path)} | rec

    if action_lower == "migrate":
        return {"action": "migrate", "db_path": str(ledger.db_path)} | ledger.migrate_legacy(
            dry_run=bool((filters or {}).get("dry_run", False)))

    if action_lower == "get":
        if not run_ids or len(run_ids) != 1:
            return {
                "error": "run_ids must contain exactly one ID for 'get' action.",
                "run_ids": run_ids,
            }
        run = ledger.get_run(run_ids[0])
        if run is None:
            return {"error": f"Run not found: {run_ids[0]}"}
        return {
            "action": "get",
            "run_id": run_ids[0],
            "run": run,
            "db_path": ledger.db_path,
        }

    if action_lower == "compare":
        if not run_ids or len(run_ids) < 2:
            return {
                "error": "run_ids must contain at least two IDs for 'compare' action.",
                "run_ids": run_ids,
            }
        comparison = ledger.compare_runs(run_ids=run_ids)
        return {
            "action": "compare",
            "run_ids": run_ids,
            "comparison": comparison,
            "db_path": ledger.db_path,
        }

    if action_lower == "summary":
        summary = ledger.export_summary()
        return {
            "action": "summary",
            "summary": summary,
            "db_path": ledger.db_path,
        }

    return {"error": f"Unknown action: {action}. Must be one of: list, log, get, compare, summary"}


# ---------------------------------------------------------------------------
# Tool 6: chiplab_custom_circuit_simulate
# ---------------------------------------------------------------------------


def chiplab_custom_circuit_simulate(
    circuit_name: str = "custom_circuit",
    node_nm: float = 28.0,
    corner: str = "TT",
    temperature_k: float = 300.0,
    template: Optional[str] = None,
    components: Optional[List[Dict[str, Any]]] = None,
    spice_netlist: Optional[str] = None,
    netlist: Optional[str] = None,
    analysis: str = "transient",
    t_stop_s: float = 1e-9,
    dt_s: float = 2e-11,
    compare_nodes_nm: Optional[List[float]] = None,
    monte_carlo_runs: int = 0,
    device: str = "auto",
    n_stages: int = 5,
    load_cap_ff: float = 5.0,
    tech: str = "memristor",
    n_rows: int = 8,
) -> Dict[str, Any]:
    """Build and simulate a custom mixed-technology circuit at any nm process node.

    Provide ONE of:
      - *template*: "cmos_inverter", "ring_oscillator", "sram_6t", "1t1r", "crossbar_column"
      - *components*: list of dicts {name, type, nodes, params?, width_nm?, length_nm?, node_nm?}
      - *spice_netlist* (or *netlist*): SPICE netlist string (R, C, L, V, I, M elements)

    Supports any *node_nm* (180 nm down to 1 nm), *corner* ("TT", "FF", "SS", "FS", "SF"),
    *compare_nodes_nm* for multi-nm scaling comparison, and *monte_carlo_runs* for Pelgrom Vt variability.
    """
    from chiplab.models.materials.custom_circuit import (
        CustomCircuit,
        build_1t1r_cell,
        build_6t_sram_cell,
        build_cmos_inverter,
        build_hybrid_crossbar_column,
        build_ring_oscillator,
    )

    if spice_netlist is None and netlist:
        spice_netlist = str(netlist)

    if spice_netlist:
        circ = CustomCircuit.from_spice_netlist(spice_netlist, node_nm=node_nm, corner=corner)
        circ.name = circuit_name
        circ.set_process_node(node_nm, corner=corner, temperature_k=temperature_k)
    elif template:
        t_low = template.lower().strip()
        if t_low in ("cmos_inverter", "inverter"):
            circ = build_cmos_inverter(node_nm=node_nm, corner=corner, load_cap_ff=load_cap_ff)
        elif t_low in ("ring_oscillator", "rosc"):
            stages = max(3, int(n_stages))
            if stages % 2 == 0:
                stages += 1
            circ = build_ring_oscillator(n_stages=stages, node_nm=node_nm, corner=corner)
        elif t_low in ("sram_6t", "6t_sram"):
            circ = build_6t_sram_cell(node_nm=node_nm, corner=corner)
        elif t_low in ("1t1r", "1t1r_memristor", "1t1r_mtj", "1t1r_pcm"):
            t_val = t_low.split("_")[1] if "_" in t_low and len(t_low.split("_")) > 1 else tech
            circ = build_1t1r_cell(node_nm=node_nm, tech=t_val, corner=corner)
        elif t_low in ("crossbar_column", "crossbar"):
            rows = max(1, int(n_rows))
            circ = build_hybrid_crossbar_column(n_rows=rows, node_nm=node_nm, tech=tech, corner=corner)
        else:
            return {"error": f"Unknown template '{template}'."}
        circ.set_process_node(node_nm, corner=corner, temperature_k=temperature_k)
    else:
        circ = CustomCircuit(name=circuit_name, node_nm=node_nm, corner=corner, temperature_k=temperature_k)
        for idx, c in enumerate(components or []):
            circ.add_component(
                name=c.get("name", f"dev_{idx}"),
                comp_type=c["type"],
                nodes=c["nodes"],
                params=c.get("params", {}),
                width_nm=c.get("width_nm"),
                length_nm=c.get("length_nm"),
                node_nm=c.get("node_nm"),
            )

    if not circ.components:
        return {"error": "No components in circuit. Provide template, components, or spice_netlist."}

    metrics: Dict[str, Any] = {
        "node_nm": circ.node_nm.value,
        "corner": circ.process_node.corner.value,
        "vdd_v": round(float(circ.process_node.vdd.value), 4),
    }

    result: Dict[str, Any] = {
        "circuit_name": circ.name,
        "node_nm": circ.node_nm.value,
        "architecture": circ.process_node.architecture.value,
        "corner": circ.process_node.corner.value,
        "spice_netlist": circ.to_spice_netlist(),
    }

    is_rosc = bool(template and template.lower().strip() in ("ring_oscillator", "rosc"))
    if analysis.lower() == "dc":
        dc_res = circ.simulate_dc(device=device)
        result["dc"] = dc_res
        metrics["static_power_w"] = dc_res.get("static_power_w")
    else:
        tran = circ.simulate_transient(t_stop=t_stop_s, dt=dt_s, device=device)
        # Keep summary + final values compact; include waveforms
        result["transient"] = tran
        metrics["avg_power_w"] = round(float(tran.get("avg_power_w", 0.0)), 6)
        metrics["peak_power_w"] = round(float(tran.get("peak_power_w", 0.0)), 6)
        metrics["total_energy_j"] = float(tran.get("total_energy_j", 0.0))
        if tran.get("delay_ps") is not None:
            metrics["delay_ps"] = round(float(tran["delay_ps"]), 4)
            if is_rosc:
                metrics["fosc_ghz"] = round(1000.0 / max(2.0 * int(n_stages) * float(tran["delay_ps"]), 1e-6), 4)

    result["metrics"] = metrics

    if monte_carlo_runs and monte_carlo_runs > 0:
        result["monte_carlo"] = circ.simulate_monte_carlo(
            n_runs=int(monte_carlo_runs), t_stop=t_stop_s, dt=dt_s, device=device
        )

    if compare_nodes_nm:
        result["node_comparison"] = circ.compare_across_nodes(
            nodes_nm=[float(n) for n in compare_nodes_nm],
            t_stop=t_stop_s,
            dt=dt_s,
            corner=corner,
            device=device,
        )

    return result


# ---------------------------------------------------------------------------
# Tool 7: chiplab_scale_technology_node
# ---------------------------------------------------------------------------


def chiplab_scale_technology_node(
    technology: str = "cmos",
    nodes_nm: Optional[List[float]] = None,
    corner: str = "TT",
    temperature_k: float = 300.0,
) -> Dict[str, Any]:
    """Compare or sweep CMOS or experimental device metrics across nm process nodes.

    Examples:
      - technology="cmos", nodes_nm=[130, 65, 28, 14, 7, 5, 3, 2, 1.4]
      - technology="memristor", nodes_nm=[65, 28, 14, 7, 3]
    """
    from chiplab.models.materials.process_nodes import sweep_technology_nodes

    if not nodes_nm:
        nodes_nm = [180.0, 130.0, 65.0, 28.0, 14.0, 7.0, 5.0, 3.0, 2.0]
    return sweep_technology_nodes(
        nodes_nm=[float(n) for n in nodes_nm],
        technology=technology,
        corner=corner,
        temperature_k=temperature_k,
    )


# ---------------------------------------------------------------------------
# Tool 8: chiplab_list_process_nodes
# ---------------------------------------------------------------------------


def _serialize_process_node(node: Any) -> Dict[str, Any]:
    """Convert a ProcessNodeParams instance into a JSON-serializable summary."""
    return {
        "pdk_key": node.pdk_key,
        "node_nm": float(node.node_nm.value),
        "architecture": node.architecture.value,
        "corner": node.corner.value,
        "temperature_k": float(node.temperature_k.value),
        "vdd_v": float(node.vdd.value),
        "vth_n_v": float(node.vth_n.value),
        "vth_p_v": float(node.vth_p.value),
        "gate_length_nm": float(node.gate_length_nm.value),
        "eot_nm": float(node.eot_nm.value),
        "subthreshold_swing_mv_dec": float(node.subthreshold_swing_mv_dec.value),
        "fo4_delay_ps": float(node.fo4_delay_ps.value),
        "gate_density_kgates_mm2": float(node.gate_density_kgates_mm2.value),
        "dynamic_energy_fj_per_gate": float(node.dynamic_energy_fj_per_gate.value),
        "leakage_nw_per_gate": float(node.leakage_nw_per_gate.value),
        "provenance": node.vdd.provenance.value,
    }


def chiplab_list_process_nodes(
    corner: str = "TT",
    temperature_k: float = 300.0,
) -> Dict[str, Any]:
    """List all PDK_LADDER nodes (180nm -> 2nm) plus interpolated/sub-2nm nodes."""
    from chiplab.library.pdk_library import PDK_LADDER
    from chiplab.models.materials.process_nodes import get_process_node

    pdk_nodes: List[Dict[str, Any]] = []
    for pdk_key, prof in sorted(PDK_LADDER.items(), key=lambda kv: -kv[1].node_nm):
        node = get_process_node(
            node_nm=prof.node_nm,
            corner=corner,
            temperature_k=temperature_k,
            pdk_key=pdk_key,
        )
        pdk_nodes.append(_serialize_process_node(node))

    extra_nm = [90.0, 45.0, 22.0, 14.0, 10.0, 5.0, 1.4, 1.0]
    interpolated_nodes: List[Dict[str, Any]] = []
    for nm in extra_nm:
        node = get_process_node(
            node_nm=nm,
            corner=corner,
            temperature_k=temperature_k,
        )
        interpolated_nodes.append(_serialize_process_node(node))

    return {
        "corner": corner.upper(),
        "temperature_k": float(temperature_k),
        "pdk_nodes": pdk_nodes,
        "interpolated_nodes": interpolated_nodes,
        "total_count": len(pdk_nodes) + len(interpolated_nodes),
    }


# ---------------------------------------------------------------------------
# Tool 9: chiplab_list_experimental_technologies
# ---------------------------------------------------------------------------


def chiplab_list_experimental_technologies() -> Dict[str, Any]:
    """List all experimental, emerging, quantum, 2D, and neuromorphic technologies."""
    categories: Dict[str, List[Dict[str, Any]]] = {
        "Neuromorphic/Memory": [
            {
                "name": "memristor",
                "module": "chiplab.models.materials.memristor",
                "description": "ReRAM / memristor crossbar VMM and filament switching",
            },
            {
                "name": "pcm",
                "module": "chiplab.models.materials.pcm",
                "description": "Phase-change memory crystallization, melt-quench, and drift",
            },
            {
                "name": "ferroelectric",
                "module": "chiplab.models.materials.ferroelectric",
                "description": "Ferroelectric FeFET/FeRAM Landau-Khalatnikov polarization dynamics",
            },
            {
                "name": "spintronic",
                "module": "chiplab.models.materials.spintronic",
                "description": "STT/SOT-MRAM switching and TMR read resistance",
            },
            {
                "name": "ftj",
                "module": "chiplab.models.materials.ftj",
                "description": "Ferroelectric Tunnel Junction synaptic crossbar and retention",
            },
            {
                "name": "stdp_synapse",
                "module": "chiplab.models.materials.stdp_synapse",
                "description": "Spike-timing-dependent plasticity (STDP) neuromorphic synapses",
            },
            {
                "name": "liquid_state",
                "module": "chiplab.models.materials.liquid_state",
                "description": "Liquid State Machine / physical reservoir computing",
            },
            {
                "name": "oscillatory_nn",
                "module": "chiplab.models.materials.oscillatory_nn",
                "description": "Oscillatory neural networks and Ising phase-locking solvers",
            },
            {
                "name": "domain_wall",
                "module": "chiplab.models.materials.domain_wall",
                "description": "Magnetic domain wall racetrack memory and skyrmion dynamics",
            },
            {
                "name": "photonic",
                "module": "chiplab.models.materials.photonic",
                "description": "Photonic MZI mesh VMM, microring weight banks, and optical links",
            },
            {
                "name": "ferrofluid",
                "module": "chiplab.models.materials.ferrofluid",
                "description": "Multi-port magnetic nanoparticle suspension: Langevin "
                               "magnetization, Neel/Brownian relaxation, coil coupling",
                "model_status": "physics-based, uncalibrated unless fitted to user data",
            },
        ],
        "Quantum/Cryogenic": [
            {
                "name": "transmon",
                "module": "chiplab.models.materials.transmon",
                "description": "Superconducting transmon qubit spectrum, Rabi/Ramsey, and gates",
            },
            {
                "name": "fluxonium",
                "module": "chiplab.models.materials.fluxonium",
                "description": "Fluxonium qubit Hamiltonian, wavefunctions, and 0-pi protection",
            },
            {
                "name": "majorana",
                "module": "chiplab.models.materials.majorana",
                "description": "Majorana zero modes topological gap, overlap, and braiding",
            },
            {
                "name": "sfq_logic",
                "module": "chiplab.models.materials.sfq_logic",
                "description": "Single Flux Quantum (RSFQ/ERSFQ/RQL) cryogenic logic",
            },
            {
                "name": "adiabatic_qc",
                "module": "chiplab.models.materials.adiabatic_qc",
                "description": "Adiabatic quantum computing, QAOA, and Ising annealing",
            },
        ],
        "2D/Beyond-CMOS": [
            {
                "name": "tmd_fet",
                "module": "chiplab.models.materials.tmd_fet",
                "description": "2D TMD monolayer FETs (MoS2, WSe2, WS2, MoSe2) with quantum capacitance",
            },
            {
                "name": "ncfet",
                "module": "chiplab.models.materials.ncfet",
                "description": "Negative Capacitance FET (FE-HfZrO2) with sub-60 mV/dec swing",
            },
            {
                "name": "tfet",
                "module": "chiplab.models.materials.tfet",
                "description": "Band-to-band tunneling FET (BTBT) steep-slope switches",
            },
            {
                "name": "graphene_pn",
                "module": "chiplab.models.materials.graphene_pn",
                "description": "Graphene p-n junction Klein tunneling and GNR bandgap FET",
            },
        ],
        "Custom Circuits & Process Nodes": [
            {
                "name": "process_nodes",
                "module": "chiplab.models.materials.process_nodes",
                "description": "Multi-node CMOS scaling (180nm -> 1nm), corners, and Pelgrom variability",
            },
            {
                "name": "custom_circuit",
                "module": "chiplab.models.materials.custom_circuit",
                "description": "Mixed-technology custom circuit & SPICE netlist DC/transient simulator",
            },
        ],
    }

    total = sum(len(items) for items in categories.values())
    return {
        "categories": categories,
        "total_technologies": total,
        "mcp_dispatch_tools": {
            "2d_beyond_cmos": ["tmd_fet", "ncfet", "tfet", "graphene_pn"],
            "quantum_cryogenic": ["transmon", "fluxonium", "majorana", "sfq_logic", "adiabatic_qc"],
            "experimental_memory_neuromorphic": [
                "ferrofluid", "memristor", "pcm", "ferroelectric", "spintronic", "photonic", "custom"
            ],
        },
    }


# ---------------------------------------------------------------------------
# Tool 10: chiplab_2d_beyond_cmos_simulate
# ---------------------------------------------------------------------------


def _compute_curve_metrics(
    vgs_grid: np.ndarray,
    ids_curve: np.ndarray,
    vdd_v: float,
) -> tuple[float, float, float, float]:
    """Compute Ion, Ioff, on/off ratio, and subthreshold swing (mV/dec) from an Id-Vgs curve."""
    abs_ids = np.abs(np.asarray(ids_curve, dtype=np.float64))
    i_off = float(max(float(abs_ids[0]), 1e-18))
    i_on = float(max(float(abs_ids[-1]), i_off))
    on_off_ratio = float(i_on / i_off)

    decades = float(np.log10(max(i_on / i_off, 1.0)))
    ss_mv_dec = float((vdd_v * 1e3) / max(decades, 0.5))
    return i_on, i_off, on_off_ratio, ss_mv_dec


def chiplab_2d_beyond_cmos_simulate(
    technology: str,
    node_nm: float = 5.0,
    vdd_v: float = 0.7,
    vgs_v: float | None = None,
    material: str | None = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """Simulate 2D and beyond-CMOS devices: tmd_fet, ncfet, tfet, or graphene_pn."""
    from chiplab.schemas.common import Provenance, TrackedValue

    tech = technology.lower().strip()
    vgs_grid = np.linspace(0.0, float(vdd_v), 25)
    vds_grid = np.full_like(vgs_grid, float(vdd_v))

    if tech in ("tmd_fet", "tmd", "mos2", "wse2", "ws2", "mose2"):
        from chiplab.models.materials.tmd_fet import (
            TMDMaterial,
            TMDFETParams,
            tmd_fet_iv,
            tmd_quantum_capacitance,
        )

        mat_choice = TMDMaterial.MOS2
        mat_key = (material or (tech if tech in ("mos2", "wse2", "ws2", "mose2") else "")).lower().strip()
        for m in TMDMaterial:
            if mat_key and (m.name.lower() == mat_key or m.value.lower() == mat_key):
                mat_choice = m
                break

        params = TMDFETParams(
            material=mat_choice,
            gate_length_nm=TrackedValue(value=float(node_nm), provenance=Provenance.MEASURED, metadata={"unit": "nm"}),
        )
        cq = tmd_quantum_capacitance(params)
        ids_curve = tmd_fet_iv(params, vgs_grid, vds_grid, device=device)
        i_on, i_off, on_off_ratio, ss_mv_dec = _compute_curve_metrics(vgs_grid, ids_curve, float(vdd_v))
        result: Dict[str, Any] = {
            "technology": "tmd_fet",
            "material": mat_choice.value,
            "node_nm": float(node_nm),
            "vdd_v": float(vdd_v),
            "ion_a": i_on,
            "ioff_a": i_off,
            "on_off_ratio": on_off_ratio,
            "subthreshold_swing_mv_dec": ss_mv_dec,
            "quantum_capacitance_f_cm2": float(cq),
            "curve_samples": [
                {"vgs_v": float(vg), "ids_a": float(cur)} for vg, cur in zip(vgs_grid, ids_curve)
            ],
        }
        if vgs_v is not None:
            result["ids_at_vgs_a"] = float(tmd_fet_iv(params, float(vgs_v), float(vdd_v), device=device))
        return result

    if tech in ("ncfet", "fe_ncfet", "negative_capacitance"):
        from chiplab.models.materials.ncfet import (
            NCFETParams,
            ncfet_capacitance_matching,
            ncfet_iv,
        )

        params = NCFETParams(
            node_nm=TrackedValue(value=float(node_nm), provenance=Provenance.ESTIMATED, metadata={"unit": "nm"}),
        )
        match = ncfet_capacitance_matching(params)
        ids_curve = ncfet_iv(params, vgs_grid, vds_grid, device=device)
        i_on, i_off, on_off_ratio, _ = _compute_curve_metrics(vgs_grid, ids_curve, float(vdd_v))
        result = {
            "technology": "ncfet",
            "node_nm": float(node_nm),
            "vdd_v": float(vdd_v),
            "ion_a": i_on,
            "ioff_a": i_off,
            "on_off_ratio": on_off_ratio,
            "subthreshold_swing_mv_dec": float(match["subthreshold_swing_mv_dec"]),
            "capacitance_matching": match,
            "curve_samples": [
                {"vgs_v": float(vg), "ids_a": float(cur)} for vg, cur in zip(vgs_grid, ids_curve)
            ],
        }
        if vgs_v is not None:
            result["ids_at_vgs_a"] = float(ncfet_iv(params, float(vgs_v), float(vdd_v), device=device))
        return result

    if tech in ("tfet", "tunnel_fet", "btbt"):
        from chiplab.models.materials.tfet import (
            TFETMaterial,
            TFETParams,
            tfet_btbt_iv,
            tfet_subthreshold_swing,
        )

        mat_choice = TFETMaterial.INGAAS_GASB
        if material:
            mat_key = material.lower().strip()
            for m in TFETMaterial:
                if m.name.lower() == mat_key or m.value.lower() == mat_key:
                    mat_choice = m
                    break

        params = TFETParams(
            material=mat_choice,
            node_nm=TrackedValue(value=float(node_nm), provenance=Provenance.ESTIMATED, metadata={"unit": "nm"}),
        )
        ids_curve = tfet_btbt_iv(params, vgs_grid, vds_grid, device=device)
        ss_mv_dec = tfet_subthreshold_swing(params, vdd=float(vdd_v))
        i_on, i_off, on_off_ratio, _ = _compute_curve_metrics(vgs_grid, ids_curve, float(vdd_v))
        result = {
            "technology": "tfet",
            "material": mat_choice.value,
            "node_nm": float(node_nm),
            "vdd_v": float(vdd_v),
            "ion_a": i_on,
            "ioff_a": i_off,
            "on_off_ratio": on_off_ratio,
            "subthreshold_swing_mv_dec": float(ss_mv_dec),
            "curve_samples": [
                {"vgs_v": float(vg), "ids_a": float(cur)} for vg, cur in zip(vgs_grid, ids_curve)
            ],
        }
        if vgs_v is not None:
            result["ids_at_vgs_a"] = float(tfet_btbt_iv(params, float(vgs_v), float(vdd_v), device=device))
        return result

    if tech in ("graphene_pn", "graphene", "gnr", "gnr_fet"):
        from chiplab.models.materials.graphene_pn import (
            GraphenePNParams,
            gnr_bandgap_ev,
            graphene_pn_iv,
        )

        params = GraphenePNParams(
            ribbon_width_nm=TrackedValue(value=float(node_nm), provenance=Provenance.MEASURED, metadata={"unit": "nm"}),
        )
        eg_ev = gnr_bandgap_ev(float(node_nm))
        ids_list = [
            float(graphene_pn_iv(params, v_bias=float(vdd_v), v_split_gate=float(vg), device=device))
            for vg in vgs_grid
        ]
        ids_curve = np.asarray(ids_list, dtype=np.float64)
        i_on, i_off, on_off_ratio, ss_mv_dec = _compute_curve_metrics(vgs_grid, ids_curve, float(vdd_v))
        result = {
            "technology": "graphene_pn",
            "node_nm": float(node_nm),
            "vdd_v": float(vdd_v),
            "ion_a": i_on,
            "ioff_a": i_off,
            "on_off_ratio": on_off_ratio,
            "subthreshold_swing_mv_dec": ss_mv_dec,
            "gnr_bandgap_ev": float(eg_ev),
            "curve_samples": [
                {"vgs_v": float(vg), "ids_a": float(cur)} for vg, cur in zip(vgs_grid, ids_list)
            ],
        }
        if vgs_v is not None:
            result["ids_at_vgs_a"] = float(
                graphene_pn_iv(params, v_bias=float(vdd_v), v_split_gate=float(vgs_v), device=device)
            )
        return result

    return {
        "error": (
            f"Unknown 2D/beyond-CMOS technology '{technology}'. "
            "Supported: tmd_fet, ncfet, tfet, graphene_pn."
        )
    }


# ---------------------------------------------------------------------------
# Tool 11: chiplab_quantum_device_simulate
# ---------------------------------------------------------------------------


def chiplab_quantum_device_simulate(
    technology: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Simulate quantum & cryogenic devices: transmon, fluxonium, majorana, sfq_logic, adiabatic_qc."""
    from chiplab.schemas.common import Provenance, TrackedValue

    p = params or {}
    tech = technology.lower().strip()
    h_planck = 6.626e-34

    if tech == "transmon":
        from chiplab.models.materials.transmon import TransmonParams, transmon_compute_spectrum

        t_params = TransmonParams()
        if "ej_ghz" in p:
            t_params.ej = TrackedValue(value=h_planck * float(p["ej_ghz"]) * 1e9, provenance=Provenance.MEASURED)
        if "ec_ghz" in p:
            t_params.ec = TrackedValue(value=h_planck * float(p["ec_ghz"]) * 1e9, provenance=Provenance.MEASURED)

        energies, freq_01, anharm = transmon_compute_spectrum(t_params)
        return {
            "technology": "transmon",
            "energies_j": [float(e) for e in energies],
            "energies_ghz": [float(e / h_planck / 1e9) for e in energies],
            "freq_01_ghz": float(freq_01 / 1e9),
            "anharmonicity_mhz": float(anharm / 1e6),
        }

    if tech == "fluxonium":
        from chiplab.models.materials.fluxonium import FluxoniumParams, fluxonium_hamiltonian

        f_params = FluxoniumParams()
        if "ej_ghz" in p:
            f_params.ej = TrackedValue(value=h_planck * float(p["ej_ghz"]) * 1e9, provenance=Provenance.MEASURED)
        if "ec_ghz" in p:
            f_params.ec = TrackedValue(value=h_planck * float(p["ec_ghz"]) * 1e9, provenance=Provenance.MEASURED)
        if "el_ghz" in p:
            f_params.el = TrackedValue(value=h_planck * float(p["el_ghz"]) * 1e9, provenance=Provenance.MEASURED)

        n_levels = int(p.get("n_levels", 10))
        flux = float(p.get("flux", 0.5))
        eigvals, _ = fluxonium_hamiltonian(f_params, n_levels=n_levels, flux=flux)
        freq_01_ghz = float((eigvals[1] - eigvals[0]) / h_planck / 1e9) if len(eigvals) >= 2 else 0.0
        return {
            "technology": "fluxonium",
            "flux_phi0": flux,
            "n_levels": n_levels,
            "energies_j": [float(e) for e in eigvals],
            "energies_ghz": [float(e / h_planck / 1e9) for e in eigvals],
            "freq_01_ghz": freq_01_ghz,
        }

    if tech in ("majorana", "mzm"):
        from chiplab.models.materials.majorana import (
            MZMParams,
            mzm_topological_gap,
            mzm_wavefunction_overlap,
        )

        m_params = MZMParams()
        b_field = float(p["b_field_t"]) if "b_field_t" in p else None
        wire_len = float(p["wire_length_um"]) if "wire_length_um" in p else None
        gap_mev = mzm_topological_gap(m_params, b_field=b_field)
        overlap_mev = mzm_wavefunction_overlap(m_params, wire_length=wire_len)
        return {
            "technology": "majorana",
            "topological_gap_mev": float(gap_mev),
            "wavefunction_overlap_mev": float(overlap_mev),
            "in_topological_phase": bool(gap_mev > 0.0),
        }

    if tech in ("sfq_logic", "sfq", "rsfq", "ersfq", "rql"):
        from chiplab.models.materials.sfq_logic import SFQFamily, SFQState, sfq_power_estimation

        default_fam = tech.upper() if tech in ("rsfq", "ersfq", "rql") else "RSFQ"
        fam_str = str(p.get("family", default_fam)).upper()
        family = SFQFamily[fam_str] if fam_str in SFQFamily.__members__ else SFQFamily.RSFQ

        sfq_state = SFQState(family=family)
        n_fluxons = int(p.get("n_fluxons", 10))
        frequency_hz = float(p.get("frequency_hz", 10e9))
        power = sfq_power_estimation(sfq_state, n_fluxons=n_fluxons, frequency=frequency_hz)
        return {
            "technology": "sfq_logic",
            "family": family.value,
            "frequency_hz": frequency_hz,
            "static_power_w": float(power["static_power_w"]),
            "dynamic_power_w": float(power["dynamic_power_w"]),
            "ac_power_w": float(power["ac_power_w"]),
            "total_power_w": float(power["total_power_w"]),
            "energy_per_op_j": float(power["energy_per_op_j"]),
            "fluxons_per_cycle": int(power["fluxons_per_cycle"]),
        }

    if tech in ("adiabatic_qc", "aqc", "annealing"):
        from chiplab.models.materials.adiabatic_qc import (
            AQCParams,
            AQCState,
            AnnealingSchedule,
            aqc_annealing_schedule,
            aqc_sample_solutions,
        )

        n_qubits = max(2, min(int(p.get("n_qubits", 4)), 8))
        t_total_s = float(p.get("t_total_ns", 100.0)) * 1e-9
        n_samples = int(p.get("n_samples", 100))

        sched_str = str(p.get("schedule", "LINEAR")).upper()
        sched_type = AnnealingSchedule[sched_str] if sched_str in AnnealingSchedule.__members__ else AnnealingSchedule.LINEAR

        aqc_params = AQCParams(n_qubits=n_qubits, schedule_type=sched_type)
        raw_h = p.get("h_bias", [0.1] * n_qubits)
        h_bias = np.zeros(n_qubits, dtype=np.float64)
        for idx in range(min(len(raw_h), n_qubits)):
            h_bias[idx] = float(raw_h[idx])

        j_coupling = np.zeros((n_qubits, n_qubits), dtype=np.float64)
        if "j_coupling" in p and isinstance(p["j_coupling"], list):
            raw_j = np.asarray(p["j_coupling"], dtype=np.float64)
            rows = min(raw_j.shape[0], n_qubits)
            cols = min(raw_j.shape[1], n_qubits) if raw_j.ndim == 2 else 0
            if cols > 0:
                j_coupling[:rows, :cols] = raw_j[:rows, :cols]
        else:
            for idx in range(n_qubits - 1):
                j_coupling[idx, idx + 1] = 1.0
                j_coupling[idx + 1, idx] = 1.0

        aqc_state = AQCState(
            params=aqc_params,
            h_bias=h_bias,
            j_coupling=j_coupling,
            classical_spins=np.ones(n_qubits, dtype=np.float64),
        )

        schedule_samples = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            a_t, b_t = aqc_annealing_schedule(frac * t_total_s, t_total_s, schedule_type=sched_type)
            schedule_samples.append({"s": frac, "a_t": float(a_t), "b_t": float(b_t)})

        sol = aqc_sample_solutions(aqc_state, n_samples=n_samples)
        return {
            "technology": "adiabatic_qc",
            "n_qubits": n_qubits,
            "schedule_type": sched_type.value,
            "schedule_samples": schedule_samples,
            "best_spins": [int(s) for s in sol["best_spins"]],
            "best_energy": float(sol["best_energy"]),
            "mean_energy": float(sol["mean_energy"]),
            "std_energy": float(sol["std_energy"]),
            "unique_solutions": int(sol["unique_solutions"]),
            "ground_state_probability": float(sol["ground_state_probability"]),
        }

    return {
        "error": (
            f"Unknown quantum technology '{technology}'. "
            "Supported: transmon, fluxonium, majorana, sfq_logic, adiabatic_qc."
        )
    }


# ---------------------------------------------------------------------------
# Tool 12: chiplab_spice_netlist_run
# ---------------------------------------------------------------------------


def chiplab_spice_netlist_run(
    netlist: str,
    node_nm: float = 28.0,
    corner: str = "TT",
    analysis: str = "dc",
    t_stop_s: float = 1e-9,
    dt_s: float = 5e-11,
    device: str = "auto",
) -> Dict[str, Any]:
    """Parse a SPICE netlist via CustomCircuit.from_spice_netlist and run DC or transient simulation."""
    from chiplab.models.materials.custom_circuit import CustomCircuit

    if not netlist or not netlist.strip():
        return {"error": "SPICE netlist must be a non-empty string."}

    circ = CustomCircuit.from_spice_netlist(netlist, node_nm=float(node_nm), corner=corner)
    if not circ.components:
        return {"error": "No valid components parsed from SPICE netlist."}

    result: Dict[str, Any] = {
        "circuit_name": circ.name,
        "node_nm": float(circ.node_nm.value),
        "corner": circ.process_node.corner.value,
        "analysis": analysis.lower(),
        "component_count": len(circ.components),
        "normalized_spice_netlist": circ.to_spice_netlist(),
    }

    analysis_low = analysis.lower().strip()
    if analysis_low == "dc":
        result["dc"] = circ.simulate_dc(device=device)
    elif analysis_low == "transient":
        result["transient"] = circ.simulate_transient(
            t_stop=float(t_stop_s),
            dt=float(dt_s),
            device=device,
        )
    elif analysis_low == "both":
        result["dc"] = circ.simulate_dc(device=device)
        result["transient"] = circ.simulate_transient(
            t_stop=float(t_stop_s),
            dt=float(dt_s),
            device=device,
        )
    else:
        return {"error": f"Unknown analysis '{analysis}'. Must be 'dc', 'transient', or 'both'."}

    return result


# ---------------------------------------------------------------------------
# Tool 13: chiplab_agent_run
# ---------------------------------------------------------------------------


def chiplab_agent_run(
    prompt: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Invoke OpenRouterAgent lazily to run a prompt with ChipLab tools."""
    if not prompt or not prompt.strip():
        return {"error": "prompt must be a non-empty string."}

    try:
        from chiplab.agent import OpenRouterAgent  # type: ignore[import-not-found]
    except ImportError as exc:
        return {
            "error": f"OpenRouterAgent is not available: {exc}",
            "prompt": prompt,
            "model": model,
        }

    agent = OpenRouterAgent(model=model) if model else OpenRouterAgent()
    output = agent.run(prompt)
    if isinstance(output, dict):
        return {
            "status": "ok",
            "model": getattr(agent, "model", model),
            "result": output,
        }
    return {
        "status": "ok",
        "model": getattr(agent, "model", model),
        "response": str(output),
    }


# ---------------------------------------------------------------------------
# Tool 14: chiplab_chipro_eda_flow
# ---------------------------------------------------------------------------


def chiplab_chipro_eda_flow(
    action: str,
    rtl_code: str = "",
    top_module: str = "top",
    pdk: str = "sky130",
    node_nm: float = 130.0,
    sdc_text: str = "",
    target_freq_mhz: float = 100.0,
    gds_path: str = "",
) -> Dict[str, Any]:
    """Run ChipRO / Open-Source EDA flow actions lazily via chiplab.agent.eda_frameworks."""
    try:
        from chiplab.agent.eda_frameworks import (  # type: ignore[import-not-found]
            chipro_open_klayout,
            chipro_rtl_lint,
            chipro_run_pnr,
            chipro_tweak_constraints,
        )
    except ImportError as exc:
        return {"error": f"eda_frameworks is not available: {exc}", "action": action}

    act = action.lower().strip()
    if act in ("lint", "rtl_lint"):
        return chipro_rtl_lint(rtl_code=rtl_code, top_module=top_module)
    if act in ("pnr", "run_pnr"):
        return chipro_run_pnr(
            rtl_code=rtl_code,
            top_module=top_module,
            pdk=pdk,
            node_nm=float(node_nm),
        )
    if act in ("tweak_constraints", "constraints", "sdc"):
        return chipro_tweak_constraints(
            sdc_text=sdc_text,
            target_freq_mhz=float(target_freq_mhz),
        )
    if act in ("klayout", "open_klayout", "gds"):
        return chipro_open_klayout(gds_path=gds_path)
    return {
        "error": f"Unknown action '{action}'. Supported: lint, pnr, tweak_constraints, klayout."
    }


# ---------------------------------------------------------------------------
# Tool 15: chiplab_mage_rtl_pipeline
# ---------------------------------------------------------------------------


def chiplab_mage_rtl_pipeline(
    rtl_code: str,
    top_module: str = "top",
    test_vectors: Optional[List[Dict[str, Any]]] = None,
    max_fix_rounds: int = 3,
) -> Dict[str, Any]:
    """Run the MAGE / VLSI-AGENT multi-agent RTL lint, simulation, and waveform debug loop."""
    if not rtl_code or not rtl_code.strip():
        return {"error": "rtl_code must be a non-empty string."}

    from chiplab.agent.rtl_co_design import mage_rtl_pipeline

    return mage_rtl_pipeline(
        rtl_code=rtl_code,
        top_module=top_module,
        test_vectors=test_vectors,
        max_fix_rounds=int(max_fix_rounds),
    )


# ---------------------------------------------------------------------------
# Tool 16: chiplab_chia_codesign_loop
# ---------------------------------------------------------------------------


def chiplab_chia_codesign_loop(
    node_nm: float = 28.0,
    target_freq_mhz: float = 500.0,
    max_area_um2: float = 50000.0,
    max_power_mw: float = 50.0,
    max_iters: int = 5,
    design_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the CHIA cyclic hardware/software co-design PPA optimization loop."""
    from chiplab.agent.rtl_co_design import chia_codesign_loop

    return chia_codesign_loop(
        design_spec=design_spec,
        node_nm=float(node_nm),
        target_freq_mhz=float(target_freq_mhz),
        max_area_um2=float(max_area_um2),
        max_power_mw=float(max_power_mw),
        max_iters=int(max_iters),
    )


# ---------------------------------------------------------------------------
# Tool 17: chiplab_local_llm_models
# ---------------------------------------------------------------------------


def chiplab_local_llm_models() -> Dict[str, Any]:
    """List available local LLM models (Ollama/vLLM/LM Studio) and free OpenRouter models."""
    from chiplab.agent.openrouter_agent import (
        list_free_openrouter_models,
        list_local_llm_models,
    )

    local_models = list_local_llm_models()
    openrouter_models = list_free_openrouter_models(fetch_live=False)
    return {
        "local_models": local_models,
        "openrouter_free_models": openrouter_models,
        "local_count": len(local_models),
        "openrouter_count": len(openrouter_models),
    }
