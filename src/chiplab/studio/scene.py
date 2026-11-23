"""Build the GPU Studio scene from a DesignSpec: one source of truth for every view.

Placement, die stacks, TSVs, interposer routing, PCB and cooler geometry are derived from the
design (plus optional ``design.metadata["studio"]`` overrides and ``["placement"]`` moves).
Power comes from ChipLab's analytical models, temperatures from ``studio.thermal`` and
activity (engine / channel / bank / link utilization) from the discrete-event simulation.
Every derived quantity that is a representative assumption rather than a design value is
listed under ``assumptions`` in the result.
"""

from __future__ import annotations

import math
from typing import Any

from ..schemas.design import ChipletSpec, DesignSpec
from ..schemas.workload import WorkloadSpec
from . import thermal as th

STUDIO_DEFAULTS: dict[str, float] = {
    "spacing_mm": 1.0, "interposer_margin_mm": 2.0, "die_thickness_um": 50.0,
    "base_die_thickness_um": 60.0, "bond_layer_um": 20.0, "compute_die_thickness_um": 775.0,
    "tsv_pitch_um": 40.0, "microbump_pitch_um": 55.0, "pcie_lanes": 16.0,
    "core_voltage_v": 0.8, "vrm_amps_per_phase": 60.0, "vrm_efficiency": 0.9,
}


def _num(v: Any) -> float:
    return float(getattr(v, "value", v)) if v is not None else 0.0


def studio_params(design: DesignSpec) -> dict[str, Any]:
    meta = (design.metadata or {}).get("studio") or {}
    p = dict(STUDIO_DEFAULTS)
    p.update({k: float(v) for k, v in meta.items() if k in STUDIO_DEFAULTS
              and isinstance(v, (int, float)) and v > 0})
    p["cooling"] = th.Cooling.from_meta(meta.get("cooling"))
    return p


def _kind(ch: ChipletSpec) -> str:
    mem, comp = ch.memory, ch.compute
    tech = (mem.technology if mem else "").lower()
    t = ch.type.lower()
    if mem and ("hbm" in tech or "hbm" in t or _num(mem.stack_layers) > 1):
        return "hbm_stack"
    if mem and comp:
        return "dram_chiplet"          # memory arrays with compute regions (ECHO-style)
    if mem:
        return "memory_chiplet"
    if comp and ("pim" in t or "pim" in (comp.architecture or "").lower()
                 or "in_memory" in (comp.architecture or "").lower()):
        return "pim_chiplet"
    if comp or t in ("compute", "gpu", "cpu", "npu", "accelerator"):
        return "compute_die"
    if any(k in t for k in ("memory", "dram", "sram")):
        return "memory_chiplet"
    if "hbm" in t:
        return "hbm_stack"
    return "io_die"


DEFAULT_AREA = {"hbm_stack": 110.0, "compute_die": 400.0, "dram_chiplet": 120.0,
                "memory_chiplet": 100.0, "pim_chiplet": 100.0, "io_die": 60.0}


def _instances(design: DesignSpec) -> list[dict[str, Any]]:
    """Chiplet instances in discrete-event node order, with footprint and per-instance power."""
    from ..engine.analytical import _estimate_compute_chiplet, _estimate_memory_chiplet
    out = []
    for ch in design.chiplets:
        count = int(_num(ch.count)) or 1
        kind = _kind(ch)
        area = _num(ch.area_mm2) or DEFAULT_AREA[kind]
        aspect = 1.1 if kind == "hbm_stack" else 1.25 if kind == "compute_die" else 1.2
        w = math.sqrt(area * aspect)
        d = area / w
        try:
            mem, _, _ = _estimate_memory_chiplet(ch)
        except ValueError:
            mem = {}
        try:
            comp, _, _ = _estimate_compute_chiplet(ch)
        except ValueError:            # spec-less chiplet: still drawn, with no power/compute
            comp = {}
        # Same power accounting as the analytical engine (no double counting).
        if ch.compute is not None or ch.memory is None:
            pw = float(comp.get("power_watts") or 0.0)
        else:
            pw = float(mem.get("power_watts") or 0.0)
        for i in range(count):
            out.append({
                "id": f"{ch.id or ch.type}[{i}]", "chiplet_id": ch.id or ch.type, "index": i,
                "kind": kind, "type": ch.type, "w": round(w, 3), "d": round(d, 3),
                "area_mm2": area, "power_w": pw, "process_nm": _num(ch.process_nm) or None,
                "capacity_gb": (mem.get("capacity_gb", 0.0) / count) if ch.memory else 0.0,
                "bandwidth_gbps": (mem.get("bandwidth_tbps", 0.0) * 1000 / count)
                if ch.memory else 0.0,
                "channels": (int(mem.get("channels") or 0) // count) if ch.memory else 0,
                "banks_per_channel": int(mem.get("banks_per_channel") or 0) if ch.memory else 0,
                "stack_layers": int(_num(ch.memory.stack_layers)) if ch.memory else 0,
                "memory_tech": ch.memory.technology if ch.memory else None,
                "latency_ns": mem.get("latency_ns") if ch.memory else None,
                "peak_tops": _num(ch.compute.peak_tops) if ch.compute else 0.0,
                "effective_tops": float(comp.get("effective_tops") or 0.0),
                "mac_units": int(_num(ch.compute.mac_units)) if ch.compute and ch.compute.mac_units else 0,
                "sram_mb": _num(ch.compute.sram_mb) if ch.compute else 0.0,
                "architecture": ch.compute.architecture if ch.compute else None,
                "precision": ch.compute.precision if ch.compute else None,
                "clock_ghz": _num(ch.compute.clock_ghz) if ch.compute else 0.0,
            })
    return out


def _auto_place(inst: list[dict[str, Any]], gap: float) -> None:
    """Compute block in the middle, memory columns left and right, the rest along the top."""
    core = [c for c in inst if c["kind"] in ("compute_die", "pim_chiplet", "dram_chiplet")]
    mem = [c for c in inst if c["kind"] in ("hbm_stack", "memory_chiplet")]
    rest = [c for c in inst if c not in core and c not in mem]
    if not core:                      # memory-only (or I/O-only) package
        core, mem = (mem, []) if mem else (rest, [])
        if core is rest:
            rest = []
    cols = max(1, math.ceil(math.sqrt(len(core))))
    rows = math.ceil(len(core) / cols)
    cw = max(c["w"] for c in core)
    cd = max(c["d"] for c in core)
    block_w = cols * cw + (cols - 1) * gap
    block_d = rows * cd + (rows - 1) * gap
    for k, c in enumerate(core):
        r, q = divmod(k, cols)
        c["x"] = -block_w / 2 + cw / 2 + q * (cw + gap)
        c["y"] = -block_d / 2 + cd / 2 + r * (cd + gap)
    left, right = mem[0::2], mem[1::2]
    for side, items in ((-1, left), (1, right)):
        if not items:
            continue
        mw = max(c["w"] for c in items)
        col_h = sum(c["d"] for c in items) + gap * (len(items) - 1)
        y = -col_h / 2
        for c in items:
            c["x"] = side * (block_w / 2 + gap + mw / 2)
            c["y"] = y + c["d"] / 2
            y += c["d"] + gap
    if rest:
        total = sum(c["w"] for c in rest) + gap * (len(rest) - 1)
        x = -total / 2
        top = max([block_d / 2] + [c["y"] + c["d"] / 2 for c in mem])
        for c in rest:
            c["x"] = x + c["w"] / 2
            c["y"] = top + gap + c["d"] / 2
            x += c["w"] + gap


def _overlaps(inst: list[dict[str, Any]]) -> list[str]:
    out = []
    for i, a in enumerate(inst):
        for b in inst[i + 1:]:
            if (abs(a["x"] - b["x"]) < (a["w"] + b["w"]) / 2 - 1e-6
                    and abs(a["y"] - b["y"]) < (a["d"] + b["d"]) / 2 - 1e-6):
                out.append(f"{a['id']} overlaps {b['id']}")
    return out


def _stack(c: dict[str, Any], p: dict[str, Any]) -> None:
    """HBM die stack: base die + memory dies, TSV and microbump counts (estimates)."""
    n = max(1, c["stack_layers"] or 8)
    die_t = p["die_thickness_um"] / 1000
    base_t = p["base_die_thickness_um"] / 1000
    bond = p["bond_layer_um"] / 1000
    layers = [{"kind": "base_die", "z": 0.0, "h": base_t, "label": "base (logic) die"}]
    z = base_t + bond
    for i in range(n):
        layers.append({"kind": "dram_die", "z": round(z, 4), "h": die_t, "label": f"DRAM die {i + 1}"})
        z += die_t + bond
    c["layers"] = layers
    c["h"] = round(z - bond, 4)
    channels = max(c["channels"], 1)
    signal_tsv = channels * 128 * 1.5                 # data + address/command/clock per channel
    power_tsv = signal_tsv * 1.0                      # comparable power/ground TSV count
    c["tsv"] = {"count_estimate": int(signal_tsv + power_tsv), "pitch_um": p["tsv_pitch_um"],
                "per_die_interfaces": n, "signal": int(signal_tsv), "power_ground": int(power_tsv)}
    c["microbumps"] = {"pitch_um": p["microbump_pitch_um"],
                       "count_estimate": int(c["area_mm2"] * 1e6 / p["microbump_pitch_um"] ** 2 * 0.25)}


def _internal(c: dict[str, Any], util: dict[str, Any]) -> dict[str, Any]:
    """Die Explorer floorplan: what is inside the die (representative, from design fields)."""
    if c["kind"] in ("compute_die", "pim_chiplet"):
        macs = c["mac_units"]
        clusters = max(4, min(144, macs // 64 if macs else round(math.sqrt(max(c["peak_tops"], 16)) * 4)))
        sram_frac = min(0.45, 0.06 + c["sram_mb"] / 400) if c["sram_mb"] else 0.08
        return {"kind": "compute", "clusters": int(clusters), "sram_fraction": sram_frac,
                "sram_mb": c["sram_mb"], "mac_units": macs, "phy_edges": [],
                "utilization": util.get("engines", {}).get(c["id"], 0.0),
                "note": "cluster count from mac_units (or peak TOPS); SRAM share from sram_mb"}
    channels = max(c["channels"], 1)
    banks = max(c["banks_per_channel"], 1)
    bank_util = util.get("banks", {}).get(c["id"]) or [[0.0] * banks for _ in range(channels)]
    bank_access = util.get("bank_access", {}).get(c["id"]) or [[0.0] * banks for _ in range(channels)]
    return {"kind": "memory", "channels": channels, "banks_per_channel": banks,
            "subarrays_per_bank": 32, "sense_amp_strips": True,
            "cim_fraction": 0.25 if c["kind"] == "dram_chiplet" else 0.0,
            "channel_util": util.get("channels", {}).get(c["id"], [0.0] * channels),
            "bank_util": bank_util, "bank_access": bank_access,
            "compute_util": util.get("engines", {}).get(c["id"]),
            "note": "subarrays and sense-amp strips are representative; bank colors are "
                    "simulated traffic (channel busy x the bank's share of accesses); "
                    "bank_util is row-activation occupancy"}


def _pcb(pkg_w: float, pkg_d: float, power_w: float, p: dict[str, Any]) -> dict[str, Any]:
    board_w = max(3.0 * pkg_w, 200.0)
    board_d = max(2.2 * pkg_d, 111.0)
    pkg_x = -board_w / 2 + pkg_w / 2 + 25.0
    amps = power_w / p["core_voltage_v"]
    phases = max(2, math.ceil(amps / p["vrm_amps_per_phase"]))
    parts, traces = [], []
    col_x = pkg_x + pkg_w / 2 + 18.0
    per_col = max(1, math.ceil(phases / 2))
    pitch = min(9.0, (board_d - 20) / per_col)
    for k in range(phases):
        col, row = divmod(k, per_col)
        x = col_x + col * 22.0
        y = -per_col * pitch / 2 + pitch / 2 + row * pitch
        parts.append({"kind": "power_stage", "id": f"vrm{k}", "x": x, "y": y, "w": 5, "d": 6, "h": 1.0,
                      "metrics": {"phase": k + 1, "amps": round(amps / phases, 1)}})
        parts.append({"kind": "inductor", "id": f"ind{k}", "x": x + 8, "y": y, "w": 7, "d": 7, "h": 5,
                      "metrics": {"phase": k + 1}})
        traces.append({"kind": "power", "width": 2.0,
                       "points": [[x, y], [pkg_x + pkg_w / 2 + 2, y * 0.5]]})
    for k in range(int(phases * 2)):
        parts.append({"kind": "capacitor", "id": f"cap{k}", "x": pkg_x - pkg_w / 2 + (k + 0.5) * pkg_w / (phases * 2),
                      "y": -pkg_d / 2 - 6, "w": 2.5, "d": 2.5, "h": 3, "metrics": {}})
    connectors = max(1, math.ceil(power_w / 600))
    for k in range(connectors):
        parts.append({"kind": "power_connector", "id": f"pwr{k}", "x": board_w / 2 - 12,
                      "y": -10 + k * 16, "w": 18, "d": 12, "h": 9,
                      "metrics": {"rating_w": 600}})
    lanes = int(p["pcie_lanes"])
    fx0 = pkg_x - 40
    for k in range(lanes):
        fx = fx0 + k * 2.2
        traces.append({"kind": "pcie", "width": 0.3,
                       "points": [[pkg_x - pkg_w / 4 + k * pkg_w / (2 * lanes), -pkg_d / 2],
                                  [fx, -board_d / 2 + 14], [fx, -board_d / 2 + 6]]})
    parts.append({"kind": "edge_connector", "id": "pcie", "x": fx0 + lanes * 1.1, "y": -board_d / 2 + 3,
                  "w": lanes * 2.2 + 4, "d": 6, "h": 0.1, "metrics": {"lanes": lanes}})
    return {"w": board_w, "d": board_d, "t": 1.6, "package_x": pkg_x, "parts": parts,
            "traces": traces, "vrm": {"phases": phases, "amps": round(amps, 1),
                                      "efficiency": p["vrm_efficiency"],
                                      "loss_w": round(power_w * (1 / p["vrm_efficiency"] - 1), 2)}}


def build_scene(design: DesignSpec, design_id: str, workload: WorkloadSpec | None = None,
                simulate: bool = True) -> dict[str, Any]:
    from ..engine.discrete_event import run_discrete_event_simulation
    from ..models.interconnect import estimate_interconnect
    from ..schemas.simulation import FidelityLevel, SimulationConfig

    p = studio_params(design)
    inst = _instances(design)
    gap = p["spacing_mm"]
    _auto_place(inst, gap)
    moves = (design.metadata or {}).get("placement") or {}
    for c in inst:
        mv = moves.get(c["id"])
        if isinstance(mv, dict):
            c["x"] = float(mv.get("x", c["x"]))
            c["y"] = float(mv.get("y", c["y"]))
            c["moved"] = True
    warnings = _overlaps(inst)

    for c in inst:
        if c["kind"] == "hbm_stack":
            _stack(c, p)
        else:
            c["h"] = (p["compute_die_thickness_um"] if c["kind"] in ("compute_die", "pim_chiplet", "io_die")
                      else 300.0) / 1000

    margin = p["interposer_margin_mm"]
    minx = min(c["x"] - c["w"] / 2 for c in inst) - margin
    maxx = max(c["x"] + c["w"] / 2 for c in inst) + margin
    miny = min(c["y"] - c["d"] / 2 for c in inst) - margin
    maxy = max(c["y"] + c["d"] / 2 for c in inst) + margin
    ipw, ipd = maxx - minx, maxy - miny
    pkg_type = str(design.package.type)
    ip_name = (design.package.interposer or "silicon").lower()
    if any(k in ip_name for k in ("cowos_l", "emib", "bridge", "lsi")):
        interposer_kind = "bridges"            # RDL/organic interposer + local silicon bridges
    elif pkg_type in ("2.5D", "3D_stacking") and any(k in ip_name for k in ("silicon", "cowos")):
        interposer_kind = "silicon_interposer"
    else:
        interposer_kind = "organic_substrate"

    inter, _, _ = estimate_interconnect(design, len(inst))
    if not any(c["power_w"] or c["capacity_gb"] for c in inst):
        warnings.append("No compute or memory specs: dies are drawn from defaults, power is 0.")
    activity: dict[str, Any] = {}
    sim_summary = None
    if simulate:
        wl = workload or WorkloadSpec(type="llm_decode", model_params_b=8, batch_size=4)
        try:
            r = run_discrete_event_simulation(design, wl, SimulationConfig(
                fidelity=FidelityLevel.DISCRETE_EVENT, num_requests=4, seed=42), design_id)
        except ValueError as exc:  # e.g. chiplets without compute/memory specs
            r = None
            warnings.append(f"Not simulated: {exc}")
        if r is None:
            pass
        elif r.success:
            activity = r.raw_details.get("resource_utilization", {})
            sim_summary = {"workload": wl.model_dump(include={"name", "type", "batch_size",
                                                              "model_params_b"}),
                           "latency_ms": r.metrics.estimated_latency_ms,
                           "throughput_tokens_per_sec": r.metrics.throughput_tokens_per_sec,
                           "bottleneck": r.bottleneck.model_dump(),
                           "events": r.raw_details.get("events"),
                           "disclaimer": r.fidelity_disclaimer}
        else:
            warnings.append(f"Discrete-event simulation failed: {r.error_message}")

    for c in inst:
        c["internal"] = _internal(c, activity)

    # Links: interposer routes / bridges between chiplets the simulated topology connects.
    links = []
    seen = set()
    by_id = {c["id"]: c for c in inst}
    for lk in activity.get("links", []):
        a, b = lk["src"], lk["dst"]
        if a in by_id and b in by_id:
            key = tuple(sorted((a, b)))
            if key in seen:
                for x in links:
                    if x["key"] == key:
                        x["util"] = max(x["util"], lk["util"])
                continue
            seen.add(key)
            links.append({"key": key, "from": a, "to": b, "util": lk["util"]})
        elif "switch" in (a, b) or a == "bus":
            node = b if a == "switch" else a
            if node in by_id and ("switch", node) not in seen:
                seen.add(("switch", node))
                links.append({"key": ("switch", node), "from": node, "to": "switch",
                              "util": lk["util"]})
            elif node in by_id:
                for x in links:
                    if x["key"] == ("switch", node):
                        x["util"] = max(x["util"], lk["util"])
    if not links and len(inst) > 1:          # no simulation: draw the topology unloaded
        for a, b in zip(inst, inst[1:]):
            links.append({"key": (a["id"], b["id"]), "from": a["id"], "to": b["id"], "util": 0.0})
    for x in links:
        x["key"] = list(x["key"])
        x["bandwidth_gbps"] = inter.get("per_link_bw_gbps")
        x["kind"] = "bridge" if interposer_kind == "bridges" else (
            "interposer_route" if interposer_kind == "silicon_interposer" else "substrate_route")

    # Interconnect power dissipates in the PHYs at the link endpoints: split it over the
    # chiplets by how many link ends each has (all chiplets equally if no links are known).
    # Link power = energy per bit x bits actually moved (simulated link utilization), plus
    # an idle PHY floor of 5% of full-rate power. (The library's flat 0.1 W per GB/s of
    # capacity ignores traffic and energy/bit and can reach kilowatts.)
    epb = float(inter.get("energy_pj_per_bit") or 0.0) * 1e-12
    bw = float(inter.get("per_link_bw_gbps") or 0.0) * 1e9 * 8          # bit/s per link
    sim_links = activity.get("links", [])
    if sim_links:
        ic_power = sum((lk["util"] + 0.05) * bw * epb for lk in sim_links)
    else:
        ic_power = 0.05 * bw * epb * len(links)
    ends = {c["id"]: 0 for c in inst}
    for x in links:
        for e in (x["from"], x["to"]):
            if e in ends:
                ends[e] += 1
    n_ends = sum(ends.values())
    for c in inst:
        c["phy_power_w"] = round(ic_power * (ends[c["id"]] / n_ends if n_ends else 1 / len(inst)), 3)
        c["power_w"] += c["phy_power_w"]
    total_die_power = sum(c["power_w"] for c in inst) - ic_power
    pkg_power = total_die_power + ic_power
    cool = p["cooling"]
    r_override = _num(design.constraints.thermal_resistance_c_per_w) or None
    ambient = _num(design.constraints.ambient_temp_c) or 25.0
    therm = th.solve((minx - 2, miny - 2, ipw + 4, ipd + 4),
                     [{"id": c["id"], "x": c["x"], "y": c["y"], "w": c["w"], "d": c["d"],
                       "power_w": c["power_w"], "thickness_mm": max(c["h"], 0.05)} for c in inst],
                     0.0, cool, ambient, r_override)
    max_t = _num(design.constraints.max_temp_c) or 85.0
    tmap = {t["id"]: t for t in therm["components"]}
    for c in inst:
        t = tmap[c["id"]]
        t["headroom_c"] = round(max_t - t["max_c"], 2)
        c["temp_c"] = round(t["max_c"], 2)
        c["temp_mean_c"] = round(t["mean_c"], 2)
        c["power_density_w_mm2"] = round(c["power_w"] / c["area_mm2"], 4) if c["area_mm2"] else 0
        c["utilization"] = (activity.get("engines", {}).get(c["id"])
                            if c["id"] in activity.get("engines", {}) else
                            (max(activity.get("channels", {}).get(c["id"]) or [0.0])
                             if c["id"] in activity.get("channels", {}) else None))
        if c["kind"] == "hbm_stack":
            n = len(c["layers"])
            share = [c["power_w"] * 0.3] + [c["power_w"] * 0.7 / (n - 1)] * (n - 1)
            prof = th.stack_profile(c["temp_c"], share, c["area_mm2"], p["bond_layer_um"])
            for layer, tc, pw in zip(c["layers"], prof, share):
                layer["temp_c"] = round(tc, 2)
                layer["power_w"] = round(pw, 3)
        if t["max_c"] > max_t:
            warnings.append(f"{c['id']} reaches {t['max_c']:.1f} °C, above max_temp_c {max_t:g} °C")

    # Inefficient = draws a large share of power while the simulation shows it mostly idle.
    for c in inst:
        u = c["utilization"]
        c["inefficient"] = bool(u is not None and pkg_power > 0
                                and c["power_w"] / pkg_power > 0.08 and u < 0.3)

    pkg_w, pkg_d = ipw + 8, ipd + 8
    pcb = _pcb(pkg_w, pkg_d, pkg_power, p)
    sink_w, sink_d = max(pkg_w + 20, 90.0), max(pkg_d + 20, 90.0)
    return {
        "design": {"id": design_id, "name": design.name, "version": design.version,
                   "package_type": pkg_type},
        "units": "mm",
        "components": inst,
        "links": links,
        "package": {
            "interposer": {"kind": interposer_kind, "x": (minx + maxx) / 2, "y": (miny + maxy) / 2,
                           "w": ipw, "d": ipd, "t": 0.1},
            "substrate": {"w": pkg_w, "d": pkg_d, "t": 1.2},
            "ihs": {"w": ipw + 4, "d": ipd + 4, "t": cool.ihs_thickness_mm},
            "heatsink": {"w": sink_w, "d": sink_d, "base_t": cool.base_thickness_mm,
                         "fins": cool.fin_count, "fin_h": cool.fin_height_mm,
                         "fin_t": cool.fin_thickness_mm},
            "fans": [{"d": cool.fan_diameter_mm, "rpm": therm["airflow"]["fan_rpm"]}
                     for _ in range(cool.fans)],
            "topology": activity.get("topology") or str(design.interconnect.topology),
        },
        "pcb": pcb,
        "power": {"package_w": round(pkg_power, 2), "dies_w": round(total_die_power, 2),
                  "interconnect_w": round(ic_power, 2), "vrm_loss_w": pcb["vrm"]["loss_w"],
                  "board_input_w": round(pkg_power + pcb["vrm"]["loss_w"], 2),
                  "by_component": {c["id"]: round(c["power_w"], 3) for c in inst}},
        "thermal": therm,
        "max_temp_c": max_t,
        "simulation": sim_summary,
        "params": {k: v for k, v in p.items() if k != "cooling"} | {"cooling": vars(cool)},
        "warnings": list(dict.fromkeys(warnings)),
        "assumptions": [
            "Placement is automatic (compute centered, memory columns at the sides) unless "
            "moved in the studio; footprints come from area_mm2 (aspect 1.1 HBM, 1.25 compute).",
            "HBM stack: base die + stack_layers DRAM dies; die/bond thickness from studio params; "
            "TSV and microbump counts are estimates from channel count and pitch.",
            "Power per die from ChipLab's analytical models; VRM phases from package power / "
            "core voltage / amps per phase.",
            "Temperatures: two-layer finite-volume package model with an airflow-derived "
            "heatsink resistance (see studio/thermal.py); not CFD.",
            "Activity: per-resource utilization from the tile-level discrete-event simulation "
            "of the shown workload.",
        ],
    }


def apply_edit(design: DesignSpec, patch: dict[str, Any]) -> DesignSpec:
    """Return a copy of the design with a studio edit applied (validated by the caller)."""
    d = design.model_copy(deep=True)
    meta = dict(d.metadata or {})
    if "placement" in patch:
        pl = dict(meta.get("placement") or {})
        for k, v in (patch["placement"] or {}).items():
            if v is None:
                pl.pop(k, None)
            else:
                pl[str(k)] = {"x": float(v["x"]), "y": float(v["y"])}
        meta["placement"] = pl
    if patch.get("reset_placement"):
        meta.pop("placement", None)
    if "studio" in patch:
        st = dict(meta.get("studio") or {})
        for k, v in (patch["studio"] or {}).items():
            if k == "cooling":
                cl = dict(st.get("cooling") or {})
                cl.update({ck: float(cv) for ck, cv in (v or {}).items()
                           if ck in vars(th.Cooling()) and float(cv) > 0})
                st["cooling"] = cl
            elif k in STUDIO_DEFAULTS:
                if float(v) <= 0:
                    raise ValueError(f"studio.{k} must be positive")
                st[k] = float(v)
            else:
                raise ValueError(f"unknown studio parameter {k!r}")
        meta["studio"] = st
    for ce in patch.get("chiplets") or []:
        target = next((c for c in d.chiplets if (c.id or c.type) == ce.get("chiplet_id")), None)
        if target is None:
            raise ValueError(f"no chiplet {ce.get('chiplet_id')!r}")
        for f in ("count", "area_mm2", "tdp_watts"):
            if f in ce:
                setattr(target, f, int(ce[f]) if f == "count" else float(ce[f]))
        if "memory" in ce and target.memory is not None:
            for f, v in ce["memory"].items():
                if f in ("channels", "banks_per_channel", "stack_layers"):
                    setattr(target.memory, f, int(v))
                elif f in ("capacity_per_chiplet_gb", "bandwidth_per_channel_gbps", "latency_ns",
                           "energy_pj_per_bit"):
                    setattr(target.memory, f, float(v))
                else:
                    raise ValueError(f"memory field {f!r} is not editable here")
    for f, v in (patch.get("constraints") or {}).items():
        if f in ("max_temp_c", "ambient_temp_c", "thermal_resistance_c_per_w", "max_power_watts"):
            setattr(d.constraints, f, None if v in (None, "") else float(v))
        else:
            raise ValueError(f"constraint {f!r} is not editable here")
    d.metadata = meta
    return d
