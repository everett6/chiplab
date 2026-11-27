"""Level 3 specialized simulation: Ramulator 2 (DRAM) + SCALE-Sim (systolic arrays).

Starts from the Level 1 analytical result (power, area, thermal and cost stay analytical)
and replaces its two idealized inputs with simulated ones:

- memory: each memory chiplet's bandwidth is multiplied by the fraction of channel peak that
  Ramulator achieves on the workload's access stream (sequential 64 B lines with the
  workload's write share), and its loaded read latency is reported;
- compute: each systolic / tensor-engine chiplet's effective TOPS is multiplied by the array
  utilization SCALE-Sim measures on the workload's GEMM shapes.

Latency = max(compute time, memory time) as in the roofline, so a specialized result is
directly comparable with the analytical one. Chiplets the tools do not cover keep their
analytical numbers and are listed under ``raw_details.specialized.not_simulated``.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any

from ..credibility import require_credible_design
from ..schemas.common import Provenance
from ..schemas.simulation import AssumptionRecord, FidelityLevel, SimulationConfig
from ..schemas.simulation import SimulationResult
from ..schemas.workload import WorkloadSpec


class SpecializedUnavailable(RuntimeError):
    """Neither tool applies to this design (not installed, or no matching chiplets)."""


def _hidden_dim(wl: WorkloadSpec) -> int:
    if wl.model_params_b:
        # Dense transformer: ~12 * layers * hidden^2 parameters.
        h = math.sqrt(wl.model_params_b * 1e9 / (12 * max(1, wl.num_layers)))
        return max(128, int(round(h / 128.0)) * 128)
    return int(wl.hidden_dim)


def _workload_gemms(wl: WorkloadSpec) -> list[dict[str, Any]]:
    from ..backends.scale_sim import transformer_gemms
    from ..schemas.workload import is_llm_decode
    moe = "moe" in wl.type.lower() and wl.num_experts > 1
    return transformer_gemms(_hidden_dim(wl), max(1, wl.batch_size), max(1, wl.sequence_length),
                             is_llm_decode(wl), max(1, wl.num_layers),
                             kv_fraction=wl.kv_cache_fraction,
                             active_experts=wl.active_experts if moe else 1)


def run_specialized_simulation(design: Any, workload: WorkloadSpec | None,
                               config: SimulationConfig, design_id: str = "unsaved"
                               ) -> SimulationResult:
    """Raises SpecializedUnavailable if neither Ramulator nor SCALE-Sim covers any chiplet."""
    from ..backends.ecore import apply_ecore_to_design
    from ..backends.ramulator import RamulatorBackend, standard_for
    from ..backends.scale_sim import ScaleSimBackend, is_systolic
    from ..models.compute import estimate_compute_chiplet
    from ..models.memory import estimate_memory_chiplet
    from ..schemas.design import plain_design
    from ..schemas.workload import expand_workload_operators, is_llm_decode
    from .analytical import run_analytical_simulation

    require_credible_design(design)
    wl = workload if workload is not None else WorkloadSpec()
    plain, _ = plain_design(design)
    plain, _, _ = apply_ecore_to_design(plain)
    ram, ss = RamulatorBackend(), ScaleSimBackend()
    ram_ok, ss_ok = ram.is_available(), ss.is_available()

    ops = expand_workload_operators(wl)
    bytes_read = sum(op.bytes_read for op in ops)
    bytes_write = sum(op.bytes_write for op in ops)
    total_bytes = bytes_read + bytes_write
    write_fraction = min(0.5, bytes_write / total_bytes) if total_bytes > 0 else 0.0

    mem_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    not_sim: list[str] = []
    records: list[AssumptionRecord] = []
    spec_bw = eff_bw = 0.0
    for idx, ch in enumerate(plain.chiplets):
        cid = ch.id or f"chiplet_{idx}"
        if ch.memory is None:
            continue
        try:
            mm, _, _ = estimate_memory_chiplet(ch)
        except ValueError:
            continue
        bw = float(mm.get("bandwidth_tbps", 0.0))
        spec_bw += bw
        std = standard_for(ch.memory.technology)
        if ram_ok and std and bw > 0:
            r = ram.measure(ch.memory.technology, "sequential", write_fraction,
                            seed=config.seed)
            eff_bw += bw * r["efficiency"]
            mem_rows.append({"chiplet": cid, "technology": ch.memory.technology,
                             "spec_bandwidth_tbps": bw,
                             "effective_bandwidth_tbps": bw * r["efficiency"], **r})
            src = f"ramulator:{r['config']['dram']}/{r['config']['timing_preset']}"
            records += [
                AssumptionRecord(parameter=f"{cid}.dram_efficiency", value=r["efficiency"],
                                 unit="fraction", provenance=Provenance.ESTIMATED, source=src,
                                 notes=f"sequential stream, write share {write_fraction:.3f}, "
                                       f"{r['n_requests']} requests, one channel"),
                AssumptionRecord(parameter=f"{cid}.loaded_read_latency_ns",
                                 value=r["avg_read_latency_ns"], unit="ns",
                                 provenance=Provenance.ESTIMATED, source=src)]
        else:
            eff_bw += bw
            not_sim.append(f"{cid}: memory {ch.memory.technology} "
                           + ("(Ramulator unavailable)" if not ram_ok else
                              "(no Ramulator preset)" if not std else "(no bandwidth)"))

    spec_tops = eff_tops = peak_tops = 0.0
    gemms = _workload_gemms(wl)
    for idx, ch in enumerate(plain.chiplets):
        cid = ch.id or f"chiplet_{idx}"
        if ch.compute is None and (ch.type in ("memory", "io") or ch.memory is not None):
            continue
        try:
            cm, _, _ = estimate_compute_chiplet(ch)
        except ValueError:
            continue
        e_tops = float(cm.get("effective_tops", 0.0)) * ch.count
        peak_tops += float(cm.get("peak_tops", 0.0)) * ch.count
        spec_tops += e_tops
        comp = ch.compute
        if (ss_ok and comp is not None and comp.ecore is None and is_systolic(comp.architecture)
                and e_tops > 0):
            macs = int(comp.mac_units) if comp.mac_units else 0
            if not macs and comp.peak_tops and comp.clock_ghz:
                macs = int(float(comp.peak_tops) * 1e12 / (2 * float(comp.clock_ghz) * 1e9))
            if macs >= 16:
                flows = [comp.dataflow] if comp.dataflow else ["ws", "os", "is"]
                runs = {df: ss.simulate_gemms(gemms, macs, float(comp.sram_mb or 8.0), df,
                                              comp.array_dim) for df in flows}
                s = max(runs.values(), key=lambda r: r["utilization"])
                s = {**s, "dataflow_utilization": {df: r["utilization"]
                                                   for df, r in runs.items()},
                     "dataflow_choice": ("specified" if comp.dataflow
                                         else "best of ws/os/is")}
                eff_tops += e_tops * s["utilization"]
                comp_rows.append({"chiplet": cid, "architecture": comp.architecture,
                                  "spec_effective_tops": e_tops,
                                  "effective_tops": e_tops * s["utilization"], **s})
                records.append(AssumptionRecord(
                    parameter=f"{cid}.array_utilization", value=s["utilization"],
                    unit="fraction", provenance=Provenance.ESTIMATED,
                    source=f"scale_sim:{s['array_side']}x{s['array_side']}x{s['n_arrays']}"
                           f"/{s['dataflow']}",
                    notes=f"{len(gemms)} GEMM shapes x layers"
                          + ("; large GEMMs extrapolated by fold count"
                             if s["extrapolated"] else "")))
                continue
        eff_tops += e_tops
        reason = ("SCALE-Sim unavailable" if not ss_ok else
                  "eCore-derived" if comp is not None and comp.ecore is not None else
                  "not a systolic/tensor array" if comp is None or not is_systolic(
                      comp.architecture) else "no mac_units or clock")
        not_sim.append(f"{cid}: compute ({reason})")

    if not mem_rows and not comp_rows:
        raise SpecializedUnavailable(
            "Ramulator/SCALE-Sim cover no chiplet of this design: " + "; ".join(not_sim))

    base = run_analytical_simulation(design, workload, config, design_id)
    if not base.success:
        return base

    total_flops = sum(op.flops for op in ops)
    if total_flops <= 0:
        total_flops = eff_tops * 1e12
    compute_ms = total_flops / (eff_tops * 1e12) * 1e3 if eff_tops > 0 else 0.0
    mem_ms = total_bytes / (eff_bw * 1e12) * 1e3 if eff_bw > 0 else 0.0
    latency_ms = max(compute_ms, mem_ms)
    tokens = wl.batch_size * (1 if is_llm_decode(wl) else wl.sequence_length)
    power = base.metrics.total_power_watts
    m = base.metrics.model_copy(update={
        "aggregate_memory_bandwidth_tbps": eff_bw,
        "effective_compute_tops": eff_tops,
        "estimated_latency_ms": latency_ms,
        "throughput_tokens_per_sec": tokens / (latency_ms / 1e3) if latency_ms > 0 else 0.0,
        "throughput_ops_per_sec": eff_tops * 1e12,
        "ridge_point_ops_per_byte": (eff_tops * 1e12) / (eff_bw * 1e12) if eff_bw > 0 else 0.0,
        "utilization_pct": (min(100.0, total_flops / (peak_tops * 1e12 * latency_ms / 1e3)
                                * 100.0) if peak_tops > 0 and latency_ms > 0 else 0.0),
        "energy_per_inference_mj": power * latency_ms,
    })
    mem_bound = mem_ms >= compute_ms
    b = base.bottleneck.model_copy(update={
        "primary_bottleneck": "memory_bandwidth" if mem_bound else "compute",
        "bound_type": "memory_bandwidth" if mem_bound else "compute",
        "details": [f"Specialized: memory time {mem_ms:.4g} ms (Ramulator-derated bandwidth "
                    f"{eff_bw:.4g} of {spec_bw:.4g} TB/s), compute time {compute_ms:.4g} ms "
                    f"(SCALE-Sim-derated {eff_tops:.4g} of {spec_tops:.4g} effective TOPS)."],
    })
    tools = ([mem_rows[0]["tool"]] if mem_rows else []) + (
        [comp_rows[0]["tool"]] if comp_rows else [])
    disclaimer = (
        "Level 3 specialized: " + " and ".join(tools) + " replace the roofline's ideal "
        "memory bandwidth"
        + (" and array efficiency" if comp_rows else "") + " for the chiplets they cover "
        "(Ramulator: one channel per memory type on a sequential stream; SCALE-Sim: QKV, "
        "attention-output and FFN GEMMs on square arrays (compute.array_dim, default <= 128), "
        "compute.dataflow or the best of ws/os/is, large GEMMs extrapolated by fold count). "
        "Latency is still max(compute, memory); power, area, thermal and cost are "
        "analytical. Not cycle-accurate end to end.")
    return SimulationResult(
        run_id=f"specialized_{design_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        design_id=design_id, design_name=base.design_name, design_version=base.design_version,
        workload_name=wl.name, fidelity=FidelityLevel.SPECIALIZED,
        fidelity_disclaimer=disclaimer, metrics=m, bottleneck=b,
        assumptions=records + list(base.assumptions),
        warnings=[f"Not simulated by Level 3 tools: {s}" for s in not_sim] + list(base.warnings),
        raw_details={**base.raw_details, "specialized": {
            "memory": mem_rows, "compute": comp_rows, "not_simulated": not_sim,
            "analytical": {"latency_ms": base.metrics.estimated_latency_ms,
                           "throughput_tokens_per_sec": base.metrics.throughput_tokens_per_sec,
                           "aggregate_memory_bandwidth_tbps":
                               base.metrics.aggregate_memory_bandwidth_tbps,
                           "effective_compute_tops": base.metrics.effective_compute_tops}}},
        success=True)
