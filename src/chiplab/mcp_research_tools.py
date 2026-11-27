"""MCP tools for the Project Brain, review gate, PIM lab, research loop, genealogy, DFT and
technology plugins.

Deliberately absent: running arbitrary code, loading plugins from files, and the agent
modification sandbox (propose/run/approve/apply are Python-only; MCP sees a read-only list).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from chiplab import dft, sandbox
from chiplab.brain import Brain, ingest_all
from chiplab.pim import builder, dram, dram_designer, sram
from chiplab.pim.workload import compile_workload
from chiplab.plugins import technology as tech
from chiplab.science import falsify, fidelity, research
from chiplab.science.genealogy import Genealogy


@lru_cache(maxsize=4)
def _brain_at(path: str) -> Brain:
    return Brain(path or None)


def _brain() -> Brain:
    return _brain_at(os.environ.get("CHIPLAB_BRAIN_DB", ""))


def _gen() -> Genealogy:
    return Genealogy(_brain())


def _ledger() -> Any:
    from chiplab.experiment_db.ledger import ExperimentLedger, default_db_path
    return ExperimentLedger(default_db_path())


def _err(e: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(e)}


_ERRS = (ValueError, KeyError, RuntimeError, TypeError)


# ---------------------------------------------------------------- brain
def brain_recall(question: str, per_kind: int = 4) -> dict[str, Any]:
    """Recall what ChipLab knows about a question: relevant project facts, architectures,
    past experiments, science (cited papers) and tools, plus known limitations and rejected
    ideas, which are always listed separately. Call this before starting new work."""
    return {"ok": True} | _brain().recall(question, per_kind=max(1, min(per_kind, 10)))


def brain_remember(kind: str, title: str, body: str = "", tags: list[str] | None = None,
                   status: str = "active", source: str = "") -> dict[str, Any]:
    """Store one memory. kind: project|architecture|experiment|science|tool. status:
    active|decision|rejected|limitation|superseded. Science entries need a source."""
    if kind == "science" and not source.strip():
        return {"ok": False, "error": "science memories need a source (paper, DOI or URL)"}
    try:
        return {"ok": True, "id": _brain().remember(kind, title, body, tags=tags or (),
                                                    status=status, source=source or "mcp")}
    except _ERRS as e:
        return _err(e)


def brain_decide(title: str, rationale: str, rejected: bool = False,
                 evidence: list[str] | None = None, tags: list[str] | None = None
                 ) -> dict[str, Any]:
    """Record a design decision (or a rejected idea, with rejected=true) and why."""
    try:
        return {"ok": True, "id": _brain().decide(title, rationale, rejected=rejected,
                                                  evidence=evidence or (), tags=tags or ())}
    except _ERRS as e:
        return _err(e)


def brain_ingest() -> dict[str, Any]:
    """Refresh the brain from the ledger, learning loop, models, tools, component library,
    IMC lab and docs (idempotent), and return memory counts."""
    return {"ok": True, "ingested": ingest_all(_brain()), "stats": _brain().stats()}


def brain_stats() -> dict[str, Any]:
    """Count memories by kind and status."""
    return {"ok": True} | _brain().stats()


# ---------------------------------------------------------------- review
def science_review_run(run_id: str) -> dict[str, Any]:
    """Run the scientific review gate on a stored ledger run. The verdict (rejected,
    unverified, plausible, validated) ignores whatever status the run claimed."""
    from chiplab.science import review_ledger_run
    runs = [r for r in _ledger().query_runs(include_duplicates=True, limit=100000)
            if r.get("run_id") == run_id]
    if not runs:
        return {"ok": False, "error": f"no run {run_id}"}
    return {"ok": True} | review_ledger_run(runs[0]).as_dict()


# ---------------------------------------------------------------- PIM
def pim_sram(action: str, spec: dict[str, Any], op_mix: dict[str, int] | None = None,
             n_ops: int = 40, seed: int = 0) -> dict[str, Any]:
    """SRAM near-memory compute. action: "predict" (flops/area estimate), "simulate" (RTL vs
    golden model in iverilog), "synthesize" (yosys + IHP liberty), "compare" (energy and data
    movement vs a CPU doing the same ops). spec: rows 2-32, lanes 1-8, lane_bits 2-16."""
    try:
        if action == "predict":
            return {"ok": True, "prediction": sram.predict(spec)}
        if action == "simulate":
            return {"ok": True} | sram.simulate(spec, n_ops=max(1, min(n_ops, 500)), seed=seed)
        if action == "synthesize":
            return {"ok": True} | sram.synthesize(spec)
        if action == "compare":
            return {"ok": True} | sram.compare_with_cpu(spec, op_mix)
        return {"ok": False, "error": "action must be predict, simulate, synthesize or compare"}
    except _ERRS as e:
        return _err(e)


def pim_dram(action: str, op: str = "and", trials: int = 20000, seed: int = 0,
             overrides: dict[str, Any] | None = None, cell_volts: list[float] | None = None,
             workload: str = "bitwise_and", n_bytes: int = 64 * 2**20,
             ops_per_second: float = 1e6) -> dict[str, Any]:
    """Commodity-DRAM PIM (1T1C charge sharing, Ambit-style triple-row activation, RowClone).
    action: "params" (cited cell/bitline parameters), "margins", "monte_carlo" (bit error
    rate with variation, noise, retention; op: copy|and|or|not|xor), "timing_energy",
    "spice_check" (ngspice cross-check of charge sharing for cell_volts), "read_disturb",
    "benchmark" (compare energy model with published Ambit/RowClone gains),
    "compare" (in-array vs near-bank vs CPU for workload: bitwise_and, bitwise_xor, copy,
    vector_add_int8, dot_int8), "report" (full designer report)."""
    try:
        if action == "params":
            return {"ok": True, "params": dram.PARAMS}
        if action == "margins":
            return {"ok": True} | dram.margins(dram.params(overrides))
        if action == "monte_carlo":
            return {"ok": True} | dram.monte_carlo(op, max(100, min(trials, 200_000)), seed,
                                                   overrides)
        if action == "timing_energy":
            return {"ok": True} | dram.timing_energy(op, overrides)
        if action == "spice_check":
            return {"ok": True} | dram.spice_check(cell_volts or [1.2, 1.2, 0.0], overrides)
        if action == "read_disturb":
            return {"ok": True} | dram.read_disturb(ops_per_second, op)
        if action == "benchmark":
            return {"ok": True} | dram.benchmark_review(op)
        if action == "compare":
            return {"ok": True} | dram_designer.compare(workload, n_bytes)
        if action == "report":
            return {"ok": True} | dram_designer.designer_report()
        return {"ok": False, "error": "unknown action"}
    except _ERRS as e:
        return _err(e)


def pim_build(memory: str, ops: list[str], config: dict[str, Any] | None = None,
              seed: int = 0) -> dict[str, Any]:
    """Build a processing-in-memory block step by step (spec, RTL, simulation, synthesis,
    review, Tiny Tapeout export). memory: sram (buildable on a digital shuttle) or dram,
    memristor, ferrofluid (returns why they cannot be built on a digital shuttle). Nothing is
    submitted anywhere."""
    try:
        out = builder.build(memory, ops, config, None, seed)
    except _ERRS as e:
        return _err(e)
    for st in out.get("stages", []):
        for k in ("rtl", "wrapper"):
            if isinstance(st.get(k), str) and len(st[k]) > 400:
                st[k] = f"<{len(st[k].splitlines())} lines>"
    return {"ok": True} | out


def workload_compile(spec: dict[str, Any], objective: str = "energy_j",
                     min_precision_bits: float | None = None) -> dict[str, Any]:
    """Map a workload (kind: matmul, attention, conv2d, reduction, sparse_mv, graph_bfs,
    bitwise, custom) to CPU, GPU, SRAM near-memory, DRAM in-array, DRAM near-bank and analog
    crossbar; rank by objective (energy_j, latency_s, area_mm2, data_movement_bytes).
    Analytical fidelity: use it to pick what to simulate in detail, not as a result."""
    try:
        return {"ok": True} | compile_workload(spec, min_precision_bits, objective)
    except _ERRS as e:
        return _err(e)


def fidelity_screen_sram(candidates: list[dict[str, Any]], objective: str = "area",
                         keep_at_least: int = 1) -> dict[str, Any]:
    """Screen 1-12 SRAM near-memory specs through a fidelity ladder (analytical -> RTL sim ->
    synthesis), pruning candidates whose optimistic bound cannot win. Reports where a cheap
    level disagreed with a more accurate one beyond its stated uncertainty.
    objective: "area" (cell area) or "energy_ratio" (NMC energy / CPU energy)."""
    try:
        if objective not in ("area", "energy_ratio"):
            return {"ok": False, "error": "objective must be area or energy_ratio"}
        cands = [(f"c{i}", c) for i, c in enumerate(candidates[:12])]
        return {"ok": True} | fidelity.run(cands, fidelity.sram_nmc_ladder(objective),
                                           keep_at_least=keep_at_least)
    except _ERRS as e:
        return _err(e)


# ---------------------------------------------------------------- research
def research_investigate(question: str, technology: str, workload: dict[str, Any],
                         metric: str = "energy_j", direction: str = "decrease",
                         min_effect: float = 0.2, control: str = "cpu",
                         parent_architecture: str | None = None) -> dict[str, Any]:
    """One research pass: recall memory, write a falsification plan (before results), evaluate
    with the workload compiler, run the review gate, judge against the plan, and record the
    result in the brain and ledger (fidelity=analytical). Usually ends "inconclusive" with
    the next experiment at the fidelity the claim needs."""
    try:
        return {"ok": True} | research.investigate(
            question, technology, workload, metric=metric, direction=direction,
            min_effect=min_effect, control=control, brain=_brain(), ledger=_ledger(),
            genealogy=_gen() if parent_architecture else None,
            parent_architecture=parent_architecture, logged_by="mcp:research_investigate")
    except _ERRS as e:
        return _err(e)


def falsification_plan(hypothesis: str, technology: str, metric: str,
                       direction: str = "decrease", min_effect: float = 0.2,
                       control: str | None = None, workload: str = "") -> dict[str, Any]:
    """Write down, before any result, what would refute a hypothesis: predictions, control,
    failure conditions, required evidence fidelity, and reasons it might fail."""
    try:
        return {"ok": True} | falsify.plan(hypothesis, technology, metric,
                                           direction=direction, min_effect=min_effect,
                                           control=control, workload=workload, brain=_brain())
    except _ERRS as e:
        return _err(e)


def falsification_judge(plan: dict[str, Any], proposed: float, control: float, fidelity_: str,
                        ratio_ci95: list[float] | None = None,
                        review_verdict: str | None = None) -> dict[str, Any]:
    """Settle a falsification plan with results: supported, refuted or inconclusive. Evidence
    below the plan's required fidelity is always inconclusive."""
    try:
        ci = (ratio_ci95[0], ratio_ci95[1]) if ratio_ci95 and len(ratio_ci95) == 2 else None
        return {"ok": True} | falsify.judge(plan, proposed, control, fidelity=fidelity_,
                                            ratio_ci95=ci, review_verdict=review_verdict)
    except _ERRS as e:
        return _err(e)


def architecture_genealogy(action: str, name: str = "", spec: dict[str, Any] | None = None,
                           parent: str = "", why: str = "",
                           improved: dict[str, Any] | None = None,
                           worsened: dict[str, Any] | None = None,
                           evidence: list[str] | None = None, reason: str = "",
                           technology: str = "") -> dict[str, Any]:
    """Architecture family tree. action: "create" (root: name, spec), "derive" (parent, name,
    spec, why, improved/worsened, evidence), "reject" (name, reason), "lineage" (name),
    "tree", "novelty" (spec: is it a repeat of something tried or rejected?)."""
    g = _gen()
    try:
        if action == "create":
            return {"ok": True, "id": g.create(name, spec or {}, technology=technology,
                                               why=why or "root architecture")}
        if action == "derive":
            return {"ok": True} | g.derive(parent, name, spec or {}, why=why, improved=improved,
                                           worsened=worsened, evidence=evidence or (),
                                           technology=technology or None)
        if action == "reject":
            g.reject(name, reason, evidence or ())
            return {"ok": True}
        if action == "lineage":
            return {"ok": True, "lineage": g.lineage(name)}
        if action == "tree":
            return {"ok": True} | g.tree()
        if action == "novelty":
            return {"ok": True} | g.check_novelty(spec or {})
        return {"ok": False, "error": "unknown action"}
    except _ERRS as e:
        return _err(e)


# ---------------------------------------------------------------- DFT
def dft_analyze(action: str, march: str = "March C-", words: int = 16, width: int = 8,
                fault: list[int] | None = None, n_flops: int = 0, base_area_um2: float = 0.0,
                die_area_mm2: float = 10.0, defect_density_per_cm2: float = 0.1,
                state_bits: int = 0, output_bits: int = 8, input_bits: int = 8,
                spec: dict[str, Any] | None = None, trials: int = 500) -> dict[str, Any]:
    """Design for test. action: "coverage" (march test fault coverage: MATS+, March X,
    March C-), "mbist" (generate a memory BIST and simulate it; fault=[word, bit, value]
    injects a stuck-at), "scan" (scan-chain area overhead from the IHP liberty), "yield"
    (Poisson / Murphy / negative binomial), "observability", "seu" (single-event upsets in the
    SRAM near-memory block, with parity detection)."""
    try:
        if action == "coverage":
            return {"ok": True} | dft.coverage(march, max(4, min(words, 256)))
        if action == "mbist":
            f = (fault[0], fault[1], fault[2]) if fault and len(fault) == 3 else None
            return {"ok": True} | dft.mbist_check(max(2, min(words, 256)),
                                                  max(1, min(width, 32)), f)
        if action == "scan":
            return {"ok": True} | dft.scan_overhead(n_flops, base_area_um2)
        if action == "yield":
            return {"ok": True} | dft.yield_models(die_area_mm2, defect_density_per_cm2)
        if action == "observability":
            return {"ok": True} | dft.observability(state_bits, output_bits, input_bits)
        if action == "seu":
            return {"ok": True} | dft.seu_injection(spec, max(10, min(trials, 5000)))
        return {"ok": False, "error": "unknown action"}
    except (*_ERRS, FileNotFoundError) as e:
        return _err(e)


# ---------------------------------------------------------------- plugins / sandbox
def technology_plugins(action: str = "describe", name: str = "",
                       inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Registered technology plugins. action: "describe" (all plugins, their equations,
    properties with sources, ports and manufacturing limits), "simulate" (name, inputs).
    Plugins are registered from Python code only."""
    try:
        if action == "describe":
            return {"ok": True, "plugins": tech.describe()}
        if action == "simulate":
            return {"ok": True} | tech.simulate(name, inputs)
        return {"ok": False, "error": "action must be describe or simulate"}
    except _ERRS as e:
        return _err(e)


def sandbox_proposals() -> dict[str, Any]:
    """List agent-proposed code changes waiting in the modification sandbox (read-only).
    Proposals are tested in isolation and need a named human's approval; protected physics
    files need an explicit physics review. Nothing can be approved through MCP."""
    return {"ok": True, "proposals": sandbox.listing()}


READ_ONLY = [brain_recall, brain_stats, science_review_run, workload_compile,
             falsification_plan, falsification_judge, technology_plugins, sandbox_proposals]
WRITES = [brain_remember, brain_decide, brain_ingest, pim_sram, pim_dram, pim_build,
          fidelity_screen_sram, research_investigate, architecture_genealogy, dft_analyze]


def register(mcp: Any, read_only: Any, writes: Any) -> None:
    tech.register(tech.example_resistive_plugin(), replace=True)
    for fn in READ_ONLY:
        mcp.tool(annotations=read_only)(fn)
    for fn in WRITES:
        mcp.tool(annotations=writes)(fn)
