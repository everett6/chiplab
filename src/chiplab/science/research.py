"""Research loop: question -> memory -> falsification plan -> evaluation -> review -> record.

`investigate()` runs one honest pass with the analytical workload compiler. Because analytical
evidence is below what most claims need (see falsify.REQUIRED_FIDELITY), the usual outcome is
"inconclusive" plus the concrete next experiment at the required fidelity. That is the point:
the loop never turns a number it generated itself into a conclusion.
"""

from __future__ import annotations

from typing import Any

from chiplab.science import falsify
from chiplab.science.review import review

IMPL_FOR_TECH = {"dram": ("dram_in_array", "dram_near_bank"), "sram": ("sram_nmc",),
                 "analog": ("analog_crossbar",), "memristor": ("analog_crossbar",),
                 "gpu": ("gpu",), "cpu": ("cpu",)}
METRICS = ("energy_j", "latency_s", "area_mm2", "data_movement_bytes")
NEXT_STEP = {
    "circuit_spice": "run the circuit at SPICE level (pim_dram spice_check for DRAM charge "
                     "sharing; chiplab_spice_netlist_run for others)",
    "rtl_sim": "generate RTL and simulate it (pim_sram simulate / pim_build)",
    "synthesis": "synthesise with yosys + a real liberty (pim_build, stage synthesis)",
    "measured": "build it and measure (hil fpga_plan, learning loop)",
    "published": "compare with a published, cited result",
    "behavioral": "run a behavioural model with variation (pim_dram monte_carlo)",
}


def _pick(impls: dict[str, Any], tech: str) -> str | None:
    for k in IMPL_FOR_TECH.get(tech, ()):
        if impls.get(k, {}).get("supported"):
            return k
    return None


def investigate(question: str, technology: str, workload: dict[str, Any], *,
                metric: str = "energy_j", direction: str = "decrease", min_effect: float = 0.2,
                control: str = "cpu", brain: Any = None, ledger: Any = None,
                genealogy: Any = None, parent_architecture: str | None = None,
                logged_by: str = "chiplab.research") -> dict[str, Any]:
    from chiplab.pim.workload import compile_workload

    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    tech = technology.lower()
    memory = brain.recall(question) if brain is not None else None
    plan = falsify.plan(question, tech, metric, direction=direction, min_effect=min_effect,
                        control=f"{control} (workload compiler)", workload=str(workload),
                        brain=brain)
    comp = compile_workload(workload, objective=metric)
    impls = comp["implementations"]
    key = _pick(impls, tech)
    out: dict[str, Any] = {"question": question, "plan": plan, "fidelity": "analytical",
                           "memory": _brief(memory), "caveats": comp.get("caveats", [])}
    if key is None or not impls.get(control, {}).get("supported"):
        out |= {"outcome": "not_evaluable",
                "why": f"no supported implementation for {tech!r} (or control {control!r}) on "
                       f"this workload: {[impls.get(k, {}).get('why') for k in IMPL_FOR_TECH.get(tech, ())]}"}
        return out
    proposed = {m: impls[key][m] for m in METRICS if isinstance(impls[key].get(m), (int, float))}
    numeric = {k: v for k, v in workload.items() if isinstance(v, (int, float))
               and not isinstance(v, bool)}

    def evaluate(x: dict[str, Any]) -> dict[str, Any]:
        r = compile_workload(workload | x, objective=metric)["implementations"][key]
        return {m: r[m] for m in proposed}

    rv = review(f"research:{tech}:{key}", proposed, fidelity="analytical",
                provenance={"tool": "chiplab.pim.workload"}, evaluate=evaluate,
                inputs=numeric).as_dict()
    verdict = falsify.judge(plan, impls[key][metric], impls[control][metric],
                            fidelity="analytical", review_verdict=rv["verdict"])
    need = plan["required_evidence"]["min_fidelity"]
    out |= {"implementation": key, "control": control, "proposed": proposed,
            "control_metrics": {m: impls[control].get(m) for m in proposed},
            "review": {"verdict": rv["verdict"], "failed": rv.get("failed", [])},
            "judgement": verdict,
            "next_experiment": NEXT_STEP.get(need, f"gather {need} evidence")}
    if ledger is not None:
        from chiplab.schemas.design import DesignSpec
        d = DesignSpec(name=f"research_{tech}_{key}", version="0.1",
                       architecture="experimental", chiplets=[],
                       description=question[:200])
        out["ledger"] = ledger.record(d, {"workload": workload, "control": control},
                                      proposed | {"review_verdict": rv["verdict"]},
                                      "analytical", "workload compiler, analytical roofline",
                                      logged_by=logged_by)
    if brain is not None:
        out["memory_id"] = brain.remember(
            "experiment", f"{tech} vs {control}: {metric} ({verdict['outcome']})",
            f"Q: {question}\nratio={verdict.get('ratio')}; review={rv['verdict']}; "
            f"{verdict.get('why', '')}\nnext: {out['next_experiment']}",
            tags=(tech, control, metric, "research"), source="chiplab.science.research",
            data={"plan_id": plan["plan_id"], "workload": workload})
    if genealogy is not None and parent_architecture:
        out["genealogy"] = genealogy.derive(
            parent_architecture, f"{tech}-{key}-{plan['plan_id'][:6]}",
            {"technology": tech, "implementation": key, "workload": workload.get("kind")},
            why=question, improved={metric: verdict.get("ratio")},
            evidence=[out.get("ledger", {}).get("run_id", "")] if ledger else [])
    return out


def _brief(memory: dict[str, Any] | None) -> dict[str, Any] | None:
    if not memory:
        return None
    out = {k: [m["title"] for m in v] for k, v in memory.get("results", {}).items()}
    for k in ("limitations", "rejected"):
        if memory.get(k):
            out[k] = [m["title"] for m in memory[k]]
    return out
